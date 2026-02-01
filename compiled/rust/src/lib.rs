#![allow(unsafe_op_in_unsafe_fn)]  // 2024 edition stricter rules about pyfunctions, silence for now
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyDict;
use ndarray::Array1;
use numpy::PyReadonlyArray1;

use crate::distributions::DistributionPrimitives;

pub mod distributions;
pub mod splitter;
pub mod scoring;

#[pyfunction(signature = (
    x,
    y,
    min_samples_leaf,
    min_child_weight,
    distribution_spec,
    eta,
    gamma,
    col_idcs=None,
    split_gain_method="map"
))]
fn find_best_split(
    py: Python<'_>,
    x: numpy::PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    distribution_spec: &PyDict,
    eta: f64,
    gamma: f64,  // Parameter for split cost complexity penalty
    col_idcs: Option<PyReadonlyArray1<i64>>,
    split_gain_method: &str
) -> PyResult<(Option<usize>, Option<f64>, f64, Option<pyo3::Py<numpy::PyArray1<bool>>>, Option<pyo3::Py<numpy::PyArray1<bool>>>, Option<pyo3::Py<PyDict>>, Option<pyo3::Py<PyDict>>)> {
    let x_array = x.as_array();
    let y_array = y.as_array();

    // Decide routing before constructing a trait object.
    let dist_type: String = distribution_spec.get_item("dist_type")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'dist_type'"))?
        .extract()?;

    let col_indices = col_idcs.as_ref().map(|arr| {
        let indices = arr.as_array();
        Array1::from_iter(
            indices.iter()
                .filter_map(|&idx| if idx >= 0 { Some(idx as usize) } else { None })
        )
    });

    let (feat_idx, threshold, loss_reduction, left_indices, right_indices, left_params, right_params) = if dist_type.as_str() == "KDE" || dist_type.as_str() == "BayesianKDE" {
        // Parse KDE-specific config (defaults keep current behavior unless user opts in)
        let kernel: String = distribution_spec.get_item("kernel")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'kernel'"))?
            .extract()?;
        let kernel = match kernel.as_str() {
            "gaussian" => distributions::kde::KernelType::Gaussian,
            "epanechnikov" => distributions::kde::KernelType::Epanechnikov,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown KDE kernel")),
        };

        let bw_item = distribution_spec.get_item("bandwidth")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'bandwidth'"))?;
        let bandwidth_rule = if let Ok(h) = bw_item.extract::<f64>() {
            distributions::kde::BandwidthRule::Fixed(h)
        } else if let Ok(s) = bw_item.extract::<String>() {
            match s.as_str() {
                "scott" => distributions::kde::BandwidthRule::Scott,
                "silverman" => distributions::kde::BandwidthRule::Silverman,
                _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown bandwidth rule")),
            }
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid bandwidth type"));
        };

        let min_bandwidth: f64 = distribution_spec.get_item("min_bandwidth")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'min_bandwidth'"))?
            .extract()?;

        // Optional knobs (backwards compatible)
        let backend: String = distribution_spec.get_item("kde_backend")
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "pairwise".to_string());
        let backend = match backend.as_str() {
            "pairwise" => splitter::KdeBackend::Pairwise,
            "fft" => splitter::KdeBackend::Fft,
            "switch" => splitter::KdeBackend::Switch,
            _ => splitter::KdeBackend::Pairwise,
        };

        let bandwidth_policy: String = distribution_spec.get_item("bandwidth_policy")
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "parent".to_string());
        let bandwidth_policy = match bandwidth_policy.as_str() {
            "parent" => splitter::BandwidthPolicy::Parent,
            "per_split" => splitter::BandwidthPolicy::PerSplit,
            _ => splitter::BandwidthPolicy::Parent,
        };

        let prior_h = distribution_spec.get_item("prior_h").and_then(|v| v.extract::<f64>().ok());
        let m_h = distribution_spec.get_item("m_h").and_then(|v| v.extract::<f64>().ok());

        let use_compact_support: bool = distribution_spec.get_item("use_compact_support")
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        // Optional FFT grid settings (computed once in Python regressor if desired)
        let fft_grid_min = distribution_spec.get_item("fft_grid_min")
            .and_then(|v| v.extract::<f64>().ok());
        let fft_grid_max = distribution_spec.get_item("fft_grid_max")
            .and_then(|v| v.extract::<f64>().ok());
        let fft_grid_points = distribution_spec.get_item("fft_grid_points")
            .and_then(|v| v.extract::<usize>().ok());

        // Auto-switch threshold: compare n*n to this value
        let kde_backend_switch_size = distribution_spec.get_item("kde_backend_switch_size")
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(2_000_000usize);
        let score_correction: Option<String> = distribution_spec.get_item("score_correction")
            .and_then(|v| v.extract::<Option<String>>().ok())
            .flatten();

        // Top-k refinement: when using parent bandwidth, refine top-k candidates with child bandwidths
        let parent_bw_refine_top_k: usize = distribution_spec.get_item("parent_bw_refine_top_k")
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(1)
            .max(1);

        let kde_config = splitter::KdeSplitConfig {
            kernel,
            bandwidth_rule,
            min_bandwidth,
            backend,
            bandwidth_policy,
            kde_backend_switch_size,
            prior_h,
            m_h,
            use_compact_support,
            fft_grid_min,
            fft_grid_max,
            fft_grid_points,
            score_correction,
            parent_bw_refine_top_k,
        };

        splitter::find_best_split_kde(
            &x_array,
            &y_array,
            min_samples_leaf,
            min_child_weight,
            &kde_config,
            eta,
            gamma,
            col_indices,
            split_gain_method,
        )
    } else {
        let distribution = create_distribution_from_spec(distribution_spec, py)?;
        let scoring_spec = distributions::ScoringSpec::from_dict(distribution_spec)?;
        splitter::find_best_split(
            &x_array,
            &y_array,
            min_samples_leaf,
            min_child_weight,
            &*distribution,
            &scoring_spec,
            eta,
            gamma,
            col_indices,
            split_gain_method,
        )
    };

    let py_left = left_indices.map(|arr| numpy::PyArray1::from_array(py, &arr).into());
    let py_right = right_indices.map(|arr| numpy::PyArray1::from_array(py, &arr).into());

    let py_left_params = left_params.map(|params| {
        let dict = PyDict::new(py);
        for (k, v) in params {
            dict.set_item(k, v).unwrap();
        }
        dict.into()
    });
    let py_right_params = right_params.map(|params| {
        let dict = PyDict::new(py);
        for (k, v) in params {
            dict.set_item(k, v).unwrap();
        }
        dict.into()
    });

    Ok((feat_idx, threshold, loss_reduction, py_left, py_right, py_left_params, py_right_params))
}

fn create_distribution_from_spec(spec: &PyDict, py: Python)
    -> PyResult<Box<dyn distributions::DistributionPrimitives>> {

    let dist_type: String = spec.get_item("dist_type")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'dist_type'"))?
        .extract()?;

    match dist_type.as_str() {
        "NormalMuNormal" => Ok(Box::new(distributions::normal::NormalMuNormal::from_spec(spec)?)),
        "NormalMuInvGammaSigmaNormal" => Ok(Box::new(distributions::normal::NormalMuInvGammaSigmaNormal::from_spec(spec)?)),
        "NormalMeanPseudoAlphaSkewNormal" => Ok(Box::new(distributions::skew_normal::NormalMeanPseudoAlphaSkewNormal::from_spec(spec)?)),
        "NormalMeanNormalGammaSkewNormal" => Ok(Box::new(distributions::skew_normal::NormalMeanNormalGammaSkewNormal::from_spec(spec)?)),
        "NormalXiNormalAlphaSkewNormalMAP" => Ok(Box::new(distributions::skew_normal::NormalXiNormalAlphaSkewNormalMAP::from_spec(spec)?)),
        "GammaABLambdaPoisson" => Ok(Box::new(distributions::poisson::GammaABLambdaPoisson::from_spec(spec)?)),
        "GammaMVLambdaPoisson" => Ok(Box::new(distributions::poisson::GammaMVLambdaPoisson::from_spec(spec)?)),
        "GammaABLambdaExponential" => Ok(Box::new(distributions::exponential::GammaABLambdaExponential::from_spec(spec)?)),
        "GammaMVLambdaExponential" => Ok(Box::new(distributions::exponential::GammaMVLambdaExponential::from_spec(spec)?)),
        "BetaABBernoulli" => Ok(Box::new(distributions::bernoulli::BetaABBernoulli::from_spec(spec)?)),
        "BetaMVBernoulli" => Ok(Box::new(distributions::bernoulli::BetaMVBernoulli::from_spec(spec)?)),
        "KDE" => Ok(Box::new(distributions::kde::Kde::from_spec(spec)?)),
        "BayesianKDE" => Ok(Box::new(distributions::kde::BayesianKde::from_spec(spec)?)),
        _ => {
            let py_obj: &pyo3::PyAny = spec.get_item("_python_object")
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing '_python_object'"))?;
            let extracted: PyObject = py_obj.extract()?;
            Ok(Box::new(distributions::python_callback::PythonCallbackDistribution::new(
                extracted.to_object(py)
            )))
        }
    }
}

#[pyfunction]
fn calculate_nll(
    py: Python<'_>,
    data: PyReadonlyArray1<f64>,
    distribution_spec: &PyDict,
) -> PyResult<f64> {
    let data_array = data.as_array();

    let dist_type: String = distribution_spec.get_item("dist_type")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'dist_type'"))?
        .extract()?;

    if dist_type.as_str() == "KDE" {
        let kde = distributions::kde::Kde::from_spec(distribution_spec)?;
        Ok(kde.nll(&data_array, false))
    } else if dist_type.as_str() == "BayesianKDE" {
        let kde = distributions::kde::BayesianKde::from_spec(distribution_spec)?;
        Ok(kde.nll(&data_array, false))
    } else {
        let distribution = create_distribution_from_spec(distribution_spec, py)?;
        let scoring_spec = distributions::ScoringSpec::from_dict(distribution_spec)?;
        Ok(distribution.nll(&data_array, scoring_spec.use_posterior_predictive))
    }
}

#[pymodule]
fn bdf_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_split, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_nll, m)?)?;
    Ok(())
}
