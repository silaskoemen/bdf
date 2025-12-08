// lib.rs - CORRECTED
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyDict;
use ndarray::Array1;
use numpy::PyReadonlyArray1;

pub mod distributions;
pub mod splitter;
pub mod scoring;

#[pyfunction]
fn find_best_split(
    py: Python<'_>,
    x: numpy::PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    distribution_spec: &PyDict,
    eta: f64,
    col_idcs: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(Option<usize>, Option<f64>, f64, Option<pyo3::Py<numpy::PyArray1<bool>>>, Option<pyo3::Py<numpy::PyArray1<bool>>>, Option<pyo3::Py<PyDict>>, Option<pyo3::Py<PyDict>>)> {
    let x_array = x.as_array();
    let y_array = y.as_array();

    let distribution = create_distribution_from_spec(distribution_spec, py)?;

    let scoring_spec = distributions::ScoringSpec::from_dict(distribution_spec)?;

    let col_indices = col_idcs.as_ref().map(|arr| {
        let indices = arr.as_array();
        Array1::from_iter(
            indices.iter()
                .filter_map(|&idx| if idx >= 0 { Some(idx as usize) } else { None })
        )
    });

    let (feat_idx, threshold, loss_reduction, left_indices, right_indices, left_params, right_params) =
        splitter::find_best_split(
            &x_array,
            &y_array,
            min_samples_leaf,
            min_child_weight,
            &*distribution,
            &scoring_spec,
            eta,
            col_indices
        );

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
        "GammaABLambdaPoisson" => Ok(Box::new(distributions::poisson::GammaABLambdaPoisson::from_spec(spec)?)),
        "GammaMVLambdaPoisson" => Ok(Box::new(distributions::poisson::GammaMVLambdaPoisson::from_spec(spec)?)),
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
fn generate_thresholds(_py: Python<'_>, data: PyReadonlyArray1<f64>, eta: f64) -> PyResult<Vec<f64>> {
    let data_array = data.as_array();
    let mut values: Vec<f64> = data_array.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_thresholds = (1.0 / eta).ceil() as usize;
    let mut thresholds: Vec<f64> = Vec::with_capacity(n_thresholds);

    for i in 0..n_thresholds {
        let q = i as f64 / (n_thresholds as f64);
        let idx = (q * (values.len() as f64)) as usize;
        if idx < values.len() {
            thresholds.push(values[idx]);
        }
    }

    thresholds.dedup();

    let mut midpoints = Vec::new();
    for i in 1..thresholds.len() {
        if thresholds[i] == thresholds[i-1] {
            continue;
        }
        midpoints.push((thresholds[i] + thresholds[i-1]) / 2.0);
    }

    Ok(midpoints)
}

#[pymodule]
fn bdf_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_split, m)?)?;
    m.add_function(wrap_pyfunction!(generate_thresholds, m)?)?;
    Ok(())
}
