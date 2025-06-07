use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict};
use ndarray::{Array1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};

pub mod distributions;
pub mod splitter;

#[pyfunction]
fn find_best_split(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    min_samples_leaf: usize,
    min_child_weight: f64,
    distribution_spec: &PyDict,
    eta: f64,
    col_idcs: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(Option<usize>, Option<f64>, f64, Option<Py<PyArray1<bool>>>, Option<Py<PyArray1<bool>>>)> {
    let x_array = x.as_array();
    let y_array = y.as_array();

    // Create distribution based on specifications
    let distribution = create_distribution_from_spec(distribution_spec, py)?;
    // Process optional column indices
    let col_indices = col_idcs.as_ref().map(|arr| {
        let indices = arr.as_array();
        let indices_usize: Vec<usize> = indices.iter()
            .filter_map(|&idx| if idx >= 0 { Some(idx as usize) } else { None })
            .collect();
        Array1::from(indices_usize)
    });

    // Call splitter implementation
    let (feat_idx, threshold, loss_reduction, left_indices, right_indices) =
        splitter::find_best_split(
            &x_array,
            &y_array,
            min_samples_leaf,
            min_child_weight,
            &*distribution,
            eta,
            col_indices
        );

    // Convert results back to Python
    let py_left = left_indices.map(|arr| PyArray1::from_array(py, &arr).into());
    let py_right = right_indices.map(|arr| PyArray1::from_array(py, &arr).into());

    Ok((feat_idx, threshold, loss_reduction, py_left, py_right))
}

// Distribution factory with explicit fallback mechanism
fn create_distribution_from_spec(spec: &PyDict, py: Python) -> PyResult<Box<dyn distributions::Distribution>> {
    // Extract the distribution type
    if let Some(Ok(dist_type)) = spec.get_item("dist_type").and_then(|dt| Some(dt.extract::<String>())) {
        // Try to create a native distribution based on type
        match dist_type.as_str() {
        "NormalMuNormal" => {
            // Use map_or to provide a default None if get_item fails
            // and convert the extraction to Option
            let mu_zero = spec.get_item("mu_zero")
                .and_then(|pm| pm.extract::<f64>().ok());
            let sigma_zero = spec.get_item("sigma_zero")
                .and_then(|ps| ps.extract::<f64>().ok());

            if let (Some(mu_zero), Some(sigma_zero)) = (mu_zero, sigma_zero) {
                let spec = distributions::NormalMuNormalSpec {
                    mu_zero,
                    sigma_zero,
                };

                return Ok(Box::new(distributions::NormalMuNormal::new(&spec)));
            }
        },
        "NormalMeanPseudoAlphaSkewNormal" => {
            let mu_zero = spec.get_item("mu_zero")
                .and_then(|m0| m0.extract::<f64>().ok());
            let sigma_zero = spec.get_item("sigma_zero")
                .and_then(|s0| s0.extract::<f64>().ok());
            let alpha_zero = spec.get_item("alpha_zero")
                .and_then(|a0| a0.extract::<f64>().ok());
            let m_alpha = spec.get_item("m_alpha")
                .and_then(|ma| ma.extract::<f64>().ok());
            if let (Some(mu_zero), Some(sigma_zero), Some(alpha_zero), Some(m_alpha)) =
                (mu_zero, sigma_zero, alpha_zero, m_alpha) {
                let spec = distributions::NormalMeanPseudoAlphaSkewNormalSpec {
                    mu_zero,
                    sigma_zero,
                    alpha_zero,
                    m_alpha,
                };

                return Ok(Box::new(distributions::NormalMeanPseudoAlphaSkewNormal::new(&spec)));
            }

        },
        "NormalMeanNormalGammaSkewNormal" => {
            let mu_zero = spec.get_item("mu_zero")
                .and_then(|m0| m0.extract::<f64>().ok());
            let sigma_zero = spec.get_item("sigma_zero")
                .and_then(|s0| s0.extract::<f64>().ok());
            let mu_gamma = spec.get_item("mu_gamma")
                .and_then(|mg| mg.extract::<f64>().ok());
            let sigma_gamma = spec.get_item("sigma_gamma")
                .and_then(|sg| sg.extract::<f64>().ok());
            if let (Some(mu_zero), Some(sigma_zero), Some(mu_gamma), Some(sigma_gamma)) =
                (mu_zero, sigma_zero, mu_gamma, sigma_gamma) {
                let spec = distributions::NormalMeanNormalGammaSkewNormalSpec {
                    mu_zero,
                    sigma_zero,
                    mu_gamma,
                    sigma_gamma,
                };

                return Ok(Box::new(distributions::NormalMeanNormalGammaSkewNormal::new(&spec)));
            }

        },
        _ => {

        }
        }
    }

    // Fallback: use Python distribution via wrapper
    if let Some(py_dist) = spec.get_item("_python_object") {
        return Ok(Box::new(distributions::PythonDistributionWrapper::new(
            py_dist.to_object(py)
        )));
    }

    // If we get here, we couldn't create any distribution
    Err(pyo3::exceptions::PyValueError::new_err(
        "Could not create distribution from spec - missing _python_object fallback"
    ))
}

#[pyfunction]
fn calculate_nll(py: Python<'_>, data: PyReadonlyArray1<f64>, distribution_spec: &PyDict) -> PyResult<f64> {
    let data_array = data.as_array();
    let distribution = create_distribution_from_spec(distribution_spec, py)?;
    Ok(distribution.nll(&data_array))
}

#[pyfunction]
fn generate_thresholds(_py: Python<'_>, data: PyReadonlyArray1<f64>, eta: f64) -> PyResult<Vec<f64>> {
    let data_array = data.as_array();
    let mut values: Vec<f64> = data_array.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_thresholds = (1.0 / eta).ceil() as usize;
    let mut thresholds: Vec<f64> = Vec::with_capacity(n_thresholds);

    // Generate thresholds from quantiles (same code as in find_best_split)
    for i in 0..n_thresholds {
        let q = i as f64 / (n_thresholds as f64);
        let idx = (q * (values.len() as f64)) as usize;
        if idx < values.len() {
            thresholds.push(values[idx]);
        }
    }

    // Deduplicate
    thresholds.dedup();

    // Return midpoints as thresholds
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
fn bdf_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_split, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_nll, m)?)?;
    m.add_function(wrap_pyfunction!(generate_thresholds, m)?)?;
    Ok(())
}
