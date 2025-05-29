use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict};
use ndarray::{Array1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};

pub mod distribution;
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
fn create_distribution_from_spec(spec: &PyDict, py: Python) -> PyResult<Box<dyn distribution::Distribution>> {
    // Extract the distribution type
    if let Some(Ok(dist_type)) = spec.get_item("dist_type").and_then(|dt| Some(dt.extract::<String>())) {
        // Try to create a native distribution based on type
        match dist_type.as_str() {
        "NormalNormal" => {
            // Use map_or to provide a default None if get_item fails
            // and convert the extraction to Option
            let prior_mean = spec.get_item("prior_mean")
                .and_then(|pm| pm.extract::<f64>().ok());
            let prior_std = spec.get_item("prior_std")
                .and_then(|ps| ps.extract::<f64>().ok());

            if let (Some(prior_mean), Some(prior_std)) = (prior_mean, prior_std) {
                let spec = distribution::NormalNormalSpec {
                    prior_mean,
                    prior_std,
                };

                return Ok(Box::new(distribution::NormalNormal::new(&spec)));
            }
        },
        "NormalEBSkewNormal" => {
            let prior_mu = spec.get_item("prior_mu")
                .and_then(|pm| pm.extract::<f64>().ok());
            let prior_sigma = spec.get_item("prior_sigma")
                .and_then(|ps| ps.extract::<f64>().ok());
            let prior_mean_alpha = spec.get_item("prior_mean_alpha")
                .and_then(|pma| pma.extract::<f64>().ok());
            let prior_m_alpha = spec.get_item("prior_m_alpha")
                .and_then(|pma| pma.extract::<f64>().ok());
            if let (Some(prior_mu), Some(prior_sigma), Some(prior_mean_alpha), Some(prior_m_alpha)) =
                (prior_mu, prior_sigma, prior_mean_alpha, prior_m_alpha) {
                let spec = distribution::NormalEBSkewNormalSpec {
                    prior_mu,
                    prior_sigma,
                    prior_mean_alpha,
                    prior_m_alpha,
                };

                return Ok(Box::new(distribution::NormalEBSkewNormal::new(&spec)));
            }

        },
        _ => {

        }
        }
    }

    // Fallback: use Python distribution via wrapper
    if let Some(py_dist) = spec.get_item("_python_object") {
        return Ok(Box::new(distribution::PythonDistributionWrapper::new(
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
