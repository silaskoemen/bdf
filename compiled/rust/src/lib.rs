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
    distribution_spec: &PyDict,  // Full spec with scoring config
    eta: f64,
    col_idcs: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(Option<usize>, Option<f64>, f64, Option<Py<PyArray1<bool>>>, Option<Py<PyArray1<bool>>>)> {

    let x_array = x.as_array();
    let y_array = y.as_array();

    // Create distribution
    let distribution = create_distribution_from_spec(distribution_spec, py)?;

    // Parse scoring spec
    let scoring_spec = distributions::ScoringSpec::from_dict(distribution_spec)?;

    // Process column indices
    let col_indices = col_idcs.as_ref().map(|arr| {
        let indices = arr.as_array();
        Array1::from_iter(
            indices.iter()
                .filter_map(|&idx| if idx >= 0 { Some(idx as usize) } else { None })
        )
    });

    // Call splitter (now with ScoringSpec)
    let (feat_idx, threshold, loss_reduction, left_indices, right_indices) =
        splitter::find_best_split(
            &x_array,
            &y_array,
            min_samples_leaf,
            min_child_weight,
            &*distribution,
            &scoring_spec,  // Pass scoring spec
            eta,
            col_indices
        );

    // Convert to Python
    let py_left = left_indices.map(|arr| PyArray1::from_array(py, &arr).into());
    let py_right = right_indices.map(|arr| PyArray1::from_array(py, &arr).into());

    Ok((feat_idx, threshold, loss_reduction, py_left, py_right))
}

// Distribution factory with explicit fallback mechanism
fn create_distribution_from_spec(spec: &PyDict, py: Python)
    -> PyResult<Box<dyn distributions::DistributionPrimitives>> {

    let dist_type = spec.get_item("dist_type")?.extract::<String>()?;

    match dist_type.as_str() {
        "NormalMuNormal" => {
            Ok(Box::new(distributions::normal::NormalMuNormal::from_spec(spec)?))
        }

        "GammaABLambdaExponential" => {
            Ok(Box::new(distributions::exponential::GammaABLambdaExponential::from_spec(spec)?))
        }

        // Fallback to Python
        _ => {
            let py_obj = spec.get_item("_python_object")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                    "Missing _python_object for unsupported distribution"
                ))?;
            Ok(Box::new(distributions::python_callback::PythonCallbackDistribution::new(
                py_obj.to_object(py)
            )))
        }
    }
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
