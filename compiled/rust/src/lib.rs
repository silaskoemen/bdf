use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};

pub mod distribution;
pub mod splitter;

#[pyfunction]
fn find_best_split_rust(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    min_samples_leaf: usize, 
    min_child_weight: f64,
    distribution: PyObject,
    eta: f64,
    col_idcs: Option<PyReadonlyArray1<usize>>,
) -> PyResult<(Option<usize>, Option<f64>, f64, Option<Py<PyArray1<bool>>>, Option<Py<PyArray1<bool>>>)> {
    let x_array = x.as_array();
    let y_array = y.as_array();
    
    // Create a wrapper for the Python distribution object
    let dist_wrapper = distribution::PythonDistributionWrapper::new(distribution);
    
    // Process optional column indices
    let col_indices = col_idcs.as_ref().map(|arr| arr.as_array());
    
    // Call our optimized implementation
    let (feat_idx, threshold, loss_reduction, left_indices, right_indices) = 
        splitter::find_best_split_rust(
            &x_array, 
            &y_array,
            min_samples_leaf,
            min_child_weight,
            &dist_wrapper,
            eta,
            col_indices
        );
    
    // Convert the boolean arrays back to Python
    let py_left = left_indices.map(|arr| PyArray1::from_array(py, &arr).into());
    let py_right = right_indices.map(|arr| PyArray1::from_array(py, &arr).into());
    
    Ok((feat_idx, threshold, loss_reduction, py_left, py_right))
}

#[pymodule]
fn bdf_optimized(_py: Python, m: &PyModule) -> PyResult<()> {
    ///m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_split_rust, m)?)?;
    Ok(())
}