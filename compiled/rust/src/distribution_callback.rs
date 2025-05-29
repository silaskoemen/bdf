// In distribution.rs
use pyo3::prelude::*;
use ndarray::{ArrayView1};
use numpy::ToPyArray;

// Define trait for distributions
pub trait Distribution {
    fn nll(&self, y: &ArrayView1<f64>) -> f64;
}

// Wrapper for Python distribution objects
pub struct PythonDistributionWrapper {
    py_dist: PyObject,
}

impl PythonDistributionWrapper {
    pub fn new(py_dist: PyObject) -> Self {
        Self { py_dist }
    }
}

impl Distribution for PythonDistributionWrapper {
    fn nll(&self, y: &ArrayView1<f64>) -> f64 {
        Python::with_gil(|py| {
            let y_array = y.to_pyarray(py);
            let result = self.py_dist
                .call_method1(py, "nll", (y_array,))
                .expect("Failed to call nll method");

            result.extract(py).expect("Failed to extract NLL value")
        })
    }
}
