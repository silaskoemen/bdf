use ndarray::ArrayView1;
use pyo3::prelude::*;
use numpy::ToPyArray;

use crate::distributions::Distribution;


// Python wrapper implementation
pub struct PythonDistributionWrapper {
    pub py_dist: PyObject,
}

unsafe impl Send for PythonDistributionWrapper {}
unsafe impl Sync for PythonDistributionWrapper {}

impl PythonDistributionWrapper {
    pub fn new(py_dist: PyObject) -> Self {
        Self { py_dist }
    }
}

impl Distribution for PythonDistributionWrapper {
    fn nll(&self, data: &ArrayView1<f64>) -> f64 {
        Python::with_gil(|py| {
            let y_array = data.to_pyarray(py);
            let result = self.py_dist
                .call_method1(py, "nll", (y_array,))
                .expect("Failed to call nll method");

            result.extract(py).expect("Failed to extract NLL value")
        })
    }
}
