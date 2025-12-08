//! Python callback distribution for falling back to Python implementations.
//!
//! This is used when a distribution type is not natively implemented in Rust.
//! It delegates all calls to the Python object, incurring GIL overhead.

use super::{DistributionPrimitives};
use ndarray::{ArrayView1, Array1};
use numpy::{ToPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Wrapper around a Python distribution object.
/// Delegates all trait methods to Python via the GIL.
pub struct PythonCallbackDistribution {
    py_obj: PyObject,
}

// Safety: PyO3's PyObject is thread-safe when accessed with the GIL
unsafe impl Send for PythonCallbackDistribution {}
unsafe impl Sync for PythonCallbackDistribution {}

impl PythonCallbackDistribution {
    pub fn new(py_obj: PyObject) -> Self {
        Self { py_obj }
    }
}

impl DistributionPrimitives for PythonCallbackDistribution {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        Python::with_gil(|py| {
            let py_array = data.to_pyarray(py);

            // Call Python's calc_posterior_params method
            let result = self.py_obj
                .call_method1(py, "calc_posterior_params", (py_array,))
                .expect("Failed to call calc_posterior_params");

            // Convert Python dict to HashMap
            let py_dict: &PyDict = result.extract(py)
                .expect("calc_posterior_params should return a dict");

            let mut params = HashMap::new();
            for (key, value) in py_dict.iter() {
                let k: String = key.extract().expect("Key should be string");
                let v: f64 = value.extract().expect("Value should be float");
                params.insert(k, v);
            }
            params
        })
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>)
        -> Array1<f64> {
        Python::with_gil(|py| {
            let py_array = data.to_pyarray(py);

            let py_params = PyDict::new(py);
            for (k, v) in params {
                py_params.set_item(k, v).unwrap();
            }

            let result = self.py_obj
                .call_method1(py, "plugin_log_likelihood", (py_array, py_params))
                .expect("Failed to call plugin_log_likelihood");

            let owned = {
                let py_result: &PyArray1<f64> = result.extract(py)
                    .expect("Failed to extract PyArray1 result");
                py_result.readonly().as_array().to_owned()
            };
            owned
        })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>,
        params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        Python::with_gil(|py| {
            let py_array = data.to_pyarray(py);

            // Convert HashMap to PyDict
            let py_params = PyDict::new(py);
            for (k, v) in params {
                py_params.set_item(k, v).unwrap();
            }

            // Check if method exists and call it
            let result = self.py_obj
                .call_method1(py, "posterior_predictive_log_likelihood", (py_array, py_params));

            match result {
                Ok(res) => {
                    // Check for None return
                    if res.is_none(py) {
                        return None;
                    }
                    let py_result: &PyArray1<f64> = res.extract(py)
                        .expect("posterior_predictive_log_likelihood should return ndarray or None");
                    Some(py_result.readonly().as_array().to_owned())
                }
                Err(_) => None, // Method doesn't exist or failed
            }
        })
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        Python::with_gil(|py| {
            let py_array = data.to_pyarray(py);

            let result = self.py_obj
                .call_method1(py, "nle", (py_array,));

            match result {
                Ok(res) => {
                    if res.is_none(py) {
                        return None;
                    }
                    Some(res.extract(py).expect("nle should return float or None"))
                }
                Err(_) => None,
            }
        })
    }

    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        Python::with_gil(|py| {
            let py_array = data.to_pyarray(py);

            // Try calling nll with the use_posterior_predictive flag
            let result = self.py_obj
                .call_method1(py, "nll", (py_array, use_posterior_predictive));

            match result {
                Ok(res) => res.extract(py).expect("nll should return float"),
                Err(_) => {
                    // Fallback: try without the flag (old API)
                    let result = self.py_obj
                        .call_method1(py, "nll", (py_array,))
                        .expect("Failed to call nll");
                    result.extract(py).expect("nll should return float")
                }
            }
        })
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>,
        use_posterior_predictive: bool) -> f64 {
        Python::with_gil(|py| {
            let train_array = train.to_pyarray(py);
            let test_array = test.to_pyarray(py);

            let result = self.py_obj
                .call_method1(py, "nll_train_test", (train_array, test_array, use_posterior_predictive));

            match result {
                Ok(res) => res.extract(py).expect("nll_train_test should return float"),
                Err(_) => {
                    // Fallback: use default implementation
                    let params = self.calc_posterior_params(train);
                    let ll = if use_posterior_predictive {
                        self.posterior_predictive_log_likelihood(test, &params)
                            .unwrap_or_else(|| self.plugin_log_likelihood(test, &params))
                    } else {
                        self.plugin_log_likelihood(test, &params)
                    };
                    -ll.sum()
                }
            }
        })
    }

    // Note: nll_suff_stats and nle_suff_stats return None (not supported via Python callback)
    // The sufficient stats optimization is only available for native Rust implementations
}
