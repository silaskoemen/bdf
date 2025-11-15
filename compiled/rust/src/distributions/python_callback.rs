use ndarray::ArrayView1;
use pyo3::prelude::*;
use numpy::ToPyArray;
use super::{Distribution, PosteriorParams, ScoreMethod, ScoreCorrection};

pub struct PythonParams;
impl PosteriorParams for PythonParams {
    fn num_params(&self) -> usize { 1 } // Fallback default
}

pub struct PythonDistributionWrapper {
    pub py_dist: PyObject,
}

unsafe impl Send for PythonDistributionWrapper {}
unsafe impl Sync for PythonDistributionWrapper {}

impl Distribution for PythonDistributionWrapper {
    type Params = PythonParams;

    fn calc_posterior_params(&self, _data: &ArrayView1<f64>) -> PythonParams {
        PythonParams
    }

    fn score(&self, data: &ArrayView1<f64>, method: ScoreMethod) -> f64 {
        Python::with_gil(|py| {
            let y_array = data.to_pyarray(py);

            // Call Python's score() method directly
            let result = self.py_dist
                .call_method1(py, "score", (y_array,))
                .expect("Failed to call score method");

            result.extract(py).expect("Failed to extract score value")
        })
    }

    fn nll(&self, data: &ArrayView1<f64>) -> f64 {
        Python::with_gil(|py| {
            let y_array = data.to_pyarray(py);
            let result = self.py_dist
                .call_method1(py, "nll", (y_array,))
                .expect("Failed to call nll method");
            result.extract(py).expect("Failed to extract NLL value")
        })
    }

    fn log_likelihood_single(&self, x: f64, _train_data: &ArrayView1<f64>) -> f64 {
        // Fallback: not used if score() is overridden
        0.0
    }
}
