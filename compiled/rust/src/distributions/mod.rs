//! Distribution trait for BDF.
//!
//! Key design: Only ONE method needed: score().
//! Native implementations for hot-path distributions (Normal, Poisson, Bernoulli).
//! Python callback fallback for everything else.

pub mod normal;
pub mod poisson;
pub mod bernoulli;
pub mod python_callback;

use ndarray::ArrayView1;
use pyo3::PyResult;

/// Distribution scoring trait.
pub trait Distribution: Send + Sync {
    /// Score a dataset (lower is better).
    ///
    /// This includes:
    /// - NLE (if conjugate)
    /// - NLL + corrections (AIC, BIC, etc.)
    /// - CV scores
    ///
    /// Distribution handles all internal logic.
    fn score(&self, data: ArrayView1<f64>) -> PyResult<f64>;

    /// Name for debugging.
    fn name(&self) -> &'static str;
}

/// Create distribution from Python spec.
pub fn create_distribution(
    dist_type: &str,
    spec: &pyo3::types::PyDict,
) -> PyResult<Box<dyn Distribution>> {
    match dist_type {
        // Native implementations (HOT PATH)
        "NormalMuNormal" => Ok(Box::new(normal::NormalMuNormal::from_spec(spec)?)),
        "GammaABLambdaPoisson" => Ok(Box::new(poisson::GammaPoisson::from_spec(spec)?)),
        "BetaABBernoulli" => Ok(Box::new(bernoulli::BetaBernoulli::from_spec(spec)?)),

        // Python callback fallback (COLD PATH)
        _ => {
            let py_obj = spec.get_item("_python_object")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                    "Missing '_python_object' for non-native distribution"
                ))?
                .extract()?;
            Ok(Box::new(python_callback::PythonCallbackDistribution::new(py_obj)))
        }
    }
}
