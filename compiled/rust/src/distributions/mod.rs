//! Distribution trait for BDF.
//!
//! Key design: Only ONE method needed: score().
//! Native implementations for hot-path distributions (Normal, Poisson, Bernoulli).
//! Python callback fallback for everything else.

pub mod normal;
pub mod poisson;
pub mod bernoulli;
pub mod python_callback;

use pyo3::PyResult;
use ndarray::{ArrayView1, Array1};
use std::collections::HashMap;


pub trait DistributionPrimitives: Send + Sync {
    /// Compute posterior parameters (returns ALL params needed for both plug-in and PP)
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64>;

    /// Plug-in log-likelihood: log p(x | θ_MAP)
    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>)
        -> Array1<f64>;

    /// Posterior predictive log-likelihood: log ∫ p(x | θ) p(θ | data) dθ
    /// Returns None if not supported (non-conjugate model)
    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>,
        params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        None
    }

    /// Log evidence: log p(data | prior) (closed-form for conjugate models)
    /// Returns None if not available
    fn log_evidence(&self, data: &ArrayView1<f64>) -> Option<f64> {
        None
    }
}

/// Scoring specification (from Python validation)
#[derive(Clone)]
pub struct ScoringSpec {
    pub score_method: String,           // "nle" or "nll"
    pub score_correction: Option<String>, // None, "aic", "bic", "loo_cv", "kfold_cv"
    pub use_posterior_predictive: bool,
    pub num_parameters: usize,          // Pre-computed in Python!
    pub cv_folds: usize,
    pub cv_shuffle: bool,
    pub cv_seed: u64,
}

impl ScoringSpec {
    /// Parse from Python dict (all validation already done!)
    pub fn from_dict(dict: &PyDict) -> PyResult<Self> {
        Ok(Self {
            score_method: dict.get_item("score_method")?.extract()?,
            score_correction: dict.get_item("score_correction")?.extract()?,
            use_posterior_predictive: dict.get_item("use_posterior_predictive")?.extract()?,
            num_parameters: dict.get_item("num_parameters")?.extract()?,
            cv_folds: dict.get_item("cv_folds")?.extract()?,
            cv_shuffle: dict.get_item("cv_shuffle")?.extract()?,
            cv_seed: dict.get_item("cv_seed")?.extract()?,
        })
    }
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
