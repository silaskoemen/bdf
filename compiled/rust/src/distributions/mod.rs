//! Distribution trait for BDF.
//!
//! Key design: Only ONE method needed: score().
//! Native implementations for hot-path distributions (Normal, Poisson, Bernoulli).
//! Python callback fallback for everything else.

pub mod normal;
pub mod bernoulli;
pub mod poisson;
pub mod kde;
pub mod python_callback;

use pyo3::PyResult;
use pyo3::types::PyDict;
use ndarray::{ArrayView1, Array1};
use std::collections::HashMap;


#[derive(Clone, Copy, Default, Debug)]
pub struct SufficientStats {
    pub n: f64,
    pub sum: f64,
    pub sum_sq: f64,
}

impl SufficientStats {
    #[inline]
    pub fn add(&mut self, val: f64) {
        self.n += 1.0;
        self.sum += val;
        self.sum_sq += val * val;
    }

    #[inline]
    pub fn remove(&mut self, val: f64) {
        self.n -= 1.0;
        self.sum -= val;
        self.sum_sq -= val * val;
    }
}


pub trait DistributionPrimitives: Send + Sync {
    /// Compute posterior parameters (returns ALL params needed for both plug-in and PP)
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64>;

    /// Plug-in log-likelihood: log p(x | θ_MAP)
    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>)
        -> Array1<f64>;

    /// Posterior predictive log-likelihood: log ∫ p(x | θ) p(θ | data) dθ
    /// Returns None if not supported (non-conjugate model)
    #[allow(unused_variables)]
    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>,
        params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        None
    }

    /// Log evidence: log p(data | prior) (closed-form for conjugate models)
    /// Returns None if not available
    #[allow(unused_variables)]
    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        None
    }

    /// FAST PATH: Calculate NLE from sufficient statistics (if possible).
    #[allow(unused_variables)]
    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        None
    }

    /// FAST PATH: Calculate Negative Log Likelihood (NLL) directly.
    /// This avoids HashMap allocations in the hot loop.
    /// Default implementation falls back to the slow calc_posterior_params method.
    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(data);

        let ll = if use_posterior_predictive {
            self.posterior_predictive_log_likelihood(data, &params)
                .unwrap_or_else(|| self.plugin_log_likelihood(data, &params))
        } else {
            self.plugin_log_likelihood(data, &params)
        };

        -ll.sum()
    }

    /// FAST PATH: Calculate NLL on TEST data after training on TRAIN data.
    /// Essential for efficient Cross-Validation.
    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(train);

        let ll = if use_posterior_predictive {
            self.posterior_predictive_log_likelihood(test, &params)
                .unwrap_or_else(|| self.plugin_log_likelihood(test, &params))
        } else {
            self.plugin_log_likelihood(test, &params)
        };

        -ll.sum()
    }

    /// FAST PATH: Calculate NLL from sufficient statistics (if possible).
    /// Returns None if not supported.
    #[allow(unused_variables)]
    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
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
            score_method: dict.get_item("score_method").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing score_method in scoring spec"
            ))?.extract()?,
            score_correction: dict.get_item("score_correction").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing score_correction in scoring spec"
            ))?.extract()?,
            use_posterior_predictive: dict.get_item("use_posterior_predictive").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing use_posterior_predictive in scoring spec"
            ))?.extract()?,
            num_parameters: dict.get_item("num_parameters").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing num_parameters in scoring spec"
            ))?.extract()?,
            cv_folds: dict.get_item("cv_folds").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing cv_folds in scoring spec"
            ))?.extract()?,
            cv_shuffle: dict.get_item("cv_shuffle").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing cv_shuffle in scoring spec"
            ))?.extract()?,
            cv_seed: dict.get_item("cv_seed").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing cv_seed in scoring spec"
            ))?.extract()?,
        })
    }
}
