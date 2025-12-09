//! Distribution trait for BDF.
//!
//! Key design: Only ONE method needed: score().
//! Native implementations for hot-path distributions (Normal, Poisson, Bernoulli).
//! Python callback fallback for everything else.

pub mod normal;
pub mod bernoulli;
pub mod poisson;
pub mod exponential;
pub mod kde;
pub mod skew_normal;
pub mod python_callback;

use pyo3::PyResult;
use pyo3::types::PyDict;
use ndarray::{ArrayView1, Array1};
use std::collections::HashMap;


/// Sufficient statistics for O(1) incremental split evaluation.
///
/// `moment_order` determines which fields are valid:
/// - 0: No stats (KDE, MAP) - skip entirely
/// - 1: Only `n`, `sum` (e.g., Bernoulli only needs sum for k)
/// - 2: `n`, `sum`, `sum_sq` (Normal, Poisson, etc.)
/// - 3: Also `sum_cu` (skew-normal modular)
/// - 4: Also `sum_qu` (kurtosis-based distributions)
#[derive(Clone, Copy, Default, Debug)]
pub struct SufficientStats {
    pub n: f64,
    pub sum: f64,
    pub sum_sq: f64,
    pub sum_cu: f64,  // sum of cubes (for skewness)
    pub sum_qu: f64,  // sum of fourth powers (for kurtosis)
}

impl SufficientStats {
    /// Create stats tracker for a given moment order.
    /// Order 0 means no tracking needed.
    #[inline]
    pub fn new(_order: u8) -> Self {
        Self::default()
    }

    /// Add a value. For order 0, this is a no-op (but caller should skip).
    #[inline]
    pub fn add(&mut self, val: f64, order: u8) {
        self.n += 1.0;
        if order >= 1 {
            self.sum += val;
        }
        if order >= 2 {
            self.sum_sq += val * val;
        }
        if order >= 3 {
            self.sum_cu += val * val * val;
        }
        if order >= 4 {
            self.sum_qu += val * val * val * val;
        }
    }

    /// Remove a value (for sliding window during split scan).
    #[inline]
    pub fn remove(&mut self, val: f64, order: u8) {
        self.n -= 1.0;
        if order >= 1 {
            self.sum -= val;
        }
        if order >= 2 {
            self.sum_sq -= val * val;
        }
        if order >= 3 {
            self.sum_cu -= val * val * val;
        }
        if order >= 4 {
            self.sum_qu -= val * val * val * val;
        }
    }

    /// Convenience: add with default order 2 (backwards compatibility)
    #[inline]
    pub fn add_default(&mut self, val: f64) {
        self.add(val, 2);
    }

    /// Convenience: remove with default order 2 (backwards compatibility)
    #[inline]
    pub fn remove_default(&mut self, val: f64) {
        self.remove(val, 2);
    }

    // ========================================================================
    // Derived statistics (computed from raw sums)
    // ========================================================================

    #[inline]
    pub fn mean(&self) -> f64 {
        if self.n > 0.0 { self.sum / self.n } else { 0.0 }
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.n > 1.0 {
            (self.sum_sq - self.sum * self.sum / self.n) / (self.n - 1.0)
        } else {
            0.0
        }
    }

    /// Sample skewness (requires order >= 3)
    #[inline]
    pub fn skewness(&self) -> f64 {
        if self.n < 3.0 { return 0.0; }
        let mean = self.mean();
        let var = self.variance();
        if var <= 0.0 { return 0.0; }

        // E[(X - μ)³] = E[X³] - 3μE[X²] + 2μ³
        let m3 = self.sum_cu / self.n
               - 3.0 * mean * self.sum_sq / self.n
               + 2.0 * mean.powi(3);

        m3 / var.powf(1.5)
    }

    /// Sample kurtosis (requires order >= 4)
    #[inline]
    pub fn kurtosis(&self) -> f64 {
        if self.n < 4.0 { return 0.0; }
        let mean = self.mean();
        let var = self.variance();
        if var <= 0.0 { return 0.0; }

        // E[(X - μ)⁴] using raw moments
        let m4 = self.sum_qu / self.n
               - 4.0 * mean * self.sum_cu / self.n
               + 6.0 * mean.powi(2) * self.sum_sq / self.n
               - 3.0 * mean.powi(4);

        m4 / var.powi(2) - 3.0  // Excess kurtosis
    }
}


pub trait DistributionPrimitives: Send + Sync {
    /// Compute posterior parameters (returns ALL params needed for both plug-in and PP)
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64>;

    fn supports_rust_params(&self) -> bool {
        true  // Default: most distributions only need scalars
    }

    fn required_moment_order(&self) -> usize {
        2  // Default: mean and variance for sufficient stats calculation
    }

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
    pub score_cv_folds: usize,
    pub score_cv_shuffle: bool,
    pub score_cv_seed: u64,
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
            score_cv_folds: dict.get_item("score_cv_folds").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing score_cv_folds in scoring spec"
            ))?.extract()?,
            score_cv_shuffle: dict.get_item("score_cv_shuffle").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing score_cv_shuffle in scoring spec"
            ))?.extract()?,
            score_cv_seed: dict.get_item("score_cv_seed").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing score_cv_seed in scoring spec"
            ))?.extract()?,
        })
    }
}
