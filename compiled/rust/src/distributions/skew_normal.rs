use super::{DistributionPrimitives, SufficientStats};
use ndarray::{ArrayView1, Array1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};
use statrs::function::erf;


// ============================================================================
// Helper functions for skew-normal calculations
// ============================================================================
// TODO: Could add supports_nll_suff_stats returning false to avoid fast path?
// BUT still use moment order to compute suff stats for param calc, offers speedup
// of parameter calculation even if NLL scoring needs full data.

/// Convert sample skewness (gamma) to delta parameter
#[inline]
fn gamma_to_delta(gamma: f64) -> f64 {
    let gamma_clamped = gamma.clamp(-0.99, 0.99);
    let abs_gamma = gamma_clamped.abs();
    let term = (4.0 - PI) / 2.0;

    gamma_clamped.signum() * (
        PI / 2.0 * abs_gamma.powf(2.0 / 3.0) /
        (abs_gamma.powf(2.0 / 3.0) + term.powf(2.0 / 3.0))
    ).sqrt()
}

/// Convert delta to alpha parameter
#[inline]
fn delta_to_alpha(delta: f64) -> f64 {
    if delta.abs() >= 1.0 {
        return delta.signum() * 10.0; // Clamp to reasonable value
    }
    delta.signum() * (delta.abs() / (1.0 - delta.powi(2)).sqrt()).powf(1.0 / 3.0)
}

/// Calculate omega from sample variance and delta
#[inline]
fn calc_omega(sample_var: f64, delta: f64) -> f64 {
    let omega = sample_var.sqrt() / (1.0 - 2.0 * delta.powi(2) / PI).sqrt();
    if omega.is_nan() || !omega.is_finite() || omega <= 0.0 {
        sample_var.sqrt().max(1e-7)
    } else {
        omega
    }
}

/// Calculate xi from posterior mean, omega, and alpha
#[inline]
fn calc_xi(posterior_mean: f64, omega: f64, alpha: f64) -> f64 {
    posterior_mean - omega * alpha / (1.0 + alpha.powi(2)).sqrt() * (2.0 / PI).sqrt()
}

/// Skew-normal log-pdf for a single point
#[inline]
fn skewnorm_logpdf(x: f64, alpha: f64, xi: f64, omega: f64) -> f64 {
    let z = (x - xi) / omega;
    let log_phi = -0.5 * z * z - 0.5 * (2.0 * PI).ln();  // log of standard normal pdf
    let phi_cap = 0.5 * (1.0 + erf::erf(alpha * z / SQRT_2)); // standard normal CDF
    let log_phi_cap = phi_cap.max(1e-300).ln();

    (2.0_f64).ln() - omega.ln() + log_phi + log_phi_cap
}


// ============================================================================
// NormalMeanPseudoAlphaSkewNormal (Moment-based with shrinkage on alpha)
// ============================================================================

pub struct NormalMeanPseudoAlphaSkewNormal {
    mu_mu: f64,       // Prior mean for data mean
    sigma_mu: f64,    // Prior std for data mean
    prior_alpha: f64,  // Shrinkage target for alpha
    m_alpha: f64,     // Shrinkage strength (pseudo sample size)
}

impl NormalMeanPseudoAlphaSkewNormal {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            mu_mu: spec.get_item("mu_mu")
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            sigma_mu: spec.get_item("sigma_mu")
                .and_then(|v| v.extract().ok())
                .unwrap_or(1.0),
            prior_alpha: spec.get_item("prior_alpha")
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            m_alpha: spec.get_item("m_alpha")
                .and_then(|v| v.extract().ok())
                .unwrap_or(10.0),
        })
    }

    /// Core calculation from sufficient statistics
    fn calc_params_from_stats(&self, n: f64, sum: f64, sum_sq: f64, sum_cu: f64) -> HashMap<String, f64> {
        if n == 0.0 {
            let mut params = HashMap::new();
            params.insert("posterior_alpha".to_string(), self.prior_alpha);
            params.insert("posterior_xi".to_string(), self.mu_mu);
            params.insert("posterior_omega".to_string(), 1.0);
            return params;
        }

        let sample_mean = sum / n;
        let sample_var = if n > 1.0 {
            (sum_sq - n * sample_mean.powi(2)) / (n - 1.0)
        } else {
            1e-10
        };

        // Calculate sample skewness from raw moments
        let std_dev = sample_var.max(1e-10).sqrt();
        // E[(X - μ)³] = E[X³] - 3μE[X²] + 2μ³
        let m3 = sum_cu / n - 3.0 * sample_mean * sum_sq / n + 2.0 * sample_mean.powi(3);
        let sample_skewness = (m3 / std_dev.powi(3)).clamp(-0.99, 0.99);

        // Step 1: Posterior mean for location (Normal prior update)
        let precision_prior = 1.0 / (self.sigma_mu * self.sigma_mu);
        let precision_data = n / sample_var.max(1e-10);
        let posterior_mean = (precision_prior * self.mu_mu + precision_data * sample_mean)
            / (precision_prior + precision_data);

        // Step 2: Convert sample skewness to alpha via moment-matching
        let delta = gamma_to_delta(sample_skewness);
        let alpha_mle = delta_to_alpha(delta);

        // Apply shrinkage prior
        let posterior_alpha = (n / (n + self.m_alpha)) * alpha_mle
            + (self.m_alpha / (n + self.m_alpha)) * self.prior_alpha;

        // Step 3: Estimate omega from sample variance adjusted for skewness
        let posterior_omega = calc_omega(sample_var, delta);

        // Step 4: Recover xi from posterior mean
        let posterior_xi = calc_xi(posterior_mean, posterior_omega, posterior_alpha);

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), posterior_alpha);
        params.insert("posterior_xi".to_string(), posterior_xi);
        params.insert("posterior_omega".to_string(), posterior_omega);
        params
    }
}

impl DistributionPrimitives for NormalMeanPseudoAlphaSkewNormal {
    fn required_moment_order(&self) -> usize {
        0  // Need skewness (sum of cubes) - currently stats for params only not implemented
    }

    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let sum: f64 = data.sum();
        let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
        let sum_cu: f64 = data.iter().map(|&x| x * x * x).sum();

        self.calc_params_from_stats(n, sum, sum_sq, sum_cu)
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        data.mapv(|x| skewnorm_logpdf(x, alpha, xi, omega))
    }

    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(data);
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        -data.iter().map(|&x| skewnorm_logpdf(x, alpha, xi, omega)).sum::<f64>()
    }

    fn nll_suff_stats(&self, _stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        // For NLL from suff stats, we need to compute the log-likelihood
        // This requires iterating over data, which we don't have.
        // However, we can compute a closed-form approximation for the normal part
        // and only the CDF term requires data.

        // Unfortunately, the CDF term (log Φ(α·z)) cannot be computed from suff stats alone.
        // So we return None and fall back to the slow path for scoring.
        //
        // BUT: we can still use suff stats for the parameter calculation in the fast path!
        // The splitter will use this for parent/child param calculation, then fall back
        // to data-based NLL scoring.
        None
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(train);
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        -test.iter().map(|&x| skewnorm_logpdf(x, alpha, xi, omega)).sum::<f64>()
    }
}


// ============================================================================
// NormalMeanNormalGammaSkewNormal (Normal prior on both mean and skewness)
// ============================================================================

pub struct NormalMeanNormalGammaSkewNormal {
    mu_mu: f64,       // Prior mean for data mean
    sigma_mu: f64,    // Prior std for data mean
    mu_gamma: f64,    // Prior mean for skewness gamma
    sigma_gamma: f64, // Prior std for skewness gamma
}

impl NormalMeanNormalGammaSkewNormal {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            mu_mu: spec.get_item("mu_mu")
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            sigma_mu: spec.get_item("sigma_mu")
                .and_then(|v| v.extract().ok())
                .unwrap_or(1.0),
            mu_gamma: spec.get_item("mu_gamma")
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            sigma_gamma: spec.get_item("sigma_gamma")
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.5),
        })
    }

    /// Core calculation from sufficient statistics
    fn calc_params_from_stats(&self, n: f64, sum: f64, sum_sq: f64, sum_cu: f64) -> HashMap<String, f64> {
        if n == 0.0 {
            let mut params = HashMap::new();
            params.insert("posterior_alpha".to_string(), 0.0);
            params.insert("posterior_xi".to_string(), self.mu_mu);
            params.insert("posterior_omega".to_string(), 1.0);
            return params;
        }

        let sample_mean = sum / n;
        let sample_var = if n > 1.0 {
            (sum_sq - n * sample_mean.powi(2)) / (n - 1.0)
        } else {
            1e-10
        };

        // Calculate sample skewness from raw moments
        let std_dev = sample_var.max(1e-10).sqrt();
        let m3 = sum_cu / n - 3.0 * sample_mean * sum_sq / n + 2.0 * sample_mean.powi(3);
        let sample_skewness = m3 / std_dev.powi(3);

        // Variance of skewness estimator
        let var_skewness = if n > 3.0 {
            (6.0 * n * (n - 1.0)) / ((n - 2.0) * (n + 1.0) * (n + 3.0))
        } else {
            1.0
        };

        // Step 1: Posterior skewness gamma (Normal prior update)
        let precision_prior_gamma = 1.0 / (self.sigma_gamma * self.sigma_gamma);
        let precision_data_gamma = n / var_skewness.max(1e-7);
        let posterior_skewness = (precision_prior_gamma * self.mu_gamma + precision_data_gamma * sample_skewness)
            / (precision_prior_gamma + precision_data_gamma);
        let posterior_skewness = posterior_skewness.clamp(-0.995, 0.995);

        // Step 2: Posterior mean for location (Normal prior update)
        let precision_prior_mu = 1.0 / (self.sigma_mu * self.sigma_mu);
        let precision_data_mu = n / sample_var.max(1e-7);
        let posterior_mean = (precision_prior_mu * self.mu_mu + precision_data_mu * sample_mean)
            / (precision_prior_mu + precision_data_mu);

        // Step 3: Convert skewness to alpha via moment-matching
        let delta = gamma_to_delta(posterior_skewness);
        let posterior_alpha = delta_to_alpha(delta);

        // Step 4: Estimate omega from sample variance adjusted for skewness
        let posterior_omega = calc_omega(sample_var, delta);

        // Step 5: Recover xi from posterior mean
        let posterior_xi = calc_xi(posterior_mean, posterior_omega, posterior_alpha);

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), posterior_alpha);
        params.insert("posterior_xi".to_string(), posterior_xi);
        params.insert("posterior_omega".to_string(), posterior_omega);
        params
    }
}

impl DistributionPrimitives for NormalMeanNormalGammaSkewNormal {
    fn required_moment_order(&self) -> usize {
        0  // Need skewness (sum of cubes)
    }

    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let sum: f64 = data.sum();
        let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
        let sum_cu: f64 = data.iter().map(|&x| x * x * x).sum();

        self.calc_params_from_stats(n, sum, sum_sq, sum_cu)
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        data.mapv(|x| skewnorm_logpdf(x, alpha, xi, omega))
    }

    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(data);
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        -data.iter().map(|&x| skewnorm_logpdf(x, alpha, xi, omega)).sum::<f64>()
    }

    fn nll_suff_stats(&self, _stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        // Same as above: CDF term cannot be computed from suff stats
        // Parameters CAN be computed, but NLL scoring requires data
        None
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(train);
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        -test.iter().map(|&x| skewnorm_logpdf(x, alpha, xi, omega)).sum::<f64>()
    }
}


// ============================================================================
// NormalXiNormalAlphaSkewNormalMAP (MAP estimation via optimization)
// ============================================================================

pub struct NormalXiNormalAlphaSkewNormalMAP {
    mu_xi: f64,       // Prior mean for location xi
    sigma_xi: f64,    // Prior std for location xi
    mu_alpha: f64,    // Prior mean for skewness alpha
    sigma_alpha: f64, // Prior std for skewness alpha
}

impl NormalXiNormalAlphaSkewNormalMAP {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            mu_xi: spec.get_item("mu_xi")
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            sigma_xi: spec.get_item("sigma_xi")
                .and_then(|v| v.extract().ok())
                .unwrap_or(1.0),
            mu_alpha: spec.get_item("mu_alpha")
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            sigma_alpha: spec.get_item("sigma_alpha")
                .and_then(|v| v.extract().ok())
                .unwrap_or(5.0),
        })
    }

    /// Simple Nelder-Mead-like optimization (coordinate descent for speed)
    fn optimize_map(&self, data: &ArrayView1<f64>, xi_init: f64, alpha_init: f64, omega: f64) -> (f64, f64) {
        let mut xi = xi_init;
        let mut alpha = alpha_init;

        // Simplified optimization: a few iterations of coordinate descent
        for _ in 0..5 {
            // Optimize xi with alpha fixed
            let step_xi = 0.1 * omega;
            let current_nlp = self.neg_log_posterior(data, xi, alpha, omega);

            let nlp_plus = self.neg_log_posterior(data, xi + step_xi, alpha, omega);
            let nlp_minus = self.neg_log_posterior(data, xi - step_xi, alpha, omega);

            if nlp_plus < current_nlp && nlp_plus < nlp_minus {
                xi += step_xi;
            } else if nlp_minus < current_nlp {
                xi -= step_xi;
            }

            // Optimize alpha with xi fixed
            let step_alpha = 0.2;
            let current_nlp = self.neg_log_posterior(data, xi, alpha, omega);

            let nlp_plus = self.neg_log_posterior(data, xi, alpha + step_alpha, omega);
            let nlp_minus = self.neg_log_posterior(data, xi, alpha - step_alpha, omega);

            if nlp_plus < current_nlp && nlp_plus < nlp_minus {
                alpha += step_alpha;
            } else if nlp_minus < current_nlp {
                alpha -= step_alpha;
            }
        }

        (xi, alpha)
    }

    fn neg_log_posterior(&self, data: &ArrayView1<f64>, xi: f64, alpha: f64, omega: f64) -> f64 {
        // Prior terms (Gaussian priors)
        let prior_xi = -0.5 * ((xi - self.mu_xi) / self.sigma_xi).powi(2);
        let prior_alpha = -0.5 * ((alpha - self.mu_alpha) / self.sigma_alpha).powi(2);

        // Likelihood term
        let likelihood: f64 = data.iter().map(|&x| skewnorm_logpdf(x, alpha, xi, omega)).sum();

        if !likelihood.is_finite() {
            return 1e10;
        }

        -(prior_xi + prior_alpha + likelihood)
    }
}

impl DistributionPrimitives for NormalXiNormalAlphaSkewNormalMAP {
    fn required_moment_order(&self) -> usize {
        0  // MAP uses numerical optimization, cannot use sufficient statistics
    }

    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        if n == 0.0 {
            let mut params = HashMap::new();
            params.insert("posterior_alpha".to_string(), self.mu_alpha);
            params.insert("posterior_xi".to_string(), self.mu_xi);
            params.insert("posterior_omega".to_string(), 1.0);
            return params;
        }

        let sample_mean = data.mean().unwrap();
        let sum_sq: f64 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();
        let sample_var = if n > 1.0 { sum_sq / (n - 1.0) } else { 1e-10 };

        // Initial estimates from moment matching
        let sum_cu: f64 = data.iter().map(|&x| ((x - sample_mean) / sample_var.max(1e-10).sqrt()).powi(3)).sum();
        let sample_skewness = (sum_cu / n).clamp(-0.995, 0.995);

        let delta = gamma_to_delta(sample_skewness);
        let alpha_init = delta_to_alpha(delta);
        let omega_init = calc_omega(sample_var, delta);
        let xi_init = calc_xi(sample_mean, omega_init, alpha_init);

        // Optimize
        let (posterior_xi, posterior_alpha) = self.optimize_map(data, xi_init, alpha_init, omega_init);

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), posterior_alpha);
        params.insert("posterior_xi".to_string(), posterior_xi);
        params.insert("posterior_omega".to_string(), omega_init);
        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        data.mapv(|x| skewnorm_logpdf(x, alpha, xi, omega))
    }

    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(data);
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        -data.iter().map(|&x| skewnorm_logpdf(x, alpha, xi, omega)).sum::<f64>()
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(train);
        let alpha = params["posterior_alpha"];
        let xi = params["posterior_xi"];
        let omega = params["posterior_omega"];

        -test.iter().map(|&x| skewnorm_logpdf(x, alpha, xi, omega)).sum::<f64>()
    }
}
