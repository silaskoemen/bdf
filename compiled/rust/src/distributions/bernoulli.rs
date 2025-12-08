use super::{DistributionPrimitives, SufficientStats};
use ndarray::{ArrayView1, Array1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// Log of Beta function: B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
fn log_beta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

// Lanczos approximation for log-gamma function
fn lgamma(x: f64) -> f64 {
    use std::f64::consts::PI;
    let p = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ];
    let g = 7.0;
    if x < 0.5 {
        PI.ln() - (PI * x).sin().ln() - lgamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut a = p[0];
        for i in 1..9 {
            a += p[i] / (z + i as f64);
        }
        let t = z + g + 0.5;
        0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
    }
}

// ============================================================================
// BetaABBernoulli (Concentration Parameterization)
// ============================================================================

/// Beta-Bernoulli conjugate model with concentration (α, β) parameterization.
///
/// Prior: p ~ Beta(alpha_p, beta_p)
/// Likelihood: y | p ~ Bernoulli(p)
/// Posterior: p | y ~ Beta(alpha_p + k, beta_p + n - k), where k = sum(y)
pub struct BetaABBernoulli {
    pub alpha_p: f64,
    pub beta_p: f64,
}

impl BetaABBernoulli {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            alpha_p: spec.get_item("alpha_p").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing alpha_p in BetaABBernoulli spec"
            ))?.extract()?,
            beta_p: spec.get_item("beta_p").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing beta_p in BetaABBernoulli spec"
            ))?.extract()?,
        })
    }
}

impl DistributionPrimitives for BetaABBernoulli {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let k: f64 = data.iter().sum(); // Number of successes

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);
        let prob_post = alpha_post / (alpha_post + beta_post);

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), alpha_post);
        params.insert("posterior_beta".to_string(), beta_post);
        params.insert("posterior_prob".to_string(), prob_post);
        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let prob = params["posterior_prob"].clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        data.mapv(|y| if y > 0.5 { log_p1 } else { log_p0 })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        // For Beta-Bernoulli, PP probability is the posterior mean
        // P(y=1 | data) = alpha_post / (alpha_post + beta_post)
        let alpha_post = params["posterior_alpha"];
        let beta_post = params["posterior_beta"];
        let ab_sum = alpha_post + beta_post;

        let prob_1 = (alpha_post / ab_sum).clamp(1e-10, 1.0 - 1e-10);
        let prob_0 = (beta_post / ab_sum).clamp(1e-10, 1.0 - 1e-10);

        let log_p1 = prob_1.ln();
        let log_p0 = prob_0.ln();

        Some(data.mapv(|y| if y > 0.5 { log_p1 } else { log_p0 }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let k: f64 = data.iter().sum();

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        // Log Evidence: log B(alpha_post, beta_post) - log B(alpha, beta)
        let log_evidence = log_beta(alpha_post, beta_post) - log_beta(self.alpha_p, self.beta_p);

        Some(-log_evidence) // Return NLE (negative log evidence)
    }

    #[allow(unused_variables)]
    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let k: f64 = data.iter().sum();

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        let prob = (alpha_post / (alpha_post + beta_post)).clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        // NLL = -sum(y * log(p) + (1-y) * log(1-p))
        //     = -k * log(p) - (n-k) * log(1-p)
        -(k * log_p1 + (n - k) * log_p0)
    }

    #[allow(unused_variables)]
    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n_train = train.len() as f64;
        let k_train: f64 = train.iter().sum();

        let alpha_post = self.alpha_p + k_train;
        let beta_post = self.beta_p + (n_train - k_train);

        let prob = (alpha_post / (alpha_post + beta_post)).clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        let k_test: f64 = test.iter().sum();
        let n_test = test.len() as f64;

        -(k_test * log_p1 + (n_test - k_test) * log_p0)
    }

    // ========================================================================
    // FAST PATH: Sufficient Statistics
    // ========================================================================

    fn nll_suff_stats(&self, stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        // For Bernoulli: sum = number of successes (k)
        let k = stats.sum;

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        let prob = (alpha_post / (alpha_post + beta_post)).clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        let nll = -(k * log_p1 + (n - k) * log_p0);
        Some(nll)
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let k = stats.sum;

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        let log_evidence = log_beta(alpha_post, beta_post) - log_beta(self.alpha_p, self.beta_p);

        Some(-log_evidence)
    }
}

// ============================================================================
// BetaMVBernoulli (Mean-Variance Parameterization)
// ============================================================================

/// Beta-Bernoulli conjugate model with mean-variance parameterization.
///
/// Prior specified via:
/// - mean_p = E[p] (expected success probability)
/// - var_p = Var[p] (uncertainty about p)
///
/// Internally converts to α, β using:
/// m = mean_p(1-mean_p)/var_p - 1
/// α = mean_p · m
/// β = (1-mean_p) · m
pub struct BetaMVBernoulli {
    pub mean_p: f64,
    pub var_p: f64,
    // Derived concentration parameters
    alpha_p: f64,
    beta_p: f64,
}

impl BetaMVBernoulli {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let mean_p: f64 = spec.get_item("mean_p").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing mean_p in BetaMVBernoulli spec"
        ))?.extract()?;
        let var_p: f64 = spec.get_item("var_p").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing var_p in BetaMVBernoulli spec"
        ))?.extract()?;

        // Convert mean-variance to concentration parameters
        let m = (mean_p * (1.0 - mean_p) / var_p) - 1.0;
        let alpha_p = mean_p * m;
        let beta_p = (1.0 - mean_p) * m;

        Ok(Self {
            mean_p,
            var_p,
            alpha_p,
            beta_p,
        })
    }
}

impl DistributionPrimitives for BetaMVBernoulli {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let k: f64 = data.iter().sum();

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);
        let prob_post = alpha_post / (alpha_post + beta_post);

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), alpha_post);
        params.insert("posterior_beta".to_string(), beta_post);
        params.insert("posterior_prob".to_string(), prob_post);
        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let prob = params["posterior_prob"].clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        data.mapv(|y| if y > 0.5 { log_p1 } else { log_p0 })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        let alpha_post = params["posterior_alpha"];
        let beta_post = params["posterior_beta"];
        let ab_sum = alpha_post + beta_post;

        let prob_1 = (alpha_post / ab_sum).clamp(1e-10, 1.0 - 1e-10);
        let prob_0 = (beta_post / ab_sum).clamp(1e-10, 1.0 - 1e-10);

        let log_p1 = prob_1.ln();
        let log_p0 = prob_0.ln();

        Some(data.mapv(|y| if y > 0.5 { log_p1 } else { log_p0 }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let k: f64 = data.iter().sum();

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        let log_evidence = log_beta(alpha_post, beta_post) - log_beta(self.alpha_p, self.beta_p);

        Some(-log_evidence)
    }

    #[allow(unused_variables)]
    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let k: f64 = data.iter().sum();

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        let prob = (alpha_post / (alpha_post + beta_post)).clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        -(k * log_p1 + (n - k) * log_p0)
    }

    #[allow(unused_variables)]
    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n_train = train.len() as f64;
        let k_train: f64 = train.iter().sum();

        let alpha_post = self.alpha_p + k_train;
        let beta_post = self.beta_p + (n_train - k_train);

        let prob = (alpha_post / (alpha_post + beta_post)).clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        let k_test: f64 = test.iter().sum();
        let n_test = test.len() as f64;

        -(k_test * log_p1 + (n_test - k_test) * log_p0)
    }

    #[allow(unused_variables)]
    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let k = stats.sum;

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        let prob = (alpha_post / (alpha_post + beta_post)).clamp(1e-10, 1.0 - 1e-10);
        let log_p1 = prob.ln();
        let log_p0 = (1.0 - prob).ln();

        let nll = -(k * log_p1 + (n - k) * log_p0);
        Some(nll)
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let k = stats.sum;

        let alpha_post = self.alpha_p + k;
        let beta_post = self.beta_p + (n - k);

        let log_evidence = log_beta(alpha_post, beta_post) - log_beta(self.alpha_p, self.beta_p);

        Some(-log_evidence)
    }
}
