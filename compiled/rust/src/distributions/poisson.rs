use super::{DistributionPrimitives, SufficientStats};
use ndarray::{ArrayView1, Array1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::f64::consts::PI;

// Lanczos approximation for log-gamma function
fn lgamma(x: f64) -> f64 {
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
// GammaABLambdaPoisson (Shape-Rate Parameterization)
// ============================================================================

/// Gamma-Poisson conjugate model with shape-rate (α, β) parameterization.
///
/// Prior: λ ~ Gamma(alpha_lambda, beta_lambda)  [shape-rate]
/// Likelihood: y | λ ~ Poisson(λ)
/// Posterior: λ | y ~ Gamma(alpha_lambda + Σy, beta_lambda + n)
/// Posterior Predictive: y_new | y ~ NegativeBinomial(α_post, β_post/(1+β_post))
pub struct GammaABLambdaPoisson {
    pub alpha_lambda: f64,  // Shape parameter
    pub beta_lambda: f64,   // Rate parameter
}

impl GammaABLambdaPoisson {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            alpha_lambda: spec.get_item("alpha_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing alpha_lambda in GammaABLambdaPoisson spec"
            ))?.extract()?,
            beta_lambda: spec.get_item("beta_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing beta_lambda in GammaABLambdaPoisson spec"
            ))?.extract()?,
        })
    }
}

impl DistributionPrimitives for GammaABLambdaPoisson {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let sum_y: f64 = data.iter().sum();

        // Conjugate update: Gamma(α, β) → Gamma(α + Σy, β + n)
        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        // Posterior mean of λ
        let lambda_post = alpha_post / beta_post;

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), alpha_post);
        params.insert("posterior_beta".to_string(), beta_post);
        params.insert("posterior_lambda".to_string(), lambda_post);
        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let lambda = params["posterior_lambda"].max(1e-10);
        let log_lambda = lambda.ln();

        // Poisson PMF: log P(k | λ) = k*log(λ) - λ - log(k!)
        data.mapv(|k| {
            k * log_lambda - lambda - lgamma(k + 1.0)
        })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        // Posterior predictive is Negative Binomial
        // NB(k | r, p) where r = α_post, p = β_post / (1 + β_post)
        let alpha_post = params["posterior_alpha"];
        let beta_post = params["posterior_beta"];

        let r = alpha_post;
        let p = beta_post / (1.0 + beta_post);
        let log_p = p.ln();
        let log_1mp = (1.0 - p).ln();

        // NB PMF: log P(k | r, p) = log Γ(k+r) - log Γ(k+1) - log Γ(r) + r*log(p) + k*log(1-p)
        Some(data.mapv(|k| {
            lgamma(k + r) - lgamma(k + 1.0) - lgamma(r) + r * log_p + k * log_1mp
        }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        // Log Evidence for Gamma-Poisson:
        // log p(y) = log Γ(α_post) - log Γ(α) + α*log(β) - α_post*log(β_post) - Σ log(y!)
        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln()
            - data.iter().map(|&k| lgamma(k + 1.0)).sum::<f64>();

        Some(-log_evidence) // Return NLE
    }

    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        if use_posterior_predictive {
            // Negative Binomial
            let r = alpha_post;
            let p = beta_post / (1.0 + beta_post);
            let log_p = p.ln();
            let log_1mp = (1.0 - p).ln();

            let mut nll = 0.0;
            for &k in data {
                let log_pmf = lgamma(k + r) - lgamma(k + 1.0) - lgamma(r) + r * log_p + k * log_1mp;
                nll -= log_pmf;
            }
            nll
        } else {
            // Plug-in Poisson
            let lambda = alpha_post / beta_post;
            let log_lambda = lambda.max(1e-10).ln();

            let mut nll = 0.0;
            for &k in data {
                let log_pmf = k * log_lambda - lambda - lgamma(k + 1.0);
                nll -= log_pmf;
            }
            nll
        }
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n_train = train.len() as f64;
        let sum_y_train: f64 = train.iter().sum();

        let alpha_post = self.alpha_lambda + sum_y_train;
        let beta_post = self.beta_lambda + n_train;

        if use_posterior_predictive {
            let r = alpha_post;
            let p = beta_post / (1.0 + beta_post);
            let log_p = p.ln();
            let log_1mp = (1.0 - p).ln();

            let mut nll = 0.0;
            for &k in test {
                let log_pmf = lgamma(k + r) - lgamma(k + 1.0) - lgamma(r) + r * log_p + k * log_1mp;
                nll -= log_pmf;
            }
            nll
        } else {
            let lambda = alpha_post / beta_post;
            let log_lambda = lambda.max(1e-10).ln();

            let mut nll = 0.0;
            for &k in test {
                let log_pmf = k * log_lambda - lambda - lgamma(k + 1.0);
                nll -= log_pmf;
            }
            nll
        }
    }

    // ========================================================================
    // FAST PATH: Sufficient Statistics
    // ========================================================================

    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        // For Poisson: sum = Σy (sum of counts)
        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        if use_posterior_predictive {
            // Negative Binomial NLL requires individual k values for lgamma(k + r)
            // Cannot be computed from sufficient stats alone
            return None;
        } else {
            // Plug-in Poisson: NLL = n*λ - Σy*log(λ) + Σlog(y!)
            // The Σlog(y!) term requires individual data points, so we cannot compute it exactly.
            // However, for SPLIT COMPARISON, this term is constant and cancels out!
            // We return the comparable part only.
            let lambda = (alpha_post / beta_post).max(1e-10);
            let log_lambda = lambda.ln();

            // NLL (comparable part) = n*λ - Σy*log(λ)
            // Note: We omit -Σlog(y!) because it's constant across splits
            let nll = n * lambda - sum_y * log_lambda;
            Some(nll)
        }
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        // Log Evidence (comparable part, omitting -Σlog(y!)):
        // log p(y) = log Γ(α_post) - log Γ(α) + α*log(β) - α_post*log(β_post)
        // The -Σlog(y!) term cancels when comparing splits
        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln();

        Some(-log_evidence)
    }
}

// ============================================================================
// GammaMVLambdaPoisson (Mean-Variance Parameterization)
// ============================================================================

/// Gamma-Poisson conjugate model with mean-variance parameterization.
///
/// Prior specified via:
/// - mean_lambda = E[λ] (prior expected rate)
/// - var_lambda = Var[λ] (prior uncertainty about rate)
///
/// Internally converts to α = mean²/var, β = mean/var
pub struct GammaMVLambdaPoisson {
    pub mean_lambda: f64,
    pub var_lambda: f64,
    // Derived shape-rate parameters
    alpha_lambda: f64,
    beta_lambda: f64,
}

impl GammaMVLambdaPoisson {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let mean_lambda: f64 = spec.get_item("mean_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing mean_lambda in GammaMVLambdaPoisson spec"
        ))?.extract()?;
        let var_lambda: f64 = spec.get_item("var_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing var_lambda in GammaMVLambdaPoisson spec"
        ))?.extract()?;

        // Convert mean-variance to shape-rate
        // E[λ] = α/β, Var[λ] = α/β²
        // → α = mean²/var, β = mean/var
        let alpha_lambda = mean_lambda * mean_lambda / var_lambda;
        let beta_lambda = mean_lambda / var_lambda;

        Ok(Self {
            mean_lambda,
            var_lambda,
            alpha_lambda,
            beta_lambda,
        })
    }
}

impl DistributionPrimitives for GammaMVLambdaPoisson {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;
        let lambda_post = alpha_post / beta_post;

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), alpha_post);
        params.insert("posterior_beta".to_string(), beta_post);
        params.insert("posterior_lambda".to_string(), lambda_post);
        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let lambda = params["posterior_lambda"].max(1e-10);
        let log_lambda = lambda.ln();

        data.mapv(|k| {
            k * log_lambda - lambda - lgamma(k + 1.0)
        })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        let alpha_post = params["posterior_alpha"];
        let beta_post = params["posterior_beta"];

        let r = alpha_post;
        let p = beta_post / (1.0 + beta_post);
        let log_p = p.ln();
        let log_1mp = (1.0 - p).ln();

        Some(data.mapv(|k| {
            lgamma(k + r) - lgamma(k + 1.0) - lgamma(r) + r * log_p + k * log_1mp
        }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln()
            - data.iter().map(|&k| lgamma(k + 1.0)).sum::<f64>();

        Some(-log_evidence)
    }

    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        if use_posterior_predictive {
            let r = alpha_post;
            let p = beta_post / (1.0 + beta_post);
            let log_p = p.ln();
            let log_1mp = (1.0 - p).ln();

            let mut nll = 0.0;
            for &k in data {
                let log_pmf = lgamma(k + r) - lgamma(k + 1.0) - lgamma(r) + r * log_p + k * log_1mp;
                nll -= log_pmf;
            }
            nll
        } else {
            let lambda = alpha_post / beta_post;
            let log_lambda = lambda.max(1e-10).ln();

            let mut nll = 0.0;
            for &k in data {
                let log_pmf = k * log_lambda - lambda - lgamma(k + 1.0);
                nll -= log_pmf;
            }
            nll
        }
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n_train = train.len() as f64;
        let sum_y_train: f64 = train.iter().sum();

        let alpha_post = self.alpha_lambda + sum_y_train;
        let beta_post = self.beta_lambda + n_train;

        if use_posterior_predictive {
            let r = alpha_post;
            let p = beta_post / (1.0 + beta_post);
            let log_p = p.ln();
            let log_1mp = (1.0 - p).ln();

            let mut nll = 0.0;
            for &k in test {
                let log_pmf = lgamma(k + r) - lgamma(k + 1.0) - lgamma(r) + r * log_p + k * log_1mp;
                nll -= log_pmf;
            }
            nll
        } else {
            let lambda = alpha_post / beta_post;
            let log_lambda = lambda.max(1e-10).ln();

            let mut nll = 0.0;
            for &k in test {
                let log_pmf = k * log_lambda - lambda - lgamma(k + 1.0);
                nll -= log_pmf;
            }
            nll
        }
    }

    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        if use_posterior_predictive {
            return None;
        } else {
            let lambda = (alpha_post / beta_post).max(1e-10);
            let log_lambda = lambda.ln();
            let nll = n * lambda - sum_y * log_lambda;
            Some(nll)
        }
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln();

        Some(-log_evidence)
    }
}
