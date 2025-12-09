use super::{DistributionPrimitives, SufficientStats};
use ndarray::{ArrayView1, Array1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

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
// GammaABLambdaExponential (Shape-Rate Parameterization)
// ============================================================================

/// Gamma-Exponential conjugate model with shape-rate (α, β) parameterization.
///
/// Prior: λ ~ Gamma(alpha_lambda, beta_lambda)  [shape-rate]
/// Likelihood: y | λ ~ Exponential(λ)
/// Posterior: λ | y ~ Gamma(alpha_lambda + n, beta_lambda + Σy)
/// Posterior Predictive: y_new | y ~ Lomax(α_post, β_post)
pub struct GammaABLambdaExponential {
    pub alpha_lambda: f64,  // Shape parameter
    pub beta_lambda: f64,   // Rate parameter
}

impl GammaABLambdaExponential {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            alpha_lambda: spec.get_item("alpha_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing alpha_lambda in GammaABLambdaExponential spec"
            ))?.extract()?,
            beta_lambda: spec.get_item("beta_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing beta_lambda in GammaABLambdaExponential spec"
            ))?.extract()?,
        })
    }
}

impl DistributionPrimitives for GammaABLambdaExponential {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let sum_y: f64 = data.iter().sum();

        // Conjugate update: Gamma(α, β) → Gamma(α + n, β + Σy)
        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;

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

        // Exponential PDF: log p(x | λ) = log(λ) - λ*x
        data.mapv(|x| log_lambda - lambda * x)
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        // Posterior predictive is Lomax (Pareto Type II)
        // Lomax PDF: f(x | α, β) = (α/β) * (1 + x/β)^(-(α+1))
        // log f(x | α, β) = log(α) - log(β) - (α+1) * log(1 + x/β)
        let alpha_post = params["posterior_alpha"];
        let beta_post = params["posterior_beta"];

        let log_alpha = alpha_post.ln();
        let log_beta = beta_post.ln();

        Some(data.mapv(|x| {
            log_alpha - log_beta - (alpha_post + 1.0) * (1.0 + x / beta_post).ln()
        }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;

        // Log Evidence for Gamma-Exponential:
        // log p(y) = log Γ(α_post) - log Γ(α) + α*log(β) - α_post*log(β_post)
        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln();

        Some(-log_evidence) // Return NLE (negative log evidence)
    }

    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;
        let lambda_post = alpha_post / beta_post;

        if use_posterior_predictive {
            // Lomax log-likelihood
            let log_alpha = alpha_post.ln();
            let log_beta = beta_post.ln();

            let mut nll = 0.0;
            for &x in data {
                let log_pdf = log_alpha - log_beta - (alpha_post + 1.0) * (1.0 + x / beta_post).ln();
                nll -= log_pdf;
            }
            nll
        } else {
            // Plug-in Exponential
            let log_lambda = lambda_post.max(1e-10).ln();

            let mut nll = 0.0;
            for &x in data {
                let log_pdf = log_lambda - lambda_post * x;
                nll -= log_pdf;
            }
            nll
        }
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n_train = train.len() as f64;
        let sum_y_train: f64 = train.iter().sum();

        let alpha_post = self.alpha_lambda + n_train;
        let beta_post = self.beta_lambda + sum_y_train;
        let lambda_post = alpha_post / beta_post;

        if use_posterior_predictive {
            let log_alpha = alpha_post.ln();
            let log_beta = beta_post.ln();

            let mut nll = 0.0;
            for &x in test {
                let log_pdf = log_alpha - log_beta - (alpha_post + 1.0) * (1.0 + x / beta_post).ln();
                nll -= log_pdf;
            }
            nll
        } else {
            let log_lambda = lambda_post.max(1e-10).ln();

            let mut nll = 0.0;
            for &x in test {
                let log_pdf = log_lambda - lambda_post * x;
                nll -= log_pdf;
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

        // For Exponential: sum = Σy (sum of observations)
        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;

        if use_posterior_predictive {
            // Lomax NLL requires individual x values for log(1 + x/β)
            // Cannot be computed from sufficient stats alone
            return None;
        } else {
            // Plug-in Exponential: NLL = n*λ - n*log(λ) + λ*Σy - n*log(λ)
            // Actually: log p(x | λ) = log(λ) - λ*x
            // Sum: Σ log p(x_i | λ) = n*log(λ) - λ*Σx
            // NLL = -n*log(λ) + λ*Σx
            let lambda = (alpha_post / beta_post).max(1e-10);
            let log_lambda = lambda.ln();

            // NLL = -n*log(λ) + λ*Σx = λ*Σx - n*log(λ)
            let nll = lambda * sum_y - n * log_lambda;
            Some(nll)
        }
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;

        // Log Evidence:
        // log p(y) = log Γ(α_post) - log Γ(α) + α*log(β) - α_post*log(β_post)
        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln();

        Some(-log_evidence)
    }
}

// ============================================================================
// GammaMVLambdaExponential (Mean-Variance Parameterization)
// ============================================================================

/// Gamma-Exponential conjugate model with mean-variance parameterization.
///
/// Prior specified via:
/// - mean_lambda = E[λ] (prior expected rate)
/// - var_lambda = Var[λ] (prior uncertainty about rate)
///
/// Internally converts to α = mean²/var, β = mean/var
pub struct GammaMVLambdaExponential {
    pub mean_lambda: f64,
    pub var_lambda: f64,
    // Derived shape-rate parameters
    alpha_lambda: f64,
    beta_lambda: f64,
}

impl GammaMVLambdaExponential {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let mean_lambda: f64 = spec.get_item("mean_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing mean_lambda in GammaMVLambdaExponential spec"
        ))?.extract()?;
        let var_lambda: f64 = spec.get_item("var_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing var_lambda in GammaMVLambdaExponential spec"
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

impl DistributionPrimitives for GammaMVLambdaExponential {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;
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

        data.mapv(|x| log_lambda - lambda * x)
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        let alpha_post = params["posterior_alpha"];
        let beta_post = params["posterior_beta"];

        let log_alpha = alpha_post.ln();
        let log_beta = beta_post.ln();

        Some(data.mapv(|x| {
            log_alpha - log_beta - (alpha_post + 1.0) * (1.0 + x / beta_post).ln()
        }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;

        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln();

        Some(-log_evidence)
    }

    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let sum_y: f64 = data.iter().sum();

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;
        let lambda_post = alpha_post / beta_post;

        if use_posterior_predictive {
            let log_alpha = alpha_post.ln();
            let log_beta = beta_post.ln();

            let mut nll = 0.0;
            for &x in data {
                let log_pdf = log_alpha - log_beta - (alpha_post + 1.0) * (1.0 + x / beta_post).ln();
                nll -= log_pdf;
            }
            nll
        } else {
            let log_lambda = lambda_post.max(1e-10).ln();

            let mut nll = 0.0;
            for &x in data {
                let log_pdf = log_lambda - lambda_post * x;
                nll -= log_pdf;
            }
            nll
        }
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n_train = train.len() as f64;
        let sum_y_train: f64 = train.iter().sum();

        let alpha_post = self.alpha_lambda + n_train;
        let beta_post = self.beta_lambda + sum_y_train;
        let lambda_post = alpha_post / beta_post;

        if use_posterior_predictive {
            let log_alpha = alpha_post.ln();
            let log_beta = beta_post.ln();

            let mut nll = 0.0;
            for &x in test {
                let log_pdf = log_alpha - log_beta - (alpha_post + 1.0) * (1.0 + x / beta_post).ln();
                nll -= log_pdf;
            }
            nll
        } else {
            let log_lambda = lambda_post.max(1e-10).ln();

            let mut nll = 0.0;
            for &x in test {
                let log_pdf = log_lambda - lambda_post * x;
                nll -= log_pdf;
            }
            nll
        }
    }

    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;

        if use_posterior_predictive {
            return None;
        } else {
            let lambda = (alpha_post / beta_post).max(1e-10);
            let log_lambda = lambda.ln();
            let nll = lambda * sum_y - n * log_lambda;
            Some(nll)
        }
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + n;
        let beta_post = self.beta_lambda + sum_y;

        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln();

        Some(-log_evidence)
    }
}
