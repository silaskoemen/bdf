//! Native Normal-Normal implementation (SIMD-optimized).

use super::Distribution;
use crate::utils::stats::single_pass_stats;
use ndarray::ArrayView1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

pub struct NormalMuNormal {
    mu_mu: f64,
    sigma_mu: f64,
    score_method: ScoringMethod,
}

enum ScoringMethod {
    NLE,
    NLL { correction: Option<Correction> },
}

enum Correction {
    AIC,
    BIC,
}

impl NormalMuNormal {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let mu_mu: f64 = spec.get_item("mu_mu")?.unwrap().extract()?;
        let sigma_mu: f64 = spec.get_item("sigma_mu")?.unwrap().extract()?;

        let score_method_str: String = spec.get_item("score_method")?.unwrap().extract()?;
        let correction_str: Option<String> = spec.get_item("score_correction")?
            .and_then(|v| v.extract().ok());

        let score_method = match score_method_str.as_str() {
            "nle" => ScoringMethod::NLE,
            "nll" => {
                let correction = match correction_str.as_deref() {
                    Some("aic") => Some(Correction::AIC),
                    Some("bic") => Some(Correction::BIC),
                    _ => None,
                };
                ScoringMethod::NLL { correction }
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown score_method: {}", score_method_str)
            ))
        };

        Ok(Self { mu_mu, sigma_mu, score_method })
    }

    /// Compute NLE using single-pass stats.
    #[inline]
    fn compute_nle(&self, data: ArrayView1<f64>) -> f64 {
        let (n, sum, sum_sq, mean, variance) = single_pass_stats(data);
        if n == 0 { return 0.0; }

        let n_f64 = n as f64;

        // Marginal variance of sample mean
        let marginal_var = (variance / n_f64) + self.sigma_mu * self.sigma_mu;

        // Log evidence for sample mean
        let mut log_ev = -0.5 * (2.0 * PI * marginal_var).ln();
        log_ev -= 0.5 * (mean - self.mu_mu).powi(2) / marginal_var;

        // Log evidence for deviations
        if n > 1 {
            log_ev -= 0.5 * (n_f64 - 1.0) * (1.0 + (2.0 * PI * variance).ln());
        }

        -log_ev  // Return NLE (lower is better)
    }

    /// Compute NLL using single-pass stats.
    #[inline]
    fn compute_nll(&self, data: ArrayView1<f64>) -> f64 {
        let (n, sum, sum_sq, mean, variance) = single_pass_stats(data);
        if n == 0 { return 0.0; }

        let n_f64 = n as f64;

        // Posterior mean (weighted average)
        let prior_precision = 1.0 / (self.sigma_mu * self.sigma_mu);
        let data_precision = n_f64 / variance.max(1e-10);
        let posterior_precision = prior_precision + data_precision;
        let posterior_variance = 1.0 / posterior_precision;
        let posterior_mean = posterior_variance *
            (prior_precision * self.mu_mu + data_precision * mean);

        // NLL using posterior mean
        // Single-pass residual sum: Σ(x - μ)² = Σx² - 2μΣx + nμ²
        let residual_sum = sum_sq - 2.0 * posterior_mean * sum + n_f64 * posterior_mean * posterior_mean;

        let nll = 0.5 * n_f64 * (2.0 * PI).ln() +
                  0.5 * n_f64 * posterior_variance.ln() +
                  0.5 * residual_sum / posterior_variance;

        nll
    }
}

impl Distribution for NormalMuNormal {
    fn score(&self, data: ArrayView1<f64>) -> PyResult<f64> {
        let score = match &self.score_method {
            ScoringMethod::NLE => self.compute_nle(data),

            ScoringMethod::NLL { correction } => {
                let mut nll = self.compute_nll(data);

                // Apply correction
                if let Some(corr) = correction {
                    let n = data.len() as f64;
                    let k = 1.0;  // Only μ estimated

                    nll += match corr {
                        Correction::AIC => k,
                        Correction::BIC => 0.5 * k * n.ln(),
                    };
                }

                nll
            }
        };

        Ok(score)
    }

    fn name(&self) -> &'static str {
        "NormalMuNormal"
    }
}
