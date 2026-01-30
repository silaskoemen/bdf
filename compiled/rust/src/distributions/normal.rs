use super::{DistributionPrimitives, SufficientStats};
use ndarray::{ArrayView1, Array1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::f64::consts::PI;

// Lanczos approximation for log-gamma function (needed for Student's t and Evidence)
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
// NormalMuNormal (Known Variance / Estimated separately)
// ============================================================================

pub struct NormalMuNormal {
    mu_mu: f64,      // Prior mean
    sigma_mu: f64,   // Prior std
}

impl NormalMuNormal {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            mu_mu: spec.get_item("mu_mu").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing mu_mu in NormalMuNormal spec"
            ))?.extract()?,
            sigma_mu: spec.get_item("sigma_mu").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                "Missing sigma_mu in NormalMuNormal spec"
            ))?.extract()?,
        })
    }
}

impl DistributionPrimitives for NormalMuNormal {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;

        // Sample statistics
        if n == 0.0 {
            // No data: posterior = prior
            let mut params = HashMap::new();
            params.insert("posterior_mu".to_string(), self.mu_mu);
            params.insert("posterior_sigma_mu".to_string(), self.sigma_mu);
            params.insert("sample_std".to_string(), 1e-10); // Avoid zero std
            return params;
        }
        let sample_mean = data.mean().unwrap();
        // Handle n=1 case: sample variance undefined, use small value (matches Python)
        let sample_var_raw = if n > 1.0 {
            let sum_sq: f64 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();
            sum_sq / (n - 1.0)
        } else {
            1e-20  // Matches Python's (sample_std=1e-10)**2
        };
        let sample_std = sample_var_raw.sqrt().max(1e-10);
        // Python uses max(sample_std**2, 1e-10) for precision calculation
        let sample_var_for_precision = sample_var_raw.max(1e-10);

        // Bayesian update
        let precision_prior = 1.0 / (self.sigma_mu * self.sigma_mu);
        let precision_data = n / sample_var_for_precision;
        let precision_posterior = precision_prior + precision_data;

        let posterior_mu = (precision_prior * self.mu_mu + precision_data * sample_mean)
            / precision_posterior;
        let posterior_sigma_mu = (1.0 / precision_posterior).sqrt();

        let mut params = HashMap::new();
        params.insert("posterior_mu".to_string(), posterior_mu);
        params.insert("posterior_sigma_mu".to_string(), posterior_sigma_mu);
        params.insert("sample_std".to_string(), sample_std);
        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>)
        -> Array1<f64> {
        let mu = params["posterior_mu"];
        let sigma = params["sample_std"].max(1e-10);
        let log_sigma = sigma.ln();
        const LOG_2PI: f64 = 1.8378770664093453;

        data.mapv(|x| {
            let z = (x - mu) / sigma;
            -0.5 * LOG_2PI - log_sigma - 0.5 * z * z
        })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>,
        params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        let mu = params["posterior_mu"];
        let sigma_mu = params["posterior_sigma_mu"];
        let sample_std = params["sample_std"];

        // Matches Python: pred_var = sigma_mu**2 + sample_std**2
        let pred_var = sigma_mu * sigma_mu + sample_std * sample_std;
        let pred_std = pred_var.sqrt();
        let log_pred_std = pred_std.ln();
        const LOG_2PI: f64 = 1.8378770664093453;

        Some(data.mapv(|x| {
            let z = (x - mu) / pred_std;
            -0.5 * LOG_2PI - log_pred_std - 0.5 * z * z
        }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sample_mean = data.mean().unwrap();
        let sample_var = if n > 1.0 {
            let sum_sq: f64 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();
            sum_sq / (n - 1.0)
        } else {
            0.0
        };

        let marginal_var = (sample_var / n) + self.sigma_mu * self.sigma_mu;

        let log_ev_mean = -0.5 * (2.0 * PI * marginal_var).ln()
                          - 0.5 * (sample_mean - self.mu_mu).powi(2) / marginal_var;

        // The data terms relative to the sample mean (independent of Mu)
        // Sum log N(x_i | x_bar, sample_var)
        let log_ev_residuals = if n > 1.0 {
             -0.5 * (n - 1.0) * (2.0 * PI * sample_var).ln() - 0.5 * (n - 1.0)
        } else {
            0.0
        };

        Some(-(log_ev_mean + log_ev_residuals))  // Return NLE (negative log evidence)
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sample_mean = stats.sum / n;
        let sample_var = if n > 1.0 {
            (stats.sum_sq - n * sample_mean.powi(2)) / (n - 1.0)
        } else {
            0.0
        };

        let marginal_var = (sample_var / n) + self.sigma_mu * self.sigma_mu;

        let log_ev_mean = -0.5 * (2.0 * PI * marginal_var).ln()
                          - 0.5 * (sample_mean - self.mu_mu).powi(2) / marginal_var;

        // The data terms relative to the sample mean (independent of Mu)
        // Sum log N(x_i | x_bar, sample_var)
        let log_ev_residuals = if n > 1.0 {
             -0.5 * (n - 1.0) * (2.0 * PI * sample_var).ln() - 0.5 * (n - 1.0)
        } else {
            0.0
        };

        Some(-(log_ev_mean + log_ev_residuals))  // Return NLE (negative log evidence)
    }

    /// Optimized stack-based NLL calculation
    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }
        let sample_mean = data.mean().unwrap();
        let sum_sq: f64 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();
        // For n=1, use 1e-20 to match Python (sample_std=1e-10)
        let sample_var_raw = if n > 1.0 { sum_sq / (n - 1.0) } else { 1e-20 };
        let sample_std = sample_var_raw.sqrt().max(1e-10);
        // Python uses max(sample_std**2, 1e-10) for precision calculation
        let sample_var_for_precision = sample_var_raw.max(1e-10);

        let precision_prior = 1.0 / (self.sigma_mu * self.sigma_mu);
        let precision_data = n / sample_var_for_precision;
        let precision_posterior = precision_prior + precision_data;

        let posterior_mu = (precision_prior * self.mu_mu + precision_data * sample_mean) / precision_posterior;
        let posterior_sigma_mu = (1.0 / precision_posterior).sqrt();

        let (mu, sigma) = if use_posterior_predictive {
            // Python uses sample_std**2 for PP variance, NOT the floored sample_var
            let pred_var = posterior_sigma_mu.powi(2) + sample_std.powi(2);
            (posterior_mu, pred_var.sqrt())
        } else {
            (posterior_mu, sample_std)
        };

        let log_sigma = sigma.ln();
        const LOG_2PI: f64 = 1.8378770664093453;

        let mut nll = 0.0;
        for &x in data {
            let z = (x - mu) / sigma;
            nll -= -0.5 * LOG_2PI - log_sigma - 0.5 * z * z;
        }
        nll
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        // 1. Train (calculate posterior params from train set)
        let n = train.len() as f64;
        if n == 0.0 { return 0.0; }

        let sample_mean = train.mean().unwrap();
        let sum_sq: f64 = train.iter().map(|&x| (x - sample_mean).powi(2)).sum();
        let sample_var_raw = if n > 1.0 { sum_sq / (n - 1.0) } else { 1e-20 };
        let sample_std = sample_var_raw.sqrt().max(1e-10);
        let sample_var_for_precision = sample_var_raw.max(1e-10);

        let precision_prior = 1.0 / (self.sigma_mu * self.sigma_mu);
        let precision_data = n / sample_var_for_precision;
        let precision_posterior = precision_prior + precision_data;

        let posterior_mu = (precision_prior * self.mu_mu + precision_data * sample_mean) / precision_posterior;
        let posterior_sigma_mu = (1.0 / precision_posterior).sqrt();

        let (mu, sigma) = if use_posterior_predictive {
            let pred_var = posterior_sigma_mu.powi(2) + sample_std.powi(2);
            (posterior_mu, pred_var.sqrt())
        } else {
            (posterior_mu, sample_std)
        };

        let log_sigma = sigma.ln();
        const LOG_2PI: f64 = 1.8378770664093453;

        // 2. Test (evaluate NLL on test set)
        let mut nll = 0.0;
        for &x in test {
            let z = (x - mu) / sigma;
            nll -= -0.5 * LOG_2PI - log_sigma - 0.5 * z * z;
        }
        nll
    }

    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sample_mean = stats.sum / n;
        // variance = (sum_sq - n*mean^2) / (n-1)
        let ss_diff = stats.sum_sq - n * sample_mean.powi(2);
        let sample_var_raw = if n > 1.0 { ss_diff / (n - 1.0) } else { 1e-20 };
        let sample_std = sample_var_raw.sqrt().max(1e-10);
        let sample_var_for_precision = sample_var_raw.max(1e-10);

        // Bayesian update (same as calc_posterior_params but on stack)
        let precision_prior = 1.0 / (self.sigma_mu * self.sigma_mu);
        let precision_data = n / sample_var_for_precision;
        let precision_posterior = precision_prior + precision_data;

        let posterior_mu = (precision_prior * self.mu_mu + precision_data * sample_mean) / precision_posterior;
        let posterior_sigma_mu = (1.0 / precision_posterior).sqrt();

        let (mu, sigma) = if use_posterior_predictive {
            let pred_var = posterior_sigma_mu.powi(2) + sample_std.powi(2);
            (posterior_mu, pred_var.sqrt())
        } else {
            (posterior_mu, sample_std)
        };

        let log_sigma = sigma.ln();
        const LOG_2PI: f64 = 1.8378770664093453;

        // NLL Sum = 0.5 * sum((x - mu)^2) / sigma^2 + n * log_sigma + 0.5 * n * LOG_2PI
        // Expansion: sum((x - mu)^2) = sum(x^2) - 2*mu*sum(x) + n*mu^2
        let sum_sq_diff_mu = stats.sum_sq - 2.0 * mu * stats.sum + n * mu * mu;

        let nll = 0.5 * sum_sq_diff_mu / (sigma * sigma) + n * log_sigma + 0.5 * n * LOG_2PI;
        Some(nll)
    }
}

// ============================================================================
// NormalMuInvGammaSigmaNormal (Unknown Mean and Variance)
// ============================================================================

pub struct NormalMuInvGammaSigmaNormal {
    mu_mu: f64,
    n_mu: f64,
    nu_sigma: f64,
    phi_sigma: f64,
}

impl NormalMuInvGammaSigmaNormal {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            mu_mu: spec.get_item("mu_mu")
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'mu_mu'"))?
                .extract()?,
            n_mu: spec.get_item("n_mu")
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'n_mu'"))?
                .extract()?,
            nu_sigma: spec.get_item("nu_sigma")
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'nu_sigma'"))?
                .extract()?,
            phi_sigma: spec.get_item("phi_sigma")
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'phi_sigma'"))?
                .extract()?,
        })
    }
}

impl DistributionPrimitives for NormalMuInvGammaSigmaNormal {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        if n == 0.0 {
            return HashMap::from([
                ("post_mu".into(), self.mu_mu),
                ("post_nu".into(), self.nu_sigma),
                ("post_phi".into(), self.phi_sigma),
                ("post_n".into(), self.n_mu),
                ("map_sigma".into(), (self.phi_sigma / (self.nu_sigma - 2.0)).max(1e-12).sqrt()),
                ("pred_scale".into(), (self.phi_sigma * (self.n_mu + 1.0)
                    / (self.n_mu * (self.nu_sigma - 2.0))).max(1e-12).sqrt()),
            ]);
        }
        let sample_mean = data.mean().unwrap();

        let ssd = {
            data.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>()
        };

        // Posterior parameters
        let post_n = self.n_mu + n;
        let post_nu = self.nu_sigma + n;
        let post_mu = (self.n_mu * self.mu_mu + n * sample_mean) / post_n;

        let prior_sum_sq = self.nu_sigma * self.phi_sigma;
        let interaction = (self.n_mu * n / post_n) * (sample_mean - self.mu_mu).powi(2);
        let post_sum_sq = prior_sum_sq + ssd + interaction;
        let post_phi = post_sum_sq / post_nu;

        let mut params = HashMap::new();
        params.insert("post_mu".to_string(), post_mu);
        params.insert("post_nu".to_string(), post_nu);
        params.insert("post_phi".to_string(), post_phi);
        params.insert("post_n".to_string(), post_n);

        // MAP estimates for plug-in
        // Mode of InvGamma(alpha, beta) is beta / (alpha + 1)
        // alpha = nu/2, beta = nu*phi/2
        let mut map_sigma2 = post_phi * post_nu / (post_nu - 2.0);
        if post_nu <= 2.0 {
            map_sigma2 = post_phi / post_nu;  // fall back to mean
        }
        let map_sigma = map_sigma2.max(1e-12).sqrt();
        params.insert("map_sigma".to_string(), map_sigma);

        // Predictive scale for Student's t
        // Scale = sqrt(phi * (1 + 1/n_n))
        let pred_var = if post_nu > 2.0 {
            post_phi * (1.0 + 1.0 / post_n)
        } else {
            post_phi * (1.0 + 1.0 / post_n) * post_nu / (post_nu - 2.0)  // Adjust for undefined variance
        };
        let pred_scale = pred_var.max(1e-12).sqrt();
        params.insert("pred_scale".to_string(), pred_scale);

        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let mu = params["post_mu"];
        let sigma = params["map_sigma"];
        let log_sigma = sigma.ln();
        const LOG_2PI: f64 = 1.8378770664093453;

        data.mapv(|x| {
            let z = (x - mu) / sigma;
            -0.5 * LOG_2PI - log_sigma - 0.5 * z * z
        })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        let mu = params["post_mu"];
        let nu = params["post_nu"];
        let scale = params["pred_scale"].max(1e-10);
        let log_scale = scale.ln();

        // Log PDF of Student's t(nu, loc, scale)
        // C = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(pi*nu) - log(scale)
        let log_c = lgamma((nu + 1.0) / 2.0)
                  - lgamma(nu / 2.0)
                  - 0.5 * (PI * nu).ln()
                  - log_scale;

        let half_nu_plus_1 = (nu + 1.0) / 2.0;

        Some(data.mapv(|x| {
            let z = (x - mu) / scale;
            log_c - half_nu_plus_1 * (1.0 + z * z / nu).ln()
        }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        // Murphy MLAPP eq 4.127: Normal-Inverse-Gamma marginal likelihood
        // log p(D) = -n/2·log(2π) + 1/2·log(κ₀/κₙ) + logΓ(αₙ) - logΓ(α₀)
        //          + α₀·log(β₀) - αₙ·log(βₙ)
        // where α = ν/2, β = νφ/2
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sample_mean = data.mean().unwrap();
        let ssd: f64 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();

        // Posterior parameters
        let kappa_n = self.n_mu + n;  // κₙ = κ₀ + n
        let nu_n = self.nu_sigma + n;  // νₙ = ν₀ + n
        let interaction = (self.n_mu * n / kappa_n) * (sample_mean - self.mu_mu).powi(2);

        // Beta parameters (using Murphy's parameterization: β = νφ/2)
        let alpha_0 = self.nu_sigma / 2.0;  // α₀ = ν₀/2
        let alpha_n = nu_n / 2.0;            // αₙ = νₙ/2
        let beta_0 = self.nu_sigma * self.phi_sigma / 2.0;  // β₀ = ν₀φ₀/2
        let beta_n = beta_0 + 0.5 * ssd + 0.5 * interaction; // βₙ = β₀ + S/2 + interaction/2

        let log_ev = -0.5 * n * (2.0 * PI).ln()
            + 0.5 * (self.n_mu.ln() - kappa_n.ln())
            + lgamma(alpha_n) - lgamma(alpha_0)
            + alpha_0 * beta_0.ln()
            - alpha_n * beta_n.ln();

        Some(-log_ev)  // NLE = negative log evidence
    }

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sample_mean = stats.sum / n;
        let ss_diff = stats.sum_sq - n * sample_mean.powi(2);
        let ssd = ss_diff;

        let kappa_n = self.n_mu + n;
        let nu_n = self.nu_sigma + n;

        let interaction = (self.n_mu * n / kappa_n) * (sample_mean - self.mu_mu).powi(2);

        // Murphy's parameterization: α = ν/2, β = νφ/2
        let alpha_0 = self.nu_sigma / 2.0;
        let alpha_n = nu_n / 2.0;
        let beta_0 = self.nu_sigma * self.phi_sigma / 2.0;
        let beta_n = beta_0 + 0.5 * ssd + 0.5 * interaction;

        let log_ev = -0.5 * n * (2.0 * PI).ln()
            + 0.5 * (self.n_mu.ln() - kappa_n.ln())
            + lgamma(alpha_n) - lgamma(alpha_0)
            + alpha_0 * beta_0.ln()
            - alpha_n * beta_n.ln();

        Some(-log_ev)
    }

    /// Optimized stack-based NLL calculation
    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        // ...existing code...
        // (Keep existing implementation)
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }
        let sample_mean = data.mean().unwrap();
        let ssd: f64 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();

        let post_n = self.n_mu + n;
        let post_nu = self.nu_sigma + n;
        let post_mu = (self.n_mu * self.mu_mu + n * sample_mean) / post_n;

        let prior_sum_sq = self.nu_sigma * self.phi_sigma;
        let interaction = (self.n_mu * n / post_n) * (sample_mean - self.mu_mu).powi(2);
        let post_sum_sq = prior_sum_sq + ssd + interaction;
        let post_phi = post_sum_sq / post_nu;

        if use_posterior_predictive {
            let scale_sq = post_phi * (1.0 + 1.0 / post_n);
            let scale = scale_sq.sqrt();
            let log_scale = scale.ln();

            let log_c = lgamma((post_nu + 1.0) / 2.0)
                      - lgamma(post_nu / 2.0)
                      - 0.5 * (PI * post_nu).ln()
                      - log_scale;

            let half_nu_plus_1 = (post_nu + 1.0) / 2.0;

            let mut nll = 0.0;
            for &x in data {
                let z = (x - post_mu) / scale;
                let log_pdf = log_c - half_nu_plus_1 * (1.0 + z * z / post_nu).ln();
                nll -= log_pdf;
            }
            nll
        } else {
            let map_sigma2 = if post_nu > 2.0 {
                post_phi * post_nu / (post_nu - 2.0)
            } else {
                post_phi / post_nu
            };
            let sigma = map_sigma2.sqrt();
            let log_sigma = sigma.ln();
            const LOG_2PI: f64 = 1.8378770664093453;

            let mut nll = 0.0;
            for &x in data {
                let z = (x - post_mu) / sigma;
                nll -= -0.5 * LOG_2PI - log_sigma - 0.5 * z * z;
            }
            nll
        }
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        // 1. Train
        let n = train.len() as f64;
        if n == 0.0 { return 0.0; }

        let sample_mean = train.mean().unwrap();
        let ssd: f64 = train.iter().map(|&x| (x - sample_mean).powi(2)).sum();

        let post_n = self.n_mu + n;
        let post_nu = self.nu_sigma + n;
        let post_mu = (self.n_mu * self.mu_mu + n * sample_mean) / post_n;

        let prior_sum_sq = self.nu_sigma * self.phi_sigma;
        let interaction = (self.n_mu * n / post_n) * (sample_mean - self.mu_mu).powi(2);
        let post_sum_sq = prior_sum_sq + ssd + interaction;
        let post_phi = post_sum_sq / post_nu;

        // 2. Test
        if use_posterior_predictive {
            let scale_sq = post_phi * (1.0 + 1.0 / post_n);
            let scale = scale_sq.sqrt();
            let log_scale = scale.ln();

            let log_c = lgamma((post_nu + 1.0) / 2.0)
                      - lgamma(post_nu / 2.0)
                      - 0.5 * (PI * post_nu).ln()
                      - log_scale;

            let half_nu_plus_1 = (post_nu + 1.0) / 2.0;

            let mut nll = 0.0;
            for &x in test {
                let z = (x - post_mu) / scale;
                let log_pdf = log_c - half_nu_plus_1 * (1.0 + z * z / post_nu).ln();
                nll -= log_pdf;
            }
            nll
        } else {
            let map_sigma2 = if post_nu > 2.0 {
                post_phi * post_nu / (post_nu - 2.0)
            } else {
                post_phi / post_nu
            };
            let sigma = map_sigma2.sqrt();
            let log_sigma = sigma.ln();
            const LOG_2PI: f64 = 1.8378770664093453;

            let mut nll = 0.0;
            for &x in test {
                let z = (x - post_mu) / sigma;
                nll -= -0.5 * LOG_2PI - log_sigma - 0.5 * z * z;
            }
            nll
        }
    }

    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sample_mean = stats.sum / n;
        let ssd = stats.sum_sq - n * sample_mean.powi(2);

        let post_n = self.n_mu + n;
        let post_nu = self.nu_sigma + n;
        let post_mu = (self.n_mu * self.mu_mu + n * sample_mean) / post_n;

        let prior_sum_sq = self.nu_sigma * self.phi_sigma;
        let interaction = (self.n_mu * n / post_n) * (sample_mean - self.mu_mu).powi(2);
        let post_sum_sq = prior_sum_sq + ssd + interaction;
        let post_phi = post_sum_sq / post_nu;

        if use_posterior_predictive {
            // Student's t NLL requires iterating data points (log(1 + z^2/nu))
            // It does NOT have a clean sufficient statistic form for the sum of logs.
            // Fallback to slow path.
            return None;
        } else {
            // Plug-in Normal NLL IS compatible with sufficient stats
            let map_sigma2 = if post_nu > 2.0 {
                post_phi * post_nu / (post_nu - 2.0)
            } else {
                post_phi / post_nu
            };
            let sigma = map_sigma2.sqrt();
            let log_sigma = sigma.ln();
            const LOG_2PI: f64 = 1.8378770664093453;

            let sum_sq_diff_mu = stats.sum_sq - 2.0 * post_mu * stats.sum + n * post_mu * post_mu;
            let nll = 0.5 * sum_sq_diff_mu / (sigma * sigma) + n * log_sigma + 0.5 * n * LOG_2PI;
            Some(nll)
        }
    }
}
