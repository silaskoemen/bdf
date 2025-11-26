use super::DistributionPrimitives;
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
            mu_mu: spec.get_item("mu_mu")?.extract()?,
            sigma_mu: spec.get_item("sigma_mu")?.extract()?,
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
        let sample_var = {
            let sum_sq: f64 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();
            sum_sq / (n - 1.0)
        };
        let sample_std = sample_var.sqrt().max(1e-10);

        // Bayesian update
        let precision_prior = 1.0 / (self.sigma_mu * self.sigma_mu);
        let precision_data = n / sample_var.max(1e-10);
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
        let sample_std = params["sample_std"].max(1e-10);

        let pred_var = sigma_mu * sigma_mu + sample_std * sample_std;
        let pred_std = pred_var.sqrt().max(1e-10);
        let log_pred_std = pred_std.ln();
        const LOG_2PI: f64 = 1.8378770664093453;

        Some(data.mapv(|x| {
            let z = (x - mu) / pred_std;
            -0.5 * LOG_2PI - log_pred_std - 0.5 * z * z
        }))
    }

    fn log_evidence(&self, data: &ArrayView1<f64>) -> Option<f64> {
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

        let mut log_ev = -0.5 * (2.0 * PI * marginal_var).ln();
        log_ev -= 0.5 * (sample_mean - self.mu_mu).powi(2) / marginal_var;

        if n > 1.0 {
            log_ev -= 0.5 * (n - 1.0) * (2.0 * PI * sample_var).ln();
            log_ev -= 0.5 * (n - 1.0);
        }

        Some(log_ev)
    }
}

// ============================================================================
// NormGammaNormal (Unknown Mean and Variance)
// ============================================================================

pub struct NormGammaNormal {
    mu_zero: f64,
    prior_n: f64,
    prior_nu: f64,
    prior_phi: f64,
}

impl NormGammaNormal {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        Ok(Self {
            mu_zero: spec.get_item("mu_zero")?.extract()?,
            prior_n: spec.get_item("prior_n")?.extract()?,
            prior_nu: spec.get_item("prior_nu")?.extract()?,
            prior_phi: spec.get_item("prior_phi")?.extract()?,
        })
    }
}

impl DistributionPrimitives for NormGammaNormal {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        if n == 0.0 {
            return HashMap::from([
                ("post_mu".into(), self.mu_zero),
                ("post_nu".into(), self.prior_nu),
                ("post_phi".into(), self.prior_phi),
                ("post_n".into(), self.prior_n),
                ("map_sigma".into(), (self.prior_phi / (self.prior_nu - 2.0)).max(1e-12).sqrt()),
                ("pred_scale".into(), (self.prior_phi * (self.prior_n + 1.0)
                    / (self.prior_n * (self.prior_nu - 2.0))).max(1e-12).sqrt()),
            ]);
        }
        let sample_mean = data.mean().unwrap();

        let ssd = {
            data.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>()
        };

        // Posterior parameters
        let post_n = self.prior_n + n;
        let post_nu = self.prior_nu + n;
        let post_mu = (self.prior_n * self.mu_zero + n * sample_mean) / post_n;

        let prior_sum_sq = self.prior_nu * self.prior_phi;
        let interaction = (self.prior_n * n / post_n) * (sample_mean - self.mu_zero).powi(2);
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
        let mut map_sigma2 = post_phi * (post_nu - 2.0) / post_nu;
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

    fn log_evidence(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sample_mean = data.mean().unwrap();
        let ssd = if n > 1.0 {
            data.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>()
        } else {
            0.0
        };

        let post_n = self.prior_n + n;
        let post_nu = self.prior_nu + n;

        let prior_sum_sq = self.prior_nu * self.prior_phi;
        let interaction = (self.prior_n * n / post_n) * (sample_mean - self.mu_zero).powi(2);
        let post_sum_sq = prior_sum_sq + ssd + interaction;

        // Alpha/Beta parameterization for evidence formula
        let alpha_0 = self.prior_nu / 2.0;
        let beta_0 = self.prior_nu * self.prior_phi / 2.0;
        let alpha_n = post_nu / 2.0;
        let beta_n = post_sum_sq / 2.0;

        let log_ev = -0.5 * n * (PI).ln()
            + 0.5 * (self.prior_n.ln() - post_n.ln())
            + lgamma(post_nu / 2.0) - lgamma(self.prior_nu / 2.0)
            + (self.prior_nu / 2.0) * self.prior_phi.ln()
            - (post_nu / 2.0) * post_phi.ln();

        Some(log_ev)
    }
}
