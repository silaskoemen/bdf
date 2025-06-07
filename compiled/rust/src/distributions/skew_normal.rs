use crate::distributions::Distribution;
use ndarray::ArrayView1;
use std::f64::consts::{PI, SQRT_2};
use statrs::function::erf;
use log::{warn};


#[derive(Debug, Clone)]
pub struct NormalMeanPseudoAlphaSkewNormalSpec {
    pub mu_zero: f64,
    pub sigma_zero: f64,
    pub alpha_zero: f64,
    pub m_alpha: f64,
}

pub struct NormalMeanPseudoAlphaSkewNormal {
    mu_zero: f64,
    sigma_zero: f64,
    alpha_zero: f64,
    m_alpha: f64,
}

impl NormalMeanPseudoAlphaSkewNormal {
    pub fn new(spec: &NormalMeanPseudoAlphaSkewNormalSpec) -> Self {
        Self {
            mu_zero: spec.mu_zero,
            sigma_zero: spec.sigma_zero,
            alpha_zero: spec.alpha_zero,
            m_alpha: spec.m_alpha,
        }
    }
}

impl Distribution for NormalMeanPseudoAlphaSkewNormal {
    fn nll(&self, data: &ArrayView1<f64>) -> f64 {
        // Implement the NLL calculation for the Skew Normal distribution
        // This is a placeholder implementation; replace with actual logic
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let sum_x: f64 = data.sum();
        let sum_x_squared: f64 = data.iter().map(|&x| x * x).sum();

        // Placeholder logic for demonstration purposes
        let sample_mean = sum_x / n;
        let sample_var: f64 = if n > 1.0 {
            (sum_x_squared - n * sample_mean * sample_mean) / (n - 1.0)
        } else {
            0.0
        };

        if sample_var.is_nan() || sample_var <= 0.0 {
            warn!("Warning: Invalid sample variance: {}", sample_var);
            return std::f64::INFINITY;
        }

        let prior_precision = 1.0 / (self.sigma_zero * self.sigma_zero);
        let data_precision = n / sample_var.max(1e-10);
        let posterior_precision = prior_precision + data_precision;
        let posterior_var = 1.0 / posterior_precision;
        let posterior_mean = posterior_var * (prior_precision * self.mu_zero + data_precision * sample_mean);

        // Calculate skewness, then update alpha and lastly posterior xi
        let mut skewness_sum = 0.0;
        let std_dev = sample_var.max(1e-7).sqrt();
        for &x in data.iter() {
            let z = (x - sample_mean) / std_dev;
            skewness_sum += z.powi(3);
        }
        let gamma = skewness_sum / n;
        let gamma = gamma.clamp(-0.99, 0.99);
        let delta = gamma.signum() * (
            PI / 2. * gamma.abs().powf(2./3.) / (
                (gamma.abs().powf(2./3.) + ((4. - PI)/2.).powf(2./3.))
            )
        ).sqrt();

        let alpha = delta.signum() * (delta.abs() / (1. - delta.powi(2))).powf(1./3.);
        let posterior_alpha = n / (n + self.m_alpha) * alpha + self.alpha_zero * self.m_alpha / (n + self.m_alpha);
        let mut posterior_omega = sample_var.sqrt() / (1.0 - 2. * delta.powi(2) / PI);
        if posterior_omega.is_nan() || !posterior_omega.is_finite() {
            warn!("Warning: Invalid posterior_omega: {}", posterior_omega);
            return std::f64::INFINITY;
        } else if posterior_omega <= 0.0 {
            warn!("Warning: Non-positive posterior_omega: {}", posterior_omega);
            posterior_omega = 1e-5; // Avoid division by zero
        }
        let posterior_xi = posterior_mean - posterior_omega * posterior_alpha / (1. + posterior_alpha.powi(2)).sqrt() * (2./PI).sqrt();



        // Extract non-pdf/cdf calculations to closed form, iterate over data for others, use statrs for pdf/cdf
        - n * (2.0_f64).ln() + n * posterior_omega.ln() + 0.5 * n * (2.0_f64 * PI).ln() -
        data.iter().map(|&x| {
            let zi = (x - posterior_xi) / posterior_omega;
            -0.5 * zi * zi + (0.5 * (1.0 + erf::erf(posterior_alpha * zi / SQRT_2))).ln()
        }).sum::<f64>()
    }
}


#[derive(Debug, Clone)]
pub struct NormalMeanNormalGammaSkewNormalSpec {
    pub mu_zero: f64,
    pub sigma_zero: f64,
    pub mu_gamma: f64,
    pub sigma_gamma: f64,
}

pub struct NormalMeanNormalGammaSkewNormal {
    mu_zero: f64,
    sigma_zero: f64,
    mu_gamma: f64,
    sigma_gamma: f64,
}

impl NormalMeanNormalGammaSkewNormal {
    pub fn new(spec: &NormalMeanNormalGammaSkewNormalSpec) -> Self {
        Self {
            mu_zero: spec.mu_zero,
            sigma_zero: spec.sigma_zero,
            mu_gamma: spec.mu_gamma,
            sigma_gamma: spec.sigma_gamma,
        }
    }
}

impl Distribution for NormalMeanNormalGammaSkewNormal {
    fn nll(&self, data: &ArrayView1<f64>) -> f64 {
        // Implement the NLL calculation for the Skew Normal distribution
        // This is a placeholder implementation; replace with actual logic
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let sum_x: f64 = data.sum();
        let sum_x_squared: f64 = data.iter().map(|&x| x * x).sum();

        // Placeholder logic for demonstration purposes
        let sample_mean = sum_x / n;
        let sample_var: f64 = if n > 1.0 {
            (sum_x_squared - n * sample_mean * sample_mean) / (n - 1.0)
        } else {
            0.0
        };

        if sample_var.is_nan() || sample_var < 0.0 {
            warn!("Warning: Invalid sample variance: {}", sample_var);
            return std::f64::INFINITY;
        }

        let mut skewness_sum = 0.0;
        let std_dev = sample_var.max(1e-7).sqrt();
        for &x in data.iter() {
            let z = (x - sample_mean) / std_dev;
            skewness_sum += z.powi(3);
        }
        let mut sample_skewness = skewness_sum / n;
        let var_skewness = (6. * n * (n - 1.0)) /
            ((n - 2.0) * (n + 1.0) * (n + 3.0));
        let posterior_skewness = (self.mu_gamma / self.sigma_gamma.powi(2) * n * sample_skewness / var_skewness.max(1e-7)) /
            (1.0 / self.sigma_gamma.powi(2) + n / var_skewness.max(1e-7));
        sample_skewness = posterior_skewness.clamp(-0.995, 0.995);

        let posterior_mean = (
            self.mu_zero / self.sigma_zero.powi(2) +
            n * sample_mean / sample_var.max(1e-7)
        ) / (
            1.0 / self.sigma_zero.powi(2) + n / sample_var.max(1e-7)
        );

        let delta = sample_skewness.signum() * (
            PI / 2. * sample_skewness.abs().powf(2./3.) / (
                (sample_skewness.abs().powf(2./3.) + ((4. - PI)/2.).powf(2./3.))
            )
        ).sqrt();

        let posterior_alpha = delta.signum() * (delta.abs() / (1. - delta.powi(2))).powf(1./3.);
        let mut posterior_omega = sample_var.sqrt() / (1.0 - 2. * delta.powi(2) / PI);
        if posterior_omega.is_nan() || !posterior_omega.is_finite() {
            warn!("Warning: Invalid posterior_omega: {}", posterior_omega);
            return std::f64::INFINITY;
        } else if posterior_omega <= 0.0 {
            warn!("Warning: Non-positive posterior_omega: {}", posterior_omega);
            posterior_omega = 1e-7; // Avoid division by zero
        }
        let posterior_xi = posterior_mean - posterior_omega * posterior_alpha / (1. + posterior_alpha.powi(2)).sqrt() * (2./PI).sqrt();

        // Extract non-pdf/cdf calculations to closed form, iterate over data for others, use statrs for pdf/cdf
        - n * (2.0_f64).ln() + n * posterior_omega.ln() + 0.5 * n * (2.0_f64 * PI).ln() -
        data.iter().map(|&x| {
            let zi = (x - posterior_xi) / posterior_omega;
            -0.5 * zi * zi + (0.5 * (1.0 + erf::erf(posterior_alpha * zi / SQRT_2))).ln()
        }).sum::<f64>()
    }
}
