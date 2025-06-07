use crate::distributions::Distribution;
use ndarray::ArrayView1;
use std::f64::consts::PI;

// Specs for different distribution types
#[derive(Debug, Clone)]
pub struct NormalMuNormalSpec {
    pub mu_zero: f64,
    pub sigma_zero: f64,
}

// Native implementation
pub struct NormalMuNormal {
    mu_zero: f64,
    sigma_zero: f64,
}

impl NormalMuNormal {
    pub fn new(spec: &NormalMuNormalSpec) -> Self {
        Self {
            mu_zero: spec.mu_zero,
            sigma_zero: spec.sigma_zero,
        }
    }
}

// Update your NormalNormal implementation to use SIMD
impl Distribution for NormalMuNormal {
    fn nll(&self, data: &ArrayView1<f64>) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        // Use SIMD acceleration for these calculations
        let sum_x: f64 = data.sum();
        let sum_x_squared: f64 = data.iter().map(|&x| x * x).sum();

        let prior_precision = 1.0 / (self.sigma_zero * self.sigma_zero);
        let sample_mean = sum_x / n;
        let sample_var: f64 = if n > 1.0 {
            (sum_x_squared - n * sample_mean * sample_mean) / (n - 1.0)
        } else {
            0.0
        };

        // Rest of your calculation remains the same...
        let data_precision= n / sample_var.max(1e-10);
        let posterior_precision = prior_precision + data_precision;
        let posterior_var = 1.0 / posterior_precision;
        let posterior_mean = posterior_var * (prior_precision * self.mu_zero + data_precision * sample_mean);

        // Calculate final NLL - this could also use SIMD but would require another helper function
        0.5 * n * (2.0 * PI).ln() +
        0.5 * n * posterior_var.ln() +
        0.5 / posterior_var * data.iter()
            .map(|&x| (x - posterior_mean).powi(2))
            .sum::<f64>()
    }
}
