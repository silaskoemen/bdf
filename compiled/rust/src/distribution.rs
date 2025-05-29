use ndarray::ArrayView1;
use pyo3::prelude::*;
use numpy::ToPyArray;
use std::f64::consts::PI;
//use statrs::function::erf;
use statrs::distribution::{Normal, ContinuousCDF, Continuous};

// Unified Distribution trait
pub trait Distribution: Sync + Send {
    fn nll(&self, data: &ArrayView1<f64>) -> f64;
}

// Specs for different distribution types
#[derive(Debug, Clone)]
pub struct NormalNormalSpec {
    pub prior_mean: f64,
    pub prior_std: f64,
}

#[derive(Debug, Clone)]
pub struct NormalEBSkewNormalSpec {
    pub prior_mu: f64,
    pub prior_sigma: f64,
    pub prior_mean_alpha: f64,
    pub prior_m_alpha: f64,
}

// Native implementation
pub struct NormalNormal {
    prior_mean: f64,
    prior_std: f64,
}

impl NormalNormal {
    pub fn new(spec: &NormalNormalSpec) -> Self {
        Self {
            prior_mean: spec.prior_mean,
            prior_std: spec.prior_std,
        }
    }
}

// Update your NormalNormal implementation to use SIMD
impl Distribution for NormalNormal {
    fn nll(&self, data: &ArrayView1<f64>) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        // Use SIMD acceleration for these calculations
        let sum_x: f64 = data.sum();
        let sum_x_squared: f64 = data.iter().map(|&x| x * x).sum();

        let prior_precision = 1.0 / (self.prior_std * self.prior_std);
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
        let posterior_mean = posterior_var * (prior_precision * self.prior_mean + data_precision * sample_mean);

        // Calculate final NLL - this could also use SIMD but would require another helper function
        0.5 * n * (2.0 * PI).ln() +
        0.5 * n * posterior_var.ln() +
        0.5 / posterior_var * data.iter()
            .map(|&x| (x - posterior_mean).powi(2))
            .sum::<f64>()
    }
}

pub struct NormalEBSkewNormal {
    prior_mu: f64,
    prior_sigma: f64,
    prior_mean_alpha: f64,
    prior_m_alpha: f64,
}

impl NormalEBSkewNormal {
    pub fn new(spec: &NormalEBSkewNormalSpec) -> Self {
        Self {
            prior_mu: spec.prior_mu,
            prior_sigma: spec.prior_sigma,
            prior_mean_alpha: spec.prior_mean_alpha,
            prior_m_alpha: spec.prior_m_alpha,
        }
    }
}

impl Distribution for NormalEBSkewNormal {
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
            println!("Warning: Invalid sample variance: {}", sample_var);
            return std::f64::INFINITY;
        }

        let prior_precision = 1.0 / (self.prior_sigma * self.prior_sigma);
        let data_precision = n / sample_var.max(1e-10);
        let posterior_precision = prior_precision + data_precision;
        let posterior_var = 1.0 / posterior_precision;
        let posterior_mean = posterior_var * (prior_precision * self.prior_mu + data_precision * sample_mean);

        // Calculate skewness, then update alpha and lastly posterior xi
        let mut skewness_sum = 0.0;
        let std_dev = sample_var.max(1e-7).sqrt();
        for &x in data.iter() {
            let z = (x - sample_mean) / std_dev;
            skewness_sum += z.powi(3);
        }
        let gamma = skewness_sum / n;
        let gamma = gamma.clamp(-0.995, 0.995);
        let delta = gamma.signum() * (
            PI / 2. * gamma.abs().powf(2./3.) / (
                (gamma.abs().powf(2./3.) + ((4. - PI)/2.).powf(2./3.))
            )
        ).sqrt();

        let alpha = delta.signum() * (delta.abs() / (1. - delta.powi(2))).powf(1./3.);
        let posterior_alpha = n / (n + self.prior_m_alpha) * alpha + self.prior_mean_alpha * self.prior_m_alpha / (n + self.prior_m_alpha);
        let posterior_omega = sample_var.sqrt() / (1.0 - 2. * delta.powi(2) / PI);
        let posterior_xi = posterior_mean - posterior_omega * posterior_alpha / (1. + posterior_alpha.powi(2)).sqrt() * (2./PI).sqrt();
        println!("Posterior xi: {}, omega: {}, alpha: {}", posterior_xi, posterior_omega, posterior_alpha);

        // Standardize the data
        // Final NLL calculation, leverage implementations of normal pdf and cdf from statrs
        let log_2 = (2.0_f64).ln();
        let log_posterior_omega = posterior_omega.ln();

        if log_posterior_omega.is_nan() || log_posterior_omega.is_infinite() {
            println!("Warning: Invalid log_posterior_omega: {}", log_posterior_omega);
            return std::f64::INFINITY;
        }

        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut nll = 0.0;

        for &x in data.iter() {
            let z = (x - posterior_xi) / posterior_omega;
            let cdf_value = normal.cdf(posterior_alpha * z).max(1e-10);
            let point_nll = -log_2 + log_posterior_omega - normal.ln_pdf(z) - cdf_value.ln();

            // Check for invalid values
            if point_nll.is_nan() || point_nll.is_infinite() {
                println!("Warning: Point NLL is invalid: {} for z={}, cdf={}",
                        point_nll, z, cdf_value);
                continue; // Skip this point
            }

            nll += point_nll;
        }

        println!("Final NLL: {} | xi: {}, omega: {}, alpha: {}",
                nll, posterior_xi, posterior_omega, posterior_alpha);
        nll
    }
}

// Python wrapper implementation
pub struct PythonDistributionWrapper {
    pub py_dist: PyObject,
}

unsafe impl Send for PythonDistributionWrapper {}
unsafe impl Sync for PythonDistributionWrapper {}

impl PythonDistributionWrapper {
    pub fn new(py_dist: PyObject) -> Self {
        Self { py_dist }
    }
}

impl Distribution for PythonDistributionWrapper {
    fn nll(&self, data: &ArrayView1<f64>) -> f64 {
        Python::with_gil(|py| {
            let y_array = data.to_pyarray(py);
            let result = self.py_dist
                .call_method1(py, "nll", (y_array,))
                .expect("Failed to call nll method");

            result.extract(py).expect("Failed to extract NLL value")
        })
    }
}
