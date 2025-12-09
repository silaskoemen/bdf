use super::{DistributionPrimitives, SufficientStats};
use ndarray::{ArrayView1, Array1};
use pyo3::types::PyDict;
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
}

#[derive(Clone, Copy, Debug)]
pub enum BandwidthRule {
    Scott,
    Silverman,
    Fixed(f64),
}

/// Standard KDE implementation
pub struct Kde {
    pub kernel: KernelType,
    pub bandwidth_rule: BandwidthRule,
    pub min_bandwidth: f64,
}

impl Kde {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let kernel_any = spec.get_item("kernel")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'kernel'"))?;
        let kernel_str: String = kernel_any.extract()?;
        let kernel = match kernel_str.as_str() {
            "gaussian" => KernelType::Gaussian,
            "epanechnikov" => KernelType::Epanechnikov,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown KDE kernel")),
        };

        let bw_item = spec.get_item("bandwidth")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'bandwidth'"))?;

        let bandwidth_rule = if let Ok(h) = bw_item.extract::<f64>() {
            BandwidthRule::Fixed(h)
        } else if let Ok(s) = bw_item.extract::<String>() {
            match s.as_str() {
                "scott" => BandwidthRule::Scott,
                "silverman" => BandwidthRule::Silverman,
                _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown bandwidth rule")),
            }
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid bandwidth type"));
        };

        let min_bandwidth: f64 = spec.get_item("min_bandwidth")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'min_bandwidth'"))?
            .extract()?;

        Ok(Self { kernel, bandwidth_rule, min_bandwidth })
    }

    pub fn compute_bandwidth(&self, data: &ArrayView1<f64>) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 {
            return self.min_bandwidth;
        }

        match self.bandwidth_rule {
            BandwidthRule::Fixed(h) => h.max(self.min_bandwidth),
            BandwidthRule::Scott => {
                // Scott's rule: h = σ * n^(-1/5)
                let std = sample_std(data);
                (std * n.powf(-0.2)).max(self.min_bandwidth)
            }
            BandwidthRule::Silverman => {
                // Silverman's rule: h = 0.9 * min(σ, IQR/1.349) * n^(-1/5)
                let std = sample_std(data);

                let mut sorted_data = data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let q25 = percentile_sorted(&sorted_data, 0.25);
                let q75 = percentile_sorted(&sorted_data, 0.75);
                let iqr = q75 - q25;

                let scale = if iqr > 0.0 {
                    std.min(iqr / 1.349)
                } else {
                    std
                };

                let used_scale = if scale <= 0.0 { std.max(self.min_bandwidth) } else { scale };

                (0.9 * used_scale * n.powf(-0.2)).max(self.min_bandwidth)
            }
        }
    }

    fn log_kernel_row(&self, x: f64, ref_data: &ArrayView1<f64>, h: f64) -> Array1<f64> {
        match self.kernel {
            KernelType::Gaussian => {
                let log_h = h.ln();
                const LOG_2PI: f64 = 1.8378770664093453; // ln(2*pi)

                ref_data.mapv(|y| {
                    let diff = (x - y) / h;
                    -0.5 * diff * diff - 0.5 * LOG_2PI - log_h
                })
            },
            KernelType::Epanechnikov => {
                let log_h = h.ln();
                let log_075 = 0.75f64.ln();

                ref_data.mapv(|y| {
                    let u = (x - y) / h;
                    if u.abs() <= 1.0 {
                        let one_minus_u2 = 1.0 - u * u;
                        if one_minus_u2 > 0.0 {
                            log_075 + one_minus_u2.ln() - log_h
                        } else {
                            f64::NEG_INFINITY
                        }
                    } else {
                        f64::NEG_INFINITY
                    }
                })
            }
        }
    }

    fn plugin_log_likelihood_impl(&self, eval_points: &ArrayView1<f64>, ref_data: &ArrayView1<f64>, h: f64) -> Array1<f64> {
        let n_ref = ref_data.len() as f64;
        if n_ref == 0.0 {
            return Array1::from_elem(eval_points.len(), f64::NEG_INFINITY);
        }
        let log_n_ref = n_ref.ln();

        // For each evaluation point, compute logsumexp of kernel contributions
        eval_points.mapv(|x| {
            let log_k = self.log_kernel_row(x, ref_data, h);

            // LogSumExp for numerical stability
            let max_val = log_k.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if max_val == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                let sum_exp: f64 = log_k.mapv(|v| (v - max_val).exp()).sum();
                max_val + sum_exp.ln() - log_n_ref
            }
        })
    }
}

fn sample_std(data: &ArrayView1<f64>) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean: f64 = data.sum() / n;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

fn percentile_sorted(sorted_data: &[f64], q: f64) -> f64 {
    let n = sorted_data.len();
    if n == 0 { return 0.0; }
    let idx = (n as f64 - 1.0) * q;
    let i = idx.floor() as usize;
    let f = idx - i as f64;
    if i + 1 < n {
        sorted_data[i] * (1.0 - f) + sorted_data[i + 1] * f
    } else {
        sorted_data[i]
    }
}

impl DistributionPrimitives for Kde {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len();
        let mut params = HashMap::new();
        if n < 2 {
            return params;
        }
        let h = self.compute_bandwidth(data);
        params.insert("bandwidth".to_string(), h);
        params.insert("n".to_string(), n as f64);
        params
    }

    fn supports_rust_params(&self) -> bool {
        false // Can't add the full data array to HashMap<String, f64>
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, _params: &HashMap<String, f64>) -> Array1<f64> {
        // Recompute bandwidth from data (self-consistent plug-in)
        let h = self.compute_bandwidth(data);
        self.plugin_log_likelihood_impl(data, data, h)
    }

    fn posterior_predictive_log_likelihood(&self, _data: &ArrayView1<f64>,
        _params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        None // KDE doesn't have posterior predictive
    }

    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let h = self.compute_bandwidth(data);
        let ll = self.plugin_log_likelihood_impl(data, data, h);
        -ll.sum()
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let h = self.compute_bandwidth(train);
        let ll = self.plugin_log_likelihood_impl(test, train, h);
        -ll.sum()
    }

    fn nll_suff_stats(&self, _stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        None // KDE cannot use sufficient statistics
    }
}

/// Bayesian KDE implementation (pseudo-Bayesian bandwidth)
pub struct BayesianKde {
    base: Kde,
    prior_h: f64,
    m_h: f64,
}

impl BayesianKde {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let base = Kde::from_spec(spec)?;
        let prior_h: f64 = spec.get_item("prior_h")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'prior_h'"))?
            .extract()?;
        let m_h: f64 = spec.get_item("m_h")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'm_h'"))?
            .extract()?;
        Ok(Self { base, prior_h, m_h })
    }
}

impl DistributionPrimitives for BayesianKde {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let mut params = HashMap::new();
        if n < 2.0 {
            return params;
        }

        let data_h = self.base.compute_bandwidth(data);

        // Bayesian bandwidth update: posterior_h = (m_h * prior_h + n * data_h) / (m_h + n)
        let posterior_h = (self.m_h * self.prior_h + n * data_h) / (self.m_h + n);

        params.insert("bandwidth".to_string(), posterior_h);
        params.insert("n".to_string(), n);
        params
    }

    fn supports_rust_params(&self) -> bool {
        false // Can't add the full data array to HashMap<String, f64>
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let h = if let Some(&bw) = params.get("bandwidth") {
            bw
        } else {
            self.base.compute_bandwidth(data)
        };
        self.base.plugin_log_likelihood_impl(data, data, h)
    }

    fn posterior_predictive_log_likelihood(&self, _data: &ArrayView1<f64>, _params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        None
    }

    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(data);
        let h = *params.get("bandwidth").unwrap_or(&self.base.min_bandwidth);

        let ll = self.base.plugin_log_likelihood_impl(data, data, h);
        -ll.sum()
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let params = self.calc_posterior_params(train);
        let h = *params.get("bandwidth").unwrap_or(&self.base.min_bandwidth);

        let ll = self.base.plugin_log_likelihood_impl(test, train, h);
        -ll.sum()
    }

    fn nll_suff_stats(&self, _stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        None
    }
}
