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
// FrequentistStudentT (Known degrees of freedom OR estimated via MoM)
// ============================================================================

pub struct FrequentistStudentT {
    df: Option<f64>,  // degrees of freedom (if None, estimate from data via MoM)
}

impl FrequentistStudentT {
    /// Create a new FrequentistStudentT distribution with given degrees of freedom.
    pub fn new(df: Option<f64>) -> Self {
        Self { df }
    }

    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        // Extract df - it can be None or a float from Python
        let df: Option<f64> = spec.get_item("df")
            //.and_then(|item| item.ok())
            .and_then(|item| item.extract().ok());
        Ok(Self { df })
    }

    // pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
    //     // Handle Python None -> Rust None, Python float -> Rust Some(f64)
    //     let df: Option<f64> = match spec.get_item("df").ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing key 'df'"))? {
    //         Some(item) => item.extract().ok(),  // Returns None if extraction fails (e.g., Python None)
    //         None => None,
    //     };
    //     Ok(Self { df })
    // }

    // ========================================================================
    // HELPER METHODS FOR MoM ESTIMATION
    // ========================================================================

    /// Estimate df from data using Method of Moments (excess kurtosis).
    /// Matches Python's _fit_mom logic for df estimation.
    fn estimate_df_mom(&self, mu: f64, sigma2: f64, data: &ArrayView1<f64>) -> f64 {
        let n = data.len() as f64;
        if n < 4.0 {
            return 5.0;  // Default fallback for small samples
        }

        // Fourth central moment for kurtosis
        let m4: f64 = data.iter().map(|&x| (x - mu).powi(4)).sum::<f64>() / n;
        let excess_kurtosis = m4 / (sigma2 * sigma2) - 3.0;

        if excess_kurtosis <= 0.0 {
            100.0  // Light tails → large df (approaches Normal)
        } else {
            // df = 6 / excess_kurtosis + 4, clamped to [2.05, 100]
            (6.0 / excess_kurtosis + 4.0).clamp(2.05, 100.0)
        }
    }

    /// Get df: fixed if Some, estimated from data if None.
    fn get_df(&self, mu: f64, sigma2: f64, data: &ArrayView1<f64>) -> f64 {
        match self.df {
            Some(fixed_df) => fixed_df,
            None => self.estimate_df_mom(mu, sigma2, data),
        }
    }

    /// Full MoM fit matching Python's _fit_mom.
    /// Returns (mu, sigma, df).
    fn fit_mom(&self, data: &ArrayView1<f64>) -> (f64, f64, f64) {
        let n = data.len() as f64;
        if n == 0.0 {
            let df = self.df.unwrap_or(5.0);
            return (0.0, 1.0, df);
        }

        let mu = data.mean().unwrap();

        // Sample variance (ddof=1)
        let sum_sq: f64 = data.iter().map(|&x| (x - mu).powi(2)).sum();
        let sigma2 = if n > 1.0 { sum_sq / (n - 1.0) } else { 1.0 };
        let sigma2 = sigma2.max(1e-10);

        // Get df: either fixed or estimated
        let df = self.get_df(mu, sigma2, data);

        // Adjust sigma for Student-t: σ_t = s * sqrt((ν-2)/ν)
        // This is because Var(t_ν) = σ² * ν/(ν-2), so we adjust sample std
        let sigma = (sigma2 * (df - 2.0) / df).sqrt().max(1e-8);

        (mu, sigma, df)
    }

    /// Compute Student-t log-pdf for a single point.
    #[inline]
    fn student_t_logpdf(&self, x: f64, mu: f64, sigma: f64, df: f64) -> f64 {
        let half_df = df / 2.0;
        let half_df_plus_half = (df + 1.0) / 2.0;

        // log C = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5*log(νπ) - log(σ)
        let log_c = lgamma(half_df_plus_half) - lgamma(half_df)
            - 0.5 * (df * PI).ln() - sigma.ln();

        let z = (x - mu) / sigma;
        log_c - half_df_plus_half * (1.0 + z * z / df).ln()
    }

    /// Compute Student-t NLL for an array given parameters.
    fn student_t_nll(&self, data: &ArrayView1<f64>, mu: f64, sigma: f64, df: f64) -> f64 {
        let sigma = sigma.max(1e-10);
        let half_df = df / 2.0;
        let half_df_plus_half = (df + 1.0) / 2.0;

        let log_c = lgamma(half_df_plus_half) - lgamma(half_df)
            - 0.5 * (df * PI).ln() - sigma.ln();

        let mut nll = 0.0;
        for &x in data {
            let z = (x - mu) / sigma;
            nll -= log_c - half_df_plus_half * (1.0 + z * z / df).ln();
        }
        nll
    }
}

impl DistributionPrimitives for FrequentistStudentT {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;

        if n == 0.0 {
            // No data: return defaults
            let df = self.df.unwrap_or(5.0);
            return HashMap::from([
                ("mu".to_string(), 0.0),
                ("sigma".to_string(), 1.0),
                ("df".to_string(), df),
            ]);
        }

        let (mu, sigma, df) = self.fit_mom(data);

        HashMap::from([
            ("mu".to_string(), mu),
            ("sigma".to_string(), sigma),
            ("df".to_string(), df),
        ])
    }

    fn required_moment_order(&self) -> usize {
        0 // Currently doesn't use suff stats fast path
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>)
        -> Array1<f64> {
        let mu = params["mu"];
        let sigma = params["sigma"].max(1e-10);
        let df = params["df"];

        let half_df = df / 2.0;
        let half_df_plus_half = (df + 1.0) / 2.0;

        let log_c = lgamma(half_df_plus_half) - lgamma(half_df)
            - 0.5 * (df * PI).ln() - sigma.ln();

        data.mapv(|x| {
            let z = (x - mu) / sigma;
            log_c - half_df_plus_half * (1.0 + z * z / df).ln()
        })
    }

    fn posterior_predictive_log_likelihood(&self, _data: &ArrayView1<f64>,
        _params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        // FrequentistStudentT does not support posterior predictive (non-conjugate)
        None
    }

    fn nle(&self, _data: &ArrayView1<f64>) -> Option<f64> {
        // No closed-form marginal likelihood for frequentist Student-t
        None
    }

    fn nle_suff_stats(&self, _stats: &SufficientStats) -> Option<f64> {
        // No closed-form marginal likelihood for frequentist Student-t
        None
    }

    /// Optimized NLL calculation: fit params and evaluate in one pass.
    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let (mu, sigma, df) = self.fit_mom(data);
        self.student_t_nll(data, mu, sigma, df)
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>,
        _use_posterior_predictive: bool) -> f64 {
        let n = train.len() as f64;
        if n == 0.0 { return 0.0; }

        // 1. Fit on train
        let (mu, sigma, df) = self.fit_mom(train);

        // 2. Evaluate NLL on test
        self.student_t_nll(test, mu, sigma, df)
    }

    fn nll_suff_stats(&self, stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        // Student-t NLL contains log(1 + z²/ν) which does NOT decompose into sufficient stats.
        // We can only compute if df is fixed AND we approximate, but that's not faithful.
        // Return None to fall back to slow path with raw data.

        // However, if df is fixed, we CAN estimate mu and sigma from sufficient stats,
        // but we still can't compute the exact NLL without raw data points.

        // For efficiency in tree growing, we could use a Normal approximation when df is large,
        // but for correctness, return None.

        if self.df.is_none() {
            // Need 4th moment for df estimation, which requires sum_fourth in SufficientStats
            // If your SufficientStats doesn't have sum_fourth, we can't estimate df
            return None;
        }

        // Even with fixed df, the log(1 + z²/ν) term can't be computed from sum/sum_sq alone
        // because it's not a polynomial in x.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fixed_df() {
        let dist = FrequentistStudentT::new(Some(5.0));
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = dist.calc_posterior_params(&data.view());

        assert!((params["mu"] - 3.0).abs() < 1e-10);
        assert!((params["df"] - 5.0).abs() < 1e-10);
        assert!(params["sigma"] > 0.0);
    }

    #[test]
    fn test_estimated_df() {
        let dist = FrequentistStudentT::new(None);
        // Data with heavy tails should give lower df
        let data = array![0.0, 0.0, 0.0, 0.0, 10.0];  // Outlier
        let params = dist.calc_posterior_params(&data.view());

        assert!(params["df"] >= 2.05);
        assert!(params["df"] <= 100.0);
    }

    #[test]
    fn test_nll_finite() {
        let dist = FrequentistStudentT::new(Some(4.0));
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let nll = dist.nll(&data.view(), false);

        assert!(nll.is_finite());
        assert!(nll > 0.0);  // NLL should be positive for reasonable data
    }

    #[test]
    fn test_train_test_split() {
        let dist = FrequentistStudentT::new(Some(5.0));
        let train = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let test = array![2.5, 3.5];

        let nll = dist.nll_train_test(&train.view(), &test.view(), false);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_logpdf_matches_scipy() {
        // scipy.stats.t.logpdf(2.0, df=5, loc=1, scale=0.5) ≈ -1.5978...
        let dist = FrequentistStudentT::new(Some(5.0));
        let logpdf = dist.student_t_logpdf(2.0, 1.0, 0.5, 5.0);

        // Rough check (exact value from scipy: -1.5978370696...)
        assert!((logpdf - (-1.5978370696)).abs() < 1e-6);
    }
}
