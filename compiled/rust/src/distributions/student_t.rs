use super::{DistributionPrimitives, SufficientStats};
use ndarray::{ArrayView1, Array1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::f64::consts::PI;

// ============================================================================
// SHARED HELPERS (used by both FrequentistStudentT and NormalMeanStudentT)
// ============================================================================

/// Lanczos approximation for log-gamma function.
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

/// Compute Student-t log-pdf for a single point.
#[inline]
fn student_t_logpdf_val(x: f64, mu: f64, sigma: f64, df: f64) -> f64 {
    let half_df = df / 2.0;
    let half_df_plus_half = (df + 1.0) / 2.0;

    let log_c = lgamma(half_df_plus_half) - lgamma(half_df)
        - 0.5 * (df * PI).ln() - sigma.ln();

    let z = (x - mu) / sigma;
    log_c - half_df_plus_half * (1.0 + z * z / df).ln()
}

/// Compute Student-t NLL for an array given parameters.
fn student_t_nll_array(data: &ArrayView1<f64>, mu: f64, sigma: f64, df: f64) -> f64 {
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

/// Compute plug-in log-likelihood array for Student-t.
fn student_t_plugin_ll(data: &ArrayView1<f64>, mu: f64, sigma: f64, df: f64) -> Array1<f64> {
    let sigma = sigma.max(1e-10);
    let half_df = df / 2.0;
    let half_df_plus_half = (df + 1.0) / 2.0;

    let log_c = lgamma(half_df_plus_half) - lgamma(half_df)
        - 0.5 * (df * PI).ln() - sigma.ln();

    data.mapv(|x| {
        let z = (x - mu) / sigma;
        log_c - half_df_plus_half * (1.0 + z * z / df).ln()
    })
}

/// Estimate df from data using Method of Moments (excess kurtosis).
fn estimate_df_mom(mu: f64, sigma2: f64, data: &ArrayView1<f64>) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return 5.0;  // Default fallback for small samples
    }

    let m4: f64 = data.iter().map(|&x| (x - mu).powi(4)).sum::<f64>() / n;
    let excess_kurtosis = m4 / (sigma2 * sigma2) - 3.0;

    if excess_kurtosis <= 0.0 {
        100.0  // Light tails -> large df (approaches Normal)
    } else {
        (6.0 / excess_kurtosis + 4.0).clamp(2.05, 100.0)
    }
}

/// Compute sample mean and variance (ddof=1) from data.
#[inline]
fn sample_mean_var(data: &ArrayView1<f64>) -> (f64, f64) {
    let n = data.len() as f64;
    let mu = data.mean().unwrap();
    let sum_sq: f64 = data.iter().map(|&x| (x - mu).powi(2)).sum();
    let sigma2 = if n > 1.0 { sum_sq / (n - 1.0) } else { 1.0 };
    (mu, sigma2.max(1e-10))
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
        let df: Option<f64> = spec.get_item("df")
            .and_then(|item| item.extract().ok());
        Ok(Self { df })
    }

    /// Get df: fixed if Some, estimated from data if None.
    fn get_df(&self, mu: f64, sigma2: f64, data: &ArrayView1<f64>) -> f64 {
        match self.df {
            Some(fixed_df) => fixed_df,
            None => estimate_df_mom(mu, sigma2, data),
        }
    }

    /// Full MoM fit. Returns (mu, sigma, df).
    fn fit_mom(&self, data: &ArrayView1<f64>) -> (f64, f64, f64) {
        let n = data.len() as f64;
        if n == 0.0 {
            let df = self.df.unwrap_or(5.0);
            return (0.0, 1.0, df);
        }

        let (mu, sigma2) = sample_mean_var(data);
        let df = self.get_df(mu, sigma2, data);
        let sigma = (sigma2 * (df - 2.0) / df).sqrt().max(1e-8);

        (mu, sigma, df)
    }
}

impl DistributionPrimitives for FrequentistStudentT {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;

        if n == 0.0 {
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
        0
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>)
        -> Array1<f64> {
        student_t_plugin_ll(data, params["mu"], params["sigma"], params["df"])
    }

    fn posterior_predictive_log_likelihood(&self, _data: &ArrayView1<f64>,
        _params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        None
    }

    fn nle(&self, _data: &ArrayView1<f64>) -> Option<f64> {
        None
    }

    fn nle_suff_stats(&self, _stats: &SufficientStats) -> Option<f64> {
        None
    }

    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let (mu, sigma, df) = self.fit_mom(data);
        student_t_nll_array(data, mu, sigma, df)
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>,
        _use_posterior_predictive: bool) -> f64 {
        let n = train.len() as f64;
        if n == 0.0 { return 0.0; }

        let (mu, sigma, df) = self.fit_mom(train);
        student_t_nll_array(test, mu, sigma, df)
    }

    fn nll_suff_stats(&self, _stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        None
    }
}

// ============================================================================
// NormalMeanStudentT (CLT moment prior on the mean)
// ============================================================================

pub struct NormalMeanStudentT {
    mu_mean: f64,      // prior mean for mu
    sigma_mean: f64,   // prior std for mu
    df: Option<f64>,   // degrees of freedom (if None, estimate via MoM)
}

impl NormalMeanStudentT {
    pub fn new(mu_mean: f64, sigma_mean: f64, df: Option<f64>) -> Self {
        Self { mu_mean, sigma_mean, df }
    }

    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let mu_mean: f64 = spec.get_item("mu_mean")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'mu_mean'"))?
            .extract()?;
        let sigma_mean: f64 = spec.get_item("sigma_mean")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'sigma_mean'"))?
            .extract()?;
        let df: Option<f64> = spec.get_item("df")
            .and_then(|item| item.extract().ok());
        Ok(Self { mu_mean, sigma_mean, df })
    }

    /// Fit with Normal prior on mean + MoM for df/sigma.
    /// Returns (mu, sigma, df).
    fn fit_with_prior(&self, data: &ArrayView1<f64>) -> (f64, f64, f64) {
        let n = data.len() as f64;
        if n < 2.0 {
            let df = self.df.unwrap_or(5.0);
            return (self.mu_mean, 1.0, df);
        }

        let (sample_mean, sample_var) = sample_mean_var(data);

        // 1. Bayesian update on mean (Normal-Normal conjugate, CLT approximation)
        let prior_prec = 1.0 / (self.sigma_mean * self.sigma_mean);
        let data_prec = n / sample_var;
        let post_prec = prior_prec + data_prec;
        let post_mu = (prior_prec * self.mu_mean + data_prec * sample_mean) / post_prec;

        // 2. Estimate df
        let df = match self.df {
            Some(fixed_df) => fixed_df,
            None => estimate_df_mom(sample_mean, sample_var, data),
        };

        // 3. Recover sigma from variance identity: Var(Y) = sigma^2 * df/(df-2)
        let sigma = (sample_var * (df - 2.0) / df).sqrt().max(1e-8);

        (post_mu, sigma, df)
    }
}

impl DistributionPrimitives for NormalMeanStudentT {
    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;

        if n == 0.0 {
            let df = self.df.unwrap_or(5.0);
            return HashMap::from([
                ("mu".to_string(), self.mu_mean),
                ("sigma".to_string(), 1.0),
                ("df".to_string(), df),
                ("posterior_mean_mu".to_string(), self.mu_mean),
            ]);
        }

        let (mu, sigma, df) = self.fit_with_prior(data);

        HashMap::from([
            ("mu".to_string(), mu),
            ("sigma".to_string(), sigma),
            ("df".to_string(), df),
            ("posterior_mean_mu".to_string(), mu),
        ])
    }

    fn required_moment_order(&self) -> usize {
        0  // log(1 + z^2/nu) is non-polynomial -> cannot use suff stats
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>)
        -> Array1<f64> {
        student_t_plugin_ll(data, params["mu"], params["sigma"], params["df"])
    }

    fn posterior_predictive_log_likelihood(&self, _data: &ArrayView1<f64>,
        _params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        None
    }

    fn nle(&self, _data: &ArrayView1<f64>) -> Option<f64> {
        None
    }

    fn nle_suff_stats(&self, _stats: &SufficientStats) -> Option<f64> {
        None
    }

    fn nll(&self, data: &ArrayView1<f64>, _use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let (mu, sigma, df) = self.fit_with_prior(data);
        student_t_nll_array(data, mu, sigma, df)
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>,
        _use_posterior_predictive: bool) -> f64 {
        let n = train.len() as f64;
        if n == 0.0 { return 0.0; }

        let (mu, sigma, df) = self.fit_with_prior(train);
        student_t_nll_array(test, mu, sigma, df)
    }

    fn nll_suff_stats(&self, _stats: &SufficientStats, _use_posterior_predictive: bool) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ========================================================================
    // FrequentistStudentT tests
    // ========================================================================

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
        let data = array![0.0, 0.0, 0.0, 0.0, 10.0];
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
        assert!(nll > 0.0);
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
        // scipy.stats.t.logpdf(2.0, df=5, loc=1, scale=0.5) = -2.0388324032
        let logpdf = student_t_logpdf_val(2.0, 1.0, 0.5, 5.0);
        assert!((logpdf - (-2.0388324032)).abs() < 1e-6);
    }

    // ========================================================================
    // NormalMeanStudentT tests
    // ========================================================================

    #[test]
    fn test_normal_mean_prior_shrinkage() {
        // Prior at 0, data mean at 10 -> posterior mean should be between 0 and 10
        let dist = NormalMeanStudentT::new(0.0, 1.0, Some(5.0));
        let data = array![8.0, 9.0, 10.0, 11.0, 12.0];
        let params = dist.calc_posterior_params(&data.view());

        let post_mu = params["mu"];
        assert!(post_mu > 0.0, "Posterior mean should be > prior mean");
        assert!(post_mu < 10.0, "Posterior mean should be < data mean");
    }

    #[test]
    fn test_normal_mean_weak_prior() {
        // Very wide prior -> posterior should be close to data mean
        let dist = NormalMeanStudentT::new(0.0, 1000.0, Some(5.0));
        let data = array![8.0, 9.0, 10.0, 11.0, 12.0];
        let params = dist.calc_posterior_params(&data.view());

        assert!((params["mu"] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_normal_mean_strong_prior() {
        // Very tight prior -> posterior should be close to prior mean
        let dist = NormalMeanStudentT::new(0.0, 0.001, Some(5.0));
        let data = array![8.0, 9.0, 10.0, 11.0, 12.0];
        let params = dist.calc_posterior_params(&data.view());

        assert!(params["mu"].abs() < 1.0, "Strong prior should pull mu close to 0");
    }

    #[test]
    fn test_normal_mean_fixed_df() {
        let dist = NormalMeanStudentT::new(0.0, 1.0, Some(5.0));
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = dist.calc_posterior_params(&data.view());

        assert!((params["df"] - 5.0).abs() < 1e-10);
        assert!(params["sigma"] > 0.0);
    }

    #[test]
    fn test_normal_mean_estimated_df() {
        let dist = NormalMeanStudentT::new(0.0, 10.0, None);
        let data = array![0.0, 0.0, 0.0, 0.0, 10.0];
        let params = dist.calc_posterior_params(&data.view());

        assert!(params["df"] >= 2.05);
        assert!(params["df"] <= 100.0);
    }

    #[test]
    fn test_normal_mean_nll_finite() {
        let dist = NormalMeanStudentT::new(3.0, 2.0, Some(5.0));
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let nll = dist.nll(&data.view(), false);

        assert!(nll.is_finite());
        assert!(nll > 0.0);
    }

    #[test]
    fn test_normal_mean_train_test() {
        let dist = NormalMeanStudentT::new(3.0, 2.0, Some(5.0));
        let train = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let test = array![2.5, 3.5];

        let nll = dist.nll_train_test(&train.view(), &test.view(), false);
        assert!(nll.is_finite());
    }
}
