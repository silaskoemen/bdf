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
// GammaMSLambdaNegBin (Mean-Strength Parameterization)
// ============================================================================

/// Gamma-NegativeBinomial conjugate model with mean-strength parameterization.
///
/// Prior: λ ~ Gamma(α₀, β₀) where α₀ = strength·mean, β₀ = strength
/// Likelihood: y | λ, φ ~ NegativeBinomial(λ, φ)
/// Posterior: λ | y ~ Gamma(α₀ + Σy, β₀ + n)
///
/// The dispersion φ can be fixed or estimated via MoM per split.
pub struct GammaMSLambdaNegBin {
    alpha_lambda: f64,  // Shape: strength * mean
    beta_lambda: f64,   // Rate: strength
    phi: Option<f64>,   // Dispersion (None → estimate via MoM)
    // Scoring config needed to determine suff stats eligibility
    score_method: String,
    use_posterior_predictive: bool,
}

impl GammaMSLambdaNegBin {
    pub fn from_spec(spec: &PyDict) -> PyResult<Self> {
        let mean_lambda: f64 = spec.get_item("mean_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing mean_lambda in GammaMSLambdaNegBin spec"
        ))?.extract()?;
        let strength_lambda: f64 = spec.get_item("strength_lambda").ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "Missing strength_lambda in GammaMSLambdaNegBin spec"
        ))?.extract()?;

        // phi can be None (estimate via MoM) or a fixed float
        let phi: Option<f64> = spec.get_item("phi")
            .and_then(|item| item.extract().ok());

        let score_method: String = spec.get_item("score_method")
            .and_then(|item| item.extract().ok())
            .unwrap_or_else(|| "nle".to_string());
        let use_posterior_predictive: bool = spec.get_item("use_posterior_predictive")
            .and_then(|item| item.extract().ok())
            .unwrap_or(true);

        // Convert mean-strength to shape-rate: α = m·λ₀, β = m
        let alpha_lambda = strength_lambda * mean_lambda;
        let beta_lambda = strength_lambda;

        Ok(Self {
            alpha_lambda,
            beta_lambda,
            phi,
            score_method,
            use_posterior_predictive,
        })
    }

    /// Estimate dispersion φ via Method of Moments: φ = μ²/(σ²-μ)
    fn estimate_phi_mom(mean: f64, var: f64) -> f64 {
        if var <= mean {
            // No overdispersion: φ = μ² gives Var = μ + 1 (nearly Poisson).
            // Floor at 1e4 so small means don't produce tiny φ.
            (mean * mean).max(1e4)
        } else {
            (mean * mean / (var - mean)).max(1e-6)
        }
    }

    /// Estimate φ from sufficient stats (mean, variance).
    /// Note: While φ itself CAN be estimated from suff stats, the scoring formulas
    /// contain per-point terms like Σ lgamma(y_i + φ) that cannot be computed from
    /// suff stats when φ varies per split (φ_left ≠ φ_right means the terms don't cancel).
    /// This is why required_moment_order() returns 0 for phi=None.
    #[allow(dead_code)]
    fn estimate_phi_from_stats(stats: &SufficientStats) -> f64 {
        if stats.n < 2.0 {
            return 1.0;
        }
        let mean = stats.mean();
        let var = stats.variance();
        Self::estimate_phi_mom(mean, var)
    }

    /// Get φ: fixed if Some, estimated from data if None
    fn get_phi_from_data(&self, data: &ArrayView1<f64>) -> f64 {
        match self.phi {
            Some(fixed_phi) => fixed_phi,
            None => {
                let n = data.len() as f64;
                if n < 2.0 { return 1.0; }
                let mean = data.iter().sum::<f64>() / n;
                let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                Self::estimate_phi_mom(mean, var)
            }
        }
    }

}

impl DistributionPrimitives for GammaMSLambdaNegBin {
    fn required_moment_order(&self) -> usize {
        // Suff stats fast path requires:
        // 1. Fixed φ (MoM needs individual data for lgamma(y_i + φ) when φ varies)
        // 2. NLE scoring, OR NLL with plug-in (not posterior predictive)
        //    NLL + posterior predictive has per-point lgamma(y+φ) terms that don't
        //    decompose into sufficient statistics even with fixed φ.
        let phi_fixed = self.phi.is_some();
        let scoring_ok = self.score_method == "nle"
            || (self.score_method == "nll" && !self.use_posterior_predictive);

        if phi_fixed && scoring_ok { 1 } else { 0 }
    }

    fn calc_posterior_params(&self, data: &ArrayView1<f64>) -> HashMap<String, f64> {
        let n = data.len() as f64;
        let sum_y: f64 = data.iter().sum();

        // Conjugate update: Gamma(α, β) → Gamma(α + Σy, β + n)
        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;
        let lambda_post = alpha_post / beta_post;

        let phi = self.get_phi_from_data(data);

        let mut params = HashMap::new();
        params.insert("posterior_alpha".to_string(), alpha_post);
        params.insert("posterior_beta".to_string(), beta_post);
        params.insert("posterior_lambda".to_string(), lambda_post);
        params.insert("phi".to_string(), phi);
        params
    }

    fn plugin_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Array1<f64> {
        let lambda = params["posterior_lambda"].max(1e-10);
        let phi = params["phi"];

        let p = phi / (phi + lambda);
        let log_p = p.ln();
        let log_1mp = (1.0 - p).ln();
        let lgamma_phi = lgamma(phi);

        data.mapv(|y| {
            lgamma(y + phi) - lgamma(y + 1.0) - lgamma_phi + phi * log_p + y * log_1mp
        })
    }

    fn posterior_predictive_log_likelihood(&self, data: &ArrayView1<f64>, params: &HashMap<String, f64>) -> Option<Array1<f64>> {
        let alpha_post = params["posterior_alpha"];
        let beta_post = params["posterior_beta"];
        let phi = params["phi"];

        // Pre-compute constants
        let lgamma_phi = lgamma(phi);
        let lgamma_alpha = lgamma(alpha_post);
        let lgamma_beta = lgamma(beta_post);
        let lgamma_ab = lgamma(alpha_post + beta_post);
        let lgamma_bp = lgamma(beta_post + phi);

        Some(data.mapv(|y| {
            lgamma(y + phi)
                + lgamma(alpha_post + y)
                + lgamma_bp
                + lgamma_ab
                - lgamma_phi
                - lgamma(y + 1.0)
                - lgamma_alpha
                - lgamma_beta
                - lgamma(alpha_post + beta_post + y + phi)
        }))
    }

    fn nle(&self, data: &ArrayView1<f64>) -> Option<f64> {
        let n = data.len() as f64;
        if n == 0.0 { return Some(0.0); }

        let sum_y: f64 = data.iter().sum();
        let phi = self.get_phi_from_data(data);

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        // Log Evidence for Gamma-NB:
        // Core conjugate part (same as Poisson):
        let mut log_ev = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln();

        // NB-specific normalization terms (depend on φ):
        for &y_i in data {
            log_ev += lgamma(y_i + phi) - lgamma(phi) - lgamma(y_i + 1.0);
        }

        // Additional φ-dependent constant from NB-Gamma convolution
        log_ev += n * phi * phi.ln();

        Some(-log_ev) // Return NLE
    }

    fn nll(&self, data: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 { return 0.0; }

        let sum_y: f64 = data.iter().sum();
        let phi = self.get_phi_from_data(data);

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        if use_posterior_predictive {
            // Beta-Negative-Binomial
            let lgamma_phi = lgamma(phi);
            let lgamma_alpha = lgamma(alpha_post);
            let lgamma_beta = lgamma(beta_post);
            let lgamma_ab = lgamma(alpha_post + beta_post);
            let lgamma_bp = lgamma(beta_post + phi);

            let mut nll = 0.0;
            for &y in data {
                let log_pmf = lgamma(y + phi)
                    + lgamma(alpha_post + y)
                    + lgamma_bp
                    + lgamma_ab
                    - lgamma_phi
                    - lgamma(y + 1.0)
                    - lgamma_alpha
                    - lgamma_beta
                    - lgamma(alpha_post + beta_post + y + phi);
                nll -= log_pmf;
            }
            nll
        } else {
            // Plug-in NB
            let lambda = (alpha_post / beta_post).max(1e-10);
            let p = phi / (phi + lambda);
            let log_p = p.ln();
            let log_1mp = (1.0 - p).ln();
            let lgamma_phi = lgamma(phi);

            let mut nll = 0.0;
            for &y in data {
                let log_pmf = lgamma(y + phi) - lgamma(y + 1.0) - lgamma_phi
                    + phi * log_p + y * log_1mp;
                nll -= log_pmf;
            }
            nll
        }
    }

    fn nll_train_test(&self, train: &ArrayView1<f64>, test: &ArrayView1<f64>, use_posterior_predictive: bool) -> f64 {
        let n_train = train.len() as f64;
        let sum_y_train: f64 = train.iter().sum();
        let phi = self.get_phi_from_data(train);

        let alpha_post = self.alpha_lambda + sum_y_train;
        let beta_post = self.beta_lambda + n_train;

        if use_posterior_predictive {
            let lgamma_phi = lgamma(phi);
            let lgamma_alpha = lgamma(alpha_post);
            let lgamma_beta = lgamma(beta_post);
            let lgamma_ab = lgamma(alpha_post + beta_post);
            let lgamma_bp = lgamma(beta_post + phi);

            let mut nll = 0.0;
            for &y in test {
                let log_pmf = lgamma(y + phi)
                    + lgamma(alpha_post + y)
                    + lgamma_bp
                    + lgamma_ab
                    - lgamma_phi
                    - lgamma(y + 1.0)
                    - lgamma_alpha
                    - lgamma_beta
                    - lgamma(alpha_post + beta_post + y + phi);
                nll -= log_pmf;
            }
            nll
        } else {
            let lambda = (alpha_post / beta_post).max(1e-10);
            let p = phi / (phi + lambda);
            let log_p = p.ln();
            let log_1mp = (1.0 - p).ln();
            let lgamma_phi = lgamma(phi);

            let mut nll = 0.0;
            for &y in test {
                let log_pmf = lgamma(y + phi) - lgamma(y + 1.0) - lgamma_phi
                    + phi * log_p + y * log_1mp;
                nll -= log_pmf;
            }
            nll
        }
    }

    // ========================================================================
    // FAST PATH: Sufficient Statistics (only when φ is fixed)
    // ========================================================================

    fn nle_suff_stats(&self, stats: &SufficientStats) -> Option<f64> {
        // Only available when φ is fixed (moment_order=1 ensures this path is only
        // reached when phi is Some)
        let phi = self.phi?;

        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        let sum_y = stats.sum;

        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;

        // Comparable part of log evidence (omitting per-point terms that cancel across splits):
        // lgamma(α_post) - lgamma(α₀) + α₀·ln(β₀) - α_post·ln(β_post) + n·φ·ln(φ) - n·lgamma(φ)
        // The per-point Σ[lgamma(y_i+φ) - lgamma(y_i+1)] terms cancel in split comparison
        let log_evidence = lgamma(alpha_post) - lgamma(self.alpha_lambda)
            + self.alpha_lambda * self.beta_lambda.ln()
            - alpha_post * beta_post.ln()
            + n * phi * phi.ln()
            - n * lgamma(phi);

        Some(-log_evidence)
    }

    fn nll_suff_stats(&self, stats: &SufficientStats, use_posterior_predictive: bool) -> Option<f64> {
        // Only available when φ is fixed
        let phi = self.phi?;

        let n = stats.n;
        if n == 0.0 { return Some(0.0); }

        if use_posterior_predictive {
            // Beta-NB requires individual y values for lgamma(y+φ) etc.
            return None;
        }

        // Plug-in NB NLL (comparable part, omitting per-point terms that cancel):
        // NLL = -n·φ·ln(p) - Σy·ln(1-p) + n·lgamma(φ) - Σlgamma(y+φ) + Σlgamma(y+1)
        // where p = φ/(φ+λ), λ = α_post/β_post
        //
        // The terms Σlgamma(y+φ) and Σlgamma(y+1) are per-point and cancel across splits.
        // Comparable part: -n·φ·ln(p) - Σy·ln(1-p) + n·lgamma(φ)
        let sum_y = stats.sum;
        let alpha_post = self.alpha_lambda + sum_y;
        let beta_post = self.beta_lambda + n;
        let lambda = (alpha_post / beta_post).max(1e-10);

        let p = phi / (phi + lambda);
        let log_p = p.ln();
        let log_1mp = (1.0 - p).ln();

        // NLL comparable = -n·φ·ln(p) - Σy·ln(1-p) + n·lgamma(φ)
        // (sign: NLL = -Σ log_pmf, so these terms are positive contributions)
        let nll = -n * phi * log_p - sum_y * log_1mp + n * lgamma(phi);
        Some(nll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_dist(phi: Option<f64>) -> GammaMSLambdaNegBin {
        GammaMSLambdaNegBin {
            alpha_lambda: 2.0,  // strength=2, mean=1 → α=2
            beta_lambda: 2.0,   // β=strength=2
            phi,
            score_method: "nle".to_string(),
            use_posterior_predictive: false,
        }
    }

    #[test]
    fn test_fixed_phi_moment_order() {
        let dist = make_dist(Some(5.0));
        assert_eq!(dist.required_moment_order(), 1);
    }

    #[test]
    fn test_none_phi_moment_order() {
        let dist = make_dist(None);
        assert_eq!(dist.required_moment_order(), 0);
    }

    #[test]
    fn test_fixed_phi_nll_pp_moment_order() {
        // Fixed φ + NLL + posterior predictive → no suff stats (per-point lgamma terms)
        let dist = GammaMSLambdaNegBin {
            alpha_lambda: 2.0, beta_lambda: 2.0, phi: Some(5.0),
            score_method: "nll".to_string(), use_posterior_predictive: true,
        };
        assert_eq!(dist.required_moment_order(), 0);
    }

    #[test]
    fn test_fixed_phi_nll_plugin_moment_order() {
        // Fixed φ + NLL + plugin → suff stats OK
        let dist = GammaMSLambdaNegBin {
            alpha_lambda: 2.0, beta_lambda: 2.0, phi: Some(5.0),
            score_method: "nll".to_string(), use_posterior_predictive: false,
        };
        assert_eq!(dist.required_moment_order(), 1);
    }

    #[test]
    fn test_posterior_params() {
        let dist = make_dist(Some(3.0));
        let data = array![1.0, 2.0, 3.0, 0.0, 1.0];
        let params = dist.calc_posterior_params(&data.view());

        // α_post = 2 + 7 = 9, β_post = 2 + 5 = 7
        assert!((params["posterior_alpha"] - 9.0).abs() < 1e-10);
        assert!((params["posterior_beta"] - 7.0).abs() < 1e-10);
        assert!((params["posterior_lambda"] - 9.0 / 7.0).abs() < 1e-10);
        assert!((params["phi"] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nll_finite() {
        let dist = make_dist(Some(3.0));
        let data = array![1.0, 2.0, 3.0, 0.0, 1.0];
        let nll = dist.nll(&data.view(), false);
        assert!(nll.is_finite());
        assert!(nll > 0.0);
    }

    #[test]
    fn test_nll_posterior_predictive_finite() {
        let dist = make_dist(Some(3.0));
        let data = array![1.0, 2.0, 3.0, 0.0, 1.0];
        let nll = dist.nll(&data.view(), true);
        assert!(nll.is_finite());
        assert!(nll > 0.0);
    }

    #[test]
    fn test_nle_finite() {
        let dist = make_dist(Some(3.0));
        let data = array![1.0, 2.0, 3.0, 0.0, 1.0];
        let nle = dist.nle(&data.view());
        assert!(nle.is_some());
        assert!(nle.unwrap().is_finite());
    }

    #[test]
    fn test_nle_suff_stats_with_fixed_phi() {
        let dist = make_dist(Some(3.0));
        let data = array![1.0, 2.0, 3.0, 0.0, 1.0];

        // Build stats
        let mut stats = SufficientStats::default();
        for &v in data.iter() {
            stats.add(v, 1);
        }

        let nle_ss = dist.nle_suff_stats(&stats);
        assert!(nle_ss.is_some());
        assert!(nle_ss.unwrap().is_finite());
    }

    #[test]
    fn test_nle_suff_stats_none_phi() {
        let dist = make_dist(None);
        let stats = SufficientStats { n: 5.0, sum: 7.0, ..Default::default() };
        // Should return None since phi is not fixed
        assert!(dist.nle_suff_stats(&stats).is_none());
    }

    #[test]
    fn test_nll_suff_stats_with_fixed_phi() {
        let dist = make_dist(Some(3.0));

        let mut stats = SufficientStats::default();
        for v in [1.0, 2.0, 3.0, 0.0, 1.0] {
            stats.add(v, 2);
        }

        let nll_ss = dist.nll_suff_stats(&stats, false);
        assert!(nll_ss.is_some());
        assert!(nll_ss.unwrap().is_finite());
    }

    #[test]
    fn test_train_test_split() {
        let dist = make_dist(Some(3.0));
        let train = array![1.0, 2.0, 3.0, 0.0, 1.0];
        let test = array![2.0, 1.0];
        let nll = dist.nll_train_test(&train.view(), &test.view(), false);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_mom_phi_estimation() {
        // Data with mean=2, var=6 → φ = 4/(6-2) = 1.0
        let phi = GammaMSLambdaNegBin::estimate_phi_mom(2.0, 6.0);
        assert!((phi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mom_phi_no_overdispersion() {
        // var <= mean → φ = μ² (Poisson-like), floored at 1e4
        let phi = GammaMSLambdaNegBin::estimate_phi_mom(5.0, 3.0);
        assert!((phi - 1e4).abs() < 1e-10); // 5²=25 < 1e4, so floor applies

        let phi_large = GammaMSLambdaNegBin::estimate_phi_mom(2000.0, 1000.0);
        assert!((phi_large - 4e6).abs() < 1e-10); // 2000² = 4e6
    }
}
