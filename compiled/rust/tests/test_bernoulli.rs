use distributions::{DistributionPrimitives, BetaABBernoulli, BetaMVBernoulli, SufficientStats};
use ndarray::{ArrayView1};
use pyo3::PyResult;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_beta_ab_posterior_params() {
        let dist = BetaABBernoulli { alpha_p: 1.0, beta_p: 1.0 };
        let data = array![1.0, 1.0, 0.0, 1.0, 0.0];
        let params = dist.calc_posterior_params(&data.view());

        // 3 successes, 2 failures
        assert!((params["posterior_alpha"] - 4.0).abs() < 1e-10);
        assert!((params["posterior_beta"] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_ab_nle_consistency() {
        let dist = BetaABBernoulli { alpha_p: 2.0, beta_p: 3.0 };
        let data = array![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];

        let nle_full = dist.nle(&data.view()).unwrap();

        let mut stats = SufficientStats::default();
        for &val in data.iter() { stats.add(val); }
        let nle_stats = dist.nle_suff_stats(&stats).unwrap();

        assert!((nle_full - nle_stats).abs() < 1e-10, "NLE mismatch: {} vs {}", nle_full, nle_stats);
    }

    #[test]
    fn test_beta_ab_nll_consistency() {
        let dist = BetaABBernoulli { alpha_p: 1.0, beta_p: 1.0 };
        let data = array![1.0, 0.0, 1.0, 1.0, 0.0];

        let nll_full = dist.nll(&data.view(), false);

        let mut stats = SufficientStats::default();
        for &val in data.iter() { stats.add(val); }
        let nll_stats = dist.nll_suff_stats(&stats, false).unwrap();

        assert!((nll_full - nll_stats).abs() < 1e-10, "NLL mismatch: {} vs {}", nll_full, nll_stats);
    }

    #[test]
    fn test_beta_mv_conversion() {
        // mean_p=0.5, var_p=0.05 should give alpha=beta=2
        // m = 0.5*0.5/0.05 - 1 = 5 - 1 = 4
        // alpha = 0.5 * 4 = 2
        // beta = 0.5 * 4 = 2
        let dist = BetaMVBernoulli {
            mean_p: 0.5,
            var_p: 0.05,
            alpha_p: 2.0,
            beta_p: 2.0,
        };

        assert!((dist.alpha_p - 2.0).abs() < 1e-10);
        assert!((dist.beta_p - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_mv_nle_matches_ab() {
        // Both should produce identical results for same effective alpha/beta
        let ab_dist = BetaABBernoulli { alpha_p: 2.0, beta_p: 2.0 };
        let mv_dist = BetaMVBernoulli {
            mean_p: 0.5,
            var_p: 0.05,
            alpha_p: 2.0,
            beta_p: 2.0,
        };

        let data = array![1.0, 0.0, 1.0, 1.0, 0.0, 0.0];

        let nle_ab = ab_dist.nle(&data.view()).unwrap();
        let nle_mv = mv_dist.nle(&data.view()).unwrap();

        assert!((nle_ab - nle_mv).abs() < 1e-10, "NLE mismatch between AB and MV");
    }
}
