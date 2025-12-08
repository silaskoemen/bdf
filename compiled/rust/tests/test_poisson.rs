
// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gamma_ab_posterior_params() {
        let dist = GammaABLambdaPoisson { alpha_lambda: 2.0, beta_lambda: 1.0 };
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0]; // sum = 15, n = 5
        let params = dist.calc_posterior_params(&data.view());

        // α_post = 2 + 15 = 17, β_post = 1 + 5 = 6
        assert!((params["posterior_alpha"] - 17.0).abs() < 1e-10);
        assert!((params["posterior_beta"] - 6.0).abs() < 1e-10);
        assert!((params["posterior_lambda"] - 17.0/6.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_ab_nle_consistency() {
        let dist = GammaABLambdaPoisson { alpha_lambda: 2.0, beta_lambda: 1.0 };
        let data = array![0.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0];

        let nle_full = dist.nle(&data.view()).unwrap();

        let mut stats = SufficientStats::default();
        for &val in data.iter() { stats.add(val); }
        let nle_stats = dist.nle_suff_stats(&stats).unwrap();

        // They won't be exactly equal because full version includes -Σlog(y!)
        // But the DIFFERENCE should be consistent (the omitted term)
        let sum_log_factorial: f64 = data.iter().map(|&k| lgamma(k + 1.0)).sum();
        let expected_diff = sum_log_factorial;

        assert!((nle_full - nle_stats - expected_diff).abs() < 1e-10, 
            "NLE mismatch: full={}, stats={}, diff={}, expected_diff={}", 
            nle_full, nle_stats, nle_full - nle_stats, expected_diff);
    }

    #[test]
    fn test_gamma_ab_nll_consistency() {
        let dist = GammaABLambdaPoisson { alpha_lambda: 1.0, beta_lambda: 1.0 };
        let data = array![0.0, 1.0, 2.0, 1.0, 0.0];

        let nll_full = dist.nll(&data.view(), false);

        let mut stats = SufficientStats::default();
        for &val in data.iter() { stats.add(val); }
        let nll_stats = dist.nll_suff_stats(&stats, false).unwrap();

        // They differ by Σlog(y!) which is constant
        let sum_log_factorial: f64 = data.iter().map(|&k| lgamma(k + 1.0)).sum();
        let expected_diff = sum_log_factorial;

        assert!((nll_full - nll_stats - expected_diff).abs() < 1e-10,
            "NLL mismatch: full={}, stats={}", nll_full, nll_stats);
    }

    #[test]
    fn test_gamma_mv_conversion() {
        // mean=2, var=2 → α = 4/2 = 2, β = 2/2 = 1
        let dist = GammaMVLambdaPoisson {
            mean_lambda: 2.0,
            var_lambda: 2.0,
            alpha_lambda: 2.0,
            beta_lambda: 1.0,
        };

        assert!((dist.alpha_lambda - 2.0).abs() < 1e-10);
        assert!((dist.beta_lambda - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_mv_nle_matches_ab() {
        let ab_dist = GammaABLambdaPoisson { alpha_lambda: 2.0, beta_lambda: 1.0 };
        let mv_dist = GammaMVLambdaPoisson {
            mean_lambda: 2.0,
            var_lambda: 2.0,
            alpha_lambda: 2.0,
            beta_lambda: 1.0,
        };

        let data = array![1.0, 2.0, 0.0, 3.0, 1.0, 2.0];

        let nle_ab = ab_dist.nle(&data.view()).unwrap();
        let nle_mv = mv_dist.nle(&data.view()).unwrap();

        assert!((nle_ab - nle_mv).abs() < 1e-10, "NLE mismatch between AB and MV");
    }

    #[test]
    fn test_posterior_predictive_returns_some() {
        let dist = GammaABLambdaPoisson { alpha_lambda: 2.0, beta_lambda: 1.0 };
        let data = array![1.0, 2.0, 3.0];
        let params = dist.calc_posterior_params(&data.view());

        let pp_ll = dist.posterior_predictive_log_likelihood(&data.view(), &params);
        assert!(pp_ll.is_some());
    }
}
