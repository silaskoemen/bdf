"""Tests that ALL distributions must pass.

These tests verify the basic contract of BDFDistribution:
- Required abstract methods are implemented and return correct types
- Validation works correctly
- Scoring methods are available as declared
- Basic sanity checks pass

Inspired by sklearn's check_estimator pattern.
"""

import numpy as np
import pytest

from bdf.distributions.bdf_distribution import BDFDistribution
from bdf.distributions.distribution_manager import DistributionManager
from bdf.utils.distribution_helpers import import_all_distributions

# Ensure all distributions are registered
import_all_distributions()


def get_test_data_for_distribution(dist_name: str) -> np.ndarray:
    """Generate appropriate test data for a distribution.

    Different distributions have different data requirements:
    - Normal: continuous, any real values
    - Poisson: non-negative integers
    - Bernoulli: 0 or 1
    - Exponential: positive values
    - KDE: continuous, any real values (needs at least 2 points)
    """
    rng = np.random.default_rng(42)

    if "Bernoulli" in dist_name or "Beta" in dist_name:
        return rng.choice([0.0, 1.0], size=50).astype(float)
    elif "Poisson" in dist_name or "Gamma" in dist_name and "Lambda" in dist_name:
        return rng.poisson(5, size=50).astype(float)
    elif "Exponential" in dist_name:
        return rng.exponential(1.0, size=50)
    elif "KDE" in dist_name:
        return rng.normal(0, 1, size=50)
    else:
        # Default: normal data
        return rng.normal(0, 1, size=50)


def get_default_params_for_distribution(dist_name: str) -> dict:
    """Get sensible default parameters for each distribution."""
    defaults = {
        "NormalMuNormal": {"mu_mu": 0.0, "sigma_mu": 1.0},
        "NormalMuInvGammaSigmaNormal": {"mu_mu": 0.0, "n_mu": 1.0, "nu_sigma": 3.0, "phi_sigma": 1.0},
        "GammaABLambdaPoisson": {"alpha_lambda": 2.0, "beta_lambda": 1.0},
        "GammaMVLambdaPoisson": {"mean_lambda": 5.0, "var_lambda": 5.0},
        "BetaABBernoulli": {"alpha_p": 1.0, "beta_p": 1.0},
        "BetaMVBernoulli": {"mean_p": 0.5, "var_p": 0.1},
        "GammaABLambdaExponential": {"alpha_lambda": 2.0, "beta_lambda": 1.0},
        "GammaMVLambdaExponential": {"mean_lambda": 1.0, "var_lambda": 1.0},
        "KDE": {"bandwidth": 0.5, "kernel": "gaussian"},
        "BayesianKDE": {"bandwidth": 0.5, "kernel": "gaussian", "prior_h": 1.0, "m_h": 1.0},
        "SkewNormalOmega": {},
        "BayesianSkewNormalOmega": {},
        "SkewNormalAlpha": {},
    }
    return defaults.get(dist_name, {})


# Get all distribution names for parametrization
ALL_DISTRIBUTIONS = list(BDFDistribution._registry.keys())

# Filter to distributions we have default params for
TESTABLE_DISTRIBUTIONS = [
    d
    for d in ALL_DISTRIBUTIONS
    if d in get_default_params_for_distribution("NormalMuNormal")
    or d
    in {
        "NormalMuNormal",
        "NormalMuInvGammaSigmaNormal",
        "GammaABLambdaPoisson",
        "GammaMVLambdaPoisson",
        "BetaABBernoulli",
        "BetaMVBernoulli",
        "GammaABLambdaExponential",
        "GammaMVLambdaExponential",
        "KDE",
        "BayesianKDE",
    }
]


class TestAllDistributionsBasicContract:
    """Tests that every distribution must pass."""

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_distribution_instantiation(self, dist_name: str):
        """Test that distribution can be instantiated with default params."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        assert dist is not None
        assert isinstance(dist, BDFDistribution)

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_calc_posterior_params_returns_dict(self, dist_name: str):
        """Test that calc_posterior_params returns a dict with float values."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        posterior_params = dist.calc_posterior_params(data)

        assert isinstance(posterior_params, dict)
        for key, value in posterior_params.items():
            assert isinstance(key, str), f"Key {key} should be string"
            assert isinstance(value, (int, float)), f"Value for {key} should be numeric"
            assert np.isfinite(value), f"Value for {key} should be finite"

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_get_posterior_mean_returns_float(self, dist_name: str):
        """Test that get_posterior_mean returns a finite float."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        mean = dist.get_posterior_mean(data=data)

        assert isinstance(mean, float)
        assert np.isfinite(mean)

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_get_posterior_variance_returns_non_negative(self, dist_name: str):
        """Test that get_posterior_variance returns a non-negative finite float."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        variance = dist.get_posterior_variance(data=data)

        assert isinstance(variance, float)
        assert np.isfinite(variance)
        assert variance >= 0, f"Variance should be non-negative, got {variance}"

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_num_parameters_returns_positive_int(self, dist_name: str):
        """Test that _num_parameters returns a positive integer."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        num_params = dist._num_parameters()

        assert isinstance(num_params, int)
        assert num_params > 0, f"Number of parameters should be positive, got {num_params}"

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_validate_targets_accepts_valid_data(self, dist_name: str):
        """Test that validate_targets accepts valid data without raising."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)

        # Should not raise
        dist.validate_targets(data)

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_validate_targets_rejects_nan(self, dist_name: str):
        """Test that validate_targets rejects data with NaN."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        data_with_nan = data.copy()
        data_with_nan[0] = np.nan

        with pytest.raises(ValueError):
            dist.validate_targets(data_with_nan)

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_validate_targets_rejects_inf(self, dist_name: str):
        """Test that validate_targets rejects data with infinity."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        data_with_inf = data.copy()
        data_with_inf[0] = np.inf

        with pytest.raises(ValueError):
            dist.validate_targets(data_with_inf)

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_nll_returns_float(self, dist_name: str):
        """Test that NLL returns a finite float."""
        params = get_default_params_for_distribution(dist_name)
        # Ensure nll scoring
        params["score_method"] = "nll"
        params["use_posterior_predictive"] = False
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        nll = dist.nll(data)

        assert isinstance(nll, float)
        assert np.isfinite(nll), f"NLL should be finite, got {nll}"

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_score_returns_float(self, dist_name: str):
        """Test that score() returns a finite float."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        score = dist.score(data)

        assert isinstance(score, float)
        assert np.isfinite(score), f"Score should be finite, got {score}"

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_log_likelihood_returns_array(self, dist_name: str):
        """Test that log_likelihood returns array of same length as data."""
        params = get_default_params_for_distribution(dist_name)
        params["use_posterior_predictive"] = False
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        ll = dist.log_likelihood(data)

        assert isinstance(ll, np.ndarray)
        assert ll.shape == data.shape, f"Shape mismatch: {ll.shape} vs {data.shape}"
        assert np.all(np.isfinite(ll)), "All log-likelihoods should be finite"

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_to_rust_spec_returns_dict(self, dist_name: str):
        """Test that to_rust_spec returns a valid dictionary."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        spec = dist.to_rust_spec()

        assert isinstance(spec, dict)
        assert "dist_type" in spec
        assert spec["dist_type"] == dist_name
        assert "num_parameters" in spec
        assert "_python_object" in spec


class TestAllDistributionsSamplingContract:
    """Tests for sampling functionality."""

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_sample_posterior_returns_array(self, dist_name: str):
        """Test that sample_posterior returns array of requested size."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        samples = dist.sample_posterior(data=data, size=100, random_state=42)

        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100,) or samples.shape[0] == 100
        assert np.all(np.isfinite(samples)), "All samples should be finite"

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_sample_posterior_is_deterministic(self, dist_name: str):
        """Test that sample_posterior is deterministic with same seed."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)

        samples1 = dist.sample_posterior(data=data, size=100, random_state=42)
        samples2 = dist.sample_posterior(data=data, size=100, random_state=42)

        np.testing.assert_array_equal(samples1, samples2)


class TestConjugateDistributionsNLE:
    """Tests specific to conjugate distributions (those supporting NLE)."""

    # Distributions that should support NLE
    CONJUGATE_DISTRIBUTIONS = [
        "NormalMuNormal",
        "NormalMuInvGammaSigmaNormal",
        "GammaABLambdaPoisson",
        "GammaMVLambdaPoisson",
        "BetaABBernoulli",
        "BetaMVBernoulli",
        "GammaABLambdaExponential",
        "GammaMVLambdaExponential",
    ]

    @pytest.mark.parametrize("dist_name", CONJUGATE_DISTRIBUTIONS)
    def test_supports_nle_flag_is_true(self, dist_name: str):
        """Test that conjugate distributions have _supports_nle = True."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        assert dist._supports_nle is True, f"{dist_name} should support NLE"

    @pytest.mark.parametrize("dist_name", CONJUGATE_DISTRIBUTIONS)
    def test_nle_returns_finite_float(self, dist_name: str):
        """Test that NLE returns a finite float for conjugate distributions."""
        params = get_default_params_for_distribution(dist_name)
        params["score_method"] = "nle"
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        nle = dist.nle(data)

        assert isinstance(nle, float)
        assert np.isfinite(nle), f"NLE should be finite, got {nle}"

    @pytest.mark.parametrize("dist_name", CONJUGATE_DISTRIBUTIONS)
    def test_log_evidence_returns_finite_float(self, dist_name: str):
        """Test that log_evidence returns a finite float."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        log_ev = dist.log_evidence(data)

        assert isinstance(log_ev, float)
        assert np.isfinite(log_ev), f"Log evidence should be finite, got {log_ev}"

    @pytest.mark.parametrize("dist_name", CONJUGATE_DISTRIBUTIONS)
    def test_nle_equals_negative_log_evidence(self, dist_name: str):
        """Test that NLE = -log_evidence."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        nle = dist.nle(data)
        log_ev = dist.log_evidence(data)

        np.testing.assert_allclose(nle, -log_ev, rtol=1e-10)


class TestPosteriorPredictiveDistributions:
    """Tests for distributions supporting posterior predictive."""

    PP_DISTRIBUTIONS = [
        "NormalMuNormal",
        "NormalMuInvGammaSigmaNormal",
        "GammaABLambdaPoisson",
        "GammaMVLambdaPoisson",
        "BetaABBernoulli",
        "BetaMVBernoulli",
        "GammaABLambdaExponential",
        "GammaMVLambdaExponential",
    ]

    @pytest.mark.parametrize("dist_name", PP_DISTRIBUTIONS)
    def test_supports_posterior_predictive_flag(self, dist_name: str):
        """Test that distributions have correct _supports_posterior_predictive flag."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        assert dist._supports_posterior_predictive is True

    @pytest.mark.parametrize("dist_name", PP_DISTRIBUTIONS)
    def test_posterior_predictive_log_likelihood_returns_array(self, dist_name: str):
        """Test PP log-likelihood returns array of same shape as data."""
        params = get_default_params_for_distribution(dist_name)
        params["use_posterior_predictive"] = True
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        posterior_params = dist.calc_posterior_params(data)
        pp_ll = dist._posterior_predictive_log_likelihood(data, posterior_params)

        assert isinstance(pp_ll, np.ndarray)
        assert pp_ll.shape == data.shape
        assert np.all(np.isfinite(pp_ll))

    @pytest.mark.parametrize("dist_name", PP_DISTRIBUTIONS)
    def test_plugin_vs_pp_are_different(self, dist_name: str):
        """Test that plug-in and PP log-likelihoods differ (generally)."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        posterior_params = dist.calc_posterior_params(data)

        plugin_ll = dist._plugin_log_likelihood(data, posterior_params)
        pp_ll = dist._posterior_predictive_log_likelihood(data, posterior_params)

        # They should be different (PP accounts for parameter uncertainty)
        # Note: This might fail for very large n where they converge
        # We use a weak test here
        assert not np.allclose(plugin_ll, pp_ll, rtol=1e-6) or len(data) > 1000


class TestDistributionConsistency:
    """Tests for internal consistency of distribution computations."""

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_nll_equals_sum_negative_log_likelihood(self, dist_name: str):
        """Test that NLL = -sum(log_likelihood)."""
        params = get_default_params_for_distribution(dist_name)
        params["use_posterior_predictive"] = False
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)

        nll = dist.nll(data)
        ll = dist.log_likelihood(data)
        expected_nll = -np.sum(ll)

        np.testing.assert_allclose(nll, expected_nll, rtol=1e-10)

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_likelihood_equals_exp_log_likelihood(self, dist_name: str):
        """Test that likelihood = exp(log_likelihood)."""
        params = get_default_params_for_distribution(dist_name)
        params["use_posterior_predictive"] = False
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)

        ll = dist.log_likelihood(data)
        lik = dist.likelihood(data)

        np.testing.assert_allclose(lik, np.exp(ll), rtol=1e-10)

    @pytest.mark.parametrize("dist_name", TESTABLE_DISTRIBUTIONS)
    def test_posterior_params_with_data_vs_params(self, dist_name: str):
        """Test that methods work with both data= and params= arguments."""
        params = get_default_params_for_distribution(dist_name)
        data = get_test_data_for_distribution(dist_name)

        dist = DistributionManager.create_distribution(dist_name, params, y=data)
        posterior_params = dist.calc_posterior_params(data)

        # Both should give same result
        mean_from_data = dist.get_posterior_mean(data=data)
        mean_from_params = dist.get_posterior_mean(params=posterior_params)

        np.testing.assert_allclose(mean_from_data, mean_from_params, rtol=1e-10)

        var_from_data = dist.get_posterior_variance(data=data)
        var_from_params = dist.get_posterior_variance(params=posterior_params)

        np.testing.assert_allclose(var_from_data, var_from_params, rtol=1e-10)
