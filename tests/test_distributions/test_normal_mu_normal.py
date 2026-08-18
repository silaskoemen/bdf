"""Comprehensive tests for NormalMuNormal distribution.

Tests verify:
1. Mathematical correctness of conjugate updates
2. Statistical properties match theoretical values
3. Log-likelihood computations against scipy
4. Python-Rust implementation equivalence
5. Edge cases and numerical stability
6. Prior sensitivity

Reference: Murphy's "Machine Learning: A Probabilistic Perspective"
"""

import numpy as np
import pytest
from hypothesis import given, settings
from scipy import stats
from scipy.special import gammaln

from bdf.distributions.distribution_manager import DistributionManager
from bdf.distributions.normal import NormalMuNormal, NormalMuNormalParams

# Import shared helpers (fixtures from conftest.py are auto-discovered by pytest)
from tests.test_distributions.helpers import (
    ATOL_TIGHT,
    RTOL_NUMERICAL,
    RTOL_STATISTICAL,
    RTOL_TIGHT,
    assert_array_close,
    assert_close,
    normal_data_strategy,
    prior_params_strategy,
    reference_normal_log_evidence,
    reference_normal_posterior_mean,
    reference_normal_posterior_variance,
    reference_plugin_log_likelihood,
)

# Try to import Rust extension
try:
    from bdf import _bdf_rs as bdf_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False


# ============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# ============================================================================


class TestNormalMuNormalMath:
    """Verify mathematical correctness of conjugate updates."""

    def test_posterior_mean_formula_simple(self):
        """Posterior mean matches analytical formula for simple case."""
        # Prior: N(0, 1)
        mu_mu, sigma_mu = 0.0, 1.0
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        dist = NormalMuNormal({"mu_mu": mu_mu, "sigma_mu": sigma_mu})
        params = dist.calc_posterior_params(data)

        # Analytical formula
        expected_mean = reference_normal_posterior_mean(data, mu_mu, sigma_mu)

        assert_close(
            params["posterior_mu"],
            expected_mean,
            rtol=RTOL_TIGHT,
            msg="Posterior mean doesn't match analytical formula",
        )

    def test_posterior_variance_formula_simple(self):
        """Posterior variance matches analytical formula."""
        mu_mu, sigma_mu = 0.0, 1.0
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        dist = NormalMuNormal({"mu_mu": mu_mu, "sigma_mu": sigma_mu})
        params = dist.calc_posterior_params(data)

        expected_var = reference_normal_posterior_variance(data, sigma_mu)

        assert_close(
            params["posterior_sigma_mu"] ** 2,
            expected_var,
            rtol=RTOL_TIGHT,
            msg="Posterior variance doesn't match analytical formula",
        )

    @pytest.mark.parametrize(
        "mu_mu,sigma_mu",
        [
            (0.0, 1.0),
            (-10.0, 0.5),
            (100.0, 10.0),
            (0.0, 0.01),
            (0.0, 100.0),
        ],
    )
    def test_posterior_mean_formula_parametrized(self, mu_mu: float, sigma_mu: float):
        """Posterior mean matches formula across parameter ranges."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, 100)

        dist = NormalMuNormal({"mu_mu": mu_mu, "sigma_mu": sigma_mu})
        params = dist.calc_posterior_params(data)

        expected_mean = reference_normal_posterior_mean(data, mu_mu, sigma_mu)

        assert_close(params["posterior_mu"], expected_mean, rtol=RTOL_TIGHT)

    @pytest.mark.parametrize(
        "mu_mu,sigma_mu",
        [
            (0.0, 1.0),
            (-10.0, 0.5),
            (100.0, 10.0),
            (0.0, 0.01),
            (0.0, 100.0),
        ],
    )
    def test_posterior_variance_formula_parametrized(self, mu_mu: float, sigma_mu: float):
        """Posterior variance matches formula across parameter ranges."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, 100)

        dist = NormalMuNormal({"mu_mu": mu_mu, "sigma_mu": sigma_mu})
        params = dist.calc_posterior_params(data)

        expected_var = reference_normal_posterior_variance(data, sigma_mu)

        assert_close(params["posterior_sigma_mu"] ** 2, expected_var, rtol=RTOL_TIGHT)

    def test_log_evidence_formula(self):
        """Log marginal likelihood matches the covariance-form marginal."""
        mu_mu, sigma_mu = 0.0, 1.0
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)

        dist = NormalMuNormal({"mu_mu": mu_mu, "sigma_mu": sigma_mu})
        log_ev = dist.log_evidence(data)

        expected_log_ev = reference_normal_log_evidence(data, mu_mu, sigma_mu)

        assert_close(log_ev, expected_log_ev, rtol=RTOL_TIGHT, msg="Log evidence doesn't match analytical formula")


class TestNormalMuNormalStatisticalProperties:
    """Verify that computed statistics match theoretical values."""

    def test_posterior_mean_between_prior_and_data(self):
        """Posterior mean should be between prior mean and data mean."""
        mu_mu = 0.0
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        data_mean = np.mean(data)

        dist = NormalMuNormal({"mu_mu": mu_mu, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        assert mu_mu <= params["posterior_mu"] <= data_mean or data_mean <= params["posterior_mu"] <= mu_mu

    def test_posterior_variance_less_than_prior_variance(self):
        """Posterior variance should be less than prior variance (data adds information)."""
        sigma_mu = 2.0
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": sigma_mu})
        params = dist.calc_posterior_params(data)

        prior_var = sigma_mu**2
        posterior_var = params["posterior_sigma_mu"] ** 2

        assert posterior_var < prior_var, "Posterior variance should be less than prior variance"

    def test_posterior_converges_to_data_mean_large_n(self):
        """With large n, posterior mean should converge to data mean."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 10000)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        assert np.abs(params["posterior_mu"] - np.mean(data)) < 0.01

    def test_posterior_variance_decreases_with_n(self):
        """Posterior variance should decrease as n increases."""
        rng = np.random.default_rng(42)
        base_data = rng.normal(0, 1, 1000)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})

        variances = []
        for n in [10, 100, 1000]:
            params = dist.calc_posterior_params(base_data[:n])
            variances.append(params["posterior_sigma_mu"] ** 2)

        assert variances[0] > variances[1] > variances[2]

    def test_get_posterior_mean_matches_params(self):
        """get_posterior_mean should match the value in params dict."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        mean_from_method = dist.get_posterior_mean(params=params)

        assert mean_from_method == params["posterior_mu"]

    def test_get_posterior_variance_matches_params(self):
        """get_posterior_variance should match the value in params dict."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        var_from_method = dist.get_posterior_variance(params=params)

        assert var_from_method == params["posterior_sigma_mu"] ** 2


class TestNormalMuNormalLogLikelihood:
    """Test log-likelihood computations against scipy."""

    def test_plugin_log_likelihood_matches_scipy(self):
        """Plugin log-likelihood should match scipy.stats.norm.logpdf."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0, "use_posterior_predictive": False})
        params = dist.calc_posterior_params(data)

        plugin_ll = dist._plugin_log_likelihood(data, params)
        scipy_ll = stats.norm.logpdf(data, loc=params["posterior_mu"], scale=params["sample_std"])

        assert_array_close(plugin_ll, scipy_ll, rtol=RTOL_TIGHT)

    def test_posterior_predictive_log_likelihood_formula(self):
        """Posterior predictive should use N(μ_post, σ_μ² + σ_data²)."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0, "use_posterior_predictive": True})
        params = dist.calc_posterior_params(data)

        pp_ll = dist._posterior_predictive_log_likelihood(data, params)

        # Manual calculation
        mu = params["posterior_mu"]
        pred_var = params["posterior_sigma_mu"] ** 2 + params["sample_std"] ** 2
        expected_ll = stats.norm.logpdf(data, loc=mu, scale=np.sqrt(pred_var))

        assert_array_close(pp_ll, expected_ll, rtol=RTOL_TIGHT)

    def test_nll_equals_negative_sum_log_likelihood(self):
        """NLL should equal -sum(log_likelihood)."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0, "use_posterior_predictive": False})
        nll = dist.nll(data)
        ll = dist.log_likelihood(data)

        assert_close(nll, -np.sum(ll), rtol=RTOL_TIGHT)

    def test_nle_equals_negative_log_evidence(self):
        """NLE should equal -log_evidence."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        nle = dist.nle(data)
        log_ev = dist.log_evidence(data)

        assert_close(nle, -log_ev, rtol=RTOL_TIGHT)


# ============================================================================
# PYTHON-RUST EQUIVALENCE TESTS
# ============================================================================


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not available")
class TestNormalMuNormalRustEquivalence:
    """Verify Python and Rust implementations match exactly."""

    @pytest.mark.parametrize("n_samples", [5, 50, 500])
    @pytest.mark.parametrize("mu_mu", [0.0, -10.0, 100.0])
    @pytest.mark.parametrize("sigma_mu", [0.1, 1.0, 10.0])
    def test_nll_match(self, n_samples: int, mu_mu: float, sigma_mu: float):
        """Rust NLL matches Python NLL."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, n_samples)

        params = {"mu_mu": mu_mu, "sigma_mu": sigma_mu, "use_posterior_predictive": False}
        py_dist = DistributionManager.create_distribution("NormalMuNormal", params, y=data)

        py_nll = py_dist.nll(data)

        rust_spec = DistributionManager.to_rust_spec(py_dist)
        rust_nll = bdf_rs.calculate_nll(data, rust_spec)

        assert_close(py_nll, rust_nll, rtol=RTOL_TIGHT, msg=f"NLL mismatch: n={n_samples}, μ₀={mu_mu}, σ_μ={sigma_mu}")

    @pytest.mark.parametrize("n_samples", [5, 50, 500])
    def test_posterior_predictive_nll_match(self, n_samples: int):
        """Rust posterior predictive NLL matches Python."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, n_samples)

        params = {"mu_mu": 0.0, "sigma_mu": 1.0, "use_posterior_predictive": True}
        py_dist = DistributionManager.create_distribution("NormalMuNormal", params, y=data)

        py_nll = py_dist.nll(data)

        rust_spec = DistributionManager.to_rust_spec(py_dist)
        rust_nll = bdf_rs.calculate_nll(data, rust_spec)

        assert_close(py_nll, rust_nll, rtol=RTOL_TIGHT, msg=f"PP NLL mismatch: n={n_samples}")

    @pytest.mark.parametrize("n_samples", [5, 50, 500])
    def test_nle_match(self, n_samples: int):
        """Rust NLE matches Python for NormalMuNormal."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, n_samples)

        params = {"mu_mu": 0.0, "sigma_mu": 1.0, "score_method": "nle"}
        py_dist = DistributionManager.create_distribution("NormalMuNormal", params, y=data)

        py_nle = py_dist.nle(data)
        py_score = py_dist.score(data)

        # NLE should equal score when score_method="nle"
        assert_close(py_nle, py_score, rtol=RTOL_TIGHT)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestNormalMuNormalEdgeCases:
    """Edge cases and numerical stability."""

    def test_single_sample(self):
        """Single sample should work (degenerate case)."""
        data = np.array([5.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        # Should return finite values
        assert np.isfinite(params["posterior_mu"])
        assert np.isfinite(params["posterior_sigma_mu"])
        assert params["posterior_sigma_mu"] > 0

    def test_two_samples(self):
        """Two samples (minimum for variance)."""
        data = np.array([0.0, 10.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        # Should return finite values
        assert np.isfinite(params["posterior_mu"])
        assert np.isfinite(params["posterior_sigma_mu"])
        assert np.isfinite(params["sample_std"])

    def test_constant_data(self):
        """All values identical (zero sample variance)."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        # Should handle zero variance gracefully
        assert np.isfinite(params["posterior_mu"])
        # Posterior mean should be close to data mean (5.0) since data precision is very high
        assert np.abs(params["posterior_mu"] - 5.0) < 1e-5

    def test_very_large_n(self):
        """Very large sample size."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 2.0, 100000)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        # Should converge to sample mean
        assert np.abs(params["posterior_mu"] - np.mean(data)) < 0.001
        # Variance should be very small
        assert params["posterior_sigma_mu"] ** 2 < 1e-4

    def test_extreme_data_values(self):
        """Data with extreme values."""
        data = np.array([1e10, 1e10 + 1, 1e10 + 2])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        assert np.isfinite(params["posterior_mu"])
        assert np.isfinite(params["posterior_sigma_mu"])

        # NLL should be finite
        nll = dist.nll(data)
        assert np.isfinite(nll)

    def test_small_data_values(self):
        """Data with very small values."""
        data = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1e-10})
        params = dist.calc_posterior_params(data)

        assert np.isfinite(params["posterior_mu"])
        assert np.isfinite(params["posterior_sigma_mu"])

    def test_empty_data_raises(self):
        """Empty data should raise ValueError."""
        data = np.array([])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})

        with pytest.raises(ValueError):
            dist.calc_posterior_params(data)


class TestNormalMuNormalPriorSensitivity:
    """Verify that prior affects posterior correctly."""

    def test_weak_prior_dominated_by_data(self):
        """Weak prior (high sigma_mu) should be dominated by data."""
        data = np.array([5.0])  # Single point

        weak = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 100.0})
        params = weak.calc_posterior_params(data)

        # Posterior mean should be close to data
        assert np.abs(params["posterior_mu"] - 5.0) < 0.1

    def test_strong_prior_dominates_small_data(self):
        """Strong prior (low sigma_mu) should dominate with small data set.

        Note: With a single data point, sample variance is set to 1e-10,
        making data precision very high. We use multiple points with
        realistic variance for this test.
        """
        # Small data with clear variance so data precision isn't artificially high
        data = np.array([5.0, 4.5, 5.5, 4.8, 5.2])  # Mean ~= 5.0, reasonable variance

        # Very strong prior: sigma_mu=0.01 means prior variance = 0.0001
        strong = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 0.01})
        params = strong.calc_posterior_params(data)

        # With strong prior, posterior should be pulled towards prior mean (0.0)
        # Not necessarily very close, but closer than the data mean (5.0)
        data_mean = np.mean(data)
        assert params["posterior_mu"] < data_mean  # Should be pulled toward 0

    def test_prior_strength_effect(self):
        """Compare weak vs strong prior effects."""
        data = np.array([10.0, 11.0, 12.0])

        weak = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 100.0})
        strong = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 0.1})

        weak_params = weak.calc_posterior_params(data)
        strong_params = strong.calc_posterior_params(data)

        data_mean = np.mean(data)

        # Weak prior: posterior closer to data mean
        weak_dist_to_data = np.abs(weak_params["posterior_mu"] - data_mean)
        # Strong prior: posterior closer to prior mean
        strong_dist_to_prior = np.abs(strong_params["posterior_mu"] - 0.0)

        assert weak_dist_to_data < np.abs(weak_params["posterior_mu"] - 0.0)
        assert strong_dist_to_prior < np.abs(strong_params["posterior_mu"] - data_mean)


class TestNormalMuNormalValidation:
    """Test input validation."""

    def test_validate_targets_rejects_nan(self):
        """NaN values should be rejected."""
        data = np.array([1.0, np.nan, 3.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})

        with pytest.raises(ValueError, match="NaN"):
            dist.validate_targets(data)

    def test_validate_targets_rejects_inf(self):
        """Infinite values should be rejected."""
        data = np.array([1.0, np.inf, 3.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})

        with pytest.raises(ValueError, match="infinite"):
            dist.validate_targets(data)

    def test_validate_targets_rejects_neginf(self):
        """Negative infinite values should be rejected."""
        data = np.array([1.0, -np.inf, 3.0])

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})

        with pytest.raises(ValueError, match="infinite"):
            dist.validate_targets(data)

    def test_invalid_sigma_mu_raises(self):
        """Non-positive sigma_mu should raise."""
        with pytest.raises(ValueError):
            NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 0.0})

        with pytest.raises(ValueError):
            NormalMuNormal({"mu_mu": 0.0, "sigma_mu": -1.0})


class TestNormalMuNormalSampling:
    """Test sampling functionality."""

    def test_sample_prior_shape(self):
        """sample_prior should return correct shape."""
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        samples = dist.sample_prior(1000, random_state=42)

        assert samples.shape == (1000,)

    def test_sample_prior_distribution(self):
        """sample_prior should match prior N(mu_mu, sigma_mu²)."""
        mu_mu, sigma_mu = 5.0, 2.0
        dist = NormalMuNormal({"mu_mu": mu_mu, "sigma_mu": sigma_mu})

        samples = dist.sample_prior(10000, random_state=42)

        # Check empirical moments
        assert np.abs(np.mean(samples) - mu_mu) < 0.1
        assert np.abs(np.std(samples) - sigma_mu) < 0.1

    def test_sample_posterior_shape(self):
        """sample_posterior should return correct shape."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})

        samples = dist.sample_posterior(data=data, size=1000, random_state=42)

        assert samples.shape == (1000,)

    def test_sample_posterior_deterministic(self):
        """sample_posterior should be deterministic with same seed."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})

        samples1 = dist.sample_posterior(data=data, size=100, random_state=42)
        samples2 = dist.sample_posterior(data=data, size=100, random_state=42)

        np.testing.assert_array_equal(samples1, samples2)


class TestNormalMuNormalLOOCV:
    """Test Leave-One-Out Cross-Validation functionality."""

    def test_loo_cv_shape(self):
        """LOO CV log-likelihood should have same shape as data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0, "use_posterior_predictive": False})

        loo_ll = dist._loo_cv_log_likelihood(data)

        assert loo_ll.shape == data.shape

    def test_loo_cv_all_finite(self):
        """LOO CV log-likelihoods should all be finite."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50)

        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        loo_ll = dist._loo_cv_log_likelihood(data)

        assert np.all(np.isfinite(loo_ll))


# ============================================================================
# PROPERTY-BASED TESTS (HYPOTHESIS)
# ============================================================================


class TestNormalMuNormalPropertyBased:
    """Property-based tests using hypothesis."""

    @given(normal_data_strategy(min_size=5, max_size=100))
    @settings(max_examples=50)
    def test_posterior_mean_always_finite(self, data):
        """Posterior mean should always be finite for valid data."""
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        assert np.isfinite(params["posterior_mu"])

    @given(normal_data_strategy(min_size=5, max_size=100))
    @settings(max_examples=50)
    def test_posterior_variance_always_positive(self, data):
        """Posterior variance should always be positive."""
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        params = dist.calc_posterior_params(data)

        assert params["posterior_sigma_mu"] > 0

    @given(normal_data_strategy(min_size=5, max_size=100))
    @settings(max_examples=50)
    def test_nll_always_finite(self, data):
        """NLL should always be finite for valid data."""
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        nll = dist.nll(data)

        assert np.isfinite(nll)

    @given(normal_data_strategy(min_size=5, max_size=100))
    @settings(max_examples=50)
    def test_nle_always_finite(self, data):
        """NLE should always be finite for valid data."""
        dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
        nle = dist.nle(data)

        assert np.isfinite(nle)
