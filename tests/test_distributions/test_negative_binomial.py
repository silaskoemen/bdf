"""Comprehensive tests for Negative Binomial distributions.

Tests verify:
1. Mathematical correctness of conjugate Gamma-NB updates
2. Exact linear shrinkage formula
3. Statistical properties match theoretical values
4. Log-likelihood and log-evidence computations
5. Edge cases and numerical stability
6. Prior sensitivity and dispersion parameter handling

Reference implementations tested:
- FrequentistNegativeBinomial: MLE/MoM estimation
- NormalMeanNegativeBinomial: Hybrid Normal prior approach
- GammaMSLambdaNegBin: Conjugate Gamma prior with mean-strength parameterization
"""

import numpy as np
import pytest
from scipy import stats
from scipy.special import gammaln
from scipy.stats import nbinom

from bdf.distributions.negative_binomial import (
    FrequentistNegativeBinomial,
    FrequentistNegativeBinomialParams,
    GammaMSLambdaNegBin,
    GammaMSLambdaNegBinParams,
    NormalMeanNegativeBinomial,
    NormalMeanNegativeBinomialParams,
)

# Import shared helpers
from tests.test_distributions.helpers import (
    RTOL_NUMERICAL,
    RTOL_STATISTICAL,
    RTOL_TIGHT,
    assert_close,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def nb_data_small():
    """Small NB dataset (n=10) for quick tests."""
    # Generate NB(λ=5, φ=2) data
    np.random.seed(42)
    r, p = 2.0, 2.0 / (2.0 + 5.0)
    return nbinom.rvs(n=r, p=p, size=10)


@pytest.fixture
def nb_data_large():
    """Large NB dataset (n=1000) for statistical tests."""
    np.random.seed(42)
    r, p = 2.0, 2.0 / (2.0 + 5.0)
    return nbinom.rvs(n=r, p=p, size=1000)


@pytest.fixture
def nb_data_overdispersed():
    """Highly overdispersed NB dataset (φ=0.5)."""
    np.random.seed(42)
    r, p = 0.5, 0.5 / (0.5 + 10.0)
    return nbinom.rvs(n=r, p=p, size=100)


@pytest.fixture
def nb_data_poisson_like():
    """NB dataset with large φ (approaches Poisson)."""
    np.random.seed(42)
    r, p = 100.0, 100.0 / (100.0 + 5.0)
    return nbinom.rvs(n=r, p=p, size=100)


# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================


def reference_gamma_nb_posterior_lambda(data: np.ndarray, mean_lambda: float, strength_lambda: float) -> float:
    """Reference implementation of exact shrinkage formula.

    E[λ|y] = (m·λ₀ + n·ȳ) / (m + n)
    """
    m = strength_lambda
    lambda_0 = mean_lambda
    n = len(data)
    y_bar = np.mean(data)
    return (m * lambda_0 + n * y_bar) / (m + n)


def reference_phi_mom(data: np.ndarray) -> float:
    """Reference MoM estimator for φ.

    φ = μ²/(σ²-μ)
    """
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    if var <= mean:
        return 100.0  # No overdispersion
    return (mean**2) / (var - mean)


# ============================================================================
# TEST GAMMA-MEAN-STRENGTH NEGATIVE BINOMIAL
# ============================================================================


class TestGammaMSLambdaNegBinMath:
    """Verify mathematical correctness of conjugate Gamma-NB updates."""

    def test_exact_shrinkage_formula(self, nb_data_small):
        """Posterior mean matches exact linear shrinkage formula."""
        mean_lambda, strength_lambda = 5.0, 1.0

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": mean_lambda,
                "strength_lambda": strength_lambda,
                "phi": None,
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        # Exact formula
        expected = reference_gamma_nb_posterior_lambda(nb_data_small, mean_lambda, strength_lambda)

        assert_close(
            params["posterior_lambda"],
            expected,
            rtol=RTOL_TIGHT,
            msg="Posterior lambda doesn't match exact shrinkage formula",
        )

    @pytest.mark.parametrize(
        "mean_lambda,strength_lambda",
        [
            (1.0, 0.1),  # Weak prior
            (5.0, 1.0),  # Moderate prior
            (10.0, 10.0),  # Strong prior
            (2.5, 5.0),  # Asymmetric
        ],
    )
    def test_shrinkage_formula_parametrized(self, nb_data_large, mean_lambda, strength_lambda):
        """Shrinkage formula holds across parameter ranges."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": mean_lambda,
                "strength_lambda": strength_lambda,
                "phi": None,
            }
        )
        params = dist.calc_posterior_params(nb_data_large)

        expected = reference_gamma_nb_posterior_lambda(nb_data_large, mean_lambda, strength_lambda)

        assert_close(params["posterior_lambda"], expected, rtol=RTOL_TIGHT)

    def test_conjugate_update_alpha_beta(self, nb_data_small):
        """Posterior Gamma parameters match conjugate update formulas."""
        mean_lambda, strength_lambda = 5.0, 1.0

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": mean_lambda,
                "strength_lambda": strength_lambda,
                "phi": 2.0,
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        # Conjugate update: Gamma(α₀, β₀) → Gamma(α₀ + Σy, β₀ + n)
        alpha_0 = strength_lambda * mean_lambda
        beta_0 = strength_lambda
        n = len(nb_data_small)
        sum_y = np.sum(nb_data_small)

        expected_alpha = alpha_0 + sum_y
        expected_beta = beta_0 + n

        assert_close(params["posterior_alpha"], expected_alpha, rtol=RTOL_TIGHT)
        assert_close(params["posterior_beta"], expected_beta, rtol=RTOL_TIGHT)

    def test_phi_estimation_mom(self, nb_data_overdispersed):
        """Dispersion φ estimated correctly via MoM."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 10.0,
                "strength_lambda": 1.0,
                "phi": None,  # Estimate
            }
        )
        params = dist.calc_posterior_params(nb_data_overdispersed)

        expected_phi = reference_phi_mom(nb_data_overdispersed)

        assert_close(params["phi"], expected_phi, rtol=RTOL_NUMERICAL)

    def test_phi_fixed_preserved(self, nb_data_small):
        """Fixed φ is preserved (not estimated)."""
        fixed_phi = 3.5

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": fixed_phi,
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        assert params["phi"] == fixed_phi

    def test_prior_limit_large_strength(self, nb_data_small):
        """Large strength → posterior dominated by prior."""
        mean_lambda, strength_lambda = 5.0, 1000.0

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": mean_lambda,
                "strength_lambda": strength_lambda,
                "phi": None,
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        # With m >> n, E[λ|y] ≈ λ₀
        assert_close(params["posterior_lambda"], mean_lambda, rtol=0.01)

    def test_data_limit_weak_prior(self, nb_data_large):
        """Weak prior + large n → posterior dominated by data."""
        mean_lambda, strength_lambda = 10.0, 0.1  # Weak prior

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": mean_lambda,
                "strength_lambda": strength_lambda,
                "phi": None,
            }
        )
        params = dist.calc_posterior_params(nb_data_large)

        # With m << n, E[λ|y] ≈ ȳ
        sample_mean = np.mean(nb_data_large)
        assert_close(params["posterior_lambda"], sample_mean, rtol=0.01)


class TestGammaMSLambdaNegBinNumParameters:
    """Test parameter counting for BIC/AIC corrections."""

    def test_num_parameters_phi_none(self):
        """When φ=None (estimated), should return 2 parameters."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        assert dist._num_parameters() == 2  # λ and φ

    def test_num_parameters_phi_fixed(self):
        """When φ is fixed, should return 1 parameter."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
            }
        )
        assert dist._num_parameters() == 1  # Only λ


class TestGammaMSLambdaNegBinStatistical:
    """Statistical properties and posterior predictive tests."""

    def test_posterior_mean_variance(self, nb_data_large):
        """Posterior mean and variance methods return correct values."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
                "use_posterior_predictive": False,
            }
        )
        params = dist.calc_posterior_params(nb_data_large)

        # Get via method
        mean_via_method = dist.get_posterior_mean(params=params)
        var_via_method = dist.get_posterior_variance(params=params)

        # Expected values
        lambda_post = params["posterior_lambda"]
        phi = params["phi"]

        expected_mean = lambda_post
        expected_var = lambda_post + (lambda_post**2) / phi

        assert_close(mean_via_method, expected_mean, rtol=RTOL_TIGHT)
        assert_close(var_via_method, expected_var, rtol=RTOL_TIGHT)

    def test_posterior_predictive_wider_variance(self, nb_data_large):
        """Posterior predictive has wider variance than plug-in (accounts for λ uncertainty)."""
        dist_plugin = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
                "use_posterior_predictive": False,
            }
        )

        dist_pp = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
                "use_posterior_predictive": True,
            }
        )

        var_plugin = dist_plugin.get_posterior_variance(data=nb_data_large)
        var_pp = dist_pp.get_posterior_variance(data=nb_data_large)

        assert var_pp > var_plugin, "Posterior predictive should have wider variance than plug-in"

    def test_posterior_predictive_samples_distribution(self, nb_data_small):
        """Posterior predictive samples have correct empirical moments."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
                "use_posterior_predictive": True,
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        # Generate many samples
        samples = dist._sample_posterior_params(params, size=10000, random_state=42)

        # Expected moments
        expected_mean = dist.get_posterior_mean(params=params)
        expected_var = dist.get_posterior_variance(params=params)

        # Empirical moments
        empirical_mean = np.mean(samples)
        empirical_var = np.var(samples)

        assert_close(empirical_mean, expected_mean, rtol=RTOL_STATISTICAL)
        assert_close(empirical_var, expected_var, rtol=RTOL_STATISTICAL)


class TestGammaMSLambdaNegBinLogEvidence:
    """Test log evidence computation."""

    def test_log_evidence_computed(self, nb_data_small):
        """Log evidence can be computed without errors."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
            }
        )
        log_ev = dist.log_evidence(nb_data_small)

        assert np.isfinite(log_ev), "Log evidence should be finite"

    def test_log_evidence_larger_with_more_data(self):
        """Log evidence generally increases with more data (when model fits well)."""
        np.random.seed(42)
        r, p = 2.0, 2.0 / 7.0
        data_small = nbinom.rvs(n=r, p=p, size=10)
        data_large = nbinom.rvs(n=r, p=p, size=100)

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
            }
        )

        log_ev_small = dist.log_evidence(data_small)
        log_ev_large = dist.log_evidence(data_large)

        # More data should increase evidence (positive log likelihood contribution)
        # Note: This is not always true if model is misspecified, but should hold for well-matched data
        assert log_ev_large > log_ev_small


class TestGammaMSLambdaNegBinEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_sample_n_equals_1(self):
        """Handles single observation gracefully."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([3])
        params = dist.calc_posterior_params(data)

        # Should not crash, should return reasonable values
        assert np.isfinite(params["posterior_lambda"])
        assert np.isfinite(params["phi"])

    def test_small_sample_n_equals_2(self):
        """Handles two observations (minimum for variance)."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([2, 5])
        params = dist.calc_posterior_params(data)

        assert np.isfinite(params["posterior_lambda"])
        assert np.isfinite(params["phi"])

    def test_no_overdispersion_var_equals_mean(self):
        """When var ≈ mean (Poisson-like), φ should be large."""
        # Generate Poisson data (var = mean)
        np.random.seed(42)
        data = stats.poisson.rvs(mu=10, size=100)

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 10.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        params = dist.calc_posterior_params(data)

        # φ should be large (approaching Poisson)
        assert params["phi"] >= 10.0, "φ should be large for Poisson-like data"

    def test_constant_data(self):
        """Handles constant data (zero variance)."""
        data = np.array([5, 5, 5, 5, 5])

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        params = dist.calc_posterior_params(data)

        # Should return large φ (no overdispersion)
        # For constant data, var=0 <= mean, so phi = mean^2 = 25.0
        assert params["phi"] >= 25.0
        assert np.isfinite(params["posterior_lambda"])

    def test_zeros_in_data(self):
        """Handles zeros in count data."""
        data = np.array([0, 0, 1, 2, 0, 3])

        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 1.0,
                "strength_lambda": 1.0,
                "phi": 2.0,
            }
        )
        params = dist.calc_posterior_params(data)

        assert np.isfinite(params["posterior_lambda"])
        assert params["posterior_lambda"] >= 0


class TestGammaMSLambdaNegBinValidation:
    """Test data validation."""

    def test_rejects_negative_data(self):
        """Rejects negative values."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([1, -2, 3])

        with pytest.raises(ValueError, match="non-negative"):
            dist.validate_targets(data)

    def test_rejects_non_integer_data(self):
        """Rejects non-integer values."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([1.5, 2.3, 3.0])

        with pytest.raises(ValueError, match="integer"):
            dist.validate_targets(data)

    def test_rejects_non_finite_data(self):
        """Rejects NaN and inf values."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([1.0, np.inf, 3.0])

        with pytest.raises(ValueError, match="non-finite"):
            dist.validate_targets(data)

    def test_rejects_empty_data(self):
        """Rejects empty arrays."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([])

        with pytest.raises(ValueError, match="empty"):
            dist.validate_targets(data)

    def test_rejects_multidimensional_data(self):
        """Rejects non-1D arrays."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="1-dimensional"):
            dist.validate_targets(data)

    def test_accepts_valid_data(self):
        """Accepts valid integer count data."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        data = np.array([0, 1, 2, 3, 5, 10])

        # Should not raise
        dist.validate_targets(data)


class TestGammaMSLambdaNegBinCapabilities:
    """Test capability flags."""

    def test_supports_nle(self):
        """Confirms NLE support."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        assert dist._supports_nle is True

    def test_supports_posterior_predictive(self):
        """Confirms posterior predictive support."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        assert dist._supports_posterior_predictive is True

    def test_no_fast_loo_cv(self):
        """Confirms no fast LOO-CV (not yet implemented)."""
        dist = GammaMSLambdaNegBin(
            {
                "mean_lambda": 5.0,
                "strength_lambda": 1.0,
                "phi": None,
            }
        )
        assert dist._has_fast_loo_cv is False


class TestGammaMSLambdaNegBinAutoParams:
    """Test auto-parameter resolution."""

    def test_auto_mean_lambda_resolves_to_sample_mean(self, nb_data_large):
        """Auto-resolution of mean_lambda uses sample mean."""
        expected = float(np.mean(nb_data_large))
        resolved = GammaMSLambdaNegBin.resolve_auto_params("mean_lambda", nb_data_large)

        assert_close(resolved, expected, rtol=RTOL_TIGHT)

    def test_auto_unknown_param_raises(self, nb_data_small):
        """Unknown auto parameter raises error."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            GammaMSLambdaNegBin.resolve_auto_params("unknown_param", nb_data_small)


# ============================================================================
# TEST FREQUENTIST NEGATIVE BINOMIAL (basic smoke tests)
# ============================================================================


class TestFrequentistNegativeBinomial:
    """Basic tests for MLE/MoM Negative Binomial."""

    def test_mom_estimation(self, nb_data_small):
        """MoM estimation returns valid parameters."""
        dist = FrequentistNegativeBinomial(
            {
                "estimation_method": "mom",
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        assert params["r"] > 0
        assert 0 < params["p"] < 1

    def test_mle_estimation(self, nb_data_small):
        """MLE estimation returns valid parameters."""
        dist = FrequentistNegativeBinomial(
            {
                "estimation_method": "mle",
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        assert params["r"] > 0
        assert 0 < params["p"] < 1

    def test_num_parameters(self):
        """Returns 2 parameters (r and p)."""
        dist = FrequentistNegativeBinomial({})
        assert dist._num_parameters() == 2


# ============================================================================
# TEST NORMAL-MEAN NEGATIVE BINOMIAL (basic smoke tests)
# ============================================================================


class TestNormalMeanNegativeBinomial:
    """Basic tests for hybrid Normal prior NB."""

    def test_basic_estimation(self, nb_data_small):
        """Normal-mean estimation returns valid parameters."""
        dist = NormalMeanNegativeBinomial(
            {
                "prior_mean": 5.0,
                "prior_std": 2.0,
            }
        )
        params = dist.calc_posterior_params(nb_data_small)

        assert params["r"] > 0
        assert 0 < params["p"] < 1
        assert "posterior_mean_mu" in params

    def test_num_parameters(self):
        """Returns 2 parameters (r and p)."""
        dist = NormalMeanNegativeBinomial(
            {
                "prior_mean": 5.0,
                "prior_std": 2.0,
            }
        )
        assert dist._num_parameters() == 2
