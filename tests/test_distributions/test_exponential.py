import numpy as np
import pytest
from scipy.stats import expon

from bdf.distributions.exponential import (
    GammaABLambdaExponential,
    GammaABLambdaExponentialParams,
    GammaMVLambdaExponential,
    GammaMVLambdaExponentialParams,
)


@pytest.fixture
def get_lambda():
    """Fixture to return a lambda value for testing,
    gives mean of 10."""
    return 0.1


@pytest.fixture
def get_alpha_beta():
    """Fixture to return alpha and beta values for testing,
    gives mean of 5."""
    return 10.0, 2.0


@pytest.fixture
def data(request):
    """Retrieve params from fixture name."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def get_large_data(get_lambda):
    return expon.rvs(scale=1 / get_lambda, size=1000)


@pytest.fixture
def get_small_data():
    """Fixture to return a small dataset for testing."""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def get_zeros_data():
    """Fixture to return a dataset with zeros for testing."""
    return np.zeros(100)


@pytest.fixture
def get_single_value_data(get_lambda):
    """Fixture to return a dataset with a single value for testing."""
    return np.array([1 / get_lambda] * 100)


@pytest.fixture
def get_none_data():
    """Fixture to return a None dataset for testing."""
    return None


@pytest.fixture
def get_invalid_bounds_data():
    """Fixture to return a dataset with invalid bounds for testing."""
    return np.array([-1.0, 0.0, 1.0])


@pytest.fixture
def get_empty_data():
    """Fixture to return an empty dataset for testing."""
    return np.array([])


@pytest.fixture
def get_inf_data():
    """Fixture to return a dataset with infinite values for testing."""
    return np.array([1.0, np.inf, 3.0])


@pytest.fixture
def get_none_values():
    """Fixture to return a dataset with None values for testing."""
    return np.array([1.0, 2.0, None, 4.0, 5.0])


class TestGammaABLambdaExponential:
    @pytest.fixture
    def get_bdf_dist_params(self, get_alpha_beta):
        """Fixture to return BDF parameters for testing."""
        alpha, beta = get_alpha_beta
        return GammaABLambdaExponentialParams(alpha_lambda=alpha, beta_lambda=beta)

    def test_params_initialization(self, get_alpha_beta):
        alpha, beta = get_alpha_beta
        params = GammaABLambdaExponentialParams(alpha_lambda=alpha, beta_lambda=beta)
        dist = GammaABLambdaExponential(params=params)
        assert dist.alpha_lambda == alpha
        assert dist.beta_lambda == beta
        assert dist.params == params
        # Assert `params` if used for distribution

    @pytest.mark.parametrize("data", ["get_small_data", "get_large_data"], indirect=True)
    def test_calc_posterior_params(self, get_alpha_beta, data):
        alpha_lambda, beta_lambda = get_alpha_beta
        params = GammaABLambdaExponentialParams(alpha_lambda=alpha_lambda, beta_lambda=beta_lambda)
        posterior_params = GammaABLambdaExponential(params=params).calc_posterior_params(data)
        posterior_alpha = alpha_lambda + len(data)
        posterior_beta = beta_lambda + np.sum(data)
        expected_posterior_lambda = posterior_alpha / posterior_beta
        assert posterior_params["posterior_lambda"] == expected_posterior_lambda  # type: ignore

    def test_calc_posterior_params_none_data(self, get_bdf_dist_params):
        """None data should raise an error (no .shape attribute)."""
        with pytest.raises((AttributeError, TypeError)):
            GammaABLambdaExponential(params=get_bdf_dist_params).calc_posterior_params(None)

    def test_correct_likelihoods(self, get_small_data, get_alpha_beta):
        """Check plugin log-likelihoods against scipy reference."""
        alpha, beta = get_alpha_beta
        params = GammaABLambdaExponentialParams(alpha_lambda=alpha, beta_lambda=beta)
        dist = GammaABLambdaExponential(params=params)
        posterior = dist.calc_posterior_params(get_small_data)
        post_lambda = posterior["posterior_lambda"]

        plugin_ll = dist._plugin_log_likelihood(get_small_data, posterior)
        scipy_ll = expon.logpdf(get_small_data, scale=1 / post_lambda)
        np.testing.assert_allclose(plugin_ll, scipy_ll, rtol=1e-10)

    def test_nll_overflow(self, get_bdf_dist_params):
        """NLL should not overflow for large values."""
        data = np.array([1e6, 2e6, 3e6])
        dist = GammaABLambdaExponential(params=get_bdf_dist_params)
        posterior = dist.calc_posterior_params(data)
        nll = dist.nll(data, posterior)
        assert np.isfinite(nll)

    def test_sample_posterior_shape_dtype(self, get_bdf_dist_params, get_small_data):
        """Posterior samples should have correct shape and float dtype."""
        dist = GammaABLambdaExponential(params=get_bdf_dist_params)
        samples = dist.sample_posterior(data=get_small_data, size=100)
        assert samples.shape == (100,)
        assert samples.dtype == np.float64

    def test_sample_posterior_params_vs_data(self, get_bdf_dist_params, get_small_data):
        """Posterior samples should be positive (exponential support)."""
        dist = GammaABLambdaExponential(params=get_bdf_dist_params)
        samples = dist.sample_posterior(data=get_small_data, size=500)
        assert np.all(samples > 0)

    def test_sample_posterior_mean_variance(self, get_bdf_dist_params, get_large_data):
        """Posterior sample mean should be close to posterior mean."""
        dist = GammaABLambdaExponential(params=get_bdf_dist_params)
        samples = dist.sample_posterior(data=get_large_data, size=5000)
        post_mean = dist.get_posterior_mean(data=get_large_data)
        assert np.abs(samples.mean() - post_mean) / post_mean < 0.1

    def test_sample_posterior_ks_test(self, get_bdf_dist_params, get_large_data):
        """Posterior samples should pass a KS test against the expected distribution."""
        from scipy.stats import kstest

        dist = GammaABLambdaExponential(params=get_bdf_dist_params)
        posterior = dist.calc_posterior_params(get_large_data)
        post_lambda = posterior["posterior_lambda"]
        samples = dist.sample_posterior(data=get_large_data, size=2000)
        _, p_value = kstest(samples, "expon", args=(0, 1 / post_lambda))
        assert p_value > 0.01

    def test_get_posterior_mean_variance(self, get_bdf_dist_params, get_small_data):
        """Posterior mean and variance should be finite and positive."""
        dist = GammaABLambdaExponential(params=get_bdf_dist_params)
        mean = dist.get_posterior_mean(data=get_small_data)
        var = dist.get_posterior_variance(data=get_small_data)
        assert np.isfinite(mean) and mean > 0
        assert np.isfinite(var) and var > 0

    def test_prior_strength(self, get_alpha_beta, get_large_data):
        """Strong prior should pull posterior lambda away from data MLE."""
        data_mle_lambda = 1.0 / np.mean(get_large_data)

        # Weak prior (barely informative)
        weak = GammaABLambdaExponential(params=GammaABLambdaExponentialParams(alpha_lambda=0.01, beta_lambda=0.01))
        weak_post = weak.calc_posterior_params(get_large_data)

        # Strong prior centered far from data MLE (prior mean lambda = 100)
        strong = GammaABLambdaExponential(
            params=GammaABLambdaExponentialParams(alpha_lambda=10000.0, beta_lambda=100.0)
        )
        strong_post = strong.calc_posterior_params(get_large_data)

        # Strong prior should pull posterior lambda further from MLE
        assert abs(strong_post["posterior_lambda"] - data_mle_lambda) > abs(
            weak_post["posterior_lambda"] - data_mle_lambda
        )

    def test_posterior_params(self, get_bdf_dist_params, get_small_data):
        """Posterior params should have expected keys and finite values."""
        dist = GammaABLambdaExponential(params=get_bdf_dist_params)
        posterior = dist.calc_posterior_params(get_small_data)
        assert "posterior_lambda" in posterior
        assert np.isfinite(posterior["posterior_lambda"])
        assert posterior["posterior_lambda"] > 0
