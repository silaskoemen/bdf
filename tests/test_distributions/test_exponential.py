import numpy as np
import pytest
from scipy.stats import expon

from bdf.distributions.exponential import (
    GammaABLambdaExponential,
    GammaABLambdaExponentialParams,
    GammaABLambdaExponentialPP,
    GammaABLambdaExponentialPPParams,
    GammaMVLambdaExponential,
    GammaMVLambdaExponentialParams,
    GammaMVLambdaExponentialPP,
    GammaMVLambdaExponentialPPParams,
    PseudoLambdaExponential,
    PseudoLambdaExponentialParams,
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
    def get_bdf_params(self, get_alpha_beta):
        """Fixture to return BDF parameters for testing."""
        alpha, beta = get_alpha_beta
        return GammaABLambdaExponentialParams(alpha_lambda=alpha, beta_lambda=beta)

    def test_params_initialization(self, get_alpha_beta):
        alpha, beta = get_alpha_beta
        params = GammaABLambdaExponentialParams(alpha_lambda=alpha, beta_lambda=beta)
        dist = GammaABLambdaExponential(prior_params=params)
        assert dist.alpha_lambda == alpha
        assert dist.beta_lambda == beta
        assert dist.prior_params == params
        # Assert `params` if used for distribution

    @pytest.mark.parametrize("data", [get_small_data, get_large_data])
    def test_calc_posterior_params(self, get_alpha_beta, data):
        alpha_lambda, beta_lambda = get_alpha_beta
        prior_params = GammaABLambdaExponentialParams(alpha_lambda=alpha_lambda, beta_lambda=beta_lambda)
        posterior_params = GammaABLambdaExponential(prior_params=prior_params).calc_posterior_params(
            data, return_dict=True
        )
        posterior_alpha = alpha_lambda + len(data)
        posterior_beta = beta_lambda + np.sum(data)
        expected_posterior_lambda = posterior_alpha / posterior_beta
        assert posterior_params["posterior_lambda"] == expected_posterior_lambda

    @pytest.mark.parametrize(
        "data",
        [
            get_zeros_data,
            get_single_value_data,
            get_none_values,
            get_none_data,
            get_invalid_bounds_data,
            get_empty_data,
            get_inf_data,
        ],
        ids=["zeros", "single_value", "none_values", "none_data", "invalid_bounds", "empty", "inf_data"],
    )
    def test_calc_posterior_params_edge_cases(self, get_alpha_beta):
        assert False

    def test_correct_likelihoods(self, get_small_data, get_alpha_beta):
        # Check whether (log-)likelihoods are correct for certain values,
        # use closed form calculations to compare with scipy
        assert False

    def test_nll_overflow(self):
        # Check whether NLL does not overflow for large values
        assert False

    def test_sample_posterior_shape_dtype(self, get_alpha_beta):
        assert False

    def test_sample_posterior_params_vs_data(self, get_alpha_beta, get_small_data):
        assert False

    def test_sample_posterior_mean_variance(self, get_alpha_beta, get_small_data):
        assert False

    def test_sample_posterior_ks_test(self, get_alpha_beta, get_small_data):
        assert False

    def test_get_posterior_mean_variance(self):
        assert False

    def test_prior_strength(self, get_alpha_beta, get_large_data):
        assert False

    def test_posterior_params(self):
        assert False

    def test_invalid_data_dtype(self, get_bdf_params):
        # Only float accepted, test against int, str, bool
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            GammaABLambdaExponential(prior_params=get_bdf_params).calc_posterior_params(data)

        data = np.array(["a", "b", "c"])
        with pytest.raises(ValueError):
            GammaABLambdaExponential(prior_params=get_bdf_params).calc_posterior_params(data)

        data = np.array([True, False, True])
        with pytest.raises(ValueError):
            GammaABLambdaExponential(prior_params=get_bdf_params).calc_posterior_params(data)
