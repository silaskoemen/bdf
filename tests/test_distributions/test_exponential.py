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


class TestGammaABLambdaExponential:
    @pytest.fixture
    def get_lambda(self):
        """Fixture to return a lambda value for testing,
        gives mean of 10."""
        return 0.1

    @pytest.fixture
    def get_alpha_beta(self):
        """Fixture to return alpha and beta values for testing,
        gives mean of 5."""
        return 10.0, 2.0

    @pytest.fixture
    def get_exponential_samples(self, get_lambda):
        return expon.rvs(scale=1 / get_lambda, size=1000)

    def test_params_initialization(self):
        params = GammaABLambdaExponentialParams(alpha_lambda=2.0, beta_lambda=3.0)
        dist = GammaABLambdaExponential(prior_params=params)
        assert dist.alpha_lambda == 2.0
        assert dist.beta_lambda == 3.0

    def test_init(self):
        params = GammaABLambdaExponentialParams(alpha_lambda=2.0, beta_lambda=3.0)
        dist = GammaABLambdaExponential(prior_params=params)
        assert dist.alpha_lambda == 2.0
        assert dist.beta_lambda == 3.0

    def test_calc_posterior_params_small(self, get_alpha_beta):
        data = np.array([1.0, 2.0, 3.0])
        alpha_lambda, beta_lambda = get_alpha_beta
        prior_params = GammaABLambdaExponentialParams(alpha_lambda=alpha_lambda, beta_lambda=beta_lambda)
        posterior_params = GammaABLambdaExponential(prior_params=prior_params).calc_posterior_params(
            data, return_dict=True
        )
        posterior_alpha = alpha_lambda + len(data)
        posterior_beta = beta_lambda + np.sum(data)
        expected_posterior_lambda = posterior_alpha / posterior_beta
        assert posterior_params["posterior_lambda"] == expected_posterior_lambda

    def test_calc_posterior_params_large(self, get_alpha_beta, get_exponential_samples):
        # Sample from given exponetial distribution, then verify posterior parameters
        # given specific prior parameters.
        alpha_lambda, beta_lambda = get_alpha_beta
        prior_params = GammaABLambdaExponentialParams(alpha_lambda=alpha_lambda, beta_lambda=beta_lambda)
        posterior_params = GammaABLambdaExponential(prior_params=prior_params).calc_posterior_params(
            get_exponential_samples, return_dict=True
        )
        posterior_alpha = alpha_lambda + len(get_exponential_samples)
        posterior_beta = beta_lambda + np.sum(get_exponential_samples)
        expected_posterior_lambda = posterior_alpha / posterior_beta
        assert posterior_params["posterior_lambda"] == expected_posterior_lambda

    def test_sample_posterior(self, get_alpha_beta):
        alpha_lambda, beta_lambda = get_alpha_beta
        params = GammaABLambdaExponentialParams(alpha_lambda=alpha_lambda, beta_lambda=beta_lambda)
        dist = GammaABLambdaExponential(prior_params=params)

        # Test output shape and type

        # Test params vs data equality

        # Test mean within tolerance of expected mean

        #
        samples = dist.sample_posterior(size=10, random_state=42)
        assert len(samples) == 10

    def test_prior_strength(self):
        # Test whether larger amounts of data correctly influence the posterior mean
        # more given the fixed prior.
        assert False

    def test_log_likelihood(self):
        params = GammaABLambdaExponentialParams(alpha_lambda=2.0, beta_lambda=3.0)
        dist = GammaABLambdaExponential(prior_params=params)
        data = np.array([1.0, 2.0, 3.0])
        log_likelihoods = dist.log_likelihood(data)
        assert len(log_likelihoods) == len(data)

    def test_likelihood(self):
        params = GammaABLambdaExponentialParams(alpha_lambda=2.0, beta_lambda=3.0)
        dist = GammaABLambdaExponential(prior_params=params)
        data = np.array([1.0, 2.0, 3.0])
        likelihoods = dist.likelihood(data)
        assert len(likelihoods) == len(data)
