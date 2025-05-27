from bdf.distributions.bdf_distribution import BDFDistribution
from typing import Dict, Any
import numpy as np
import math


class BetaBinomial(BDFDistribution):
    """ Beta-Binomial distribution class for Bayesian Distributional Forests.
    Beta prior on the probability of success given a known number of trials.
    """
    def __init__(self, prior_params: Dict[str, Any], params: tuple | None = None):
        """ Initialize the Beta-Binomial distribution with prior parameters.
        
        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'alpha' and 'beta'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        super().__init__(prior_params, params)
        self.prior_alpha = prior_params.get('alpha', 1.0)
        self.prior_beta = prior_params.get('beta', 1.0)

    def calc_posterior_params(self, data: np.ndarray) -> tuple[float, float]:
        """ Calculate posterior parameters based on the data.
        
        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        
        Returns
        -------
        tuple[float, float]
            A tuple containing the posterior alpha and posterior beta.
        """
        n_successes = np.sum(data)
        n_trials = data.shape[0]
        
        posterior_alpha = self.prior_alpha + n_successes
        posterior_beta = self.prior_beta + n_trials - n_successes
        
        return posterior_alpha, posterior_beta

    def nll(self, data: np.ndarray) -> float:
        """ Compute the negative log-likelihood of the data given the distribution.
        
        Args
        ----
        `data` : np.ndarray
            The data to compute the negative log-likelihood for.
        
        Returns
        -------
        float
            The negative log-likelihood value.
        """
        return -np.sum(self.log_likelihood(data))

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """ Compute the likelihood of the data given the distribution.
        
        Args
        ----
        `data` : np.ndarray
            The data to compute the likelihood for.
        
        Returns
        -------
        float
            The likelihood value.
        """
        posterior_alpha, posterior_beta = self.calc_posterior_params(data)
        
        # Using the Beta-Binomial probability mass function (PMF)
        return (
            data ** (posterior_alpha - 1) * (1 - data) ** (posterior_beta - 1) /
            (math.gamma(posterior_alpha) * math.gamma(posterior_beta)) * math.gamma(posterior_alpha + posterior_beta)
        )
    
    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """ Compute the log-likelihood of the data given the distribution.
        Args
        ----
        `data` : np.ndarray
            The data to compute the log-likelihood for.
        Returns
        -------
        float
            The log-likelihood value.
        """
        posterior_alpha, posterior_beta = self.calc_posterior_params(data)
        
        # Using the log of the Beta-Binomial PMF
        return (
            (posterior_alpha - 1) * np.log(data) + (posterior_beta - 1) * np.log(1 - data) -
            (math.lgamma(posterior_alpha) + math.lgamma(posterior_beta)) + math.lgamma(posterior_alpha + posterior_beta)
        )
    
    def sample_prior(self, size: int) -> np.ndarray:
        """ Sample from the prior distribution.
        Args
        ----
        `size` : int
            The number of samples to draw from the prior distribution.
        Returns
        -------
        np.ndarray
            An array of samples drawn from the prior distribution.
        """
        return np.random.beta(self.prior_alpha, self.prior_beta, size=size)
    
    def sample_posterior(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None, size: int = 1) -> np.ndarray:
        """ Sample from the posterior distribution.
        
        Args
        ----
        `data` : np.ndarray | None
            The data to calculate the posterior parameters from, if available.
        `params` : dict[str, float] | None
            The posterior parameters to sample from, if available.
        `size` : int
            The number of samples to generate.
        
        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution.
        """
        if params is not None:
            assert 'alpha' in params and 'beta' in params, "params must contain 'alpha' and 'beta' keys"
            assert params['beta'] > 0, "Beta parameter must be positive"
            assert params['alpha'] > 0, "Alpha parameter must be positive"
            return np.random.beta(params['alpha'], params['beta'], size=size)
        elif data is not None:
            posterior_alpha, posterior_beta = self.calc_posterior_params(data)
            return np.random.beta(posterior_alpha, posterior_beta, size=size)
        else:
            raise ValueError("Either 'data' or 'params' must be provided to generate samples from the posterior distribution.")