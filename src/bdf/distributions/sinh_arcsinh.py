"""
File for the sinh-arcsinh (SAS/SHASH) distribution.

Due to no closed-form MLE/MoM estimates, BOBYQA is used for parameter estimation
(also in Rust calculations). Permits an additional parameter for the maximum number
of iterations in the optimization process.

Uses the reparameterization of sigma/delta as sigma_delta in estimation,
then changes it back for NLL evaluation.

Given that MLE estimates are used, asymptotic normality can be used for
independent Normal-Normal posterior calculations, although this is only implemented
for epsilon and delta as sigma is considered fixed and mu is used to recover the
posterior mean.

In the different versions, likelihood will be abbreviated as SHASH. Permits:
- NormalMeanPseudoEpsilonSHASH
- NormalMeanPseudoEpsilonPseudoDeltaSHASH
- NormalMeanNormalEpsilonSHASH
- NormalMeanNormalEpsilonNormalDeltaSHASH

Could also implement mixtures of normal and pseudo for epsilon and delta, but
currently no evidence of this being useful.
"""

import numpy as np
from scipy.stats import norm

from bdf.distributions.bdf_distribution import BDFDistribution
from bdf.utils.constants import RANDOM_SEED


# Define helper functions for likelihood calculations
def c_epsilon_delta(z, epsilon, delta):
    return np.cosh(epsilon + delta * np.arcsinh(z))


def s_epsilon_delta(z, epsilon, delta):
    return np.sinh(epsilon + delta * np.arcsinh(z))


# Helper functions for asumptotic variance calculations
def fm_delta(delta) -> float:
    raise NotImplementedError("fm_delta is not implemented yet.")


def fs_delta(delta) -> float:
    raise NotImplementedError("fs_delta is not implemented yet.")


def fc_delta(delta) -> float:
    raise NotImplementedError("fc_delta is not implemented yet.")


def fd_delta(delta) -> float:
    raise NotImplementedError("fd_delta is not implemented yet.")


def fisher_info_mu(delta, sigma):
    return fm_delta(delta) / sigma**2


def fisher_info_sigma(delta, sigma):
    return fs_delta(delta) / sigma**2


def fisher_info_delta(delta, sigma):
    return fd_delta(delta) / sigma**2


class SHASHBase(BDFDistribution):
    """Base class for all SHASH distribution implementations.

    All implementations use parameters alpha, xi and omega, so sampling,
    (log)likelihoods and return functions are all identical.
    """

    # This is just a placeholder - child classes will have their own init
    def __init__(self, params):
        super().__init__(params)

    def calc_posterior_params(self, data):
        raise NotImplementedError("Subclass must implement calc_posterior_params.")

    def log_likelihood(self, data: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the log-likelihood for.
        `params` : dict | None
            Optional pre-computed posterior parameters.

        Returns
        -------
        np.ndarray
            A numpy array containing the log-likelihood values for each data point.
        """
        if params is None:
            params = self.calc_posterior_params(data)
        posterior_mu, posterior_sigma, posterior_epsilon, posterior_delta = (
            params["posterior_mu"],
            params["posterior_sigma"],
            params["posterior_epsilon"],
            params["posterior_delta"],
        )
        if posterior_sigma <= 0 or posterior_delta <= 0:
            raise ValueError("Scale sigma and tailweight delta must be positive.")

        z = (data - posterior_mu) / posterior_sigma
        # Calculate the log-likelihood using the skew-normal distribution
        return (
            -np.log(posterior_sigma)
            - 0.5 * np.log(2 * np.pi)
            + np.log(posterior_delta)
            + np.log(c_epsilon_delta(z, posterior_epsilon, posterior_delta))
            - 0.5 * np.log(1 + z**2)
            - 0.5 * s_epsilon_delta(z, posterior_epsilon, posterior_delta) ** 2
        )

    def likelihood(self, data: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Compute the likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the likelihood for.
        `params` : dict | None
            Optional pre-computed posterior parameters.

        Returns
        -------
        np.ndarray
            A numpy array containing the likelihood values for each data point.
        """
        if params is None:
            params = self.calc_posterior_params(data)
        posterior_mu, posterior_sigma, posterior_epsilon, posterior_delta = (
            params["posterior_mu"],
            params["posterior_sigma"],
            params["posterior_epsilon"],
            params["posterior_delta"],
        )
        if posterior_sigma <= 0 or posterior_delta <= 0:
            raise ValueError("Scale sigma and tailweight delta must be positive.")

        z = (data - posterior_mu) / posterior_sigma
        # Calculate the likelihood using the skew-normal distribution
        return (
            1
            / np.sqrt(2 * np.pi * posterior_sigma**2)
            * posterior_delta
            * c_epsilon_delta(z, posterior_epsilon, posterior_delta)
            / np.sqrt(1 + z**2)
            * np.exp(-0.5 * s_epsilon_delta(z, posterior_epsilon, posterior_delta) ** 2)
        )

    def nll(self, data: np.ndarray, params: dict | None = None) -> float:
        """Compute the negative log-likelihood of the data given the distribution.

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

    def sample_posterior(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict | None = None,
        size: int = 1,
        random_state: int = RANDOM_SEED,
    ) -> np.ndarray:
        """Sample from the posterior distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to condition the posterior on. If None, uses prior parameters.
        `params` : dict | None, optional
            Additional parameters for sampling.
        `size` : int, optional
            The number of samples to generate.
        `random_state` : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Samples from the posterior distribution.
        """
        if params is not None:
            return self._sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self._sample_posterior_data(data, size=size, random_state=random_state)
        else:
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution given parameters.

        Args
        ----
        `params` : dict[str, float]
            Parameters for the posterior distribution.
        `size` : int, optional
            The number of samples to generate, default is 1.
        `random_state` : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Samples from the posterior distribution.
        """
        mu, sigma, epsilon, delta = (
            params.get("mu", params.get("posterior_mu")),
            params.get("sigma", params.get("posterior_sigma")),
            params.get("epsilon", params.get("posterior_epsilon", 0.0)),
            params.get("delta", params.get("posterior_delta", 0.0)),
        )

        if mu is None or sigma is None or epsilon is None or delta is None:
            raise ValueError("params must contain 'mu', 'sigma', 'epsilon', and 'delta' keys.")

        assert sigma > 0 and delta > 0, f"sigma and delta must be positive, got {sigma =}, {delta =}"

        # Generate samples from the SHASH distribution
        z = norm.rvs(size=size, random_state=random_state)
        # NOTE: Summary article uses + epsilon, could change
        return mu + sigma * np.sinh((np.arcsinh(z) - epsilon) / delta)

    def _sample_posterior_data(self, data: np.ndarray, *, size: int = 1, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample from the posterior distribution given data.

        Args
        ----
        `data` : np.ndarray
            The data to condition the posterior on.
        `size` : int, optional
            The number of samples to generate, default is 1.
        `random_state` : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Samples from the posterior distribution.
        """
        posterior_params = self.calc_posterior_params(data)
        return self._sample_posterior_params(posterior_params, size=size, random_state=random_state)
