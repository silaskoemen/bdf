from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from bdf.utils.constants import RANDOM_SEED


class BDFDistributionParams(BaseModel, ABC):
    pass


class BDFDistribution(ABC):
    """Abstract base class for all BDF distributions."""

    def __init__(self, prior_params: BDFDistributionParams, params: tuple | None = None):
        self.prior_params = prior_params
        self.params = params

    @abstractmethod
    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = True) -> dict | tuple:
        """Get the posterior parameters of the distribution given the data."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def nll(self, data: np.ndarray) -> float:
        """Compute the negative log-likelihood of the data given the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the likelihood of the data given the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def sample_prior(self, size: int) -> np.ndarray:
        """Sample from the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def sample_posterior(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
        size: int = 1,
        random_state: int = RANDOM_SEED,
    ) -> np.ndarray:
        """Sample from the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior mean of the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_posterior_std(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior standard deviation of the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters of the distribution."""
        raise NotImplementedError("Subclasses must implement this method.")

    def __repr__(self):
        return f"{self.__class__.__name__}(prior_params={self.prior_params}, params={self.params})"

    def __str__(self):
        return f"{self.__class__.__name__}(prior_params={self.prior_params}, params={self.params})"
