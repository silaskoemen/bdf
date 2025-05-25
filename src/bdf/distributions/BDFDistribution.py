import numpy
from abc import ABC, abstractmethod


class BDFDistribution(ABC):
    """ Abstract base class for all BDF distributions.
    """
    def __init__(self, prior_params: dict, params: tuple = None):
        self.prior_params = prior_params
        self.params = params
    
    @abstractmethod
    def calc_posterior_params(self, data: numpy.ndarray) -> dict:
        """ Get the posterior parameters of the distribution given the data.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def nll(self, data: numpy.ndarray) -> float:
        """ Compute the negative log-likelihood of the data given the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def likelihood(self, data: numpy.ndarray) -> float:
        """ Compute the likelihood of the data given the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def log_likelihood(self, data: numpy.ndarray) -> float:
        """ Compute the log-likelihood of the data given the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def sample_prior(self, size: int) -> numpy.ndarray:
        """ Sample from the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def sample_posterior(self, size: int) -> numpy.ndarray:
        """ Sample from the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def get_posterior_params(self) -> dict:
        """ Get the posterior parameters of the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")     
    
    def __repr__(self):
        return f"{self.__class__.__name__}(prior_params={self.prior_params}, params={self.params})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(prior_params={self.prior_params}, params={self.params})"