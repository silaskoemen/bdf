import numpy as np
from bdf.distributions.bdf_distribution import BDFDistribution


class NormalNormal(BDFDistribution):
    """ Normal distribution class for Bayesian Distributional Forests.
    """
    def __init__(self, prior_params: dict, params: tuple | None = None, var_ddof: int = 1):
        """ Initialize the Normal distribution with prior parameters.
        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'mean' and 'std'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        `var_ddof` : int, optional
            Degrees of freedom for variance calculation, default is 1 (sample standard deviation).
        """
        super().__init__(prior_params, params)
        self.prior_mean = prior_params.get('mean', 0)
        self.prior_std = prior_params.get('std', 1)
        self.var_ddof = var_ddof  # Degrees of freedom for sample variance calculation

    def calc_posterior_params(self, data: np.ndarray, eps: float = 1e-5) -> tuple[float, float]:
        """ Calculate posterior parameters based on the data.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        `eps` : float, optional
            A small value to avoid division by zero, default is 1e-5.
        
        Returns
        -------
        tuple[float, float]
            A tuple containing the posterior mean and posterior standard deviation.
        """
        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=self.var_ddof)
        posterior_mean = (
            (
                (n/(sample_std**2 + eps))*sample_mean + (1/(self.prior_std**2 + eps))*self.prior_mean
            ) / ((n/(sample_std**2 + eps)) + (1/(self.prior_std**2 + eps)))
        )
        posterior_std = np.sqrt(1/((n/(sample_std**2 + eps)) + (1/(self.prior_std**2 + eps))))
        return posterior_mean, posterior_std

    def nll(self, data: np.ndarray) -> float:
        """ Compute the negative log-likelihood of the data given the distribution.
        """
        return -np.sum(self.log_likelihood(data))

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """ Compute the likelihood of the data given the distribution.
        """
        posterior_mean, posterior_std = self.calc_posterior_params(data)
        return (1 / (posterior_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - posterior_mean) / posterior_std)**2)

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """ Compute the log-likelihood of the data given the distribution.
        """
        posterior_mean, posterior_std = self.calc_posterior_params(data)
        return -0.5 * np.log(2 * np.pi) - np.log(posterior_std) - 0.5 * ((data - posterior_mean) / posterior_std)**2

    def sample_prior(self, size: int) -> np.ndarray:
        """ Sample from the distribution.
        """
        return np.random.normal(loc=self.prior_mean, scale=self.prior_std, size=size)
    
    def sample_posterior(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None, size: int = 1) -> np.ndarray:
        """ Sample from the distribution.
        """
        if data is None and params is None:
            raise ValueError("Either 'data' or 'params' must be provided to generate samples from the posterior distribution.")
        if params is not None:
            return self.sample_posterior_params(params, size=size)
        elif data is not None:
            return self.sample_posterior_data(data, size=size)
        else:  # This case should not happen due to the initial check but is required for type safety
            raise ValueError("Either 'data' or 'params' must be provided to generate samples from the posterior distribution.")

    def sample_posterior_params(self, params: dict[str, float], size: int = 1) -> np.ndarray:
        """ Sample from the posterior distribution using provided parameters.
        
        Args
        ----
        `params` : dict[str, float]
            Dictionary containing the posterior parameters 'mean' and 'std'.
        `size` : int
            Number of samples to generate.
        
        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution.
        """
        assert 'mean' in params and 'std' in params, "params must contain 'mean' and 'std' keys"
        assert params['std'] > 0, "Standard deviation must be positive"
        return np.random.normal(loc=params['mean'], scale=params['std'], size=size)
    
    def sample_posterior_data(self, data: np.ndarray, size: int = 1) -> np.ndarray:
        """ Sample from the posterior distribution using the data.
        
        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        `size` : int
            Number of samples to generate.
        
        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution based on the data.
        """
        posterior_mean, posterior_std = self.calc_posterior_params(data)
        return np.random.normal(loc=posterior_mean, scale=posterior_std, size=size)
    
    def get_posterior_params(self, data: np.ndarray) -> dict:
        """ Get the posterior parameters of the distribution.
        """
        posterior_mean, posterior_std = self.calc_posterior_params(data)
        return {'mean': posterior_mean, 'std': posterior_std}

    def __repr__(self):
        return f"Normal(prior_params={{'mean': {self.prior_mean}, 'std': {self.prior_std}}})"
    
    def __str__(self):
        return f"Normal(prior_params={{'mean': {self.prior_mean}, 'std': {self.prior_std}}})"
    
    def __eq__(self, other):
        if not isinstance(other, NormalNormal):
            return False
        return (self.prior_mean == other.prior_mean and
                self.prior_std == other.prior_std)
    

# If consider both mu and sigma as unknowns, can use this definition as priors on both
class NormGammaNormal(BDFDistribution):
    """ Normal-Gamma distribution class for Bayesian Distributional Forests.
    This class models a Normal distribution with a Gamma prior on the variance.
    """
    
    def __init__(self, prior_params: dict, params: tuple | None = None):
        """ Initialize the Normal-Gamma distribution with prior parameters.
        
        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'mean', 'std', 'alpha', and 'beta'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        super().__init__(prior_params, params)
        self.prior_mean = prior_params.get('mean', 0)  # alternatively mu
        self.prior_n = prior_params.get('n', 1)  # number of observations for prior mean
        self.prior_nu = prior_params.get('nu', 1)  # prior for gamma on sigma
        self.prior_phi = prior_params.get('phi', 1)  # prior for gamma on sigma
    
    def calc_posterior_params(self, data: np.ndarray) -> tuple[float, float]:
        """ Calculate posterior parameters based on the data.
        
        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        
        Returns
        -------
        tuple[float, float, float, float]
            A tuple containing the posterior mean, posterior n, posterior nu, and posterior phi.
        """
        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        posterior_mean = (self.prior_n * self.prior_mean + n * sample_mean) / (self.prior_n + n)
        posterior_std = (
            (1 / (self.prior_nu + n) * 
                (
                    (n-1) * sample_var + self.prior_nu * self.prior_phi + (n * self.prior_n)/(self.prior_n + n) * (sample_mean - self.prior_mean)**2
                )
             )
        )
        return posterior_mean, posterior_std
    