import numpy as np
from bdf.distributions.BDFDistribution import BDFDistribution


class Normal(BDFDistribution):
    """ Normal distribution class for Bayesian Distributional Forests.
    """
    def __init__(self, prior_params: dict, params: tuple = None, var_ddof: int = 1):
        """ Initialize the Normal distribution with prior parameters.
        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'mean' and 'std'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        `var_ddof` : int, optional
            Degrees of freedom for sample variance calculation, default is 1 (sample standard deviation).
        """
        super().__init__(prior_params, params)
        self.prior_mean = prior_params.get('mean', 0)
        self.prior_std = prior_params.get('std', 1)
        self. var_ddof = var_ddof  # Degrees of freedom for sample variance calculation

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

    def likelihood(self, data: np.ndarray) -> float:
        """ Compute the likelihood of the data given the distribution.
        """
        posterior_mean, posterior_std = self.calc_posterior_params(data)
        return (1 / (posterior_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - posterior_mean) / posterior_std)**2)

    def log_likelihood(self, data: np.ndarray) -> float:
        """ Compute the log-likelihood of the data given the distribution.
        """
        posterior_mean, posterior_std = self.calc_posterior_params(data)
        return -0.5 * np.log(2 * np.pi) - np.log(posterior_std) - 0.5 * ((data - posterior_mean) / posterior_std)**2

    def sample_prior(self, size: int) -> np.ndarray:
        """ Sample from the distribution.
        """
        return np.random.normal(loc=self.prior_mean, scale=self.prior_std, size=size)
    
    def sample_posterior(self, data: np.ndarray, size: int) -> np.ndarray:
        """ Sample from the distribution.
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
        if not isinstance(other, Normal):
            return False
        return (self.prior_mean == other.prior_mean and
                self.prior_std == other.prior_std)