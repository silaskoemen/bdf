import numpy as np
from scipy.stats import skewnorm

from bdf.distributions.bdf_distribution import BDFDistribution
from bdf.utils.constants import RANDOM_SEED


class NormalNormalSkewNormal(BDFDistribution):
    """ Skew-Normal distribution with Normal prior on the mean `xi` and Normal prior on
    the shape `alpha`, treating omega as fixed and estimating posterior `xi` as if
    it were a Normal distribution. Posterior `alpha` is approximated using a second
    order Taylor expansion of the posterior mode of the skew-normal distribution, using 
    Score and Fisher information.
    """
    def __init__(self, prior_params: dict, params: tuple | None = None):
        """ Initialize the Skew-Normal distribution with prior parameters.
        
        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include mean and std. for `xi` and `alpha`.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        super().__init__(prior_params, params)
        self.prior_mean_xi = prior_params.get('mean_xi', prior_params.get('mu_xi', 0.0))
        self.prior_std_xi = prior_params.get('std_xi', prior_params.get('sigma_xi', 0.0))
        self.prior_mean_alpha = prior_params.get('mean_alpha', prior_params.get('mu_alpha', 0.0))
        self.prior_std_alpha = prior_params.get('std_alpha', prior_params.get('sigma_alpha', 0.0))
        if self.prior_std_xi <= 0 or self.prior_std_alpha <= 0:
            raise ValueError("Prior parameters 'std_xi' and 'std_alpha' must be positive.")
    
    def calc_posterior_params(self, data: np.ndarray) -> tuple[float, float, float]:
        """ Calculate posterior parameters based on the data.
        
        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        
        Returns
        -------
        tuple[float, float, float]
            A tuple containing the posterior location `xi`, posterior std `alpha`, and fixed `omega`.
        """
        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        
        # Posterior mean for xi (Normal prior)
        posterior_mean_xi = (
            (self.prior_mean_xi / self.prior_std_xi**2 + n * sample_mean / sample_var) /
            (1 / self.prior_std_xi**2 + n / sample_var)
        )

        z = (data - sample_mean) / np.sqrt(sample_var)

        if self.prior_mean_alpha == 0:
            phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

            # Numerically stable approximation of Phi(z)
            from scipy.stats import norm
            Phi = norm.cdf(z)

            # Avoid division by zero or very small values
            Phi_safe = np.clip(Phi, 1e-10, 1 - 1e-10)

            # Score and Fisher Information
            score = np.sum((z * phi) / Phi_safe)
            Fisher_info = np.sum((z**2 * phi**2) / (Phi_safe**2))

            # Posterior for alpha
            posterior_mean_alpha = score / (Fisher_info + 1 / self.prior_std_alpha**2)
        else:
            raise NotImplementedError(
                "Posterior mean for alpha is not implemented for non-zero prior mean_alpha."
            )

        return posterior_mean_alpha, posterior_mean_xi, np.sqrt(sample_var)
    
    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """ Compute the log-likelihood of the data given the distribution.
        
        Args
        ----
        `data` : np.ndarray
            The data to compute the log-likelihood for.
        
        Returns
        -------
        np.ndarray
            A numpy array containing the log-likelihood values for each data point.
        """
        posterio_alpha, posterior_xi, omega = self.calc_posterior_params(data)
        return skewnorm.logpdf(data, a=posterio_alpha, loc=posterior_xi, scale=omega)
    
    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """ Compute the likelihood of the data given the distribution.
        
        Args
        ----
        `data` : np.ndarray
            The data to compute the likelihood for.
        
        Returns
        -------
        np.ndarray
            A numpy array containing the likelihood values for each data point.
        """
        posterior_alpha, posterior_xi, omega = self.calc_posterior_params(data)
        return skewnorm.pdf(data, a=posterior_alpha, loc=posterior_xi, scale=omega)
    
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
    
    def sample_prior(self, size: int) -> np.ndarray:
        """ Sample from the prior distribution.
        
        Args
        ----
        `size` : int
            The number of samples to draw from the prior distribution.
        
        Returns
        -------
        np.ndarray
            A numpy array containing samples drawn from the prior distribution.
        """
        return skewnorm.rvs(a=self.prior_mean_alpha, loc=self.prior_mean_xi, scale=self.prior_std_xi, size=size, random_state=RANDOM_SEED)  # type: ignore
    
    def sample_posterior(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None, size: int = 1) -> np.ndarray:
        """ Sample from the posterior distribution.
        
        Args
        ----
        `data` : np.ndarray | None, optional
            The data to use for sampling, if available.
        `params` : dict[str, float] | None, optional
            Additional parameters for sampling, if available.
        `size` : int, optional
            The number of samples to draw from the posterior distribution, default is 1.
        
        Returns
        -------
        np.ndarray
            A numpy array containing samples drawn from the posterior distribution.
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
            The posterior parameters to sample from, must include 'alpha', 'xi', and 'omega'.
        `size` : int, optional
            The number of samples to generate, default is 1.
        
        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution.
        """
        alpha, xi, omega = params.get('alpha', params.get('posterior_alpha')), params.get('xi', params.get('posterior_xi')), params.get('omega', params.get('posterior_omega'))
        if alpha is None or xi is None or omega is None:
            raise ValueError("params must contain 'alpha', 'xi', and 'omega' keys")
        assert omega > 0, "Omega parameter must be positive"
        return skewnorm.rvs(alpha, loc=xi, scale=omega, size=size, random_state=RANDOM_SEED)  # type: ignore
    
    def sample_posterior_data(self, data: np.ndarray, size: int = 1) -> np.ndarray:
        """ Sample from the posterior distribution using the data.
        
        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        `size` : int, optional
            The number of samples to generate, default is 1.
        
        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution based on the data.
        """
        posterior_alpha, posterior_xi, omega = self.calc_posterior_params(data)
        return skewnorm.rvs(a=posterior_alpha, loc=posterior_xi, scale=omega, size=size, random_state=RANDOM_SEED)  # type: ignore
    
    def get_posterior_params(self, data: np.ndarray) -> dict:
        """ Get the posterior parameters of the distribution.
        
        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        
        Returns
        -------
        dict
            A dictionary containing the posterior parameters 'alpha', 'xi', and 'omega'.
        """
        posterior_alpha, posterior_xi, omega = self.calc_posterior_params(data)
        return {'alpha': posterior_alpha, 'xi': posterior_xi, 'omega': omega}
