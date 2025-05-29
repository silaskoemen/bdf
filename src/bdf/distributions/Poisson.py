import numpy as np

from bdf.distributions.bdf_distribution import BDFDistribution


class GammaPoisson(BDFDistribution):
    """Gamma-Poisson distribution class for Bayesian Distributional Forests.
    This class models a Poisson distribution with a Gamma prior on the rate parameter.
    """

    def __init__(self, prior_params: dict, params: tuple | None = None):
        """Initialize the Gamma-Poisson distribution with prior parameters.

        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'alpha' and 'beta'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        super().__init__(prior_params, params)
        self.prior_alpha = prior_params.get("alpha", 1.0)
        self.prior_beta = prior_params.get("beta", 1.0)
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Prior parameters 'alpha' and 'beta' must be positive.")
        self.params = (self.prior_alpha, self.prior_beta)

    def calc_posterior_params(self, data: np.ndarray) -> tuple[float, float]:
        """Calculate posterior parameters based on the data.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.

        Returns
        -------
        tuple[float, float]
            A tuple containing the posterior alpha and posterior beta.
        """
        n_events = np.sum(data)
        n_observations = data.shape[0]

        posterior_alpha = self.prior_alpha + n_events
        posterior_beta = self.prior_beta + n_observations

        return posterior_alpha, posterior_beta

    def nll(self, data: np.ndarray) -> float:
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

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the likelihood of the data given the distribution.

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

        # Resulting distribution is NegativeBinomial with params alpha and beta/(1+beta)
        return np.empty_like(data)
