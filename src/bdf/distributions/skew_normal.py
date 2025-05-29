import warnings

import numpy as np
from pydantic import Field
from scipy.stats import skewnorm

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED


class NormalEBSkewNormalParams(BDFDistributionParams):
    """Pydantic model for Skew-Normal distribution parameters."""

    mu: float = Field(default=0.0, alias="mean", description="Prior mean for mean mu of data")
    sigma: float = Field(default=1.0, gt=0, alias="std", description="Prior standard deviation for mu of data")
    mean_alpha: float = Field(default=0.0, alias="mu_alpha", description="Prior mean for alpha")
    m_alpha: float = Field(default=10.0, gt=0, alias="belief_alpha", description="Prior strength of belief for alpha")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data: dict) -> None:
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu" not in data and "mean" not in data:
            missing_fields["mu"] = self.__class__.model_fields["mu"].default
        if "sigma" not in data and "std" not in data:
            missing_fields["sigma"] = self.__class__.model_fields["sigma"].default
        if "mean_alpha" not in data and "mu_alpha" not in data:
            missing_fields["mean_alpha"] = self.__class__.model_fields["mean_alpha"].default
        if "m_alpha" not in data and "belief_alpha" not in data:
            missing_fields["m_alpha"] = self.__class__.model_fields["m_alpha"].default

        # Initialize the model
        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class NormalEBSkewNormal(BDFDistribution):
    """Skew-Normal distribution with Normal prior on the mean `xi` and Normal prior on
    the shape `alpha`, treating omega as fixed and estimating posterior `xi` as if
    it were a Normal distribution. Posterior `alpha` is approximated using a second
    order Taylor expansion of the posterior mode of the skew-normal distribution, using
    Score and Fisher information.
    """

    def __init__(self, prior_params: dict[str, float] | NormalEBSkewNormalParams, params: tuple | None = None):
        """Initialize the Skew-Normal distribution with prior parameters.

        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include mean and std. for `xi` and `alpha`.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        # Input has already been validated in the DistributionManager with Pydantic BaseModel below
        if isinstance(prior_params, dict):
            prior_params = NormalEBSkewNormalParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalEBSkewNormalParams
        ), "prior_params must be an instance of NormalEBSkewNormalParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.prior_mu = prior_params.mu
        self.prior_sigma = prior_params.sigma
        self.prior_mean_alpha = prior_params.mean_alpha
        self.prior_m_alpha = prior_params.m_alpha

    def calc_posterior_params(self, data: np.ndarray) -> tuple[float, float, float]:
        """Calculate posterior parameters based on the data.

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
        sample_skewness = np.clip(np.mean(((data - sample_mean) / np.sqrt(sample_var)) ** 3), 0.995, 0.995)

        # Posterior mean for xi (Normal prior)
        posterior_mean = (self.prior_mu / self.prior_sigma**2 + n * sample_mean / sample_var) / (  # type: ignore
            1 / self.prior_sigma**2 + n / sample_var  # type: ignore
        )

        delta = np.sign(sample_skewness) * np.sqrt(
            np.pi
            / 2
            * np.abs(sample_skewness) ** (2 / 3)
            / (np.abs(sample_skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        alpha = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 2)
        posterior_alpha = (
            n / (n + self.prior_m_alpha) * alpha + self.prior_m_alpha / (n + self.prior_m_alpha) * self.prior_mean_alpha
        )  # type: ignore
        posterior_omega = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)

        posterior_xi = posterior_mean - posterior_omega * posterior_alpha / np.sqrt(1 + posterior_alpha**2) * np.sqrt(
            2 / np.pi
        )

        return posterior_alpha, posterior_xi, posterior_omega

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the log-likelihood for.

        Returns
        -------
        np.ndarray
            A numpy array containing the log-likelihood values for each data point.
        """
        posterior_alpha, posterior_xi, omega = self.calc_posterior_params(data)
        return skewnorm.logpdf(data, a=posterior_alpha, loc=posterior_xi, scale=omega)

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the likelihood of the data given the distribution.

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

    def sample_prior(self, size: int) -> np.ndarray:
        """Sample from the prior distribution.

        Args
        ----
        `size` : int
            The number of samples to draw from the prior distribution.

        Returns
        -------
        np.ndarray
            A numpy array containing samples drawn from the prior distribution.
        """
        raise NotImplementedError(
            "Due to parameterization of global mean and EB priors on alpha, "
            "no fixed prior is available to sample from. "
        )

    def sample_posterior(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
        size: int = 1,
    ) -> np.ndarray:
        """Sample from the posterior distribution.

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
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )
        if params is not None:
            return self.sample_posterior_params(params, size=size)
        elif data is not None:
            return self.sample_posterior_data(data, size=size)
        else:  # This case should not happen due to the initial check but is required for type safety
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def sample_posterior_params(self, params: dict[str, float], size: int = 1) -> np.ndarray:
        """Sample from the posterior distribution using provided parameters.

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
        alpha, xi, omega = (
            params.get("alpha", params.get("posterior_alpha")),
            params.get("xi", params.get("posterior_xi")),
            params.get("omega", params.get("posterior_omega")),
        )
        if alpha is None or xi is None or omega is None:
            raise ValueError("params must contain 'alpha', 'xi', and 'omega' keys")
        assert omega > 0, "Omega parameter must be positive"
        return skewnorm.rvs(alpha, loc=xi, scale=omega, size=size, random_state=RANDOM_SEED)  # type: ignore

    def sample_posterior_data(self, data: np.ndarray, size: int = 1) -> np.ndarray:
        """Sample from the posterior distribution using the data.

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
        return skewnorm.rvs(  # type: ignore
            a=posterior_alpha, loc=posterior_xi, scale=omega, size=size, random_state=RANDOM_SEED
        )

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters of the distribution.

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
        return {"alpha": posterior_alpha, "xi": posterior_xi, "omega": omega}
