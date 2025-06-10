""" Beta distribution of data, reparametrized with mean mu and shape nu

Related to original parameters via alpha = mu * nu, beta = (1 - mu) * nu.

nu in terms of observed quantities is mu * (1 - mu) / var - 1.
"""
import warnings

import numpy as np
from pydantic import Field
from scipy.stats import beta as beta_dist

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED


class NormalMuBetaParams(BDFDistributionParams):
    """
    Parameters for the Beta distribution reparametrized with mean and shape.
    """

    mu_zero: float = Field(
        default=0.5, gt=0.0, lt=1.0, alias="mean", description="Prior mean of the Beta distribution (0 < mu < 1)"
    )
    sigma_zero: float = Field(
        default=0.5, gt=0.0, alias="std", description="Prior standard deviation of mean mu of the beta distribution"
    )

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data: dict) -> None:
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu_zero" not in data and "mean" not in data:
            missing_fields["mu_zero"] = self.__class__.model_fields["mu_zero"].default
        if "sigma_zero" not in data and "std" not in data:
            missing_fields["sigma_zero"] = self.__class__.model_fields["sigma_zero"].default

        # Initialize the model
        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class NormalMuBeta(BDFDistribution):
    """
    Beta distribution reparametrized with mean mu and shape nu.
    Related to original parameters via alpha = mu * nu, beta = (1 - mu) * nu.
    nu in terms of observed quantities is mu * (1 - mu) / var - 1.
    """

    def __init__(self, prior_params: dict | NormalMuBetaParams, params: tuple | None = None, var_ddof: int = 1):
        """Initialize the Normal distribution with prior parameters.
        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'mean' and 'std'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        `var_ddof` : int, optional
            Degrees of freedom for variance calculation, default is 1 (sample standard deviation).
        """
        if isinstance(prior_params, dict):
            prior_params = NormalMuBetaParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalMuBetaParams
        ), "prior_params must be an instance of NormalNormalParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mu_zero = prior_params.mu_zero
        self.sigma_zero = prior_params.sigma_zero
        self.var_ddof = var_ddof  # Degrees of freedom for sample variance calculation

    def calc_posterior_params(
        self, data: np.ndarray, return_dict: bool = True, eps: float = 1e-5
    ) -> dict[str, float] | tuple[float, float]:
        """Calculate posterior parameters based on the data.

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
        mu = ((n / (sample_std**2 + eps)) * sample_mean + (1 / (self.sigma_zero**2 + eps)) * self.mu_zero) / (
            (n / (sample_std**2 + eps)) + (1 / (self.sigma_zero**2 + eps))
        )
        nu = mu * (1 - mu) / (sample_std**2 + eps) - 1
        if return_dict:
            return {"mu": mu, "nu": nu}
        else:
            return mu, nu

    def get_alpha_beta(self, posterior_params: dict[str, float] | tuple[float, float]) -> tuple[float, float]:
        """Get the alpha and beta parameters from the posterior parameters.

        Args
        ----
        `posterior_params` : dict or tuple
            Posterior parameters containing 'mu' and 'nu' or a tuple of (mu, nu).

        Returns
        -------
        tuple[float, float]
            A tuple containing the alpha and beta parameters.
        """
        if isinstance(posterior_params, dict):
            mu = posterior_params["mu"]
            nu = posterior_params["nu"]
        else:
            mu, nu = posterior_params

        alpha = mu * nu
        beta = (1 - mu) * nu
        return alpha, beta

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Calculate the likelihood of the data given the distribution parameters.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the likelihood for.

        Returns
        -------
        np.ndarray
            The likelihood of the data.
        """
        posterior_params = self.calc_posterior_params(data, return_dict=True)
        alpha, beta = self.get_alpha_beta(posterior_params)
        return beta_dist.pdf(data, a=alpha, b=beta)

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Calculate the log likelihood of the data given the distribution parameters.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the log likelihood for.

        Returns
        -------
        np.ndarray
            The log likelihood of the data.
        """
        posterior_params = self.calc_posterior_params(data, return_dict=True)
        alpha, beta = self.get_alpha_beta(posterior_params)
        return beta_dist.logpdf(data, a=alpha, b=beta)

    def nll(self, data: np.ndarray) -> float:
        """Calculate the negative log likelihood of the data given the distribution parameters.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the negative log likelihood for.

        Returns
        -------
        float
            The negative log likelihood of the data.
        """
        return -np.sum(self.log_likelihood(data))

    def sample_posterior(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
        size: int = 1,
        random_state: int = RANDOM_SEED,
    ) -> np.ndarray:
        """Sample from the distribution."""
        if data is None and params is None:
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )
        if params is not None:
            return self.sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self.sample_posterior_data(data, size=size, random_state=random_state)  # type: ignore
        else:  # This case should not happen due to the initial check but is required for type safety
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def sample_posterior_params(self, params: dict[str, float], *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution using provided parameters.

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
        assert "mu" in params and "nu" in params, "params must contain 'mu' and 'nu' keys"
        assert params["mu"] > 0 and params["nu"] > 0, "Both 'mu' and 'nu' must be greater than 0"
        alpha, beta = self.get_alpha_beta(params)
        # Sample from the beta distribution using the calculated alpha and beta
        return beta_dist.rvs(a=alpha, b=beta, size=size, random_state=random_state)  # type: ignore

    def sample_posterior_data(self, data: np.ndarray, *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution using the data.

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
        mu, nu = self.calc_posterior_params(data, return_dict=False)
        alpha, beta = self.get_alpha_beta((mu, nu))  # type: ignore
        return beta_dist.rvs(a=alpha, b=beta, size=size, random_state=random_state)  # type: ignore

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior mean of the distribution.

        Args
        ----
        `data` : np.ndarray, optional
            The data to calculate the posterior parameters from, default is None.
        `params` : dict[str, float], optional
            Dictionary containing the posterior parameters 'mean' and 'std', default is None.

        Returns
        -------
        float
            The posterior mean of the distribution.
        """
        if params is not None:
            return params["mu"]
        elif data is not None:
            mu, _ = self.calc_posterior_params(data, return_dict=False)
            return mu  # type: ignore
        else:
            raise ValueError("Either 'data' or 'params' must be provided to get the posterior mean.")

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get the posterior standard deviation of the distribution.

        Args
        ----
        `data` : np.ndarray, optional
            The data to calculate the posterior parameters from, default is None.
        `params` : dict[str, float], optional
            Dictionary containing the posterior parameters 'mean' and 'std', default is None.

        Returns
        -------
        float
            The posterior standard deviation of the distribution.
        """
        if params is not None:
            return params["mu"] / (params["nu"] ** 2) / (params["nu"] + 1)  # type: ignore
        elif data is not None:
            mu, nu = self.calc_posterior_params(data, return_dict=False)
            return mu / (nu**2) / (nu + 1)  # type: ignore
        else:
            raise ValueError("Either 'data' or 'params' must be provided to get the posterior standard deviation.")

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters of the distribution."""
        mu, nu = self.calc_posterior_params(data)
        return {"mu": mu, "nu": nu}
