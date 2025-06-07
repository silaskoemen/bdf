import warnings

import numpy as np
from pydantic import Field
from scipy.stats import norm

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED


class NormalMuNormalParams(BDFDistributionParams):
    """Parameters for the Normal distribution in Bayesian Distributional Forests.

    Attributes
    ----------
    mean : float
        The prior mean of the Normal distribution.
    std : float
        The prior standard deviation of the Normal distribution.
    """

    mu_zero: float = Field(default=0.0, alias="mean", description="Prior mean of the Normal distribution")
    sigma_zero: float = Field(
        default=1.0, alias="std", gt=0, description="Prior standard deviation of the Normal distribution"
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


class NormalMuNormal(BDFDistribution):
    """Normal distribution class for Bayesian Distributional Forests."""

    def __init__(self, prior_params: dict | NormalMuNormalParams, params: tuple | None = None, var_ddof: int = 1):
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
            prior_params = NormalMuNormalParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalMuNormalParams
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
        posterior_mean = (
            (n / (sample_std**2 + eps)) * sample_mean + (1 / (self.sigma_zero**2 + eps)) * self.mu_zero
        ) / ((n / (sample_std**2 + eps)) + (1 / (self.sigma_zero**2 + eps)))
        posterior_std = np.sqrt(1 / ((n / (sample_std**2 + eps)) + (1 / (self.sigma_zero**2 + eps))))
        if return_dict:
            return {"posterior_mean": posterior_mean, "posterior_std": posterior_std}
        else:
            return posterior_mean, posterior_std

    def nll(self, data: np.ndarray) -> float:
        """Compute the negative log-likelihood of the data given the distribution."""
        return -np.sum(self.log_likelihood(data))

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the likelihood of the data given the distribution."""
        posterior_mean, posterior_std = self.calc_posterior_params(data, return_dict=False)
        return (1 / (posterior_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((data - posterior_mean) / posterior_std) ** 2  # type: ignore
        )

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution."""
        posterior_mean, posterior_std = self.calc_posterior_params(data, return_dict=False)
        return -0.5 * np.log(2 * np.pi) - np.log(posterior_std) - 0.5 * ((data - posterior_mean) / posterior_std) ** 2  # type: ignore

    def sample_prior(self, size: int) -> np.ndarray:
        """Sample from the distribution."""
        return np.random.normal(loc=self.mu_zero, scale=self.sigma_zero, size=size)

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
        assert "posterior_mean" in params and "posterior_std" in params, "params must contain 'mean' and 'std' keys"
        assert params["posterior_std"] > 0, "Standard deviation must be positive"
        return norm.rvs(loc=params["posterior_mean"], scale=params["posterior_std"], size=size, random_state=random_state)  # type: ignore

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
        posterior_mean, posterior_std = self.calc_posterior_params(data, return_dict=False)
        return norm.rvs(loc=posterior_mean, scale=posterior_std, size=size, random_state=random_state)  # type: ignore

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
            return params["posterior_mean"]
        elif data is not None:
            posterior_mean, _ = self.calc_posterior_params(data, return_dict=False)
            return posterior_mean  # type: ignore
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
            return params["posterior_std"] ** 2
        elif data is not None:
            _, posterior_std = self.calc_posterior_params(data, return_dict=False)
            return posterior_std**2  # type: ignore
        else:
            raise ValueError("Either 'data' or 'params' must be provided to get the posterior standard deviation.")

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters of the distribution."""
        posterior_mean, posterior_std = self.calc_posterior_params(data)
        return {"mean": posterior_mean, "std": posterior_std}

    def __repr__(self):
        return f"Normal(prior_params={{'mean': {self.mu_zero}, 'std': {self.sigma_zero}}})"

    def __str__(self):
        return f"Normal(prior_params={{'mean': {self.mu_zero}, 'std': {self.sigma_zero}}})"

    def __eq__(self, other):
        if not isinstance(other, NormalMuNormal):
            return False
        return self.mu_zero == other.mu_zero and self.sigma_zero == other.sigma_zero


class NormGammaNormalParams(BDFDistributionParams):
    """Parameters for the Normal-EBSkewNormal distribution in Bayesian Distributional Forests.

    Attributes
    ----------
    mean : float
        The prior mean of the Normal distribution.
    std : float
        The prior standard deviation of the Normal distribution.
    alpha : float
        The prior shape parameter for the skewness.
    beta : float
        The prior scale parameter for the skewness.
    """

    mean: float = Field(default=0.0, alias="mu", description="Prior mean of the Normal distribution")
    n: float = Field(
        default=1.0, alias="sigma", gt=0, description="Prior standard deviation of the Normal distribution"
    )
    nu: float = Field(default=0.0, alias="alpha", description="Prior shape parameter for skewness")
    phi: float = Field(default=1.0, alias="beta", gt=0, description="Prior scale parameter for skewness")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data: dict) -> None:
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu" not in data and "mean" not in data:
            missing_fields["mean"] = self.__class__.model_fields["mean"].default
        if "sigma" not in data and "std" not in data:
            missing_fields["std"] = self.__class__.model_fields["std"].default
        if "alpha" not in data:
            missing_fields["alpha"] = self.__class__.model_fields["alpha"].default
        if "beta" not in data:
            missing_fields["beta"] = self.__class__.model_fields["beta"].default

        # Initialize the model
        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


# If consider both mu and sigma as unknowns, can use this definition as priors on both
class NormGammaNormal(BDFDistribution):
    """Normal-Gamma distribution class for Bayesian Distributional Forests.
    This class models a Normal distribution with a Gamma prior on the variance.
    """

    def __init__(self, prior_params: dict | BDFDistributionParams, params: tuple | None = None):
        """Initialize the Normal-Gamma distribution with prior parameters.

        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'mean', 'std', 'alpha', and 'beta'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        if not isinstance(prior_params, NormGammaNormalParams):
            prior_params = NormGammaNormalParams.model_validate(prior_params)
        super().__init__(prior_params, params)
        self.mu_zero = prior_params.mean
        self.prior_n = prior_params.n
        self.prior_nu = prior_params.nu
        self.prior_phi = prior_params.phi

    def calc_posterior_params(self, data: np.ndarray) -> tuple[float, float]:
        """Calculate posterior parameters based on the data.

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
        posterior_mean = (self.prior_n * self.mu_zero + n * sample_mean) / (self.prior_n + n)
        posterior_std = (
            1
            / (self.prior_nu + n)
            * (
                (n - 1) * sample_var
                + self.prior_nu * self.prior_phi
                + (n * self.prior_n) / (self.prior_n + n) * (sample_mean - self.mu_zero) ** 2
            )
        )
        return posterior_mean, posterior_std


class InverseGammaNormal(BDFDistribution):
    """Inverse-Gamma distribution class for Bayesian Distributional Forests.
    This class models a Normal distribution with an Inverse-Gamma prior on the variance.
    """

    def __init__(self, prior_params: dict, params: tuple | None = None):
        """Initialize the Inverse-Gamma distribution with prior parameters.

        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'mean', 'std', 'alpha', and 'beta'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        pass
