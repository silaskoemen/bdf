import warnings

import numpy as np
from pydantic import Field
from scipy.stats import skewnorm

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED


class SkewNormalBase(BDFDistribution):
    """Base class for all Skew Normal distribution implementations.

    All implementations use parameters alpha, xi and omega, so sampling,
    (log)likelihoods and return functions are all identical.
    """

    # This is just a placeholder - child classes will have their own init
    def __init__(self, prior_params, params=None):
        super().__init__(prior_params, params)

    # Child classes MUST implement this method
    def calc_posterior_params(self, data, return_dict=False):
        raise NotImplementedError("Subclasses must implement calc_posterior_params")

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
        posterior_alpha, posterior_xi, posterior_omega = self.calc_posterior_params(data, return_dict=False)
        return skewnorm.logpdf(data, a=posterior_alpha, loc=posterior_xi, scale=posterior_omega)

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
        posterior_alpha, posterior_xi, posterior_omega = self.calc_posterior_params(data, return_dict=False)
        return skewnorm.pdf(data, a=posterior_alpha, loc=posterior_xi, scale=posterior_omega)

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

    def sample_prior(self, size: int, random_state: int) -> np.ndarray:
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
        random_state: int = RANDOM_SEED,
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
            return self.sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self.sample_posterior_data(data, size=size, random_state=random_state)
        else:  # This case should not happen due to the initial check but is required for type safety
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def sample_posterior_params(self, params: dict[str, float], *, size: int = 1, random_state: int) -> np.ndarray:
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
            params.get("posterior_alpha"),
            params.get("posterior_xi"),
            params.get("posterior_omega"),
        )
        if alpha is None or xi is None or omega is None:
            raise ValueError("params must contain 'posterior_alpha', 'posterior_xi', and 'posterior_omega' keys")
        assert omega > 0, "Omega parameter must be positive"
        return skewnorm.rvs(alpha, loc=xi, scale=omega, size=size, random_state=random_state)  # type: ignore

    def sample_posterior_data(self, data: np.ndarray, *, size: int = 1, random_state: int) -> np.ndarray:
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
        posterior_alpha, posterior_xi, posterior_omega = self.calc_posterior_params(data, return_dict=False)
        return skewnorm.rvs(  # type: ignore
            a=posterior_alpha, loc=posterior_xi, scale=posterior_omega, size=size, random_state=random_state
        )

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior mean of the distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to calculate the posterior mean from, if available.
        `params` : dict[str, float] | None, optional
            The posterior parameters to use for calculating the mean, if available.

        Returns
        -------
        float
            The posterior mean of the distribution.
        """
        if params is not None:
            return params.get("posterior_xi") + params.get("posterior_omega") * params.get("posterior_alpha") / np.sqrt(1 + params.get("posterior_alpha") ** 2) * np.sqrt(2 / np.pi)  # type: ignore
        elif data is not None:
            posterior_alpha, posterior_xi, posterior_omega = self.calc_posterior_params(data, return_dict=False)
            return posterior_xi + posterior_omega * posterior_alpha / np.sqrt(1 + posterior_alpha**2) * np.sqrt(2 / np.pi)  # type: ignore
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior mean.")

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get the posterior variance of the distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to calculate the posterior variance from, if available.
        `params` : dict[str, float] | None, optional
            The posterior parameters to use for calculating the variance, if available.

        Returns
        -------
        float
            The posterior variance of the distribution.
        """
        if params is not None:
            alpha, omega = (
                params.get("posterior_alpha"),
                params.get("posterior_omega"),
            )
            if alpha is None or omega is None:
                raise ValueError("params must contain 'posterior_alpha' and 'posterior_omega' keys")
            delta = alpha / np.sqrt(1 + alpha**2)
            return omega**2 * (1 - 2 * delta**2 / np.pi)
        elif data is not None:
            alpha, _, omega = self.calc_posterior_params(data, return_dict=False)
            if alpha is None or omega is None:
                raise ValueError("Posterior parameters could not be calculated from the data.")
            delta = alpha / np.sqrt(1 + alpha**2)  # type: ignore
            return omega**2 * (1 - 2 * delta**2 / np.pi)  # type: ignore
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior variance.")

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
        return self.calc_posterior_params(data, return_dict=True)  # type: ignore


class NormalMeanPseudoAlphaSkewNormalParams(BDFDistributionParams):
    """Pydantic model for Skew-Normal distribution parameters."""

    mu_zero: float = Field(default=0.0, description="Prior mean for mean mu of data")
    sigma_zero: float = Field(default=1.0, gt=0, description="Prior standard deviation for mu of data")
    alpha_zero: float = Field(default=0.0, alias="mean_alpha", description="Prior mean for alpha")
    m_alpha: float = Field(default=10.0, gt=0, alias="belief_alpha", description="Prior strength of belief for alpha")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data: dict) -> None:
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu_zero" not in data:
            missing_fields["mu_zero"] = self.__class__.model_fields["mu_zero"].default
        if "sigma_zero" not in data:
            missing_fields["sigma_zero"] = self.__class__.model_fields["sigma_zero"].default
        if "alpha_zero" not in data and "mu_alpha" not in data:
            missing_fields["alpha_zero"] = self.__class__.model_fields["alpha_zero"].default
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


class NormalMeanPseudoAlphaSkewNormal(SkewNormalBase):
    """Skew-Normal distribution with Normal prior on the mean `xi` and Normal prior on
    the shape `alpha`, treating omega as fixed and estimating posterior `xi` as if
    it were a Normal distribution. Posterior `alpha` is approximated using a second
    order Taylor expansion of the posterior mode of the skew-normal distribution, using
    Score and Fisher information.
    """

    def __init__(
        self, prior_params: dict[str, float] | NormalMeanPseudoAlphaSkewNormalParams, params: tuple | None = None
    ):
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
            prior_params = NormalMeanPseudoAlphaSkewNormalParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalMeanPseudoAlphaSkewNormalParams
        ), "prior_params must be an instance of NormalMeanPseudoAlphaSkewNormalParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mu_zero = prior_params.mu_zero
        self.sigma_zero = prior_params.sigma_zero
        self.alpha_zero = prior_params.alpha_zero
        self.m_alpha = prior_params.m_alpha

    def calc_posterior_params(
        self, data: np.ndarray, return_dict: bool = False
    ) -> dict[str, float] | tuple[float, float, float]:
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
        sample_skewness = np.clip(np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-5)) ** 3), -0.99, 0.99)

        # Posterior mean for xi (Normal prior)
        posterior_mean = (self.mu_zero / self.sigma_zero**2 + n * sample_mean / (sample_var + 1e-5)) / (  # type: ignore
            1 / self.sigma_zero**2 + n / (sample_var + 1e-5)  # type: ignore
        )

        delta = np.sign(sample_skewness) * np.sqrt(
            np.pi
            / 2
            * np.abs(sample_skewness) ** (2 / 3)
            / (np.abs(sample_skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        alpha = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 3)
        posterior_alpha = (
            n / (n + self.m_alpha) * alpha + self.m_alpha / (n + self.m_alpha) * self.alpha_zero
        )  # type: ignore
        posterior_omega = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)
        if posterior_omega <= 0 or np.isnan(posterior_omega) or np.isinf(posterior_omega):
            warnings.warn(
                "Posterior omega is non-positive or invalid, setting to a small positive value.",
                UserWarning,
            )
            posterior_omega = 1e-5  # Set a small positive value to avoid issues in sampling

        posterior_xi = posterior_mean - posterior_omega * posterior_alpha / np.sqrt(1 + posterior_alpha**2) * np.sqrt(
            2 / np.pi
        )
        if return_dict:
            return {
                "posterior_alpha": posterior_alpha,
                "posterior_xi": posterior_xi,
                "posterior_omega": posterior_omega,
            }
        else:
            return posterior_alpha, posterior_xi, posterior_omega


class NormalMeanNormalGammaSkewNormalParams(BDFDistributionParams):
    """Pydantic model for Skew-Normal distribution with Normal prior on the mean and
    Normal prior on the skewness gamma"""

    mu_zero: float = Field(default=0.0, description="Prior mean for mean of data")
    sigma_zero: float = Field(default=1.0, gt=0, description="Prior standard deviation for mean of data")
    mu_gamma: float = Field(default=0.0, alias="mean_gamma", description="Prior mean of skewness parameter gamma")
    sigma_gamma: float = Field(
        default=0.5, gt=0, alias="std_gamma", description="Prior std fof skewness parameter gamma"
    )

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data: dict) -> None:
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu_zero" not in data:
            missing_fields["mu_zero"] = self.__class__.model_fields["mu_zero"].default
        if "sigma_zero" not in data:
            missing_fields["sigma_zero"] = self.__class__.model_fields["sigma_zero"].default
        if "mu_gamma" not in data and "mean_gamma" not in data:
            missing_fields["mu_gamma"] = self.__class__.model_fields["mu_gamma"].default
        if "sigma_gamma" not in data and "std_gamma" not in data:
            missing_fields["sigma_gamma"] = self.__class__.model_fields["sigma_gamma"].default

        # Initialize the model
        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class NormalMeanNormalGammaSkewNormal(SkewNormalBase):
    """Skew-Normal data distribution with Normal prior on the mean (under assumption of no skewness)
    and Normal prior on the skewness gamma
    """

    def __init__(
        self, prior_params: dict[str, float] | NormalMeanNormalGammaSkewNormalParams, params: tuple | None = None
    ):
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
            prior_params = NormalMeanNormalGammaSkewNormalParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalMeanNormalGammaSkewNormalParams
        ), "prior_params must be an instance of NormalEBSkewNormalParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mu_zero = prior_params.mu_zero
        self.sigma_zero = prior_params.sigma_zero
        self.mu_gamma = prior_params.mu_gamma
        self.sigma_gamma = prior_params.sigma_gamma

    def calc_posterior_params(
        self, data: np.ndarray, return_dict: bool = False
    ) -> dict[str, float] | tuple[float, float, float]:
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
        sample_skewness = np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-7)) ** 3)
        var_skewness = (6 * n * (n - 1)) / ((n - 2) * (n + 1) * (n + 3))  # Variance of skewness estimator
        posterior_skewness = (self.mu_gamma / self.sigma_gamma**2 + n * sample_skewness / (var_skewness + 1e-7)) / (
            1 / self.sigma_gamma**2 + n / (var_skewness + 1e-7)
        )  # type: ignore
        sample_skewness = np.clip(posterior_skewness, -0.995, 0.995)

        # Posterior mean for xi (Normal prior)
        posterior_mean = (self.mu_zero / self.sigma_zero**2 + n * sample_mean / (sample_var + 1e-7)) / (  # type: ignore
            1 / self.sigma_zero**2 + n / (sample_var + 1e-7)  # type: ignore
        )

        delta = np.sign(sample_skewness) * np.sqrt(
            np.pi
            / 2
            * np.abs(sample_skewness) ** (2 / 3)
            / (np.abs(sample_skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        posterior_alpha = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 3)
        posterior_omega = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)
        if np.isnan(posterior_omega) or np.isinf(posterior_omega):
            warnings.warn(
                "Posterior omega invalid, setting to large value to discourage this split",
                UserWarning,
            )
            posterior_omega = 1e7  # Set a small positive value to avoid issues in sampling
        elif posterior_omega <= 0:
            warnings.warn(
                "Posterior omega is non-positive, setting to small positive value.",
                UserWarning,
            )
            posterior_omega = 1e-7

        posterior_xi = posterior_mean - posterior_omega * posterior_alpha / np.sqrt(1 + posterior_alpha**2) * np.sqrt(
            2 / np.pi
        )
        if return_dict:
            return {
                "posterior_alpha": posterior_alpha,
                "posterior_xi": posterior_xi,
                "posterior_omega": posterior_omega,
            }
        else:
            return posterior_alpha, posterior_xi, posterior_omega


class NormalXiNormalAlphaSkewNormalMAPParams(BDFDistributionParams):
    """Pydantic model for Skew-Normal distribution with Normal prior on the mean and
    Normal prior on the skewness alpha, using MAP estimation.
    """

    mu_alpha: float = Field(default=0.0, alias="mean_alpha", description="Prior mean for skewness alpha")
    sigma_alpha: float = Field(
        default=5.0, alias="std_alpha", gt=0, description="Prior standard deviation for skewness alpha"
    )
    mu_xi: float = Field(default=0.0, alias="mean_alpha", description="Prior mean of location parameter xi")
    sigma_xi: float = Field(
        default=0.1, gt=0, alias="std_alpha", description="Prior standard deviation of location parameter xi"
    )

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data: dict) -> None:
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu_alpha" not in data and "mean_alpha" not in data:
            missing_fields["mu_alpha"] = self.__class__.model_fields["mu_alpha"].default
        if "sigma_alpha" not in data and "std_alpha" not in data:
            missing_fields["sigma_alpha"] = self.__class__.model_fields["sigma_alpha"].default
        if "mu_xi" not in data and "mean_xi" not in data:
            missing_fields["mu_xi"] = self.__class__.model_fields["mu_xi"].default
        if "sigma_xi" not in data and "std_xi" not in data:
            missing_fields["sigma_xi"] = self.__class__.model_fields["sigma_xi"].default

        # Initialize the model
        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class NormalXiNormalAlphaSkewNormalMAP(SkewNormalBase):
    """Skew-Normal distribution with Normal prior on the mean `xi` and Normal prior on
    the shape `alpha`, using MAP estimation.
    """

    def __init__(
        self, prior_params: dict[str, float] | NormalXiNormalAlphaSkewNormalMAPParams, params: tuple | None = None
    ):
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
            prior_params = NormalXiNormalAlphaSkewNormalMAPParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalXiNormalAlphaSkewNormalMAPParams
        ), "prior_params must be an instance of NormalXiNormalAlphaSkewNormalMAPParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mu_alpha = prior_params.mu_alpha
        self.sigma_alpha = prior_params.sigma_alpha
        self.mu_xi = prior_params.mu_xi
        self.sigma_xi = prior_params.sigma_xi

    def calc_posterior_params(
        self, data: np.ndarray, return_dict: bool = False
    ) -> dict[str, float] | tuple[float, float, float]:
        """Calculate posterior parameters using Laplace approximation."""
        from scipy.optimize import minimize
        from scipy.stats import skewnorm

        data.shape[0]
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # 1. Get initial estimates from moment matching (you already have this code)
        sample_skewness = np.clip(np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-7)) ** 3), -0.995, 0.995)
        delta = np.sign(sample_skewness) * np.sqrt(
            np.pi
            / 2
            * np.abs(sample_skewness) ** (2 / 3)
            / (np.abs(sample_skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        alpha_init = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 3)
        omega_init = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)
        xi_init = sample_mean - omega_init * alpha_init / np.sqrt(1 + alpha_init**2) * np.sqrt(2 / np.pi)

        # 2. Define negative log posterior (combining likelihood and prior)
        def neg_log_posterior(params):
            xi, alpha = params
            # Skip omega as we'll treat it as fixed from moment estimate

            # Prior terms
            prior_xi = -0.5 * ((xi - self.mu_xi) / self.sigma_xi) ** 2
            prior_alpha = -0.5 * ((alpha - self.mu_alpha) / self.sigma_alpha) ** 2

            # Likelihood term - use skewnorm log PDF from scipy
            likelihood = np.sum(skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega_init))

            return -(prior_xi + prior_alpha + likelihood)

        # Starting point and bounds to avoid numerical issues
        x0 = np.array([xi_init, alpha_init])

        result = minimize(
            neg_log_posterior,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 3},  # Extremely limited iterations for speed
        )

        # 4. Get MAP estimates
        posterior_xi, posterior_alpha = result.x
        posterior_omega = omega_init

        # Return results
        if return_dict:
            return {
                "posterior_alpha": posterior_alpha,
                "posterior_xi": posterior_xi,
                "posterior_omega": posterior_omega,
            }
        else:
            return posterior_alpha, posterior_xi, posterior_omega
