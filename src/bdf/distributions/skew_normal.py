import warnings

import numpy as np
from pydantic import Field
from scipy.stats import skewnorm

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# TODO: Implement inference_method (modular, map, laplace, then vi, mcmc later)
# Allows flexibility in how posterior is calculated in this non-conjugate case
# For now, only modular inference is implemented by default


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
            return self._sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self._sample_posterior_data(data, size=size, random_state=random_state)
        else:  # This case should not happen due to the initial check but is required for type safety
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def _sample_posterior_params(self, params: dict[str, float], *, size: int = 1, random_state: int) -> np.ndarray:
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

    def _sample_posterior_data(self, data: np.ndarray, *, size: int = 1, random_state: int) -> np.ndarray:
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

    def validate_targets(self, data: np.ndarray):
        """Validate data for normal distribution.
        Currently only checks for NaN, easily add on more.
        """
        assert not any(np.isnan(data)), "Inputs for normal distribution may not be NaN."
        std = np.std(data)
        assert (
            np.isfinite(std) and std is not None and std >= 0.0
        ), f"Standard deviation has to be finite, not None and >=0, got {std}"


class NormalMeanPseudoAlphaSkewNormalParams(BDFDistributionParams):
    """Pydantic model for Skew-Normal distribution parameters."""

    mu_mu: float = Field(default=0.0, description="Prior mean for mean mu of data")
    sigma_mu: float = Field(default=1.0, gt=0, description="Prior standard deviation for mu of data")
    alpha_zero: float = Field(default=0.0, alias="mean_alpha", description="Prior mean for alpha")
    m_alpha: float = Field(default=10.0, gt=0, alias="belief_alpha", description="Prior strength of belief for alpha")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data: dict) -> None:
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu_mu" not in data:
            missing_fields["mu_mu"] = self.__class__.model_fields["mu_mu"].default
        if "sigma_mu" not in data:
            missing_fields["sigma_mu"] = self.__class__.model_fields["sigma_mu"].default
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
        self.mu_mu = prior_params.mu_mu
        self.sigma_mu = prior_params.sigma_mu
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
        posterior_mean = (self.mu_mu / self.sigma_mu**2 + n * sample_mean / (sample_var + 1e-5)) / (  # type: ignore
            1 / self.sigma_mu**2 + n / (sample_var + 1e-5)  # type: ignore
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

    mu_mu: float = Field(default=0.0, description="Prior mean for mean of data")
    sigma_mu: float = Field(default=1.0, gt=0, description="Prior standard deviation for mean of data")
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
        if "mu_mu" not in data:
            missing_fields["mu_mu"] = self.__class__.model_fields["mu_mu"].default
        if "sigma_mu" not in data:
            missing_fields["sigma_mu"] = self.__class__.model_fields["sigma_mu"].default
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
        self.mu_mu = prior_params.mu_mu
        self.sigma_mu = prior_params.sigma_mu
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
        posterior_mean = (self.mu_mu / self.sigma_mu**2 + n * sample_mean / (sample_var + 1e-7)) / (  # type: ignore
            1 / self.sigma_mu**2 + n / (sample_var + 1e-7)  # type: ignore
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


# import warnings
# from typing import Literal

# import numpy as np
# from pydantic import Field, model_validator
# from scipy.stats import skewnorm
# from scipy.optimize import minimize
# from scipy.special import logsumexp

# from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
# from bdf.utils.constants import RANDOM_SEED


# # ============================================================================
# # PARAMS CLASSES
# # ============================================================================

# class SkewNormalPseudoAlphaParams(BDFDistributionParams):
#     """Parameters for Skew-Normal with pseudo-prior on alpha (skewness).

#     Supports three inference methods:
#     - 'modular': Independent moment-based updates (fastest, no PP)
#     - 'map': Joint MAP via optimization (fast, no PP)
#     - 'laplace': MAP + Hessian approximation (PP available, slower)
#     """

#     # Prior hyperparameters
#     mu_mu: float = Field(default=0.0, description="Prior mean for location μ")
#     sigma_mu: float = Field(default=1.0, gt=0, description="Prior std for location μ")
#     alpha_pseudo_mean: float = Field(default=0.0, description="Pseudo-prior mean for skewness α")
#     alpha_pseudo_strength: float = Field(default=10.0, gt=0, description="Pseudo-prior strength m for α")

#     # Inference method
#     inference_method: Literal["modular", "map", "laplace"] = Field(
#         default="modular",
#         description=(
#             "Inference method:\n"
#             "  - 'modular': Independent moment-based updates (fastest, no uncertainty)\n"
#             "  - 'map': Joint MAP via optimization (fast, no uncertainty)\n"
#             "  - 'laplace': MAP + Hessian (provides uncertainty for PP)\n"
#         )
#     )

#     # Optimization settings (for MAP/Laplace)
#     optim_method: Literal["L-BFGS-B", "Nelder-Mead", "Powell"] = Field(default="L-BFGS-B")
#     optim_maxiter: int = Field(default=100, gt=0, description="Max optimization iterations")
#     optim_tol: float = Field(default=1e-6, gt=0, description="Optimization tolerance")

#     # Laplace-specific settings
#     laplace_n_samples: int = Field(
#         default=1000, gt=0, description="Monte Carlo samples for Laplace PP (only for inference_method='laplace')"
#     )

#     @model_validator(mode="after")
#     def validate_inference_config(self):
#         """Auto-configure PP support based on inference method."""
#         if self.inference_method == "laplace":
#             # Laplace supports PP
#             if not self.use_posterior_predictive:
#                 warnings.warn(
#                     f"inference_method='laplace' supports posterior predictive but use_posterior_predictive=False. "
#                     f"Consider setting use_posterior_predictive=True for better uncertainty quantification.",
#                     UserWarning
#                 )
#         elif self.inference_method in ["modular", "map"]:
#             # These don't support PP
#             if self.use_posterior_predictive:
#                 warnings.warn(
#                     f"inference_method='{self.inference_method}' does not support posterior predictive; "
#                     f"forcing use_posterior_predictive=False.",
#                     UserWarning
#                 )
#                 self.use_posterior_predictive = False

#         return self


# class SkewNormalNormalGammaParams(BDFDistributionParams):
#     """Parameters for Skew-Normal with Normal prior on mean and Normal prior on skewness γ.

#     Only supports modular inference (moment-based).
#     """

#     mu_mu: float = Field(default=0.0, description="Prior mean for location")
#     sigma_mu: float = Field(default=1.0, gt=0, description="Prior std for location")
#     mu_gamma: float = Field(default=0.0, description="Prior mean for skewness γ")
#     sigma_gamma: float = Field(default=0.5, gt=0, description="Prior std for skewness γ")

#     # This distribution only supports modular inference
#     inference_method: Literal["modular"] = Field(default="modular", description="Only 'modular' supported")

#     @model_validator(mode="after")
#     def validate_no_pp(self):
#         """Modular inference doesn't support PP."""
#         if self.use_posterior_predictive:
#             warnings.warn(
#                 "SkewNormalNormalGamma only supports modular inference (no PP); "
#                 "forcing use_posterior_predictive=False.",
#                 UserWarning
#             )
#             self.use_posterior_predictive = False
#         return self


# # ============================================================================
# # DISTRIBUTION IMPLEMENTATIONS
# # ============================================================================

# class SkewNormalPseudoAlpha(BDFDistribution):
#     """Skew-Normal distribution with pseudo-prior on skewness parameter α.

#     Supports three inference methods via params.inference_method:
#     - 'modular': Independent moment-based updates (default, fastest)
#     - 'map': Joint MAP estimation via numerical optimization
#     - 'laplace': Laplace approximation (MAP + Hessian for uncertainty)

#     The pseudo-prior on α is: p(α) ∝ N(α | α₀, 1/m)^m
#     where m is the pseudo-prior strength.
#     """

#     params_cls = SkewNormalPseudoAlphaParams

#     # Capabilities (set dynamically based on inference_method)
#     _supports_nle = False  # Not conjugate
#     _supports_posterior_predictive = False  # Set in __init__ based on inference_method
#     _has_fast_loo_cv = False
#     _has_fast_kfold_cv = False

#     def __init__(self, params: SkewNormalPseudoAlphaParams):
#         # Set PP support based on inference method
#         self._supports_posterior_predictive = (params.inference_method == "laplace")
#         super().__init__(params)

#         # Store prior hyperparameters
#         self.mu_mu = params.mu_mu
#         self.sigma_mu = params.sigma_mu
#         self.alpha_pseudo_mean = params.alpha_pseudo_mean
#         self.alpha_pseudo_strength = params.alpha_pseudo_strength

#     # ========================================================================
#     # REQUIRED METHODS
#     # ========================================================================

#     def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
#         """Calculate posterior parameters using selected inference method."""
#         match self.params.inference_method:
#             case "modular":
#                 return self._calc_modular(data)
#             case "map":
#                 return self._calc_map(data)
#             case "laplace":
#                 return self._calc_laplace(data)
#             case _:
#                 raise ValueError(f"Unknown inference_method: {self.params.inference_method}")

#     def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
#         """Plug-in log-likelihood using point estimates (MAP or modular)."""
#         # Extract point estimates (works for both modular and MAP/Laplace)
#         alpha = params.get("alpha_map", params.get("alpha"))
#         xi = params.get("xi_map", params.get("xi"))
#         omega = params.get("omega_map", params.get("omega"))

#         if alpha is None or xi is None or omega is None:
#             raise ValueError("params must contain alpha/xi/omega or alpha_map/xi_map/omega_map")

#         return skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega)

#     def _num_parameters(self) -> int:
#         """Skew-normal has 3 parameters: α, ξ, ω."""
#         return 3

#     def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
#         """Sample from posterior (plug-in: just sample from fitted skew-normal)."""
#         alpha = params.get("alpha_map", params.get("alpha"))
#         xi = params.get("xi_map", params.get("xi"))
#         omega = params.get("omega_map", params.get("omega"))

#         if alpha is None or xi is None or omega is None:
#             raise ValueError("params must contain alpha/xi/omega or alpha_map/xi_map/omega_map")

#         return skewnorm.rvs(a=alpha, loc=xi, scale=omega, size=size, random_state=random_state)

#     def validate_targets(self, data: np.ndarray):
#         """Validate data for skew-normal (supports all real values)."""
#         if np.any(np.isnan(data)):
#             raise ValueError("Data contains NaN values")
#         if not np.all(np.isfinite(data)):
#             raise ValueError("Data contains infinite values")

#     def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
#         """Get posterior mean: E[X] = ξ + ω·δ·√(2/π) where δ = α/√(1+α²)."""
#         if params is None:
#             if data is None:
#                 raise ValueError("Provide either 'data' or 'params'")
#             params = self.calc_posterior_params(data)

#         alpha = params.get("alpha_map", params.get("alpha"))
#         xi = params.get("xi_map", params.get("xi"))
#         omega = params.get("omega_map", params.get("omega"))

#         delta = alpha / np.sqrt(1 + alpha**2)
#         return xi + omega * delta * np.sqrt(2 / np.pi)

#     def get_posterior_variance(
#         self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
#     ) -> float:
#         """Get posterior variance: Var[X] = ω²(1 - 2δ²/π) where δ = α/√(1+α²)."""
#         if params is None:
#             if data is None:
#                 raise ValueError("Provide either 'data' or 'params'")
#             params = self.calc_posterior_params(data)

#         alpha = params.get("alpha_map", params.get("alpha"))
#         omega = params.get("omega_map", params.get("omega"))

#         delta = alpha / np.sqrt(1 + alpha**2)
#         return omega**2 * (1 - 2 * delta**2 / np.pi)

#     # ========================================================================
#     # OPTIONAL METHODS
#     # ========================================================================

#     def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
#         """Posterior predictive for Laplace approximation (Monte Carlo over Normal(MAP, Σ))."""
#         if self.params.inference_method != "laplace":
#             raise NotImplementedError("PP only available for inference_method='laplace'")

#         if "cov_matrix" not in params:
#             raise ValueError("Laplace params must contain 'cov_matrix'")

#         # Extract MAP and covariance
#         theta_map = np.array([params["xi_map"], params["alpha_map"]])
#         cov_matrix = np.array(params["cov_matrix"])
#         omega_map = params["omega_map"]  # Treat ω as fixed

#         # Sample θ ~ N(MAP, Σ)
#         rng = np.random.default_rng(RANDOM_SEED)
#         theta_samples = rng.multivariate_normal(theta_map, cov_matrix, size=self.params.laplace_n_samples)

#         # Monte Carlo: average likelihood over samples
#         log_liks = np.empty((data.shape[0], self.params.laplace_n_samples))
#         for i, theta in enumerate(theta_samples):
#             xi_sample, alpha_sample = theta
#             log_liks[:, i] = skewnorm.logpdf(data, a=alpha_sample, loc=xi_sample, scale=omega_map)

#         # Log-mean-exp
#         return logsumexp(log_liks, axis=1) - np.log(self.params.laplace_n_samples)

#     def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
#         """Cannot sample from pseudo-prior (no closed-form prior distribution)."""
#         raise NotImplementedError(
#             "SkewNormalPseudoAlpha uses a pseudo-prior (no true prior distribution to sample from)."
#         )

#     # ========================================================================
#     # PRIVATE INFERENCE METHODS
#     # ========================================================================

#     def _calc_modular(self, data: np.ndarray) -> dict[str, float]:
#         """Modular Bayes: independent moment-based updates (original implementation).

#         1. Update ξ using Normal prior (assuming α = 0)
#         2. Update α using pseudo-prior strength
#         3. Update ω from sample moments
#         """
#         n = data.shape[0]
#         sample_mean = np.mean(data)
#         sample_var = np.var(data, ddof=1)
#         sample_skewness = np.clip(
#             np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-7)) ** 3),
#             -0.995, 0.995
#         )

#         # Step 1: Posterior mean for ξ (Normal prior, assuming symmetric distribution)
#         posterior_mean = (
#             (self.mu_mu / self.sigma_mu**2 + n * sample_mean / (sample_var + 1e-7)) /
#             (1 / self.sigma_mu**2 + n / (sample_var + 1e-7))
#         )

#         # Step 2: Convert sample skewness to α using moment matching
#         delta = np.sign(sample_skewness) * np.sqrt(
#             np.pi / 2 * np.abs(sample_skewness)**(2/3) /
#             (np.abs(sample_skewness)**(2/3) + ((4 - np.pi) / 2)**(2/3))
#         )
#         alpha_mle = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2))**(1/3)

#         # Apply pseudo-prior shrinkage
#         alpha = (
#             n / (n + self.alpha_pseudo_strength) * alpha_mle +
#             self.alpha_pseudo_strength / (n + self.alpha_pseudo_strength) * self.alpha_pseudo_mean
#         )

#         # Step 3: Adjust ω for skewness
#         omega = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)

#         # Handle numerical issues
#         if omega <= 0 or np.isnan(omega) or np.isinf(omega):
#             warnings.warn("Posterior omega invalid, using small positive value", UserWarning)
#             omega = 1e-5

#         # Step 4: Recover ξ from location parameter
#         xi = posterior_mean - omega * alpha / np.sqrt(1 + alpha**2) * np.sqrt(2 / np.pi)

#         return {
#             "alpha": float(alpha),
#             "xi": float(xi),
#             "omega": float(omega),
#             "inference_method": "modular",
#         }

#     def _calc_map(self, data: np.ndarray) -> dict[str, float]:
#         """Joint MAP estimation via numerical optimization.

#         Optimizes log p(θ | y) = log p(y | θ) + log p(θ) jointly over (ξ, α, ω).
#         """
#         n = data.shape[0]
#         sample_mean = np.mean(data)
#         sample_var = np.var(data, ddof=1)

#         # Initialize from moment matching (same as modular)
#         modular_params = self._calc_modular(data)
#         x0 = np.array([modular_params["xi"], modular_params["alpha"], np.log(modular_params["omega"])])

#         def neg_log_posterior(theta):
#             xi, alpha, log_omega = theta
#             omega = np.exp(log_omega)  # Constrain ω > 0

#             # Prior: p(ξ) = N(ξ | μ_μ, σ_μ²)
#             prior_xi = -0.5 * ((xi - self.mu_mu) / self.sigma_mu)**2

#             # Pseudo-prior: p(α) ∝ N(α | α₀, 1/m)^m
#             prior_alpha = -0.5 * self.alpha_pseudo_strength * ((alpha - self.alpha_pseudo_mean)**2)

#             # Likelihood: ∑ log p(yᵢ | ξ, α, ω)
#             try:
#                 log_lik = np.sum(skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega))
#                 if not np.isfinite(log_lik):
#                     return 1e10
#             except:
#                 return 1e10

#             return -(prior_xi + prior_alpha + log_lik)

#         # Optimize
#         result = minimize(
#             neg_log_posterior,
#             x0,
#             method=self.params.optim_method,
#             options={"maxiter": self.params.optim_maxiter, "ftol": self.params.optim_tol}
#         )

#         if not result.success:
#             warnings.warn(f"MAP optimization did not converge: {result.message}", UserWarning)

#         xi_map, alpha_map, log_omega_map = result.x
#         omega_map = np.exp(log_omega_map)

#         return {
#             "xi_map": float(xi_map),
#             "alpha_map": float(alpha_map),
#             "omega_map": float(omega_map),
#             "inference_method": "map",
#         }

#     def _calc_laplace(self, data: np.ndarray) -> dict[str, float]:
#         """Laplace approximation: MAP + Hessian.

#         1. Find MAP via optimization
#         2. Compute Hessian at MAP (numerical)
#         3. Approximate posterior as N(θ | MAP, Σ) where Σ = H⁻¹
#         """
#         # Step 1: Get MAP
#         map_params = self._calc_map(data)
#         theta_map = np.array([map_params["xi_map"], map_params["alpha_map"]])
#         omega_map = map_params["omega_map"]  # Treat ω as fixed for simplicity

#         # Step 2: Compute Hessian (numerical approximation)
#         def neg_log_posterior_theta(theta):
#             """Negative log posterior as function of (ξ, α) only, with ω fixed."""
#             xi, alpha = theta

#             # Priors
#             prior_xi = -0.5 * ((xi - self.mu_mu) / self.sigma_mu)**2
#             prior_alpha = -0.5 * self.alpha_pseudo_strength * ((alpha - self.alpha_pseudo_mean)**2)

#             # Likelihood
#             log_lik = np.sum(skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega_map))

#             return -(prior_xi + prior_alpha + log_lik)

#         # Numerical Hessian via finite differences
#         from scipy.optimize import approx_fprime
#         eps = np.sqrt(np.finfo(float).eps)

#         def grad_wrapper(theta):
#             return approx_fprime(theta, neg_log_posterior_theta, eps)

#         hessian = np.zeros((2, 2))
#         for i in range(2):
#             theta_plus = theta_map.copy()
#             theta_plus[i] += eps
#             theta_minus = theta_map.copy()
#             theta_minus[i] -= eps
#             hessian[:, i] = (grad_wrapper(theta_plus) - grad_wrapper(theta_minus)) / (2 * eps)

#         # Symmetrize
#         hessian = 0.5 * (hessian + hessian.T)

#         # Covariance = Hessian⁻¹
#         try:
#             cov_matrix = np.linalg.inv(hessian)
#             # Ensure positive definite
#             if not np.all(np.linalg.eigvals(cov_matrix) > 0):
#                 warnings.warn("Hessian not positive definite, using diagonal approximation", UserWarning)
#                 cov_matrix = np.diag(np.abs(np.diag(cov_matrix)))
#         except np.linalg.LinAlgError:
#             warnings.warn("Hessian singular, using diagonal approximation", UserWarning)
#             cov_matrix = np.eye(2) * 0.01  # Fallback

#         return {
#             "xi_map": float(theta_map[0]),
#             "alpha_map": float(theta_map[1]),
#             "omega_map": float(omega_map),
#             "cov_matrix": cov_matrix.tolist(),  # For serialization
#             "inference_method": "laplace",
#         }


# class SkewNormalNormalGamma(BDFDistribution):
#     """Skew-Normal with Normal prior on mean and Normal prior on skewness γ.

#     Uses modular inference only (moment-based updates).
#     Similar to SkewNormalPseudoAlpha but uses different skewness parameterization.
#     """

#     params_cls = SkewNormalNormalGammaParams

#     _supports_nle = False
#     _supports_posterior_predictive = False
#     _has_fast_loo_cv = False
#     _has_fast_kfold_cv = False

#     def __init__(self, params: SkewNormalNormalGammaParams):
#         super().__init__(params)
#         self.mu_mu = params.mu_mu
#         self.sigma_mu = params.sigma_mu
#         self.mu_gamma = params.mu_gamma
#         self.sigma_gamma = params.sigma_gamma

#     # ========================================================================
#     # REQUIRED METHODS
#     # ========================================================================

#     def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
#         """Calculate posterior using moment-based modular updates."""
#         n = data.shape[0]
#         sample_mean = np.mean(data)
#         sample_var = np.var(data, ddof=1)
#         sample_skewness = np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-7))**3)

#         # Variance of skewness estimator
#         var_skewness = (6 * n * (n - 1)) / ((n - 2) * (n + 1) * (n + 3))

#         # Posterior skewness γ (Normal prior)
#         posterior_skewness = (
#             (self.mu_gamma / self.sigma_gamma**2 + n * sample_skewness / (var_skewness + 1e-7)) /
#             (1 / self.sigma_gamma**2 + n / (var_skewness + 1e-7))
#         )
#         sample_skewness = np.clip(posterior_skewness, -0.995, 0.995)

#         # Posterior mean for ξ
#         posterior_mean = (
#             (self.mu_mu / self.sigma_mu**2 + n * sample_mean / (sample_var + 1e-7)) /
#             (1 / self.sigma_mu**2 + n / (sample_var + 1e-7))
#         )

#         # Convert to α parameterization
#         delta = np.sign(sample_skewness) * np.sqrt(
#             np.pi / 2 * np.abs(sample_skewness)**(2/3) /
#             (np.abs(sample_skewness)**(2/3) + ((4 - np.pi) / 2)**(2/3))
#         )
#         alpha = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2))**(1/3)
#         omega = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)

#         # Handle numerical issues
#         if np.isnan(omega) or np.isinf(omega):
#             warnings.warn("Posterior omega invalid, using large value", UserWarning)
#             omega = 1e7
#         elif omega <= 0:
#             warnings.warn("Posterior omega non-positive, using small value", UserWarning)
#             omega = 1e-7

#         xi = posterior_mean - omega * alpha / np.sqrt(1 + alpha**2) * np.sqrt(2 / np.pi)

#         return {
#             "alpha": float(alpha),
#             "xi": float(xi),
#             "omega": float(omega),
#         }

#     def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
#         """Plug-in log-likelihood."""
#         return skewnorm.logpdf(data, a=params["alpha"], loc=params["xi"], scale=params["omega"])

#     def _num_parameters(self) -> int:
#         return 3

#     def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
#         """Sample from fitted skew-normal."""
#         return skewnorm.rvs(
#             a=params["alpha"], loc=params["xi"], scale=params["omega"],
#             size=size, random_state=random_state
#         )

#     def validate_targets(self, data: np.ndarray):
#         """Validate data."""
#         if np.any(np.isnan(data)):
#             raise ValueError("Data contains NaN values")
#         if not np.all(np.isfinite(data)):
#             raise ValueError("Data contains infinite values")

#     def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
#         """Get posterior mean."""
#         if params is None:
#             if data is None:
#                 raise ValueError("Provide either 'data' or 'params'")
#             params = self.calc_posterior_params(data)

#         alpha, xi, omega = params["alpha"], params["xi"], params["omega"]
#         delta = alpha / np.sqrt(1 + alpha**2)
#         return xi + omega * delta * np.sqrt(2 / np.pi)

#     def get_posterior_variance(
#         self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
#     ) -> float:
#         """Get posterior variance."""
#         if params is None:
#             if data is None:
#                 raise ValueError("Provide either 'data' or 'params'")
#             params = self.calc_posterior_params(data)

#         alpha, omega = params["alpha"], params["omega"]
#         delta = alpha / np.sqrt(1 + alpha**2)
#         return omega**2 * (1 - 2 * delta**2 / np.pi)

#     def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
#         """Cannot sample from prior (parameterization issue)."""
#         raise NotImplementedError(
#             "SkewNormalNormalGamma prior parameterization does not support sampling."
#         )
