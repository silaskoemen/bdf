"""Gamma distribution implementations for Bayesian Distributional Forest.

The Gamma distribution is parameterized by shape α and rate β.
Mean = α/β, Variance = α/β²

Implemented versions:
- GammaPseudoMean: Pseudo-prior on the mean with strength parameter
- GammaNormalMean: Normal prior on the CLT mean for regularization
- GammaMLE: Pure MLE/MoM estimation (no prior, for reference)

All implementations share the same likelihood/sampling logic through GammaBase.
"""

import warnings

import numpy as np
from pydantic import Field
from scipy.stats import gamma as gamma_dist

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# =============================================================================
# Base Class for All Gamma Implementations
# =============================================================================


class GammaBase(BDFDistribution):
    """Base class for all Gamma distribution implementations.

    All implementations use shape α and rate β parameters, so sampling,
    likelihoods, and return functions are identical across implementations.
    Only the calc_posterior_params method differs.
    """

    # Child classes MUST implement this method
    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False):
        raise NotImplementedError("Subclasses must implement calc_posterior_params")

    def _validate_targets(self, y: np.ndarray):
        """Gamma distribution requires strictly positive values."""
        if np.any(y <= 0):
            raise ValueError("Gamma distribution requires all target values to be strictly positive.")
        if not np.all(np.isfinite(y)):
            raise ValueError("Targets must be finite for Gamma distribution.")

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution.

        Args
        ----
        data : np.ndarray
            The data to compute the log-likelihood for.

        Returns
        -------
        np.ndarray
            Log-likelihood values for each data point.
        """
        alpha, beta = self.calc_posterior_params(data, return_dict=False)
        return gamma_dist.logpdf(data, a=alpha, scale=1 / beta)

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the likelihood of the data given the distribution.

        Args
        ----
        data : np.ndarray
            The data to compute the likelihood for.

        Returns
        -------
        np.ndarray
            Likelihood values for each data point.
        """
        params = self.calc_posterior_params(data)
        alpha = params["alpha"]
        beta = params["beta"]
        return gamma_dist.pdf(data, a=alpha, scale=1 / beta)

    def nll(self, data: np.ndarray) -> float:
        """Compute the negative log-likelihood of the data.

        Args
        ----
        data : np.ndarray
            The data to compute the negative log-likelihood for.

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        if len(data) == 0:
            return np.inf
        return -np.sum(self.log_likelihood(data))

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
        data : np.ndarray | None, optional
            The data to use for sampling, if available.
        params : dict[str, float] | None, optional
            Additional parameters for sampling, if available.
        size : int, optional
            The number of samples to draw, default is 1.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution.
        """
        if params is not None:
            return self._sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self._sample_posterior_data(data, size=size, random_state=random_state)
        else:
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def _sample_posterior_params(self, params: dict[str, float], *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior using provided parameters.

        Args
        ----
        params : dict[str, float]
            Must contain 'alpha' and 'beta' keys.
        size : int, optional
            Number of samples to generate.
        random_state : int
            Random seed.

        Returns
        -------
        np.ndarray
            Samples from Gamma(α, β).
        """
        posterior_alpha, posterior_beta = params["posterior_alpha"], params["posterior_beta"]
        if posterior_alpha is None or posterior_beta is None:
            raise ValueError("params must contain 'posterior_alpha' and 'posterior_beta' keys")
        if posterior_alpha <= 0 or posterior_beta <= 0:
            raise ValueError(f"posterior_alpha={posterior_alpha} and posterior_beta={posterior_beta} must be positive")
        return gamma_dist.rvs(a=posterior_alpha, scale=1 / posterior_beta, size=size, random_state=random_state)  # type: ignore

    def _sample_posterior_data(self, data: np.ndarray, *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior using data.

        Args
        ----
        data : np.ndarray
            Data to calculate posterior parameters from.
        size : int, optional
            Number of samples to generate.
        random_state : int
            Random seed.

        Returns
        -------
        np.ndarray
            Samples from the posterior distribution.
        """
        params = self.calc_posterior_params(data, return_dict=True)
        return self._sample_posterior_params(params, size=size, random_state=random_state)

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior mean: α/β.

        Args
        ----
        data : np.ndarray | None, optional
            Data to calculate the posterior mean from.
        params : dict[str, float] | None, optional
            Posterior parameters to use.

        Returns
        -------
        float
            The posterior mean.
        """
        if params is not None:
            posterior_alpha = params["posterior_alpha"]
            posterior_beta = params["posterior_beta"]
        elif data is not None:
            posterior_alpha, posterior_beta = self.calc_posterior_params(data, return_dict=True)
        else:
            raise ValueError("Either 'data' or 'params' must be provided.")
        return posterior_alpha / posterior_beta

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get the posterior variance: α/β².

        Args
        ----
        data : np.ndarray | None, optional
            Data to calculate the posterior variance from.
        params : dict[str, float] | None, optional
            Posterior parameters to use.

        Returns
        -------
        float
            The posterior variance.
        """
        if params is not None:
            alpha = params["alpha"]
            beta = params["beta"]
        elif data is not None:
            alpha, beta = self.calc_posterior_params(data)
        else:
            raise ValueError("Either 'data' or 'params' must be provided.")
        return alpha / (beta**2)

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters.

        Args
        ----
        data : np.ndarray
            Data to calculate posterior parameters from.

        Returns
        -------
        dict
            Dictionary containing 'alpha' and 'beta'.
        """
        return self.calc_posterior_params(data, return_dict=True)


# =============================================================================
# Gamma with Pseudo-Prior on Mean
# =============================================================================


class GammaPseudoMeanParams(BDFDistributionParams):
    """Parameters for Gamma distribution with pseudo-prior on the mean."""

    prior_mean: float = Field(default=1.0, gt=0, description="Prior belief about the mean")
    prior_strength: float = Field(default=1.0, gt=0, description="Strength of prior (like pseudo-observations)")
    min_variance: float = Field(default=1e-6, gt=0, description="Minimum variance for numerical stability")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        missing_fields = {}
        if "prior_mean" not in data:
            missing_fields["prior_mean"] = self.__class__.model_fields["prior_mean"].default
        if "prior_strength" not in data:
            missing_fields["prior_strength"] = self.__class__.model_fields["prior_strength"].default

        super().__init__(**data)

        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class GammaPseudoMean(GammaBase):
    """Gamma distribution with pseudo-prior on the mean.

    The mean is regularized towards a prior mean with configurable strength.
    Shape α is estimated from data to preserve coefficient of variation.
    Rate β is adjusted to achieve the regularized mean.
    """

    def __init__(self, prior_params: dict | GammaPseudoMeanParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = GammaPseudoMeanParams.model_validate(prior_params)
        assert isinstance(
            prior_params, GammaPseudoMeanParams
        ), "prior_params must be an instance of GammaPseudoMeanParams"
        super().__init__(prior_params, params)
        self.prior_mean = prior_params.prior_mean
        self.prior_strength = prior_params.prior_strength
        self.min_variance = prior_params.min_variance

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | tuple | dict:
        """Calculate posterior parameters with pseudo-prior regularization."""
        n = len(data)

        if n == 0:
            alpha_default = 2.0
            beta_default = alpha_default / self.prior_mean
            if return_dict:
                return {"alpha": alpha_default, "beta": beta_default, "posterior_mean": self.prior_mean}
            return beta_default

        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1)) if n > 1 else self.min_variance
        sample_var = max(sample_var, self.min_variance)

        # Estimate shape from data
        alpha_est = (sample_mean**2) / sample_var
        alpha_est = max(alpha_est, 0.1)

        # Regularize the mean
        posterior_mean = (self.prior_strength * self.prior_mean + n * sample_mean) / (self.prior_strength + n)

        # Calculate β to achieve regularized mean
        posterior_beta = alpha_est / posterior_mean

        if return_dict:
            return {"posterior_alpha": alpha_est, "posterior_beta": posterior_beta}
        return alpha_est, posterior_beta


# =============================================================================
# Gamma with Normal Prior on Mean (CLT-based)
# =============================================================================


class GammaNormalMeanParams(BDFDistributionParams):
    """Parameters for Gamma distribution with Normal prior on the mean."""

    prior_mean: float = Field(default=1.0, gt=0, description="Prior mean (center of Normal prior)")
    prior_variance: float = Field(default=1.0, gt=0, description="Prior variance (uncertainty about mean)")
    min_variance: float = Field(default=1e-6, gt=0, description="Minimum variance for numerical stability")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        missing_fields = {}
        if "prior_mean" not in data:
            missing_fields["prior_mean"] = self.__class__.model_fields["prior_mean"].default
        if "prior_variance" not in data:
            missing_fields["prior_variance"] = self.__class__.model_fields["prior_variance"].default

        super().__init__(**data)

        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class GammaNormalMean(GammaBase):
    """Gamma distribution with Normal prior on the CLT mean.

    Uses Normal-Normal conjugacy on the sample mean (via CLT).
    Shape α is estimated from data to preserve coefficient of variation.
    Rate β is calculated to achieve the posterior mean (MAP estimate).
    """

    def __init__(self, prior_params: dict | GammaNormalMeanParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = GammaNormalMeanParams.model_validate(prior_params)
        assert isinstance(
            prior_params, GammaNormalMeanParams
        ), "prior_params must be an instance of GammaNormalMeanParams"
        super().__init__(prior_params, params)
        self.prior_mean = prior_params.prior_mean
        self.prior_variance = prior_params.prior_variance
        self.min_variance = prior_params.min_variance

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict | tuple:
        """Calculate posterior parameters using Normal prior on the mean."""
        n = len(data)

        if n == 0:
            alpha_default = 2.0
            beta_default = alpha_default / self.prior_mean
            if return_dict:
                return {
                    "alpha": alpha_default,
                    "beta": beta_default,
                    "posterior_mean": self.prior_mean,
                    "posterior_variance_of_mean": self.prior_variance,
                }
            return beta_default

        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1)) if n > 1 else self.min_variance
        sample_var = max(sample_var, self.min_variance)

        # Estimate shape from data
        alpha_est = (sample_mean**2) / sample_var
        alpha_est = max(alpha_est, 0.1)

        # Normal-Normal conjugacy for the mean
        precision_prior = 1 / self.prior_variance
        precision_data = n / sample_var
        precision_post = precision_prior + precision_data
        variance_post = 1 / precision_post

        mean_post = variance_post * (precision_prior * self.prior_mean + precision_data * sample_mean)

        # Calculate β for posterior mean
        beta_post = alpha_est / mean_post

        if return_dict:
            return {
                "posterior_alpha": alpha_est,
                "posterior_beta": beta_post,
            }
        return alpha_est, beta_post


# =============================================================================
# Pure MLE (for reference/comparison)
# =============================================================================


class GammaMLEParams(BDFDistributionParams):
    """Parameters for Gamma distribution with pure MLE estimation."""

    min_variance: float = Field(default=1e-6, gt=0, description="Minimum variance for numerical stability")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        validate_by_name = True


class GammaMLE(GammaBase):
    """Gamma distribution with pure MLE/MoM estimation.

    Both α and β are estimated from data with no regularization.
    This is the maximum likelihood approach.
    """

    def __init__(self, prior_params: dict | GammaMLEParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = GammaMLEParams.model_validate(prior_params)
        assert isinstance(prior_params, GammaMLEParams), "prior_params must be an instance of GammaMLEParams"
        super().__init__(prior_params, params)
        self.min_variance = prior_params.min_variance

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict | tuple:
        """Calculate MLE parameters using method of moments."""
        n = len(data)

        if n == 0:
            return {"alpha": 1.0, "beta": 1.0} if return_dict else 1.0

        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1)) if n > 1 else self.min_variance
        sample_var = max(sample_var, self.min_variance)

        alpha_mle = (sample_mean**2) / sample_var
        beta_mle = sample_mean / sample_var

        alpha_mle = max(alpha_mle, 0.1)

        if return_dict:
            return {"posterior_alpha": alpha_mle, "posterior_beta": beta_mle}
        return alpha_mle, beta_mle
