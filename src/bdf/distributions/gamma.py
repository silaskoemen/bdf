"""Gamma distribution implementations for Bayesian Distributional Forest.

The Gamma distribution is parameterized by shape α and rate β.
Mean = α/β, Variance = α/β²

Implemented versions:
- GammaPseudoMean: Pseudo-prior on the mean with strength parameter
- GammaNormalMean: Normal prior on the CLT mean for regularization
- FrequentistGamma: Pure MLE/MoM estimation (no prior, for reference)

All implementations share the same likelihood/sampling logic through GammaBase.
"""

import warnings
from typing import Any

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
    def calc_posterior_params(self, data: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement calc_posterior_params")

    def _validate_targets(self, y: np.ndarray):
        """Gamma distribution requires strictly positive values."""
        if np.any(y <= 0):
            raise ValueError("Gamma distribution requires all target values to be strictly positive.")
        if not np.all(np.isfinite(y)):
            raise ValueError("Targets must be finite for Gamma distribution.")

    def log_likelihood(self, data: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution.

        Args
        ----
        data : np.ndarray
            The data to compute the log-likelihood for.
        params : dict | None
            Optional pre-computed posterior parameters.

        Returns
        -------
        np.ndarray
            Log-likelihood values for each data point.
        """
        if params is None:
            params = self.calc_posterior_params(data)
        return gamma_dist.logpdf(data, a=params["posterior_alpha"], scale=1 / params["posterior_beta"])

    def likelihood(self, data: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Compute the likelihood of the data given the distribution.

        Args
        ----
        data : np.ndarray
            The data to compute the likelihood for.
        params : dict | None
            Optional pre-computed posterior parameters.

        Returns
        -------
        np.ndarray
            Likelihood values for each data point.
        """
        if params is None:
            params = self.calc_posterior_params(data)
        return gamma_dist.pdf(data, a=params["posterior_alpha"], scale=1 / params["posterior_beta"])

    def nll(self, data: np.ndarray, params: dict | None = None) -> float:
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

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
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
        return gamma_dist.rvs(a=posterior_alpha, scale=1 / posterior_beta, size=size, random_state=random_state)

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
        params = self.calc_posterior_params(data)
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
            post_params = self.calc_posterior_params(data)
            posterior_alpha = post_params["posterior_alpha"]
            posterior_beta = post_params["posterior_beta"]
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
            posterior_alpha = params["posterior_alpha"]
            posterior_beta = params["posterior_beta"]
        elif data is not None:
            post_params = self.calc_posterior_params(data)
            posterior_alpha = post_params["posterior_alpha"]
            posterior_beta = post_params["posterior_beta"]
        else:
            raise ValueError("Either 'data' or 'params' must be provided.")
        return posterior_alpha / (posterior_beta**2)

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
        return self.calc_posterior_params(data)


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
    r"""Gamma distribution with pseudo-prior on the mean.

    **Usage:** ``dist="GammaPseudoMean"``

    **Model:**

    *   **Prior:** Pseudo-prior shrinking the mean toward ``prior_mean`` with
        strength ``prior_strength``
    *   **Likelihood:** :math:`y \sim \text{Gamma}(\alpha, \beta)`

    Shape :math:`\alpha` is estimated from data via MoM; rate :math:`\beta` is
    adjusted to achieve the regularized mean.

    Parameters
    ----------
    prior_mean : float, default=1.0
        Prior belief about the data mean (shrinkage target).
    prior_strength : float, default=1.0
        Strength of prior (pseudo-observations). Higher = stronger shrinkage.
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaNormalMean : Alternative with Normal prior on the CLT mean.
    FrequentistGamma : Pure MLE estimation without regularization.
    """

    def __init__(self, params: dict | GammaPseudoMeanParams):
        if isinstance(params, dict):
            params = GammaPseudoMeanParams.model_validate(params)
        assert isinstance(params, GammaPseudoMeanParams), "params must be an instance of GammaPseudoMeanParams"
        super().__init__(params)
        self.prior_mean = params.prior_mean
        self.prior_strength = params.prior_strength
        self.min_variance = params.min_variance

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, Any]:
        """Calculate posterior parameters with pseudo-prior regularization."""
        n = len(data)

        if n == 0:
            alpha_default = 2.0
            beta_default = alpha_default / self.prior_mean
            return {"posterior_alpha": alpha_default, "posterior_beta": beta_default}

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

        return {"posterior_alpha": alpha_est, "posterior_beta": posterior_beta}


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
    r"""Gamma distribution with Normal prior on the CLT mean.

    **Usage:** ``dist="GammaNormalMean"``

    **Model:**

    *   **Prior:** :math:`\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)` (CLT approximation)
    *   **Likelihood:** :math:`y \sim \text{Gamma}(\alpha, \beta)`

    Uses Normal-Normal conjugacy on the sample mean. Shape :math:`\alpha` is
    estimated from data; rate :math:`\beta` is set to achieve the posterior mean.

    Parameters
    ----------
    prior_mean : float, default=1.0
        Prior mean (center of Normal prior on the data mean).
    prior_variance : float, default=1.0
        Prior variance (uncertainty about the data mean).
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaPseudoMean : Alternative with pseudo-prior on the mean.
    FrequentistGamma : Pure MLE estimation without regularization.
    """

    def __init__(self, params: dict | GammaNormalMeanParams):
        if isinstance(params, dict):
            params = GammaNormalMeanParams.model_validate(params)
        assert isinstance(params, GammaNormalMeanParams), "params must be an instance of GammaNormalMeanParams"
        super().__init__(params)
        self.prior_mean = params.prior_mean
        self.prior_variance = params.prior_variance
        self.min_variance = params.min_variance

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, Any]:
        """Calculate posterior parameters using Normal prior on the mean."""
        n = len(data)

        if n == 0:
            alpha_default = 2.0
            beta_default = alpha_default / self.prior_mean
            return {"posterior_alpha": alpha_default, "posterior_beta": beta_default}

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

        return {"posterior_alpha": alpha_est, "posterior_beta": beta_post}


# =============================================================================
# Pure MLE (for reference/comparison)
# =============================================================================


class FrequentistGammaParams(BDFDistributionParams):
    """Parameters for Gamma distribution with pure MLE estimation."""

    min_variance: float = Field(default=1e-6, gt=0, description="Minimum variance for numerical stability")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"
        validate_by_name = True


class FrequentistGamma(GammaBase):
    r"""Gamma distribution with frequentist MoM estimation (no prior).

    **Usage:** ``dist="FrequentistGamma"``

    Both :math:`\alpha` (shape) and :math:`\beta` (rate) are estimated from data
    via Method of Moments with no regularization.

    .. math:: y \sim \text{Gamma}(\alpha, \beta)

    Parameters
    ----------
    score_method : {"nll"}
        Only NLL scoring is supported (frequentist model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaPseudoMean : Regularized version with pseudo-prior on the mean.
    GammaNormalMean : Regularized version with Normal prior on the CLT mean.
    """

    def __init__(self, params: dict | FrequentistGammaParams):
        if isinstance(params, dict):
            params = FrequentistGammaParams.model_validate(params)
        assert isinstance(params, FrequentistGammaParams), "params must be an instance of FrequentistGammaParams"
        super().__init__(params)
        self.min_variance = params.min_variance

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, Any]:
        """Calculate MLE parameters using method of moments."""
        n = len(data)

        if n == 0:
            return {"posterior_alpha": 1.0, "posterior_beta": 1.0}

        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1)) if n > 1 else self.min_variance
        sample_var = max(sample_var, self.min_variance)

        alpha_mle = (sample_mean**2) / sample_var
        beta_mle = sample_mean / sample_var

        alpha_mle = max(alpha_mle, 0.1)

        return {"posterior_alpha": alpha_mle, "posterior_beta": beta_mle}
