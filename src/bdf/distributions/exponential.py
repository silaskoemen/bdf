"""Exponential distribution implementations for BDF.

Provides conjugate Exponential-Gamma models:
- GammaABExponential: Gamma(α, β) prior on rate λ (shape-rate parameterization)
- GammaMVExponential: Gamma prior via mean/variance (more interpretable)

Both support:
- Closed-form Bayesian evidence (NLE)
- Posterior predictive (Lomax distribution)
- Efficient inference (conjugate updates)
"""

import warnings
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.stats import expon
from scipy.stats import gamma as gamma_dist
from scipy.stats import lomax

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class GammaABLambdaExponentialParams(BDFDistributionParams):
    """Parameters for Exponential with Gamma(α, β) prior on rate λ.

    Prior: λ ~ Gamma(α, β)  [shape-rate parameterization]
    Likelihood: y | λ ~ Exponential(λ)
    Posterior: λ | y ~ Gamma(α + n, β + Σy)
    Posterior predictive: y_new | y ~ Lomax(α_post, β_post)

    Notes:
    - α (alpha_lambda): shape parameter (α > 0)
    - β (beta_lambda): rate parameter (β > 0)
    - E[λ] = α/β, Var[λ] = α/β²
    """

    # Prior hyperparameters
    alpha_lambda: float = Field(default=1.0, gt=0, description="Shape parameter α of Gamma prior for rate λ")
    beta_lambda: float = Field(default=1.0, gt=0, description="Rate parameter β of Gamma prior for rate λ")

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True, description="Use Lomax posterior predictive (marginalizes λ uncertainty)."
    )


class GammaMVLambdaExponentialParams(BDFDistributionParams):
    """Parameters for Exponential with Gamma prior specified via mean/variance.

    Prior: λ ~ Gamma(α, β) where α = mean²/var, β = mean/var
    Likelihood: y | λ ~ Exponential(λ)
    Posterior: λ | y ~ Gamma(α + n, β + Σy)
    Posterior predictive: y_new | y ~ Lomax(α_post, β_post)

    This parameterization is more interpretable:
    - mean_lambda: prior expectation of rate λ
    - var_lambda: prior uncertainty about λ
    """

    # Prior hyperparameters (mean-variance parameterization)
    mean_lambda: float = Field(default=1.0, gt=0, description="Prior mean E[λ] for rate parameter")
    var_lambda: float = Field(default=1.0, gt=0, description="Prior variance Var[λ] for rate parameter")

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nle")
    use_posterior_predictive: bool = Field(default=True, description="Use Lomax posterior predictive.")


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class GammaABLambdaExponential(BDFDistribution[GammaABLambdaExponentialParams]):
    r"""Exponential-Gamma conjugate model with shape-rate parameterization.

    **Usage:** ``dist="GammaABLambdaExponential"``

    **Model:**

    *   **Prior:** :math:`\lambda \sim \text{Gamma}(\alpha, \beta)`
    *   **Likelihood:** :math:`y \mid \lambda \sim \text{Exp}(\lambda)`
    *   **Posterior:** :math:`\lambda \mid y \sim \text{Gamma}(\alpha + n, \beta + \sum y_i)`
    *   **Posterior Predictive:** :math:`y_\text{new} \mid y \sim \text{Lomax}(\alpha_\text{post}, \beta_\text{post})`

    Parameters
    ----------
    alpha_lambda : float, default=1.0
        Shape parameter :math:`\alpha` of the Gamma prior on rate :math:`\lambda`.
    beta_lambda : float, default=1.0
        Rate parameter :math:`\beta` of the Gamma prior on rate :math:`\lambda`.

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaMVLambdaExponential : Same model with mean-variance parameterization.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = GammaABLambdaExponentialParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: dict[str, Any] | GammaABLambdaExponentialParams):
        super().__init__(params)
        self.alpha_lambda = self.params.alpha_lambda
        self.beta_lambda = self.params.beta_lambda

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Gamma posterior parameters for λ.

        Returns dict with:
        - posterior_alpha: Posterior shape α_post = α + n
        - posterior_beta: Posterior rate β_post = β + Σy
        - posterior_lambda: Posterior mean E[λ | y] = α_post/β_post
        """
        n = data.shape[0]
        sum_data = np.sum(data)

        # Conjugate update: Gamma(α, β) → Gamma(α + n, β + Σy)
        alpha_post = self.alpha_lambda + n
        beta_post = self.beta_lambda + sum_data

        # Posterior mean of λ
        lambda_post = alpha_post / beta_post

        return {
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "posterior_lambda": float(lambda_post),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Exponential likelihood: Exp(x | λ_post).

        Uses posterior mean λ = α_post/β_post as point estimate.
        """
        lambda_post = params["posterior_lambda"]
        # Exponential PDF: λ exp(-λx)
        # scipy.stats.expon uses scale = 1/λ
        return expon.logpdf(data, scale=1.0 / lambda_post)

    def _num_parameters(self) -> int:
        """Only λ is estimated."""
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive (Lomax distribution).

        If use_posterior_predictive=True:
            Sample from Lomax(α_post, β_post)
        Else:
            Sample from Exponential(λ_post)
        """
        if self.params.use_posterior_predictive:
            # Posterior predictive: Lomax(α, β)
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            # scipy.stats.lomax: c=α, scale=β (shape parameter c, scale parameter scale)
            return np.array(lomax.rvs(c=alpha_post, scale=beta_post, size=size, random_state=random_state))
        else:
            # Plug-in: Exponential(λ_post)
            lambda_post = params["posterior_lambda"]
            return np.array(expon.rvs(scale=1.0 / lambda_post, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Exponential distribution."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data > 0):
            raise ValueError("Exponential data must be strictly positive (x > 0)")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean of the distribution.

        For Exponential(λ), the mean is 1/λ.
        For posterior predictive Lomax(α, β), mean = β/(α-1) if α > 1.
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            # Lomax mean: β/(α-1) for α > 1
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            if alpha_post <= 1:
                warnings.warn(
                    f"Lomax posterior predictive has undefined mean (α={alpha_post:.2f} ≤ 1). " f"Returning NaN.",
                    UserWarning,
                )
                return float("nan")
            return beta_post / (alpha_post - 1)
        else:
            # Exponential mean: 1/λ
            lambda_post = params["posterior_lambda"]
            return 1.0 / lambda_post

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance of the distribution.

        For Exponential(λ), variance = 1/λ².
        For posterior predictive Lomax(α, β), variance = β²α/[(α-1)²(α-2)] if α > 2.
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            # Lomax variance: β²α/[(α-1)²(α-2)] for α > 2
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]

            if alpha_post <= 1:
                warnings.warn(
                    f"Lomax posterior predictive has undefined variance (α={alpha_post:.2f} ≤ 1). " f"Returning NaN.",
                    UserWarning,
                )
                return float("nan")
            elif alpha_post <= 2:
                warnings.warn(
                    f"Lomax posterior predictive has infinite variance (α={alpha_post:.2f} ≤ 2). " f"Returning inf.",
                    UserWarning,
                )
                return float("inf")

            return (beta_post**2 * alpha_post) / ((alpha_post - 1) ** 2 * (alpha_post - 2))
        else:
            # Exponential variance: 1/λ²
            lambda_post = params["posterior_lambda"]
            return 1.0 / (lambda_post**2)

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Exponential-Gamma conjugate.

        p(y | prior) = ∫ p(y | λ) p(λ) dλ

        Closed form for Gamma-Exponential:
        log p(y) = log Γ(α + n) - log Γ(α) + α log β - (α + n) log(β + Σy)
        """
        from scipy.special import gammaln

        n = data.shape[0]
        sum_data = np.sum(data)

        alpha_post = self.alpha_lambda + n
        beta_post = self.beta_lambda + sum_data

        log_ev = gammaln(alpha_post) - gammaln(self.alpha_lambda)
        log_ev += self.alpha_lambda * np.log(self.beta_lambda)
        log_ev -= alpha_post * np.log(beta_post)

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Lomax(α_post, β_post).

        Integrates out uncertainty in λ over the Gamma posterior.

        Lomax PDF: f(x | α, β) = (α/β) (1 + x/β)^(-(α+1))
        """
        alpha_post = params["posterior_alpha"]
        beta_post = params["posterior_beta"]

        # scipy.stats.lomax: c=α (shape), scale=β
        return lomax.logpdf(data, c=alpha_post, scale=beta_post)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample λ from prior Gamma(α, β).

        Returns rate parameters λ (not Exponential samples).
        To get prior predictive samples, draw λ ~ Gamma(α, β) then x ~ Exp(λ).
        """
        rng = np.random.default_rng(random_state)
        # scipy.stats.gamma: a=α (shape), scale=1/β (rate→scale conversion)
        return np.array(gamma_dist.rvs(a=self.alpha_lambda, scale=1.0 / self.beta_lambda, size=size, random_state=rng))


class GammaMVLambdaExponential(BDFDistribution[GammaMVLambdaExponentialParams]):
    r"""Exponential-Gamma conjugate model with mean-variance parameterization.

    **Usage:** ``dist="GammaMVLambdaExponential"``

    Same conjugate model as :class:`GammaABLambdaExponential`, but the prior is
    specified via mean and variance. Internally converts to
    :math:`\alpha = \mu^2/\sigma^2`, :math:`\beta = \mu/\sigma^2`.

    Parameters
    ----------
    mean_lambda : float or "auto", default=1.0
        Prior mean :math:`E[\lambda]`. If ``"auto"``, set to the sample mean
        of ``y`` at fit time.
    var_lambda : float, default=1.0
        Prior variance :math:`\text{Var}[\lambda]`.

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaABLambdaExponential : Same model with shape-rate parameterization.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = GammaMVLambdaExponentialParams

    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: GammaMVLambdaExponentialParams):
        super().__init__(params)
        self.mean_lambda = self.params.mean_lambda
        self.var_lambda = self.params.var_lambda

        # Convert mean-variance to shape-rate
        self.alpha_lambda = self.mean_lambda**2 / self.var_lambda
        self.beta_lambda = self.mean_lambda / self.var_lambda

    # ========================================================================
    # REQUIRED METHODS (identical to GammaABExponential after conversion)
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Gamma posterior parameters."""
        n = data.shape[0]
        sum_data = np.sum(data)

        alpha_post = self.alpha_lambda + n
        beta_post = self.beta_lambda + sum_data
        lambda_post = alpha_post / beta_post

        return {
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "posterior_lambda": float(lambda_post),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Exponential likelihood."""
        lambda_post = params["posterior_lambda"]
        return expon.logpdf(data, scale=1.0 / lambda_post)

    def _num_parameters(self) -> int:
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive or plug-in."""
        if self.params.use_posterior_predictive:
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            return np.array(lomax.rvs(c=alpha_post, scale=beta_post, size=size, random_state=random_state))
        else:
            lambda_post = params["posterior_lambda"]
            return np.array(expon.rvs(scale=1.0 / lambda_post, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data > 0):
            raise ValueError("Exponential data must be strictly positive (x > 0)")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            if alpha_post <= 1:
                warnings.warn(f"Lomax mean undefined (α={alpha_post:.2f} ≤ 1). Returning NaN.", UserWarning)
                return float("nan")
            return beta_post / (alpha_post - 1)
        else:
            return 1.0 / params["posterior_lambda"]

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]

            if alpha_post <= 1:
                return float("nan")
            elif alpha_post <= 2:
                warnings.warn(f"Lomax variance infinite (α={alpha_post:.2f} ≤ 2). Returning inf.", UserWarning)
                return float("inf")

            return (beta_post**2 * alpha_post) / ((alpha_post - 1) ** 2 * (alpha_post - 2))
        else:
            return 1.0 / (params["posterior_lambda"] ** 2)

    # ========================================================================
    # OPTIONAL METHODS
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence."""
        from scipy.special import gammaln

        n = data.shape[0]
        sum_data = np.sum(data)

        alpha_post = self.alpha_lambda + n
        beta_post = self.beta_lambda + sum_data

        log_ev = gammaln(alpha_post) - gammaln(self.alpha_lambda)
        log_ev += self.alpha_lambda * np.log(self.beta_lambda)
        log_ev -= alpha_post * np.log(beta_post)

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Lomax."""
        alpha_post = params["posterior_alpha"]
        beta_post = params["posterior_beta"]
        return lomax.logpdf(data, c=alpha_post, scale=beta_post)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample λ from prior Gamma(α, β)."""
        rng = np.random.default_rng(random_state)
        return np.array(gamma_dist.rvs(a=self.alpha_lambda, scale=1.0 / self.beta_lambda, size=size, random_state=rng))

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        For NormalMuNormal:
        - 'mean_lambda': Use sample mean
        """
        if key == "mean_lambda":
            return float(np.mean(data))
        else:
            raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")
