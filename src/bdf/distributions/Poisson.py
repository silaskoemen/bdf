"""Poisson distribution implementations for BDF.

Provides conjugate Poisson-Gamma models:
- GammaABLambdaPoisson: Gamma(α, β) prior on rate λ (shape-rate parameterization)
- GammaMVLambdaPoisson: Gamma prior via mean/variance (more interpretable)

Both support:
- Closed-form Bayesian evidence (NLE)
- Negative Binomial posterior predictive
- Efficient inference (conjugate updates)
"""

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.stats import gamma as gamma_dist
from scipy.stats import nbinom, poisson

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class GammaABLambdaPoissonParams(BDFDistributionParams):
    """Parameters for Poisson with Gamma(α, β) prior on rate λ.

    Prior: λ ~ Gamma(α, β)  [shape-rate parameterization]
    Likelihood: y | λ ~ Poisson(λ)
    Posterior: λ | y ~ Gamma(α + Σy, β + n)
    Posterior predictive: y_new | y ~ NegativeBinomial(α_post, β_post/(1+β_post))

    Notes:
    - α (alpha_lambda): shape parameter (α > 0)
    - β (beta_lambda): rate parameter (β > 0)
    - E[λ] = α/β, Var[λ] = α/β²
    - NB parameterization: n=α_post, p=β_post/(1+β_post)
    """

    # Prior hyperparameters
    alpha_lambda: float = Field(default=1.0, gt=0, description="Shape parameter α of Gamma prior for rate λ")
    beta_lambda: float = Field(default=1.0, gt=0, description="Rate parameter β of Gamma prior for rate λ")

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True, description="Use Negative Binomial posterior predictive (marginalizes λ uncertainty)."
    )

    raise_on_non_integer: bool = Field(
        default=True, description="Raise error if non-integer data is provided for Poisson distribution."
    )


class GammaMVLambdaPoissonParams(BDFDistributionParams):
    """Parameters for Poisson with Gamma prior specified via mean/variance.

    Prior: λ ~ Gamma(α, β) where α = mean²/var, β = mean/var
    Likelihood: y | λ ~ Poisson(λ)
    Posterior: λ | y ~ Gamma(α + Σy, β + n)
    Posterior predictive: y_new | y ~ NegativeBinomial(α_post, β_post/(1+β_post))

    This parameterization is more interpretable:
    - mean_lambda: prior expectation of rate λ
    - var_lambda: prior uncertainty about λ

    Auto-parameter resolution:
    - mean_lambda: 'auto' → sample mean
    - var_lambda: 'auto' → sample_mean * var_lambda_auto_scale
      (For Poisson, Var[Y] = E[Y] = λ, so sample mean is a natural scale)
    """

    # Prior hyperparameters (mean-variance parameterization)
    mean_lambda: float = Field(default=1.0, gt=0, description="Prior mean E[λ] for rate parameter")
    var_lambda: float = Field(default=1.0, gt=0, description="Prior variance Var[λ] for rate parameter")
    var_lambda_auto_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale factor for automatic var_lambda. var_lambda = sample_mean * scale. "
        "Higher values = weaker prior (more uncertainty about λ).",
        exclude=True,
    )

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nle")
    use_posterior_predictive: bool = Field(default=True, description="Use Negative Binomial posterior predictive.")

    raise_on_non_integer: bool = Field(
        default=True, description="Raise error if non-integer data is provided for Poisson distribution."
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class GammaABLambdaPoisson(BDFDistribution[GammaABLambdaPoissonParams]):
    r"""Poisson-Gamma conjugate model with shape-rate parameterization.

    **Usage:** ``dist="GammaABLambdaPoisson"``

    **Model:**

    *   **Prior:** :math:`\lambda \sim \text{Gamma}(\alpha, \beta)`
    *   **Likelihood:** :math:`y \mid \lambda \sim \text{Poisson}(\lambda)`
    *   **Posterior:** :math:`\lambda \mid y \sim \text{Gamma}(\alpha + \sum y_i, \beta + n)`
    *   **Posterior Predictive:** :math:`y_\text{new} \mid y \sim \text{NegBin}(\alpha_\text{post}, \beta_\text{post}/(1+\beta_\text{post}))`

    Parameters
    ----------
    alpha_lambda : float, default=1.0
        Shape parameter :math:`\alpha` of the Gamma prior on rate :math:`\lambda`.
    beta_lambda : float, default=1.0
        Rate parameter :math:`\beta` of the Gamma prior on rate :math:`\lambda`.
    raise_on_non_integer : bool, default=True
        If True, raises an error when non-integer data is provided.

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaMVLambdaPoisson : Same model with more interpretable mean-variance parameterization.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = GammaABLambdaPoissonParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: GammaABLambdaPoissonParams):
        super().__init__(params)
        self.alpha_lambda = self.params.alpha_lambda
        self.beta_lambda = self.params.beta_lambda

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Gamma posterior parameters for λ.

        Returns dict with:
        - posterior_alpha: Posterior shape α_post = α + Σy
        - posterior_beta: Posterior rate β_post = β + n
        - posterior_lambda: Posterior mean E[λ | y] = α_post/β_post
        """
        n = data.shape[0]
        sum_counts = np.sum(data)

        # Conjugate update: Gamma(α, β) → Gamma(α + Σy, β + n)
        alpha_post = self.alpha_lambda + sum_counts
        beta_post = self.beta_lambda + n

        # Posterior mean of λ
        lambda_post = alpha_post / beta_post

        return {
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "posterior_lambda": float(lambda_post),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Poisson likelihood: Poisson(k | λ_post).

        Uses posterior mean λ = α_post/β_post as point estimate.
        """
        lambda_post = params["posterior_lambda"]
        # Poisson PMF: λ^k exp(-λ) / k!
        return poisson.logpmf(data, mu=lambda_post)

    def _num_parameters(self) -> int:
        """Only λ is estimated."""
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive (Negative Binomial distribution).

        If use_posterior_predictive=True:
            Sample from NegativeBinomial(n=α_post, p=β_post/(1+β_post))
        Else:
            Sample from Poisson(λ_post)
        """
        if self.params.use_posterior_predictive:
            # Posterior predictive: Negative Binomial
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]

            # scipy.stats.nbinom: n (number of successes), p (success probability)
            # Conversion: n = α_post, p = β_post/(1+β_post)
            n_nb = alpha_post
            p_nb = beta_post / (1 + beta_post)

            return np.array(nbinom.rvs(n=n_nb, p=p_nb, size=size, random_state=random_state))
        else:
            # Plug-in: Poisson(λ_post)
            lambda_post = params["posterior_lambda"]
            return np.array(poisson.rvs(mu=lambda_post, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Poisson distribution."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Poisson data must be non-negative integers (k ≥ 0)")
        if not np.all(data == np.floor(data)) and self.params.raise_on_non_integer:
            raise ValueError("Poisson data must be integers")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean of the distribution.

        For Poisson(λ), the mean is λ.
        For posterior predictive NB(α, β), mean = α(1-p)/p = α/β.
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            # Negative Binomial mean: n(1-p)/p = α/β
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            return alpha_post / beta_post
        else:
            # Poisson mean: λ
            return params["posterior_lambda"]

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance of the distribution.

        For Poisson(λ), variance = λ.
        For posterior predictive NB(α, β), variance = α(1-p)/p² = α(1+β)/β².
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            # Negative Binomial variance: n(1-p)/p² = α(1+β)/β²
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            return alpha_post * (1 + beta_post) / (beta_post**2)
        else:
            # Poisson variance: λ
            return params["posterior_lambda"]

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Poisson-Gamma conjugate.

        p(y | prior) = ∫ p(y | λ) p(λ) dλ

        Closed form for Gamma-Poisson:
        log p(y) = log Γ(α + Σy) - log Γ(α) + α log β - (α + Σy) log(β + n) - Σ log(y!)
        """
        from scipy.special import gammaln

        n = data.shape[0]
        sum_counts = np.sum(data)

        alpha_post = self.alpha_lambda + sum_counts
        beta_post = self.beta_lambda + n

        # Log evidence = log marginal likelihood
        log_ev = gammaln(alpha_post) - gammaln(self.alpha_lambda)
        log_ev += self.alpha_lambda * np.log(self.beta_lambda)
        log_ev -= alpha_post * np.log(beta_post)

        # Subtract log factorials (Poisson normalization)
        log_ev -= np.sum(gammaln(data + 1))

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Negative Binomial(α_post, β_post/(1+β_post)).

        Integrates out uncertainty in λ over the Gamma posterior.

        NB PMF: f(k | n, p) = C(k+n-1, k) p^n (1-p)^k
        where n = α_post, p = β_post/(1+β_post)
        """
        alpha_post = params["posterior_alpha"]
        beta_post = params["posterior_beta"]

        # scipy.stats.nbinom: n (number of successes), p (success probability)
        n_nb = alpha_post
        p_nb = beta_post / (1 + beta_post)

        return nbinom.logpmf(data, n=n_nb, p=p_nb)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample λ from prior Gamma(α, β).

        Returns rate parameters λ (not Poisson samples).
        To get prior predictive samples, draw λ ~ Gamma(α, β) then k ~ Poisson(λ).
        """
        rng = np.random.default_rng(random_state)
        # scipy.stats.gamma: a=α (shape), scale=1/β (rate→scale conversion)
        return np.array(gamma_dist.rvs(a=self.alpha_lambda, scale=1.0 / self.beta_lambda, size=size, random_state=rng))


class GammaMVLambdaPoisson(BDFDistribution[GammaMVLambdaPoissonParams]):
    r"""Poisson-Gamma conjugate model with mean-variance parameterization.

    **Usage:** ``dist="GammaMVLambdaPoisson"``

    Same conjugate model as :class:`GammaABLambdaPoisson`, but the prior is
    specified via interpretable mean and variance rather than shape and rate.
    Internally converts to :math:`\alpha = \mu^2/\sigma^2`, :math:`\beta = \mu/\sigma^2`.

    **Model:**

    *   **Prior:** :math:`\lambda \sim \text{Gamma}(\alpha, \beta)`
    *   **Likelihood:** :math:`y \mid \lambda \sim \text{Poisson}(\lambda)`
    *   **Posterior:** :math:`\lambda \mid y \sim \text{Gamma}(\alpha + \sum y_i, \beta + n)`

    Parameters
    ----------
    mean_lambda : float or "auto", default=1.0
        Prior mean :math:`E[\lambda]`. If ``"auto"``, set to the sample mean
        of ``y`` at fit time.
    var_lambda : float or "auto", default=1.0
        Prior variance :math:`\text{Var}[\lambda]`. If ``"auto"``, set to
        ``mean(y) * var_lambda_auto_scale`` at fit time.
    var_lambda_auto_scale : float, default=1.0
        Multiplier applied when ``var_lambda="auto"``. Ignored otherwise.
    raise_on_non_integer : bool, default=True
        If True, raises an error when non-integer data is provided.

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaABLambdaPoisson : Same model with shape-rate parameterization.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = GammaMVLambdaPoissonParams

    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: GammaMVLambdaPoissonParams):
        super().__init__(params)
        self.mean_lambda = self.params.mean_lambda
        self.var_lambda = self.params.var_lambda

        # Convert mean-variance to shape-rate
        self.alpha_lambda = self.mean_lambda**2 / self.var_lambda
        self.beta_lambda = self.mean_lambda / self.var_lambda

    # ========================================================================
    # REQUIRED METHODS (identical to GammaABLambdaPoisson after conversion)
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Gamma posterior parameters."""
        n = data.shape[0]
        sum_counts = np.sum(data)

        alpha_post = self.alpha_lambda + sum_counts
        beta_post = self.beta_lambda + n
        lambda_post = alpha_post / beta_post

        return {
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "posterior_lambda": float(lambda_post),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Poisson likelihood."""
        lambda_post = params["posterior_lambda"]
        return poisson.logpmf(data, mu=lambda_post)

    def _num_parameters(self) -> int:
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive or plug-in."""
        if self.params.use_posterior_predictive:
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]

            n_nb = alpha_post
            p_nb = beta_post / (1 + beta_post)

            return np.array(nbinom.rvs(n=n_nb, p=p_nb, size=size, random_state=random_state))
        else:
            lambda_post = params["posterior_lambda"]
            return np.array(poisson.rvs(mu=lambda_post, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Poisson data must be non-negative integers (k ≥ 0)")
        if not np.all(data == np.floor(data)) and self.params.raise_on_non_integer:
            raise ValueError("Poisson data must be integers")
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
            return alpha_post / beta_post
        else:
            return params["posterior_lambda"]

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
            return alpha_post * (1 + beta_post) / (beta_post**2)
        else:
            return params["posterior_lambda"]

    # ========================================================================
    # OPTIONAL METHODS
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence."""
        from scipy.special import gammaln

        n = data.shape[0]
        sum_counts = np.sum(data)

        alpha_post = self.alpha_lambda + sum_counts
        beta_post = self.beta_lambda + n

        log_ev = gammaln(alpha_post) - gammaln(self.alpha_lambda)
        log_ev += self.alpha_lambda * np.log(self.beta_lambda)
        log_ev -= alpha_post * np.log(beta_post)
        log_ev -= np.sum(gammaln(data + 1))

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Negative Binomial."""
        alpha_post = params["posterior_alpha"]
        beta_post = params["posterior_beta"]

        n_nb = alpha_post
        p_nb = beta_post / (1 + beta_post)

        return nbinom.logpmf(data, n=n_nb, p=p_nb)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample λ from prior Gamma(α, β)."""
        rng = np.random.default_rng(random_state)
        return np.array(gamma_dist.rvs(a=self.alpha_lambda, scale=1.0 / self.beta_lambda, size=size, random_state=rng))

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        For GammaMVLambdaPoisson:
        - 'mean_lambda': Use sample mean (natural estimate of λ for Poisson)
        - 'var_lambda': Use sample_mean * var_lambda_auto_scale
          (For Poisson, Var[Y] = E[Y] = λ, so sample mean provides natural scale)
        """
        if key == "mean_lambda":
            return float(np.mean(data))
        if key == "var_lambda":
            assert params is not None, "'params' must be provided to resolve 'var_lambda' automatically."
            assert (
                "var_lambda_auto_scale" in params
            ), "'var_lambda_auto_scale' must be defined in params to resolve 'var_lambda' automatically."
            sample_mean = np.mean(data)
            scale = params["var_lambda_auto_scale"]
            # Prior variance = sample_mean * scale
            # scale=1: prior std ≈ sqrt(mean), reasonable for Poisson
            # scale>1: weaker prior (more uncertainty)
            # scale<1: stronger prior (less uncertainty)
            return float(max(sample_mean * scale, 1e-6))  # Ensure positive
        else:
            raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")
