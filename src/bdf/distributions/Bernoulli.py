"""Bernoulli distribution implementations for BDF.

Provides conjugate Bernoulli-Beta models for binary classification:
- BetaABBernoulli: Beta(α, β) prior on success probability p (concentration parameterization)
- BetaMVBernoulli: Beta prior via mean/variance (more interpretable)

Both support:
- Closed-form Bayesian evidence (NLE)
- Beta-Bernoulli posterior predictive
- Efficient inference (conjugate updates)

Data format: Binary labels {0, 1}
"""

import warnings
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.special import betaln
from scipy.stats import beta as beta_dist

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class BetaABBernoulliParams(BDFDistributionParams):
    """Parameters for Bernoulli with Beta(α, β) prior on success probability p.

    Prior: p ~ Beta(α, β)  [concentration parameterization]
    Likelihood: y | p ~ Bernoulli(p)
    Posterior: p | y ~ Beta(α + k, β + (n-k))  where k = Σyᵢ
    Posterior predictive: P(y_new = 1 | data) = (α + k) / (α + β + n)

    Notes:
    - α (alpha_p): concentration for successes (α > 0)
    - β (beta_p): concentration for failures (β > 0)
    - E[p] = α/(α+β)
    - Var[p] = αβ/[(α+β)²(α+β+1)]
    - α = β = 1 → Uniform prior (Jeffreys prior is α = β = 0.5)
    """

    # Prior hyperparameters
    alpha_p: float = Field(default=1.0, gt=0, description="Concentration α for successes (pseudo-count of 1s)")
    beta_p: float = Field(default=1.0, gt=0, description="Concentration β for failures (pseudo-count of 0s)")

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True, description="Use Beta-Bernoulli posterior predictive (marginalizes p uncertainty)."
    )


class BetaMVBernoulliParams(BDFDistributionParams):
    """Parameters for Bernoulli with Beta prior specified via mean/variance.

    Prior: p ~ Beta(α, β) where:
    - mean_p = α/(α+β) (prior mean probability)
    - var_p = αβ/[(α+β)²(α+β+1)] (prior variance)

    Converts to: α = mean_p·m, β = (1-mean_p)·m
    where m = mean_p(1-mean_p)/var_p - 1 (equivalent sample size)

    This parameterization is more interpretable:
    - mean_p: Your prior belief about success probability
    - var_p: How uncertain you are (smaller → more concentrated)

    Example:
        mean_p=0.7, var_p=0.01
        → Prior expects 70% success rate with moderate confidence
    """

    # Prior hyperparameters (mean-variance parameterization)
    mean_p: float = Field(default=0.5, gt=0, lt=1, description="Prior mean E[p] (expected success probability)")
    var_p: float = Field(default=0.1, gt=0, description="Prior variance Var[p] (uncertainty about p)")
    raise_on_invalid_var: bool = Field(
        default=False,
        description="Raise error if var_p is invalid for given mean_p; else auto-adjust to max valid variance.",
    )

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nle")
    use_posterior_predictive: bool = Field(default=True, description="Use Beta-Bernoulli posterior predictive.")

    @model_validator(mode="after")
    def validate_var_p(self):
        """Validate variance is within valid range for Beta."""
        mean_p = self.mean_p
        var_p = self.var_p
        max_var = mean_p * (1 - mean_p)  # Maximum variance at given mean
        if var_p >= max_var:
            if self.raise_on_invalid_var:
                raise ValueError(
                    f"var_p={var_p:.4f} is too large for mean_p={mean_p:.4f}. "
                    f"Maximum allowed variance is {max_var:.4f} (occurs when α=β→0)."
                )
            else:
                warnings.warn(
                    f"var_p={var_p:.4f} is too large for mean_p={mean_p:.4f}. "
                    f"Adjusting to maximum valid variance {max_var:.4f}."
                )
                self.var_p = max_var - 1e-6  # Slightly below max to avoid edge case
        return self


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class BetaABBernoulli(BDFDistribution[BetaABBernoulliParams]):
    """Bernoulli-Beta conjugate model with concentration (α, β) parameterization.

    Supports:
    - Closed-form Bayesian evidence (NLE)
    - Beta-Bernoulli posterior predictive (integrates out p uncertainty)
    - Efficient conjugate updates

    Data format: Binary labels {0, 1}
    Example: np.array([0, 1, 0, 1, 1]) → 5 observations, 3 successes
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = BetaABBernoulliParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: dict[str, Any] | BetaABBernoulliParams):
        super().__init__(params)
        self.alpha_p = self.params.alpha_p
        self.beta_p = self.params.beta_p

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Beta posterior parameters for p.

        Returns dict with:
        - posterior_alpha: Posterior α_post = α + k (k = number of 1s)
        - posterior_beta: Posterior β_post = β + (n-k) (n-k = number of 0s)
        - posterior_prob: Posterior mean E[p | data] = α_post/(α_post + β_post)
        """
        n = data.shape[0]
        k = np.sum(data)  # Number of successes

        # Conjugate update: Beta(α, β) + Bernoulli data → Beta(α+k, β+(n-k))
        alpha_post = self.alpha_p + k
        beta_post = self.beta_p + (n - k)

        # Posterior mean probability
        prob_post = alpha_post / (alpha_post + beta_post)

        return {
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "posterior_prob": float(prob_post),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Bernoulli likelihood: log p(y | p_post).

        Uses posterior mean probability as point estimate.
        For each data point yᵢ:
        - If yᵢ=1: log(p_post)
        - If yᵢ=0: log(1 - p_post)
        """
        prob_post = params["posterior_prob"]

        # Bernoulli log-likelihood
        # log p(y=1) = log(p), log p(y=0) = log(1-p)
        log_p1 = np.log(prob_post + 1e-10)
        log_p0 = np.log(1 - prob_post + 1e-10)

        return np.where(data == 1, log_p1, log_p0)

    def _num_parameters(self) -> int:
        """Only p is estimated."""
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive (Beta-Bernoulli).

        If use_posterior_predictive=True:
            Sample p ~ Beta(α_post, β_post), then y ~ Bernoulli(p)
            Equivalent to: P(y=1) = α_post/(α_post+β_post)
        Else:
            Sample y ~ Bernoulli(p_post)
        """
        if self.params.use_posterior_predictive:
            # Posterior predictive probability
            # This marginalizes over p ~ Beta(α_post, β_post)
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]

            # Beta-Bernoulli: P(y=1 | data) = α_post/(α_post+β_post)
            # This is exactly the posterior mean, but conceptually different:
            # - Plug-in: uses p as fixed point estimate
            # - PP: integrates over uncertainty in p

            # For sampling, both give same result for Bernoulli
            # But for likelihood evaluation, PP uses different formula
            prob_predictive = alpha_post / (alpha_post + beta_post)

            rng = np.random.default_rng(random_state)
            return rng.binomial(n=1, p=prob_predictive, size=size)
        else:
            # Plug-in: Bernoulli(p_post)
            prob_post = params["posterior_prob"]
            rng = np.random.default_rng(random_state)
            return rng.binomial(n=1, p=prob_post, size=size)

    def validate_targets(self, data: np.ndarray):
        """Validate data for Bernoulli distribution."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all((data == 0) | (data == 1)):
            raise ValueError("Bernoulli data must be binary (0 or 1)")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean (expected success probability).

        Returns E[p | data] = α_post/(α_post + β_post)
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        return params["posterior_prob"]

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance (uncertainty about p).

        For Beta(α, β): Var[p] = αβ/[(α+β)²(α+β+1)]
        For posterior predictive Bernoulli: Var[y] = p(1-p)
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            # Posterior predictive variance (for y, not p)
            # Var[y] = E[p](1 - E[p]) + Var[p]
            # For Beta-Bernoulli: Var[y] = p̄(1-p̄) where p̄ = α/(α+β)
            prob = params["posterior_prob"]
            return prob * (1 - prob)
        else:
            # Variance of p itself (parameter uncertainty)
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]

            ab_sum = alpha_post + beta_post
            return (alpha_post * beta_post) / (ab_sum**2 * (ab_sum + 1))

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Bernoulli-Beta conjugate.

        p(y | prior) = ∫ p(y | p) p(p) dp

        Closed form for Beta-Bernoulli:
        log p(y) = log B(α+k, β+(n-k)) - log B(α, β)
        where B(a,b) = Γ(a)Γ(b)/Γ(a+b)

        Using log Beta function:
        log p(y) = [lnΓ(α+k) + lnΓ(β+n-k) - lnΓ(α+β+n)]
                  - [lnΓ(α) + lnΓ(β) - lnΓ(α+β)]
        """
        n = data.shape[0]
        k = np.sum(data)

        alpha_post = self.alpha_p + k
        beta_post = self.beta_p + (n - k)

        # Log evidence using Beta function
        # log B(α+k, β+n-k) - log B(α, β)
        log_ev = betaln(alpha_post, beta_post) - betaln(self.alpha_p, self.beta_p)

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Beta-Bernoulli.

        For a new observation y_new:
        P(y_new = 1 | data) = E[p | data] = α_post/(α_post + β_post)
        P(y_new = 0 | data) = 1 - E[p | data] = β_post/(α_post + β_post)

        This integrates out uncertainty in p over Beta(α_post, β_post).
        """
        alpha_post = params["posterior_alpha"]
        beta_post = params["posterior_beta"]

        ab_sum = alpha_post + beta_post

        # Predictive probabilities
        prob_1 = alpha_post / ab_sum
        prob_0 = beta_post / ab_sum

        log_prob_1 = np.log(prob_1 + 1e-10)
        log_prob_0 = np.log(prob_0 + 1e-10)

        return np.where(data == 1, log_prob_1, log_prob_0)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample p from prior Beta(α, β).

        Returns probability values p ∈ (0, 1), not binary samples.
        To get prior predictive samples, draw p ~ Beta(α, β) then y ~ Bernoulli(p).
        """
        rng = np.random.default_rng(random_state)
        return np.array(beta_dist.rvs(a=self.alpha_p, b=self.beta_p, size=size, random_state=rng))


class BetaMVBernoulli(BDFDistribution[BetaMVBernoulliParams]):
    """Bernoulli-Beta conjugate model with mean-variance parameterization.

    Same as BetaABBernoulli, but prior specified via:
    - mean_p = E[p] (expected success probability)
    - var_p = Var[p] (uncertainty about p)

    Internally converts to α, β using:
    m = mean_p(1-mean_p)/var_p - 1
    α = mean_p · m
    β = (1-mean_p) · m
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = BetaMVBernoulliParams

    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: dict[str, Any] | BetaMVBernoulliParams):
        super().__init__(params)
        self.mean_p = self.params.mean_p
        self.var_p = self.params.var_p

        # Convert mean-variance to concentration parameters
        # From Beta properties:
        # mean = α/(α+β)
        # var = αβ/[(α+β)²(α+β+1)]
        # Solving for α, β:
        m = (self.mean_p * (1 - self.mean_p) / self.var_p) - 1
        self.alpha_p = self.mean_p * m
        self.beta_p = (1 - self.mean_p) * m

    # ========================================================================
    # REQUIRED METHODS (identical to BetaABBernoulli after conversion)
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Beta posterior parameters."""
        n = data.shape[0]
        k = np.sum(data)

        alpha_post = self.alpha_p + k
        beta_post = self.beta_p + (n - k)
        prob_post = alpha_post / (alpha_post + beta_post)

        return {
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "posterior_prob": float(prob_post),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Bernoulli likelihood."""
        prob_post = params["posterior_prob"]
        log_p1 = np.log(prob_post + 1e-10)
        log_p0 = np.log(1 - prob_post + 1e-10)
        return np.where(data == 1, log_p1, log_p0)

    def _num_parameters(self) -> int:
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive or plug-in."""
        if self.params.use_posterior_predictive:
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            prob_predictive = alpha_post / (alpha_post + beta_post)
            rng = np.random.default_rng(random_state)
            return rng.binomial(n=1, p=prob_predictive, size=size)
        else:
            prob_post = params["posterior_prob"]
            rng = np.random.default_rng(random_state)
            return rng.binomial(n=1, p=prob_post, size=size)

    def validate_targets(self, data: np.ndarray):
        """Validate data."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all((data == 0.0) | (data == 1.0)):
            raise ValueError("Bernoulli data must be binary (0 or 1)")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        return params["posterior_prob"]

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        if self.params.use_posterior_predictive:
            prob = params["posterior_prob"]
            return prob * (1 - prob)
        else:
            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            ab_sum = alpha_post + beta_post
            return (alpha_post * beta_post) / (ab_sum**2 * (ab_sum + 1))

    # ========================================================================
    # OPTIONAL METHODS
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence."""
        n = data.shape[0]
        k = np.sum(data)

        alpha_post = self.alpha_p + k
        beta_post = self.beta_p + (n - k)

        log_ev = betaln(alpha_post, beta_post) - betaln(self.alpha_p, self.beta_p)
        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Beta-Bernoulli."""
        alpha_post = params["posterior_alpha"]
        beta_post = params["posterior_beta"]

        ab_sum = alpha_post + beta_post
        prob_1 = alpha_post / ab_sum
        prob_0 = beta_post / ab_sum

        log_prob_1 = np.log(prob_1 + 1e-10)
        log_prob_0 = np.log(prob_0 + 1e-10)

        return np.where(data == 1, log_prob_1, log_prob_0)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample p from prior Beta(α, β)."""
        rng = np.random.default_rng(random_state)
        return np.array(beta_dist.rvs(a=self.alpha_p, b=self.beta_p, size=size, random_state=rng))

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data."""
        if key == "mean_p":
            # Set prior mean to empirical mean of data
            return float(np.mean(data))
        else:
            raise ValueError(f"Cannot resolve 'auto' for unknown parameter '{key}'")
