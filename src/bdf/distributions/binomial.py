"""Beta-Binomial distribution implementation for BDF.

Provides:
- BetaBinomial: Conjugate Bayesian model for binomial data (fixed n trials)

Note: Beta-Binomial requires count data with known number of trials.

Parameterization: Beta(α, β) prior on p, Binomial(n, p) likelihood
- Prior: p ~ Beta(α, β)
- Likelihood: k | p ~ Binomial(n, p)
- Posterior: p | k ~ Beta(α + k, β + n - k)
- Posterior predictive: Beta-Binomial(n, α, β)
"""

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.special import betaln, gammaln
from scipy.stats import beta, betabinom

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class BetaBinomialParams(BDFDistributionParams):
    """Parameters for Beta-Binomial conjugate model.

    Prior: p ~ Beta(α, β)
    Likelihood: k ~ Binomial(n, p) where n is known
    Posterior: p | k ~ Beta(α + k, β + n - k)
    Posterior predictive: Beta-Binomial(n, α, β)
    """

    # Prior hyperparameters
    alpha: float = Field(default=1.0, gt=0, description="Prior alpha parameter (prior successes)")
    beta: float = Field(default=1.0, gt=0, description="Prior beta parameter (prior failures)")
    n_trials: int = Field(default=1, gt=0, description="Number of trials per observation")

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True, description="Use Beta-Binomial posterior predictive (marginalizes p uncertainty)."
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATION
# ============================================================================


class BetaBinomial(BDFDistribution):
    r"""Beta-Binomial conjugate model for binomial count data.

    **Usage:** ``dist="BetaBinomial"``

    **Model:**

    *   **Prior:** :math:`p \sim \text{Beta}(\alpha, \beta)`
    *   **Likelihood:** :math:`k \mid p \sim \text{Binomial}(n, p)`
    *   **Posterior:** :math:`p \mid k \sim \text{Beta}(\alpha + \sum k_i, \beta + nN - \sum k_i)`

    Parameters
    ----------
    alpha : float, default=1.0
        Prior alpha parameter (prior successes).
    beta : float, default=1.0
        Prior beta parameter (prior failures).
    n_trials : int, default=1
        Number of trials per observation. Set to 1 for Bernoulli-like data.

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    BetaABBernoulli : Specialized for binary (Bernoulli) outcomes.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = BetaBinomialParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: BetaBinomialParams):
        super().__init__(params)
        self.prior_alpha = params.alpha
        self.prior_beta = params.beta
        self.n_trials = params.n_trials

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Beta posterior parameters.

        Returns dict with:
        - posterior_alpha: α + Σk_i
        - posterior_beta: β + n*N - Σk_i
        """
        self.validate_targets(data)

        n_obs = len(data)
        sum_successes = float(np.sum(data))

        posterior_alpha = self.prior_alpha + sum_successes
        posterior_beta = self.prior_beta + (self.n_trials * n_obs) - sum_successes

        return {
            "posterior_alpha": float(posterior_alpha),
            "posterior_beta": float(posterior_beta),
            "n_trials": float(self.n_trials),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Binomial likelihood using posterior mode of p.

        For Beta(α, β), mode = (α - 1)/(α + β - 2) if α, β > 1
        Otherwise use mean = α/(α + β)
        """
        alpha = params["posterior_alpha"]
        beta_param = params["posterior_beta"]
        n = int(params["n_trials"])

        # Use mode if well-defined, else mean
        if alpha > 1 and beta_param > 1:
            p_hat = (alpha - 1) / (alpha + beta_param - 2)
        else:
            p_hat = alpha / (alpha + beta_param)

        # Binomial log-likelihood
        # log p(k | n, p) = log C(n,k) + k*log(p) + (n-k)*log(1-p)
        log_binom_coef = gammaln(n + 1) - gammaln(data + 1) - gammaln(n - data + 1)
        log_lik = log_binom_coef + data * np.log(p_hat + 1e-10) + (n - data) * np.log(1 - p_hat + 1e-10)

        return log_lik

    def _num_parameters(self) -> int:
        """Only p is estimated (n is known)."""
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from Beta-Binomial posterior predictive.

        For each sample:
        1. Draw p ~ Beta(α_post, β_post)
        2. Draw k ~ Binomial(n, p)

        Or use scipy's betabinom directly.
        """
        alpha = params["posterior_alpha"]
        beta_param = params["posterior_beta"]
        n = int(params["n_trials"])

        # Use scipy's Beta-Binomial (parameterized as n, a, b)
        return np.array(betabinom.rvs(n=n, a=alpha, b=beta_param, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Beta-Binomial."""
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Beta-Binomial requires non-negative count data (k >= 0)")
        if not np.all(data <= self.n_trials):
            raise ValueError(f"Beta-Binomial requires k <= n_trials={self.n_trials}")
        if not np.all(data == np.floor(data)):
            raise ValueError("Beta-Binomial requires integer count data")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean of p: E[p | data] = α/(α + β)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        beta_param = params["posterior_beta"]
        return alpha / (alpha + beta_param)

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance of p: Var[p | data] = αβ/[(α+β)²(α+β+1)]."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        beta_param = params["posterior_beta"]

        return (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Beta-Binomial conjugate.

        p(k_1, ..., k_N | α, β, n) = ∏ᵢ [C(n, kᵢ) * B(α + kᵢ, β + n - kᵢ) / B(α, β)]

        where B(a,b) = Γ(a)Γ(b)/Γ(a+b) is the Beta function.

        Log evidence:
        log p(data) = Σᵢ [log C(n, kᵢ) + log B(α + kᵢ, β + n - kᵢ) - log B(α, β)]
        """
        self.validate_targets(data)

        n = self.n_trials
        alpha = self.prior_alpha
        beta_param = self.prior_beta

        # Log binomial coefficients: log C(n, k)
        log_binom_coef = gammaln(n + 1) - gammaln(data + 1) - gammaln(n - data + 1)

        # Log Beta function: log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a+b)
        # For each observation: log B(α + k, β + n - k)
        log_beta_post = betaln(alpha + data, beta_param + n - data)

        # Prior log Beta: log B(α, β)
        log_beta_prior = betaln(alpha, beta_param)

        # Sum over all observations
        log_ev = np.sum(log_binom_coef + log_beta_post - log_beta_prior)

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Beta-Binomial posterior predictive: integrates out p uncertainty.

        p(k_new | k_obs) = ∫ Binomial(k_new | n, p) * Beta(p | α_post, β_post) dp
                         = Beta-Binomial(k_new | n, α_post, β_post)
        """
        alpha = params["posterior_alpha"]
        beta_param = params["posterior_beta"]
        n = int(params["n_trials"])

        # scipy's betabinom.logpmf(k, n, a, b)
        return betabinom.logpmf(data, n=n, a=alpha, b=beta_param)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample p from prior Beta(α, β)."""
        return np.array(beta.rvs(a=self.prior_alpha, b=self.prior_beta, size=size, random_state=random_state))
