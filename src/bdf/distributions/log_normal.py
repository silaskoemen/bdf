"""Log-Normal distribution with conjugate prior on log-scale mean.

When log-scale precision ρ is known (or estimated first), placing a Normal prior
on μ (log-scale mean) yields exact conjugate inference.

From Fink (1997), Section 2.12.1:
- Prior: μ ~ Normal(m, 1/p)
- Likelihood: log(x) ~ Normal(μ, 1/ρ) [ρ known]
- Posterior: μ | data ~ Normal(m', 1/p')
  where p' = p + nρ, m' = (mp + nρ·x̄_log)/(p + nρ)
"""

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.stats import lognorm, norm

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class LogNormalConjugateParams(BDFDistributionParams):
    """Log-Normal with conjugate Normal prior on μ (log-scale mean).

    Assumes log-scale precision ρ is estimated from data (MLE/MoM), then
    performs conjugate Bayesian update on μ.

    Prior hyperparameters:
    - m: prior mean for μ (log-scale mean)
    - p: prior precision for μ (1/variance on log-scale)
    """

    prior_log_mean: float = Field(
        default=0.0,
        description="Prior mean m for μ (log-scale mean)",
    )
    prior_log_precision: float = Field(
        default=1.0,
        gt=0,
        description="Prior precision p for μ (1/σ² on log-scale)",
    )
    rho_estimation: Literal["mle", "mom"] = Field(
        default="mle",
        description="How to estimate log-scale precision ρ: MLE or Method of Moments",
    )

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle",
        description="Conjugate model supports exact evidence (NLE)",
    )
    use_posterior_predictive: bool = Field(
        default=True,
        description="Use log-Normal posterior predictive (marginalizes μ uncertainty)",
    )


class FrequentistLogNormalParams(BDFDistributionParams):
    """Log-Normal with MLE (no prior)."""

    score_method: Literal["nle", "nll"] = Field(
        default="nll",
        description="Frequentist model uses NLL",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="No posterior predictive for frequentist model",
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class LogNormalConjugate(BDFDistribution):
    """Log-Normal with conjugate prior on log-scale mean μ.

    Two-stage inference:
    1. Estimate log-scale precision ρ from data (MLE or MoM)
    2. Bayesian update on μ given ρ (conjugate Normal prior)

    Advantages over global log-transform:
    - Adapts log-scale variance per leaf (heteroscedastic on log-scale)
    - Exact Bayesian evidence for μ | ρ
    - Posterior predictive integrates μ uncertainty
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = LogNormalConjugateParams

    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: LogNormalConjugateParams):
        super().__init__(params)
        self.prior_m = params.prior_log_mean
        self.prior_p = params.prior_log_precision
        self.rho_estimation = params.rho_estimation

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Two-stage estimation:
        1. Estimate ρ (log-scale precision)
        2. Conjugate update on μ given ρ
        """
        self.validate_targets(data)

        n = len(data)
        log_data = np.log(data)
        mean_log = np.mean(log_data)

        # Stage 1: Estimate ρ (log-scale precision = 1/σ²_log)
        if self.rho_estimation == "mle":
            var_log = np.var(log_data, ddof=1)
            rho = 1.0 / np.maximum(var_log, 1e-6)
        else:  # mom
            var_log = np.var(log_data, ddof=0)  # MoM uses biased estimator
            rho = 1.0 / np.maximum(var_log, 1e-6)

        # Stage 2: Conjugate Bayesian update on μ (Fink 1997, Eq 2.53)
        # Posterior: μ | data ~ Normal(m', 1/p')
        p_post = self.prior_p + n * rho
        m_post = (self.prior_m * self.prior_p + n * rho * mean_log) / p_post

        return {
            "posterior_log_mean": float(m_post),
            "posterior_log_precision": float(p_post),
            "estimated_rho": float(rho),
            "sigma_log": float(np.sqrt(1.0 / rho)),  # For convenience
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Log-Normal likelihood using posterior mean μ_post."""
        mu_log = params["posterior_log_mean"]
        sigma_log = params["sigma_log"]

        # scipy.stats.lognorm uses (s, scale=exp(μ)) parameterization
        # s = σ_log, scale = exp(μ_log)
        return lognorm.logpdf(data, s=sigma_log, scale=np.exp(mu_log))

    def _num_parameters(self) -> int:
        """Two parameters: μ_log and σ_log (even though only μ has prior)."""
        return 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from log-Normal posterior predictive.

        Two sources of uncertainty:
        1. μ ~ Normal(m_post, 1/p_post) [parameter uncertainty]
        2. x | μ ~ LogNormal(μ, 1/ρ) [data uncertainty]
        """
        m_post = params["posterior_log_mean"]
        p_post = params["posterior_log_precision"]
        sigma_log = params["sigma_log"]

        rng = np.random.default_rng(random_state)

        # Sample μ from posterior
        mu_samples = rng.normal(
            loc=m_post,
            scale=np.sqrt(1.0 / p_post),
            size=size,
        )

        # Sample x | μ from log-Normal
        samples = rng.lognormal(
            mean=mu_samples,
            sigma=sigma_log,
            size=size,
        )

        return samples

    def validate_targets(self, data: np.ndarray):
        """Log-Normal requires strictly positive data."""
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data > 0):
            raise ValueError("Log-Normal requires strictly positive data (y > 0)")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Mean of log-Normal: exp(μ + σ²/2)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        mu = params["posterior_log_mean"]
        sigma = params["sigma_log"]

        return float(np.exp(mu + 0.5 * sigma**2))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Variance of log-Normal: (exp(σ²) - 1) * exp(2μ + σ²)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        mu = params["posterior_log_mean"]
        sigma = params["sigma_log"]

        return float((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for μ | ρ (Fink 1997).

        Evidence integrates out μ uncertainty:
        p(x | m, p, ρ) = ∫ p(x | μ, ρ) p(μ | m, p) dμ

        This is Normal-Normal conjugate on log-scale, so evidence is:
        log p(x) = const - (n/2) log(2π/ρ) - (1/2) log(p'/p) - (ρ/2) RSS

        where RSS = Σ(log(xᵢ) - x̄_log)² + n·p/(p+nρ)·(x̄_log - m)²
        """
        self.validate_targets(data)

        n = len(data)
        log_data = np.log(data)
        mean_log = np.mean(log_data)

        # Estimate ρ
        if self.rho_estimation == "mle":
            var_log = np.var(log_data, ddof=1)
            rho = 1.0 / np.maximum(var_log, 1e-6)
        else:
            var_log = np.var(log_data, ddof=0)
            rho = 1.0 / np.maximum(var_log, 1e-6)

        # Posterior precision
        p_post = self.prior_p + n * rho

        # Evidence terms (on log-scale)
        # 1. Normalization for data likelihood
        log_ev = -0.5 * n * np.log(2 * np.pi / rho)

        # 2. Prior-to-posterior precision ratio
        log_ev += -0.5 * np.log(p_post / self.prior_p)

        # 3. Residual sum of squares (on log-scale)
        ss_data = np.sum((log_data - mean_log) ** 2)
        ss_prior = (n * self.prior_p / p_post) * (mean_log - self.prior_m) ** 2
        log_ev += -0.5 * rho * (ss_data + ss_prior)

        # 4. Jacobian for log-transform: -Σ log(xᵢ)
        log_ev += -np.sum(np.log(data))

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive integrates μ uncertainty.

        For new data x_new:
        p(x_new | data) = ∫ LogNormal(x_new | μ, ρ) Normal(μ | m_post, p_post) dμ

        This is log-Normal with inflated variance:
        log(x_new) ~ t-distribution or Normal with variance (1/ρ + 1/p_post)

        Approximation: Use Normal on log-scale with combined variance.
        """
        m_post = params["posterior_log_mean"]
        p_post = params["posterior_log_precision"]
        rho = params["estimated_rho"]

        # Combined variance: data variance + parameter variance
        var_combined = (1.0 / rho) + (1.0 / p_post)
        sigma_combined = np.sqrt(var_combined)

        log_data = np.log(data)

        # Log-Normal likelihood on original scale with combined variance
        log_lik = norm.logpdf(log_data, loc=m_post, scale=sigma_combined)

        # Jacobian correction: -log(x)
        log_lik -= np.log(data)

        return log_lik


class FrequentistLogNormal(BDFDistribution):
    """Log-Normal with MLE (no prior)."""

    params_cls: ClassVar[type[BDFDistributionParams]] = FrequentistLogNormalParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """MLE for log-Normal: μ̂ = mean(log(x)), σ̂² = var(log(x))."""
        self.validate_targets(data)

        log_data = np.log(data)
        mu_log = np.mean(log_data)
        sigma_log = np.std(log_data, ddof=1)

        return {
            "mu_log": float(mu_log),
            "sigma_log": float(sigma_log),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Log-Normal likelihood."""
        mu_log = params["mu_log"]
        sigma_log = params["sigma_log"]

        return lognorm.logpdf(data, s=sigma_log, scale=np.exp(mu_log))

    def _num_parameters(self) -> int:
        return 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from fitted log-Normal."""
        mu_log = params["mu_log"]
        sigma_log = params["sigma_log"]

        return np.array(lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data > 0):
            raise ValueError("Log-Normal requires strictly positive data (y > 0)")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        mu = params["mu_log"]
        sigma = params["sigma_log"]
        return float(np.exp(mu + 0.5 * sigma**2))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        mu = params["mu_log"]
        sigma = params["sigma_log"]
        return float((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
