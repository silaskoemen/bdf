"""Negative Binomial distribution implementations for BDF.

Provides:
- FrequentistNegativeBinomial: MLE estimation (no prior)
- NormalMeanNegativeBinomial: Normal prior on mean, MoM parameter mapping

Note: Negative Binomial requires non-negative integer count data (y ∈ {0, 1, 2, ...}).

Parameterization: NB(r, p) where:
- r > 0: number of successes (can be non-integer for overdispersion)
- p ∈ (0, 1): success probability
- Mean: μ = r(1-p)/p
- Variance: σ² = r(1-p)/p²
"""

import warnings
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.special import digamma, polygamma
from scipy.stats import nbinom

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class FrequentistNegativeBinomialParams(BDFDistributionParams):
    """Parameters for Frequentist Negative Binomial (MLE).

    No priors. Estimates r (dispersion) and p (success probability) via MLE using
    Method of Moments for initial values.
    """

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(
        default="nll", description="Frequentist model only supports NLL (plug-in)."
    )
    use_posterior_predictive: bool = Field(
        default=False, description="Posterior predictive not supported for frequentist model."
    )
    estimation_method: Literal["mom", "mle"] = Field(
        default="mom",
        description="Use closed-form Method of Moments ('mom') or fast Newton MLE ('mle').",
    )


class NormalMeanNegativeBinomialParams(BDFDistributionParams):
    """Parameters for Negative Binomial with Normal prior on the mean.

    Methodology:
    1. Place Normal(μ₀, σ₀²) prior on the population mean.
    2. Update posterior mean μ_post using Bayesian update (CLT-based).
    3. Estimate NegBinom parameters (r, p) using Method of Moments based on
       μ_post and sample variance.
    """

    prior_mean: float = Field(default=1.0, gt=0, description="Prior mean for the population mean (must be > 0).")
    prior_std: float = Field(default=1.0, gt=0, description="Prior standard deviation for the population mean.")

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nll", description="NLE not supported for this hybrid model.")
    use_posterior_predictive: bool = Field(default=False, description="Posterior predictive not currently supported.")


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class FrequentistNegativeBinomial(BDFDistribution):
    """Negative Binomial distribution estimated via MLE (Method of Moments initialization).

    y ~ NegativeBinomial(r, p)

    Uses MoM for fast closed-form estimation:
    - r = μ² / (σ² - μ)  (dispersion parameter)
    - p = μ / σ²         (success probability)

    where μ = sample mean, σ² = sample variance.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = FrequentistNegativeBinomialParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        self.validate_targets(data)
        if data.size < 2:
            raise ValueError("Negative Binomial requires at least two observations.")
        if self.params.estimation_method == "mom":
            return self._estimate_params_mom(data)
        return self._estimate_params_mle(data)

    def _ensure_overdispersion(self, mean: float, var: float) -> float:
        """Clamp variance so NB assumptions hold."""
        return float(np.maximum(var, mean + 1e-6))

    def _estimate_params_mle(self, data: np.ndarray) -> dict[str, float]:
        mean = float(np.mean(data))
        var = self._ensure_overdispersion(mean, float(np.var(data, ddof=1)))
        r = (mean**2) / (var - mean)  # MoM init
        for _ in range(5):
            grad = np.sum(digamma(data + r) - digamma(r)) + len(data) * np.log(r / (r + mean))
            hess = np.sum(polygamma(1, data + r) - polygamma(1, r)) + len(data) * (1 / r - 1 / (r + mean))
            delta = grad / (hess + 1e-12)
            r = np.maximum(r - delta, 1e-6)
            if abs(delta) < 1e-6:
                break
        p = r / (r + mean)
        return {"r": float(r), "p": float(np.clip(p, 1e-6, 1 - 1e-6))}

    def _estimate_params_mom(self, data: np.ndarray) -> dict[str, float]:
        mean = float(np.mean(data))
        var = self._ensure_overdispersion(mean, float(np.var(data, ddof=1)))
        p = mean / var
        r = (mean**2) / (var - mean)
        return {"r": float(np.maximum(r, 1e-6)), "p": float(np.clip(p, 1e-6, 1 - 1e-6))}

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Negative Binomial log-likelihood."""
        r = params["r"]
        p = params["p"]
        # scipy.stats.nbinom uses (n, p) parameterization where n=r
        return nbinom.logpmf(data, n=r, p=p)

    def _num_parameters(self) -> int:
        """r and p (2 parameters)."""
        return 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from fitted Negative Binomial."""
        r = params["r"]
        p = params["p"]
        return np.array(nbinom.rvs(n=r, p=p, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Negative Binomial."""
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Negative Binomial requires non-negative count data (y >= 0)")
        if not np.all(data == np.floor(data)):
            raise ValueError("Negative Binomial requires integer count data")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Mean of Negative Binomial: μ = r(1-p)/p."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / p

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Variance of Negative Binomial: σ² = r(1-p)/p²."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / (p**2)


class NormalMeanNegativeBinomial(BDFDistribution):
    """Negative Binomial with Normal prior on the mean (Hybrid MoM approach).

    Methodology:
    1. Place Normal(μ₀, σ₀²) prior on population mean.
    2. Bayesian update: posterior mean μ_post via precision-weighted average.
    3. Map (μ_post, sample_var) → NegBinom(r, p) via Method of Moments.

    This provides regularization on the mean while respecting overdispersion structure.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMeanNegativeBinomialParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: NormalMeanNegativeBinomialParams):
        super().__init__(params)
        self.prior_mean = params.prior_mean
        self.prior_std = params.prior_std

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate posterior mean, then solve for NegBinom params via MoM."""
        n = float(len(data))

        if n < 2:
            # Insufficient data: return prior-informed defaults
            return {"r": 1.0, "p": 0.5, "posterior_mean_mu": self.prior_mean}

        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # Negative Binomial requires variance > mean
        if sample_var <= sample_mean:
            warnings.warn(
                f"Negative Binomial requires variance > mean. "
                f"Got mean={sample_mean:.3f}, var={sample_var:.3f}. "
                f"Adding variance inflation.",
                UserWarning,
            )
            sample_var = max(sample_mean * 1.1, sample_var + 0.1)

        # 1. Bayesian update on mean (Normal-Normal conjugate, CLT approximation)
        # Precision-weighted average
        prior_prec = 1.0 / (self.prior_std**2)
        # Use sample variance as approximate likelihood variance (CLT)
        data_prec = n / sample_var
        post_prec = prior_prec + data_prec

        post_mu = (prior_prec * self.prior_mean + data_prec * sample_mean) / post_prec

        # 2. Method of Moments mapping (μ_post, sample_var) → (r, p)
        # Using posterior mean but sample variance (not updated by prior)
        p_est = post_mu / sample_var
        r_est = (post_mu**2) / (sample_var - post_mu)

        # Safety clamps
        p_est = np.clip(p_est, 1e-6, 1 - 1e-6)
        r_est = max(r_est, 1e-4)

        return {
            "r": float(r_est),
            "p": float(p_est),
            "posterior_mean_mu": float(post_mu),  # For debugging/inspection
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Negative Binomial log-likelihood."""
        r = params["r"]
        p = params["p"]
        return nbinom.logpmf(data, n=r, p=p)

    def _num_parameters(self) -> int:
        """r and p (2 parameters)."""
        return 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from fitted Negative Binomial."""
        r = params["r"]
        p = params["p"]
        return np.array(nbinom.rvs(n=r, p=p, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Negative Binomial."""
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Negative Binomial requires non-negative count data (y >= 0)")
        if not np.all(data == np.floor(data)):
            raise ValueError("Negative Binomial requires integer count data")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Mean of Negative Binomial: μ = r(1-p)/p."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / p

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Variance of Negative Binomial: σ² = r(1-p)/p²."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / (p**2)


"""
NOTE: Use gamma prior on precision to avoid instability for tiny leaves?
Principled? Good idea? Use standalone NormalMean and NormalMeanGammaVariance?
def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        self.validate_targets(data)
        n = float(len(data))
        if n < 2:
            raise ValueError("Need at least two observations for NB.")
        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1))

        # Normal-Normal update on μ
        prior_prec_mu = 1.0 / (self.prior_std**2)
        data_prec_mu = n / np.maximum(sample_var, 1e-6)
        post_prec_mu = prior_prec_mu + data_prec_mu
        post_mu = (prior_prec_mu * self.prior_mean + data_prec_mu * sample_mean) / post_prec_mu

        # Gamma prior on precision τ = 1/σ²
        alpha_post = self.params.prior_var_shape + 0.5 * (n - 1)
        beta_post = self.params.prior_var_rate + 0.5 * (n - 1) * sample_var
        post_var = 1.0 / (alpha_post / beta_post)  # E[σ² | data]

        post_var = float(np.maximum(post_var, post_mu + 1e-6))
        p_hat = post_mu / post_var
        r_hat = (post_mu**2) / (post_var - post_mu)

        return {
            "r": float(np.maximum(r_hat, 1e-6)),
            "p": float(np.clip(p_hat, 1e-6, 1 - 1e-6)),
            "posterior_mean_mu": float(post_mu),
            "posterior_mean_var": float(post_var),
        }
"""
