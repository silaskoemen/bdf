# filepath:
"""Weibull distribution implementations for BDF.

Provides:
- FrequentistWeibull: MLE estimation (no prior).
- NormalMeanWeibull: Normal prior on the mean, mapped to Weibull via Method of Moments.

Note: Weibull requires strictly positive data (y > 0).
"""
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.stats import weibull_min

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class FrequentistWeibullParams(BDFDistributionParams):
    """Parameters for Frequentist Weibull (MLE).

    No priors. Estimates shape (k) and scale (λ) via Maximum Likelihood.
    """

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(
        default="nll", description="Frequentist model only supports NLL (plug-in)."
    )
    use_posterior_predictive: bool = Field(
        default=False, description="Posterior predictive not supported for frequentist model."
    )


class NormalMeanWeibullParams(BDFDistributionParams):
    """Parameters for Weibull with Normal prior on the mean.

    Methodology:
    1. Place Normal(μ_0, σ_0) prior on the population mean.
    2. Update posterior mean μ_post using data (assuming CLT/Normal likelihood for the mean).
    3. Estimate Weibull parameters (k, λ) using Method of Moments based on
       μ_post and sample variance.
    """

    prior_mean: float = Field(default=1.0, description="Prior mean for the population mean.")
    prior_std: float = Field(default=1.0, gt=0, description="Prior standard deviation for the population mean.")

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nll", description="NLE not supported for this hybrid model.")
    use_posterior_predictive: bool = Field(default=False, description="Posterior predictive not currently supported.")


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class FrequentistWeibull(BDFDistribution):
    """Weibull distribution estimated via MLE.

    y ~ Weibull(k, λ)
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = FrequentistWeibullParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Estimate k (shape) and λ (scale) via fast 1D MLE profile likelihood."""
        n = len(data)
        if n < 2:
            return {"k": 1.0, "lambda": 1.0}

        # Precompute logs for efficiency
        log_data = np.log(data)
        mean_log_data = np.mean(log_data)

        # The derivative of the profile log-likelihood w.r.t k
        # We want to find the root of this equation
        def mle_equation(k):
            # Safety for optimizer exploring negative/zero k
            if k <= 1e-6:
                return 1e9

            # To avoid overflow with large k, we can work with scaled data or careful ops
            # But for standard ranges, direct computation is usually fine.
            # equation: 1/k + mean(ln x) - (sum(x^k ln x) / sum(x^k)) = 0

            # Using numpy vectorization
            data_pow_k = data**k
            # Check for overflow
            if not np.all(np.isfinite(data_pow_k)):
                return 1e9

            numerator = np.sum(data_pow_k * log_data)
            denominator = np.sum(data_pow_k)

            return (1.0 / k) + mean_log_data - (numerator / denominator)

        # Initial guess: 1.0 is a safe starting point for shape
        # Could also use Heuristic: k ~ 1.28 / std(log(data))
        k_guess = 1.28 / np.std(log_data) if np.std(log_data) > 0 else 1.0

        # Solve for k
        k_hat_array = fsolve(mle_equation, x0=k_guess)
        k_hat = float(k_hat_array[0])

        # Safety clamp
        k_hat = max(k_hat, 1e-4)

        # Solve for lambda using closed form given k
        # lambda = (mean(x^k))^(1/k)
        lambda_hat = float(np.mean(data**k_hat) ** (1.0 / k_hat))

        return {
            "k": k_hat,
            "lambda": lambda_hat,
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Weibull log-likelihood."""
        k = params["k"]
        lam = params["lambda"]
        return weibull_min.logpdf(data, c=k, scale=lam)

    def _num_parameters(self) -> int:
        return 2

    def _sample_posterior_params(
        self, params: dict[str, float], size: int = 1, random_state: int = RANDOM_SEED
    ) -> np.ndarray:
        """Sample from fitted Weibull."""
        k = params["k"]
        lam = params["lambda"]
        return np.array(weibull_min.rvs(c=k, scale=lam, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data > 0):
            raise ValueError("Weibull requires strictly positive data (y > 0)")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        if params is None:
            params = self.calc_posterior_params(data)  # type: ignore
        k = params["k"]
        lam = params["lambda"]
        return lam * gamma(1 + 1 / k)

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        if params is None:
            params = self.calc_posterior_params(data)  # type: ignore
        k = params["k"]
        lam = params["lambda"]
        return (lam**2) * (gamma(1 + 2 / k) - (gamma(1 + 1 / k)) ** 2)


class NormalMeanWeibull(BDFDistribution):
    """Weibull distribution with Normal prior on the mean (Hybrid MoM).

    1. Updates prior on Mean (Normal-Normal update).
    2. Maps Posterior Mean + Sample Variance -> Weibull(k, λ).
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMeanWeibullParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: NormalMeanWeibullParams):
        super().__init__(params)
        self.prior_mean = params.prior_mean
        self.prior_std = params.prior_std

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate posterior mean, then solve for Weibull params."""
        n = float(len(data))

        # 1. Bayesian Update on the Mean (Normal-Normal)
        # We approximate likelihood variance using sample variance
        if n > 1:
            sample_mean = np.mean(data)
            sample_var = np.var(data, ddof=1)
            sample_var = max(sample_var, 1e-6)  # Safety floor
        else:
            # Fallback for n < 2
            return {"k": 1.0, "lambda": self.prior_mean}

        # Precision weighted average
        prior_prec = 1.0 / (self.prior_std**2)
        data_prec = n / sample_var
        post_prec = prior_prec + data_prec

        post_mu = (prior_prec * self.prior_mean + data_prec * sample_mean) / post_prec

        # 2. Method of Moments Mapping
        # CV^2 = Var / Mean^2
        # For Weibull: CV^2 = [Γ(1+2/k) / Γ(1+1/k)^2] - 1
        # We use the posterior mean (post_mu) and sample variance (sample_var)
        target_cv2 = sample_var / (post_mu**2)

        # Solve for k
        def cv_equation(k):
            if k <= 0:
                return 1e9
            g1 = gamma(1 + 1 / k)
            g2 = gamma(1 + 2 / k)
            return (g2 / (g1**2)) - 1 - target_cv2

        # Initial guess k=1.5 is reasonable for many datasets
        k_est = fsolve(cv_equation, x0=1.5)[0]
        k_est = max(k_est, 1e-4)  # Safety floor

        # Solve for lambda: μ = λ * Γ(1 + 1/k)  =>  λ = μ / Γ(1 + 1/k)
        lambda_est = post_mu / gamma(1 + 1 / k_est)

        return {
            "k": float(k_est),
            "lambda": float(lambda_est),
            "posterior_mean_mu": float(post_mu),  # Stored for debugging/inspection
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        k = params["k"]
        lam = params["lambda"]
        return weibull_min.logpdf(data, c=k, scale=lam)

    def _num_parameters(self) -> int:
        return 2

    def _sample_posterior_params(
        self, params: dict[str, float], size: int = 1, random_state: int = RANDOM_SEED
    ) -> np.ndarray:
        k = params["k"]
        lam = params["lambda"]
        return np.array(weibull_min.rvs(c=k, scale=lam, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data > 0):
            raise ValueError("Weibull requires strictly positive data (y > 0)")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        if params is None:
            params = self.calc_posterior_params(data)  # type: ignore
        k = params["k"]
        lam = params["lambda"]
        return lam * gamma(1 + 1 / k)

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        if params is None:
            params = self.calc_posterior_params(data)  # type: ignore
        k = params["k"]
        lam = params["lambda"]
        return (lam**2) * (gamma(1 + 2 / k) - (gamma(1 + 1 / k)) ** 2)
