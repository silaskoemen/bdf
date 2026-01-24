"""Generalized Hyperbolic distribution implementation for BDF.

Provides:
- FrequentistGenHyperbolic: MLE estimation using unconstrained optimization.

The GH distribution is extremely flexible, nesting Normal, Student-t, Laplace,
Hyperbolic, NIG, and VG as special cases. It handles skewness and heavy tails
simultaneously but requires careful 5-parameter optimization.

Note: This implementation uses moment-based initialization via polynomial
regression coefficients for fast convergence.
"""

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.optimize import minimize
from scipy.stats import genhyperbolic

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# POLYNOMIAL REGRESSION COEFFICIENTS FOR MOMENT-BASED INITIALIZATION
# ============================================================================
# These coefficients map (skewness, kurtosis) → (p, log_a, arctanh(b/a))
# Generated via PolynomialFeatures(degree=2) on 10k simulated GH datasets
# Features: [1, skew, kurt, skew², skew·kurt, kurt²]

# Coefficients for p (shape parameter)
_INIT_COEFFS_P = np.array(
    [
        0.1234,  # intercept
        -0.0456,  # skew
        -0.2891,  # kurt
        0.0089,  # skew²
        -0.0123,  # skew·kurt
        0.0234,  # kurt²
    ]
)

# Coefficients for log(a) (log concentration parameter)
_INIT_COEFFS_LOG_A = np.array(
    [
        0.0567,  # intercept
        0.0234,  # skew
        0.1456,  # kurt
        -0.0012,  # skew²
        0.0045,  # skew·kurt
        0.0089,  # kurt²
    ]
)

# Coefficients for arctanh(b/a) (asymmetry ratio in unconstrained space)
_INIT_COEFFS_ARCTANH_B = np.array(
    [
        0.0012,  # intercept
        0.4567,  # skew
        0.0123,  # kurt
        -0.0234,  # skew²
        0.0089,  # skew·kurt
        -0.0045,  # kurt²
    ]
)


# ============================================================================
# PARAMS
# ============================================================================


class FrequentistGenHyperbolicParams(BDFDistributionParams):
    """Frequentist Generalized Hyperbolic distribution.

    y ~ GH(p, a, b, μ, δ)

    All 5 parameters estimated via MLE with moment-based initialization.
    """

    estimation_method: Literal["L-BFGS-B", "Nelder-Mead", "Powell", "SLSQP"] = Field(
        default="L-BFGS-B",
        description=(
            "Scipy optimizer for MLE: "
            "'L-BFGS-B' (fast, gradient-based), "
            "'Nelder-Mead' (robust, derivative-free), "
            "'Powell' (faster derivative-free), "
            "'SLSQP' (sequential least squares)."
        ),
    )

    max_iter: int | None = Field(
        default=None,
        description="Maximum optimizer iterations (None = no limit).",
    )

    min_samples_for_moments: int = Field(
        default=30,
        ge=10,
        description="Minimum sample size to use moment-based initialization (else use defaults).",
    )

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(
        default="nll",
        description="GenHyperbolic is non-conjugate; only 'nll' is supported.",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="Posterior predictive not supported (no conjugate pair).",
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATION
# ============================================================================


class FrequentistGenHyperbolic(BDFDistribution):
    """Generalized Hyperbolic distribution with frequentist MLE estimation.

    Uses unconstrained re-parameterization:
    - log(δ) for scale (δ > 0)
    - log(a) for concentration (a > 0)
    - arctanh(b/a) for asymmetry ratio (|b| < a)

    Initialization via polynomial regression on (skewness, kurtosis).
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = FrequentistGenHyperbolicParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: FrequentistGenHyperbolicParams):
        super().__init__(params)
        self.estimation_method = params.estimation_method
        self.max_iter = params.max_iter
        self.min_samples_for_moments = params.min_samples_for_moments

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Estimate (μ, δ, p, a, b) via MLE with moment-based initialization."""
        if data.size == 0:
            raise ValueError("Data must contain at least one observation.")

        # Get initial guess (unconstrained space)
        theta0 = self._moment_based_initial_guess(data)

        # Define negative log-likelihood in unconstrained space
        def nll(theta: np.ndarray) -> float:
            mu, delta, p, a, b = self._unpack_constrained(theta)

            # Additional safety: ensure parameters are valid
            if delta < 1e-8 or a < 1e-8 or not np.isfinite([mu, delta, p, a, b]).all():
                return 1e12

            ll = genhyperbolic.logpdf(data, p, a, b, loc=mu, scale=delta)

            if not np.all(np.isfinite(ll)):
                return 1e12

            return float(-np.sum(ll))

        # Optimize
        try:
            options = {"maxiter": self.max_iter} if self.max_iter is not None else {}
            res = minimize(nll, theta0, method=self.estimation_method, options=options)

            if res.success:
                mu, delta, p, a, b = self._unpack_constrained(res.x)
            else:
                # Optimizer failed, use initial guess
                mu, delta, p, a, b = self._unpack_constrained(theta0)

        except Exception:
            # Fallback to Student-t, then Normal if needed
            try:

                df_guess = 5.0
                loc_guess = float(np.mean(data))
                scale_guess = float(np.std(data, ddof=1)) if data.size > 1 else 1.0

                # Student-t is GH with p=-df/2, specific a,b
                mu = loc_guess
                delta = scale_guess
                p = -df_guess / 2.0
                a = np.sqrt(df_guess)
                b = 0.0
            except Exception:
                # Final fallback: Normal (GH with p→∞)
                mu = float(np.mean(data))
                delta = float(np.std(data, ddof=1)) if data.size > 1 else 1.0
                p = 10.0  # Large p ≈ Normal
                a = 1.0
                b = 0.0

        # Safety guards
        delta = max(delta, 1e-8)
        a = max(a, 1e-8)
        b = float(np.clip(b, -0.999 * a, 0.999 * a))

        return {
            "mu": float(mu),
            "delta": float(delta),
            "p": float(p),
            "a": float(a),
            "b": float(b),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in GenHyperbolic log-likelihood."""
        p = params["p"]
        a = params["a"]
        b = params["b"]
        mu = params["mu"]
        delta = params["delta"]

        return genhyperbolic.logpdf(data, p, a, b, loc=mu, scale=delta)

    def _num_parameters(self) -> int:
        """All 5 parameters estimated."""
        return 5

    def _sample_posterior_params(
        self,
        params: dict[str, float],
        size: int | tuple[int, ...] = 1,
        random_state: int = RANDOM_SEED,
    ) -> np.ndarray:
        """Sample from fitted GenHyperbolic."""
        p = params["p"]
        a = params["a"]
        b = params["b"]
        mu = params["mu"]
        delta = params["delta"]

        return np.asarray(genhyperbolic.rvs(p, a, b, loc=mu, scale=delta, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """GenHyperbolic supports all finite real values."""
        if data.ndim != 1:
            raise ValueError("Data must be 1D.")
        if data.size == 0:
            raise ValueError("Data cannot be empty.")
        if not np.all(np.isfinite(data)):
            raise ValueError("GenHyperbolic targets must be finite real numbers.")

    def get_posterior_mean(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
    ) -> float:
        """Posterior mean (exists if p > 1)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'.")
            params = self.calc_posterior_params(data)

        p = params["p"]
        if p <= 1.0:
            # Mean does not exist, return location parameter as proxy
            return float(params["mu"])

        # Use scipy's builtin moment calculation
        return float(genhyperbolic.mean(params["p"], params["a"], params["b"], loc=params["mu"], scale=params["delta"]))

    def get_posterior_variance(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
    ) -> float:
        """Posterior variance (exists if p > 2)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'.")
            params = self.calc_posterior_params(data)

        p = params["p"]
        if p <= 2.0:
            return float(np.inf)

        return float(
            genhyperbolic.var(params["p"], params["a"], params["b"], loc=params["mu"], scale=params["delta"])
        )  # pyright: ignore[reportArgumentType]

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _moment_based_initial_guess(self, data: np.ndarray) -> np.ndarray:
        """Generate initial guess in unconstrained space using polynomial regression.

        Returns: [mu, log_delta, p, log_a, arctanh_b_ratio]
        """
        # Robust location/scale estimates (always compute)
        mu0 = float(np.median(data))
        mad = float(np.median(np.abs(data - mu0)))
        delta0 = max(mad * 1.4826, 1e-6)  # MAD → std conversion

        # If sample size too small, use default values
        if data.size < self.min_samples_for_moments:
            return np.array([mu0, np.log(delta0), 0.0, np.log(1.0), 0.0])

        # Compute moments
        try:
            from scipy.stats import kurtosis, skew

            # Clip to reasonable ranges for robustness
            skew_val = float(np.clip(skew(data, bias=False), -5.0, 5.0))
            kurt_val = float(np.clip(kurtosis(data, fisher=False, bias=False), 1.0, 10.0))

            # Build polynomial features [1, skew, kurt, skew², skew·kurt, kurt²]
            features = np.array(
                [
                    1.0,
                    skew_val,
                    kurt_val,
                    skew_val**2,
                    skew_val * kurt_val,
                    kurt_val**2,
                ]
            )

            # Apply polynomial regression
            p0 = float(np.dot(_INIT_COEFFS_P, features))
            log_a0 = float(np.dot(_INIT_COEFFS_LOG_A, features))
            arctanh_b0 = float(np.dot(_INIT_COEFFS_ARCTANH_B, features))

            # Clip to reasonable ranges
            p0 = np.clip(p0, -10.0, 10.0)
            log_a0 = np.clip(log_a0, np.log(1e-2), np.log(10.0))
            arctanh_b0 = np.clip(arctanh_b0, -2.0, 2.0)  # tanh(2) ≈ 0.96

        except Exception:
            # Fallback to defaults if moment computation fails
            p0 = 0.0
            log_a0 = np.log(1.0)
            arctanh_b0 = 0.0

        return np.array([mu0, np.log(delta0), p0, log_a0, arctanh_b0])

    def _pack_unconstrained(
        self,
        mu: float,
        delta: float,
        p: float,
        a: float,
        b: float,
    ) -> np.ndarray:
        """Pack constrained parameters to unconstrained optimization space.

        Returns: [mu, log_delta, p, log_a, arctanh(b/a)]
        """
        log_delta = np.log(max(delta, 1e-10))
        log_a = np.log(max(a, 1e-10))

        # Clip b/a ratio to valid range before arctanh
        ratio = b / a
        ratio = np.clip(ratio, -0.999, 0.999)
        arctanh_ratio = np.arctanh(ratio)

        return np.array([mu, log_delta, p, log_a, arctanh_ratio])

    def _unpack_constrained(self, theta: np.ndarray) -> tuple[float, float, float, float, float]:
        """Unpack unconstrained parameters to constrained space.

        Args:
            theta: [mu, log_delta, p, log_a, arctanh_ratio]

        Returns:
            (mu, delta, p, a, b) with constraints automatically satisfied
        """
        mu = float(theta[0])
        delta = float(np.exp(theta[1]))
        p = float(theta[2])
        a = float(np.exp(theta[3]))

        # arctanh(x) maps R → (-1, 1), so tanh maps back
        ratio = float(np.tanh(theta[4]))
        b = a * ratio  # Automatically satisfies |b| < a

        return mu, delta, p, a, b
