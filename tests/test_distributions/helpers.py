"""Shared helpers and constants for distribution tests.

This module provides:
- Tolerance constants for numerical comparisons
- Helper functions for comparing values
- Reference implementations for verification
- Hypothesis strategies for property-based testing

Note: Pytest fixtures are in conftest.py (auto-discovered).
"""

from typing import Any

import numpy as np
from hypothesis import strategies as st
from scipy import stats

# ============================================================================
# TOLERANCE CONSTANTS
# ============================================================================

# For exact Python-Rust equivalence (same formula, same data)
RTOL_TIGHT = 1e-10
ATOL_TIGHT = 1e-14

# For numerical stability tests (may have some floating point error)
RTOL_NUMERICAL = 1e-6
ATOL_NUMERICAL = 1e-10

# For statistical tests (empirical vs theoretical moments)
RTOL_STATISTICAL = 0.1
ATOL_STATISTICAL = 0.05


# ============================================================================
# HYPOTHESIS STRATEGIES
# ============================================================================


@st.composite
def normal_data_strategy(draw, min_size: int = 2, max_size: int = 1000):
    """Generate normally distributed data for testing."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    mean = draw(st.floats(min_value=-100, max_value=100))
    std = draw(st.floats(min_value=0.1, max_value=10))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


@st.composite
def prior_params_strategy(draw):
    """Generate valid prior parameters for NormalMuNormal."""
    mu_mu = draw(st.floats(min_value=-100, max_value=100))
    sigma_mu = draw(st.floats(min_value=0.01, max_value=100))
    return {"mu_mu": mu_mu, "sigma_mu": sigma_mu}


@st.composite
def invgamma_prior_params_strategy(draw):
    """Generate valid prior parameters for NormalMuInvGammaSigmaNormal."""
    mu_mu = draw(st.floats(min_value=-100, max_value=100))
    n_mu = draw(st.floats(min_value=0.1, max_value=100))
    nu_sigma = draw(st.floats(min_value=3.0, max_value=100))
    phi_sigma = draw(st.floats(min_value=0.1, max_value=100))
    return {
        "mu_mu": mu_mu,
        "n_mu": n_mu,
        "nu_sigma": nu_sigma,
        "phi_sigma": phi_sigma,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def assert_close(
    actual: float, expected: float, rtol: float = RTOL_TIGHT, atol: float = ATOL_TIGHT, msg: str = ""
) -> None:
    """Assert two values are close with helpful error message."""
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        rel_err = abs(actual - expected) / (abs(expected) + 1e-15)
        abs_err = abs(actual - expected)
        raise AssertionError(
            f"{msg}\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Rel err:  {rel_err:.2e} (tol: {rtol:.2e})\n"
            f"  Abs err:  {abs_err:.2e} (tol: {atol:.2e})"
        )


def assert_array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = RTOL_TIGHT,
    atol: float = ATOL_TIGHT,
    msg: str = "",
) -> None:
    """Assert two arrays are close with helpful error message."""
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        max_rel_err = np.max(np.abs(actual - expected) / (np.abs(expected) + 1e-15))
        max_abs_err = np.max(np.abs(actual - expected))
        worst_idx = np.argmax(np.abs(actual - expected))
        raise AssertionError(
            f"{msg}\n"
            f"  Max rel err: {max_rel_err:.2e} (tol: {rtol:.2e})\n"
            f"  Max abs err: {max_abs_err:.2e} (tol: {atol:.2e})\n"
            f"  Worst index: {worst_idx}\n"
            f"  Expected[{worst_idx}]: {expected.flat[worst_idx]}\n"
            f"  Actual[{worst_idx}]:   {actual.flat[worst_idx]}"
        )


def assert_dict_close(
    actual: dict[str, float],
    expected: dict[str, float],
    rtol: float = RTOL_TIGHT,
    atol: float = ATOL_TIGHT,
    msg: str = "",
) -> None:
    """Assert two dicts have close values for common keys."""
    common_keys = set(actual.keys()) & set(expected.keys())
    for key in common_keys:
        assert_close(actual[key], expected[key], rtol=rtol, atol=atol, msg=f"{msg} (key: {key})")


# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================


def reference_normal_posterior_mean(data: np.ndarray, mu_mu: float, sigma_mu: float) -> float:
    """Reference implementation for Normal-Normal posterior mean.

    Formula: μ_posterior = (σ²μ₀ + n*σ_μ²*x̄) / (σ² + n*σ_μ²)
    where σ² is the sample variance.
    """
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1) if n > 1 else 1e-10

    precision_prior = 1 / (sigma_mu**2)
    precision_data = n / sample_var

    return (precision_prior * mu_mu + precision_data * sample_mean) / (precision_prior + precision_data)


def reference_normal_posterior_variance(data: np.ndarray, sigma_mu: float) -> float:
    """Reference implementation for Normal-Normal posterior variance.

    Formula: σ²_posterior = (σ² * σ_μ²) / (σ² + n*σ_μ²)
    """
    n = len(data)
    sample_var = np.var(data, ddof=1) if n > 1 else 1e-10

    precision_prior = 1 / (sigma_mu**2)
    precision_data = n / sample_var

    return 1 / (precision_prior + precision_data)


def reference_normal_log_evidence(data: np.ndarray, mu_mu: float, sigma_mu: float) -> float:
    """Reference implementation for Normal-Normal log evidence.

    Uses scipy for numerical integration as ground truth.
    """
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1) if n > 1 else 0.0

    # Marginal variance of sample mean
    marginal_var = (sample_var / n) + sigma_mu**2

    # Log evidence for sample mean
    log_ev = -0.5 * np.log(2 * np.pi * marginal_var)
    log_ev -= 0.5 * (sample_mean - mu_mu) ** 2 / marginal_var

    # Log evidence for deviations from mean (independent of prior)
    if n > 1:
        log_ev -= 0.5 * (n - 1) * (1 + np.log(2 * np.pi * sample_var))

    return log_ev


def reference_plugin_log_likelihood(data: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Reference plug-in log-likelihood using scipy."""
    return stats.norm.logpdf(data, loc=mu, scale=sigma)


def reference_student_t_log_likelihood(data: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    """Reference Student's t log-likelihood using scipy."""
    return stats.t.logpdf(data, df=df, loc=loc, scale=scale)


# ============================================================================
# DISTRIBUTION REGISTRY HELPERS
# ============================================================================


def get_all_distribution_names() -> list[str]:
    """Get names of all registered distributions."""
    from bdf.distributions.bdf_distribution import BDFDistribution
    from bdf.utils.distribution_helpers import import_all_distributions

    import_all_distributions()
    return list(BDFDistribution._registry.keys())


def get_conjugate_distribution_names() -> list[str]:
    """Get names of distributions that support NLE."""
    from bdf.distributions.bdf_distribution import BDFDistribution
    from bdf.utils.distribution_helpers import import_all_distributions

    import_all_distributions()

    conjugate = []
    for name, cls in BDFDistribution._registry.items():
        if cls._supports_nle:
            conjugate.append(name)
    return conjugate


def get_rust_implemented_distributions() -> list[str]:
    """Get names of distributions with Rust implementations.

    These are distributions that don't fall back to Python callbacks.
    """
    # Based on the Rust code mapping in lib.rs
    return [
        "NormalMuNormal",
        "NormalMuInvGammaSigmaNormal",
        "GammaABLambdaPoisson",
        "GammaMVLambdaPoisson",
        "BetaABBernoulli",
        "BetaMVBernoulli",
        "GammaABLambdaExponential",
        "GammaMVLambdaExponential",
        "KDE",
        "BayesianKDE",
        "SkewNormalOmega",
        "BayesianSkewNormalOmega",
        "SkewNormalAlpha",
    ]
