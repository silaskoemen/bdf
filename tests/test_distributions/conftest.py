"""Shared fixtures for distribution tests.

This module provides pytest fixtures that are auto-discovered.
For constants and helper functions, see helpers.py.
"""

import numpy as np
import pytest

from bdf.distributions.distribution_manager import DistributionManager

# Re-export constants and helpers for backward compatibility
# (tests can also import directly from helpers)
from tests.test_distributions.helpers import (
    ATOL_NUMERICAL,
    ATOL_STATISTICAL,
    ATOL_TIGHT,
    RTOL_NUMERICAL,
    RTOL_STATISTICAL,
    RTOL_TIGHT,
    assert_array_close,
    assert_close,
    assert_dict_close,
    get_all_distribution_names,
    get_conjugate_distribution_names,
    get_rust_implemented_distributions,
    invgamma_prior_params_strategy,
    normal_data_strategy,
    prior_params_strategy,
    reference_normal_log_evidence,
    reference_normal_posterior_mean,
    reference_normal_posterior_variance,
    reference_plugin_log_likelihood,
    reference_student_t_log_likelihood,
)

# ============================================================================
# DATA FIXTURES
# ============================================================================


@pytest.fixture
def small_data():
    """Small deterministic dataset for edge case testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def medium_data():
    """Medium-sized dataset from known distribution."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, 100)


@pytest.fixture
def large_data():
    """Large dataset for statistical convergence tests."""
    rng = np.random.default_rng(42)
    return rng.normal(5.0, 2.0, 10000)


@pytest.fixture
def single_point_data():
    """Single data point (edge case)."""
    return np.array([3.14])


@pytest.fixture
def constant_data():
    """All values identical (edge case)."""
    return np.array([5.0, 5.0, 5.0, 5.0, 5.0])


@pytest.fixture
def two_point_data():
    """Two data points (minimum for variance calculation)."""
    return np.array([0.0, 10.0])


@pytest.fixture
def extreme_values_data():
    """Data with very large values."""
    return np.array([1e10, 1e10 + 1, 1e10 + 2])


@pytest.fixture
def small_values_data():
    """Data with very small values."""
    return np.array([1e-10, 2e-10, 3e-10])


@pytest.fixture
def bimodal_data():
    """Bimodal data for testing distribution flexibility."""
    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-5, 1, 50), rng.normal(5, 1, 50)])


@pytest.fixture
def skewed_data():
    """Skewed data (log-normal)."""
    rng = np.random.default_rng(42)
    return rng.lognormal(0, 1, 100)


# ============================================================================
# INVALID DATA FIXTURES
# ============================================================================


@pytest.fixture
def empty_data():
    """Empty array (should raise)."""
    return np.array([])


@pytest.fixture
def nan_data():
    """Data containing NaN (should raise)."""
    return np.array([1.0, 2.0, np.nan, 4.0])


@pytest.fixture
def inf_data():
    """Data containing infinity (should raise)."""
    return np.array([1.0, 2.0, np.inf, 4.0])


@pytest.fixture
def neginf_data():
    """Data containing negative infinity (should raise)."""
    return np.array([1.0, 2.0, -np.inf, 4.0])


# ============================================================================
# DISTRIBUTION FIXTURES
# ============================================================================


@pytest.fixture
def normal_mu_normal_default():
    """NormalMuNormal with default parameters."""
    return DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0}, y=np.array([0.0]))


@pytest.fixture
def normal_mu_normal_weak_prior():
    """NormalMuNormal with weak (uninformative) prior."""
    return DistributionManager.create_distribution(
        "NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 100.0}, y=np.array([0.0])
    )


@pytest.fixture
def normal_mu_normal_strong_prior():
    """NormalMuNormal with strong (informative) prior."""
    return DistributionManager.create_distribution(
        "NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 0.01}, y=np.array([0.0])
    )


@pytest.fixture
def normal_invgamma_default():
    """NormalMuInvGammaSigmaNormal with default parameters."""
    return DistributionManager.create_distribution(
        "NormalMuInvGammaSigmaNormal",
        {"mu_mu": 0.0, "n_mu": 1.0, "nu_sigma": 3.0, "phi_sigma": 1.0},
        y=np.array([0.0]),
    )


# ============================================================================
# PYTEST MARKERS AND CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "rust: marks tests that require Rust extension")
    config.addinivalue_line("markers", "numerical: marks tests for numerical stability")
    config.addinivalue_line("markers", "statistical: marks tests that verify statistical properties")
