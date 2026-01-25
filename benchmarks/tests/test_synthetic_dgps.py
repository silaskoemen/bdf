"""Tests for synthetic DGP generators.

Validates that:
- All DGPs generate data with correct shapes
- Ground truth functions are well-defined
- Ground truth metrics can be computed
- Reproducibility via seeds
"""

import numpy as np
import pytest

from benchmarks.pipeline.synthetic_dgps import (
    DGP_REGISTRY,
    DGPType,
    SyntheticDataset,
    generate_bimodal_mixture,
    generate_heavy_tailed,
    generate_heteroscedastic_sinusoidal,
    generate_sparse_sampling,
    generate_step_function,
)


@pytest.fixture(params=list(DGP_REGISTRY.keys()))
def dgp_name(request):
    """Parametrize over all registered DGPs."""
    return request.param


def test_dgp_generation(dgp_name):
    """Test that DGP generates valid SyntheticDataset."""
    dataset = DGP_REGISTRY[dgp_name](n_samples=100, seed=42)

    assert isinstance(dataset, SyntheticDataset)
    assert len(dataset.X) == len(dataset.y)
    assert dataset.X.ndim == 2
    assert dataset.y.ndim == 1
    assert dataset.n_features == dataset.X.shape[1]
    assert dataset.n_samples == len(dataset.y)


def test_ground_truth_mean_function(dgp_name):
    """Test that ground truth mean function is callable and returns correct shape."""
    dataset = DGP_REGISTRY[dgp_name](n_samples=100, seed=42)
    X_test = dataset.X[:10]

    y_mean = dataset.ground_truth.mean_fn(X_test)

    assert y_mean.shape[0] == len(X_test)
    assert not np.any(np.isnan(y_mean))


def test_ground_truth_variance_function(dgp_name):
    """Test variance function if available."""
    dataset = DGP_REGISTRY[dgp_name](n_samples=100, seed=42)

    if dataset.ground_truth.variance_fn is not None:
        X_test = dataset.X[:10]
        y_var = dataset.ground_truth.variance_fn(X_test)

        assert y_var.shape[0] == len(X_test)
        assert np.all(y_var >= 0)  # Variance must be non-negative
        assert not np.any(np.isnan(y_var))


def test_reproducibility(dgp_name):
    """Test that same seed produces same data."""
    dataset1 = DGP_REGISTRY[dgp_name](n_samples=100, seed=42)
    dataset2 = DGP_REGISTRY[dgp_name](n_samples=100, seed=42)

    np.testing.assert_array_equal(dataset1.X, dataset2.X)
    np.testing.assert_array_equal(dataset1.y, dataset2.y)


def test_different_seeds(dgp_name):
    """Test that different seeds produce different data."""
    dataset1 = DGP_REGISTRY[dgp_name](n_samples=100, seed=42)
    dataset2 = DGP_REGISTRY[dgp_name](n_samples=100, seed=43)

    assert not np.allclose(dataset1.y, dataset2.y)


def test_heteroscedastic_variance_change():
    """Test that heteroscedastic DGP has variance change at π."""
    dataset = generate_heteroscedastic_sinusoidal(n_samples=1000, seed=42, noise_scale=0.1)

    # Check variance increases after π
    x_before_pi = np.array([[np.pi - 0.5]])
    x_after_pi = np.array([[np.pi + 0.5]])

    var_before = dataset.ground_truth.variance_fn(x_before_pi)
    var_after = dataset.ground_truth.variance_fn(x_after_pi)

    assert var_after > var_before


def test_step_function_discontinuities():
    """Test that step function has discrete levels."""
    dataset = generate_step_function(n_samples=1000, seed=42, n_steps=5)

    # Sample mean function at many points
    x_grid = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y_mean = dataset.ground_truth.mean_fn(x_grid)

    # Count unique values (should be roughly n_steps)
    unique_values = np.unique(np.round(y_mean, decimals=5))
    assert len(unique_values) == dataset.metadata["n_steps"]


def test_bimodal_mixing_probability():
    """Test that bimodal mixture has correct mixing probability."""
    dataset = generate_bimodal_mixture(n_samples=1000, seed=42)

    # At x=0, should be mostly mode 2 (π(0) = 0.1)
    # At x=1, should be mostly mode 1 (π(1) = 0.9)
    # This should reflect in the variance

    x_low = np.array([[0.1]])
    x_high = np.array([[0.9]])

    var_low = dataset.ground_truth.variance_fn(x_low)
    var_high = dataset.ground_truth.variance_fn(x_high)

    # Variance should be lower at x=0.9 (more deterministic mixing)
    # and higher at x=0.5 (50-50 mixing)
    x_mid = np.array([[0.5]])
    var_mid = dataset.ground_truth.variance_fn(x_mid)

    # At 50-50 mixing, variance from mode uncertainty is maximized
    assert var_mid > var_low
    assert var_mid > var_high


def test_heavy_tailed_higher_variance():
    """Test that heavy-tailed DGP has higher variance than normal."""
    dataset = generate_heavy_tailed(n_samples=1000, seed=42, df=3.0, scale=1.0)

    # For df=3, variance = scale² × df/(df-2) = 3
    # For normal, variance = scale² = 1
    x_test = np.array([[0.0, 0.0]])
    var_t = dataset.ground_truth.variance_fn(x_test)

    expected_var = 1.0**2 * 3.0 / (3.0 - 2.0)  # = 3.0
    np.testing.assert_allclose(var_t, expected_var, rtol=1e-5)


def test_sparse_sampling_density():
    """Test that sparse sampling produces non-uniform x distribution."""
    dataset = generate_sparse_sampling(n_samples=1000, seed=42, sparsity_type="linear_decay")

    x_flat = dataset.X.flatten()

    # For linear_decay, should have more samples near 0 than near 2π
    n_low = np.sum(x_flat < np.pi / 2)
    n_high = np.sum(x_flat > 3 * np.pi / 2)

    assert n_low > n_high  # More density at low x values


def test_quantile_function():
    """Test quantile function if available."""
    dataset = generate_heteroscedastic_sinusoidal(n_samples=100, seed=42)

    if dataset.ground_truth.quantile_fn is not None:
        x_test = np.array([[1.0]])

        # Median
        q_50 = dataset.ground_truth.quantile_fn(x_test, 0.5)
        # Should be close to mean for symmetric distribution
        mean = dataset.ground_truth.mean_fn(x_test)
        np.testing.assert_allclose(q_50, mean, rtol=0.1)

        # Lower and upper quantiles
        q_05 = dataset.ground_truth.quantile_fn(x_test, 0.05)
        q_95 = dataset.ground_truth.quantile_fn(x_test, 0.95)

        assert q_05 < q_50 < q_95


def test_sample_function():
    """Test that sample function generates samples."""
    dataset = generate_heteroscedastic_sinusoidal(n_samples=100, seed=42)

    if dataset.ground_truth.sample_fn is not None:
        x_test = dataset.X[:5]
        n_samples = 100

        samples = dataset.ground_truth.sample_fn(x_test, n_samples)

        assert samples.shape == (len(x_test), n_samples)
        assert not np.any(np.isnan(samples))

        # Check samples have reasonable mean and variance
        sample_means = np.mean(samples, axis=-1)
        true_means = dataset.ground_truth.mean_fn(x_test).flatten()

        # Allow for sampling variability
        np.testing.assert_allclose(sample_means, true_means, rtol=0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
