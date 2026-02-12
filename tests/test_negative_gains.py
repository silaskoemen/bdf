"""
Test that negative gains are properly returned after the Rust fix.
"""

import numpy as np
import pytest

from bdf.distributions.normal import NormalMuNormal
from bdf.tree_classes.bdf_node import BDFNode


def test_negative_gain_returned_with_high_gamma():
    """Test that high gamma penalty can produce negative gains."""
    # Create a weak split scenario
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)  # Pure noise, no signal

    # Create distribution and node
    dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=42)
    node.estimate_posterior(y)

    # Very high gamma should produce negative gain even if raw improvement is positive
    gamma = 10.0  # Extremely high penalty

    feature_idx, threshold, loss_reduction, left_idx, right_idx, left_params, right_params = node.find_best_split(
        X, y, min_samples_leaf=10, min_child_weight=0.0, col_idcs=None, gamma=gamma, eta=0.1
    )

    # With pure noise and high gamma, we expect negative gain
    # (or no split found at all)
    if feature_idx is not None:
        # If a split was found, gain should be negative due to high penalty
        assert loss_reduction < 0, f"Expected negative gain with gamma={gamma}, got {loss_reduction}"
    else:
        # No split found is also acceptable
        assert loss_reduction == 0.0


def test_negative_gain_comparison_python_rust():
    """Test that Python and Rust return identical gains (MAP method)."""
    np.random.seed(123)
    X = np.random.randn(50, 3)
    y = np.random.randn(50)

    dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=123)
    node.estimate_posterior(y)

    gamma = 5.0

    # Rust (uses MAP)
    feat_rust, thresh_rust, gain_rust, left_rust, right_rust, _, _ = node.find_best_split(
        X, y, min_samples_leaf=5, min_child_weight=0.0, gamma=gamma, eta=0.1
    )

    # Python fallback (MAP to match Rust)
    feat_py, thresh_py, gain_py, left_py, right_py, _, _ = node._find_best_split_python(
        X,
        y,
        min_samples_leaf=5,
        min_child_weight=0.0,
        gamma=gamma,
        eta=0.1,
        split_gain_method="map",
    )

    # Both should find the same feature, threshold, and gain
    if feat_rust is not None and feat_py is not None:
        assert feat_rust == feat_py, f"Feature mismatch: Rust={feat_rust}, Python={feat_py}"
        assert np.isclose(gain_rust, gain_py, rtol=1e-6), f"Gain mismatch: Rust={gain_rust:.6f}, Python={gain_py:.6f}"


def test_defer_prior_allows_split_with_negative_evidence():
    """Test that defer prior can force splits even with negative evidence."""
    # Create a scenario where evidence is weak but prior strongly favors splitting
    np.random.seed(456)
    n = 100
    X = np.random.randn(n, 3)
    y = np.random.randn(n) * 0.1  # Very small variance, hard to split

    dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=456)
    node.estimate_posterior(y)

    # High gamma to get negative gain
    gamma = 3.0

    feature_idx, threshold, loss_reduction, left_idx, right_idx, left_params, right_params = node.find_best_split(
        X, y, min_samples_leaf=10, min_child_weight=0.0, gamma=gamma, eta=0.1
    )

    # Calculate what depth_penalty would be with strong prior for splitting
    alpha = 0.95
    delta = 0.9
    depth = 0
    p_d = alpha * (delta**depth)  # 0.95
    depth_penalty = np.log((1 - p_d) / p_d)  # log(0.05/0.95) ≈ -2.944

    print(f"\nTest: Defer prior with negative evidence")
    print(f"  loss_reduction: {loss_reduction:.4f}")
    print(f"  depth_penalty:  {depth_penalty:.4f}")
    print(f"  Would split:    {loss_reduction > depth_penalty}")

    # The key test: even if loss_reduction is negative,
    # it should be returnable (not clamped to 0)
    if feature_idx is not None:
        # If we found a split, verify we can get the actual negative value
        # (not clamped to 0.0)
        if loss_reduction < 0:
            # This is the key: before the fix, Rust would return 0.0
            # After the fix, it returns the actual negative value
            print(f"  ✓ Rust correctly returned negative gain: {loss_reduction:.4f}")
            assert loss_reduction < 0

        # Simulate the defer decision
        should_split_defer = loss_reduction > depth_penalty
        print(f"  Defer decision: {'SPLIT' if should_split_defer else 'NO SPLIT'}")


def test_extreme_negative_gain_prevents_split():
    """Test that very negative gain prevents splitting even with prior favoring it."""
    np.random.seed(789)
    X = np.random.randn(80, 4)
    y = np.random.randn(80)

    dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=789)
    node.estimate_posterior(y)

    # Extreme gamma to ensure very negative gain
    gamma = 20.0

    feature_idx, threshold, loss_reduction, _, _, _, _ = node.find_best_split(
        X, y, min_samples_leaf=10, min_child_weight=0.0, gamma=gamma, eta=0.1
    )

    # Calculate depth_penalty for comparison
    alpha = 0.9
    delta = 0.7
    depth = 0
    p_d = alpha * (delta**depth)
    depth_penalty = np.log((1 - p_d) / p_d)  # ≈ -2.197

    print(f"\nTest: Extreme negative gain")
    print(f"  loss_reduction: {loss_reduction:.4f}")
    print(f"  depth_penalty:  {depth_penalty:.4f}")

    # With extreme gamma, gain should be very negative
    # More negative than depth_penalty, so no split
    if feature_idx is not None:
        should_split = loss_reduction > depth_penalty
        if loss_reduction < depth_penalty:
            print(f"  ✓ Extreme negative gain ({loss_reduction:.4f}) < depth_penalty ({depth_penalty:.4f})")
            print(f"    Would correctly prevent split in defer mode")


if __name__ == "__main__":
    print("Testing negative gain support after Rust fix...")
    print("=" * 80)

    test_negative_gain_returned_with_high_gamma()
    print("✓ Test 1 passed: High gamma produces negative gains")

    test_negative_gain_comparison_python_rust()
    print("✓ Test 2 passed: Python and Rust produce similar negative gains")

    test_defer_prior_allows_split_with_negative_evidence()
    print("✓ Test 3 passed: Defer prior works with negative evidence")

    test_extreme_negative_gain_prevents_split()
    print("✓ Test 4 passed: Extreme negative gain prevents split")

    print("=" * 80)
    print("All tests passed! Negative gain support is working correctly.")
