"""
Direct test of Rust negative gain handling.
"""

import numpy as np

from bdf.distributions.normal import NormalMuNormal
from bdf.tree_classes.bdf_node import BDFNode


def test_rust_returns_negative_gain_directly():
    """
    Test that Rust actually returns negative gain values.

    We create a scenario with:
    1. Random noise data (no real structure)
    2. Very high gamma penalty
    3. This should produce negative gain if penalty > raw improvement
    """
    np.random.seed(999)

    # Small dataset with pure noise
    n = 50
    X = np.random.randn(n, 3)
    y = np.random.randn(n)

    # Create node
    dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=999)
    node.estimate_posterior(y)

    # Very high gamma to force negative gain
    gamma = 15.0

    # Call Rust directly (default behavior)
    print("\n" + "=" * 80)
    print("DIRECT RUST TEST: High Gamma Penalty")
    print("=" * 80)
    print(f"Data: n={n}, pure noise")
    print(f"Gamma: {gamma}")
    print()

    result = node.find_best_split(
        X, y, min_samples_leaf=5, min_child_weight=0.0, col_idcs=None, gamma=gamma, eta=0.1, split_gain_method="map"
    )

    feature_idx, threshold, loss_reduction, left_idx, right_idx, left_params, right_params = result

    print(f"Rust returned:")
    print(f"  feature_idx:     {feature_idx}")
    print(f"  threshold:       {threshold}")
    print(f"  loss_reduction:  {loss_reduction}")
    print(f"  left_idx:        {left_idx is not None}")
    print(f"  right_idx:       {right_idx is not None}")

    if feature_idx is not None:
        print(f"\n✓ Split found!")
        print(f"  Raw gain - penalty = {loss_reduction}")

        if loss_reduction < 0:
            print(f"  ✓✓ NEGATIVE GAIN RETURNED: {loss_reduction:.4f}")
            print(f"      This is the key fix - Rust now allows negative gains!")
        else:
            print(f"  ⚠ Positive gain: {loss_reduction:.4f}")
            print(f"     With gamma={gamma}, this means the raw improvement was very large")
    else:
        print(f"\n✗ No split found")
        print(f"  loss_reduction returned: {loss_reduction}")
        if loss_reduction == 0.0:
            print(f"  ⚠ Returned 0.0 (no split found convention)")
        elif loss_reduction == float("-inf"):
            print(f"  ⚠ Returned -inf (initial value)")

    print("=" * 80)


def test_controlled_negative_gain():
    """
    Create a controlled scenario where we KNOW the gain should be negative.

    Strategy:
    - Use constant y (no variance to split on)
    - Any split will have zero improvement
    - With gamma > 0, penalty will make gain negative
    """
    np.random.seed(777)

    # Constant target - no information to split on
    n = 100
    X = np.random.randn(n, 5)
    y = np.ones(n) * 5.0  # Completely constant

    dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=777)
    node.estimate_posterior(y)

    # Moderate gamma
    gamma = 2.0

    print("\n" + "=" * 80)
    print("CONTROLLED TEST: Constant Target (Zero Improvement)")
    print("=" * 80)
    print(f"Data: n={n}, y=5.0 (constant)")
    print(f"Gamma: {gamma}")
    print(f"Expected: Any split has 0 improvement, so gain = -gamma*penalty < 0")
    print()

    result = node.find_best_split(
        X, y, min_samples_leaf=10, min_child_weight=0.0, gamma=gamma, eta=0.1, split_gain_method="map"
    )

    feature_idx, threshold, loss_reduction, _, _, _, _ = result

    print(f"Rust returned:")
    print(f"  feature_idx:     {feature_idx}")
    print(f"  loss_reduction:  {loss_reduction}")

    if feature_idx is not None:
        # With constant y, any split should have ~0 improvement
        # So gain should be approximately: 0 - gamma*log(k*m) < 0
        k = 5  # num features
        m = int(np.ceil(1.0 / 0.1))  # num thresholds
        expected_penalty = gamma * (np.log(k) + np.log(m))
        expected_gain = -expected_penalty

        print(f"\n  Expected penalty: {expected_penalty:.4f}")
        print(f"  Expected gain:    ~{expected_gain:.4f}")
        print(f"  Actual gain:      {loss_reduction:.4f}")

        if loss_reduction < 0:
            print(f"\n  ✓✓ NEGATIVE GAIN CONFIRMED!")
            print(f"     Rust correctly returns negative gains after penalty")
        else:
            print(f"\n  ✗✗ PROBLEM: Expected negative gain, got {loss_reduction:.4f}")
    else:
        print(f"\n  No split found (loss_reduction={loss_reduction})")

    print("=" * 80)


def test_compare_initial_value():
    """Test what happens when NO valid split can beat the initial threshold."""
    np.random.seed(555)

    # Tiny dataset that will fail min_samples_leaf constraints
    n = 15  # Too small to split with min_samples_leaf=10
    X = np.random.randn(n, 2)
    y = np.random.randn(n)

    dist = NormalMuNormal({"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=555)
    node.estimate_posterior(y)

    print("\n" + "=" * 80)
    print("EDGE CASE TEST: No Valid Splits (min_samples_leaf constraint)")
    print("=" * 80)
    print(f"Data: n={n}")
    print(f"min_samples_leaf: 10 (too large for n=15)")
    print()

    result = node.find_best_split(X, y, min_samples_leaf=10, min_child_weight=0.0, gamma=1.0, eta=0.1)  # Too large!

    feature_idx, threshold, loss_reduction, _, _, _, _ = result

    print(f"Rust returned:")
    print(f"  feature_idx:     {feature_idx}")
    print(f"  loss_reduction:  {loss_reduction}")

    if feature_idx is None:
        if loss_reduction == 0.0:
            print(f"\n  ✓ Correctly returns 0.0 when no split found")
        elif loss_reduction == float("-inf"):
            print(f"\n  ! Returns -inf (the initial value)")
            print(f"    This should be converted to 0.0 for 'no split' case")
        else:
            print(f"\n  ? Unexpected value: {loss_reduction}")
    else:
        print(f"\n  Unexpected: Found a split despite constraints")

    print("=" * 80)


if __name__ == "__main__":
    test_rust_returns_negative_gain_directly()
    test_controlled_negative_gain()
    test_compare_initial_value()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("If you see negative gains returned, the Rust fix is working!")
    print("If you only see 0.0, there may be additional clamping somewhere.")
    print("=" * 80)
