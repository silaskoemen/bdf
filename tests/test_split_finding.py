import bdf_rs
import numpy as np
import pytest

from bdf.distributions.distribution_manager import DistributionManager
from bdf.tree_classes.bdf_node import BDFNode
from bdf.tree_classes.bdf_regressor import BDFRegressor


def test_split_finding_basic():
    """Test basic split finding with simple dataset"""
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
            [8.0, 80.0],
        ]
    )

    # Step function response - clear split at x[0] = 4.5
    y = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])

    dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=42)

    # Setting min_samples_leaf = 4 would prevent any valid split
    feature_idx, threshold, loss, left_mask, right_mask, _, _ = node._find_best_split_python(X, y, 1, 0.0, None)

    # Should find the obvious split
    assert feature_idx == 0  # First feature
    assert 4.0 < threshold < 5.0  # type: ignore Split between 4 and 5
    assert loss > 0  # Should have positive loss reduction


def test_min_samples_leaf_constraint():
    """Test min_samples_leaf constraint is respected"""
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ]
    )

    # Would naturally split at x[0] = 3.5
    y = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=42)

    # Setting min_samples_leaf = 4 would prevent any valid split
    feature_idx, threshold, loss, left_mask, right_mask, _, _ = node._find_best_split_python(X, y, 4, 0.0, None)

    # Should not find a valid split
    assert feature_idx is None
    assert threshold is None


def test_split_mask_correctness():
    """Test split masks are correctly generated"""
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    y = np.array([1.0, 1.0, 2.0, 2.0])

    dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=42)

    # Use min_samples_leaf = 1 to allow splits (min_samples_leaf=4 would prevent any valid split with 4 samples)
    feature_idx, threshold, loss, left_mask, right_mask, _, _ = node._find_best_split_python(X, y, 1, 0.0, None)

    # Check masks are boolean arrays
    assert left_mask.dtype == np.bool_  # type: ignore
    assert right_mask.dtype == np.bool_  # type: ignore

    # Check masks are complementary
    assert np.all(left_mask != right_mask)
    assert np.all((left_mask | right_mask) == np.ones_like(left_mask))  # type: ignore

    # For this simple dataset, assuming split at x[0] ~ 2.5
    expected_left = np.array([True, True, False, False])
    expected_right = np.array([False, False, True, True])
    assert np.all(left_mask == expected_left)
    assert np.all(right_mask == expected_right)


def test_rust_python_split_equivalence():
    """Test Rust and Python split finding produce same results"""
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ]
    )

    y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    # Python implementation
    dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0})
    node = BDFNode(distribution=dist, depth=0, random_state=42)

    # Use Python implementation
    py_feature_idx, py_threshold, py_loss, py_left, py_right, py_left_params, py_right_params = (
        node._find_best_split_python(X, y, 1, 0.0, None)
    )

    # Use Rust implementation
    rust_spec = DistributionManager.to_rust_spec(dist)
    rust_feature_idx, rust_threshold, rust_loss, rust_left, rust_right, _, _ = bdf_rs.find_best_split(  # type: ignore
        X, y, 1, 0.0, rust_spec, 0.1, 0.0, None, "map"
    )

    # Check results match
    assert py_feature_idx == rust_feature_idx
    assert py_threshold is None and rust_threshold is None or abs(py_threshold - rust_threshold) < 1e-10
    assert abs(py_loss - rust_loss) < 1e-10
    if py_left is not None and rust_left is not None:
        assert np.all(py_left == rust_left)
        assert np.all(py_right == rust_right)
