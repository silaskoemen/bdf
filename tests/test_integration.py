import bdf_rust
import numpy as np
import pytest
from sklearn.datasets import make_regression

from bdf.distributions.distribution_manager import DistributionManager
from bdf.tree_classes.bdf_regressor import BDFRegressor


def test_end_to_end_regression():
    """Test the full regression pipeline"""
    # Generate synthetic data
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)  # type: ignore

    # Train model
    regressor = BDFRegressor(
        dist="normal", prior_params={"mean": 0, "std": 5}, n_trees=10, max_depth=5, min_samples_leaf=2
    )
    regressor.fit(X, y)

    # Make predictions
    preds = regressor.predict(X)

    # Basic sanity checks
    assert preds.shape == y.shape
    assert np.corrcoef(preds, y)[0, 1] > 0.7  # Strong correlation with true values

    # Check parameter passing
    assert len(regressor.trees) == 10

    # Test serialization/deserialization
    import pickle

    serialized = pickle.dumps(regressor)
    loaded = pickle.loads(serialized)

    loaded_preds = loaded.predict(X)
    assert np.allclose(preds, loaded_preds)


def test_distribution_parameters():
    """Test that distribution parameters are correctly passed through the system"""
    # Create a distribution with non-default parameters
    dist = DistributionManager.create_distribution("normal", {"mean": 3.5, "std": 4.2})

    # Convert to Rust spec
    rust_spec = DistributionManager.to_rust_spec(dist)

    # Check parameters are preserved
    assert rust_spec["prior_mean"] == 3.5
    assert rust_spec["prior_std"] == 4.2

    # Use in a simple split finding test
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    # Call Rust function
    result = bdf_rust.find_best_split(X, y, 1, 0.0, rust_spec, 0.1, None)  # type: ignore

    # Just check it runs without error - actual values tested elsewhere
    assert result is not None


def test_subsample_colsample():
    """Test that subsampling and column sampling work correctly"""
    X = np.random.rand(100, 20)
    y = np.random.rand(100)

    # Test with very low subsampling and column sampling
    regressor = BDFRegressor(
        dist="normal",
        prior_params={"mean": 0, "std": 1},
        subsample=0.5,  # Use only half the data per tree
        colsample=0.5,  # Use only half the features per tree
        n_trees=5,
    )

    # This should run without error
    regressor.fit(X, y)

    # Each tree should have different feature subsets
    feature_counts = np.zeros(20)
    for tree in regressor.trees:
        # Count which features were used
        used_features = set()
        collect_used_features(tree.root, used_features)
        for feature in used_features:
            feature_counts[feature] += 1

    # Not all features should be used in all trees
    assert np.mean(feature_counts) < 5  # Less than full usage
    assert np.max(feature_counts) <= 5  # No feature used more than n_trees times


def test_rust_python_nll_equivalence():
    """Test NLL calculations match between Python and Rust"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Normal distribution with known parameters
    mean = 2.0
    std = 3.0

    # Calculate in Python
    dist = DistributionManager.create_distribution("normal", {"mean": mean, "std": std})
    py_nll = dist.nll(data)

    # Calculate in Rust
    rust_spec = DistributionManager.to_rust_spec(dist)
    rust_nll = bdf_rust.calculate_nll(data, rust_spec)  # type: ignore

    # Should be very close
    assert abs(py_nll - rust_nll) < 1e-10


# Helper function for feature usage tracking
def collect_used_features(node, feature_set):
    if node is None:
        return

    if node.feature_idx is not None:
        feature_set.add(node.feature_idx)

    if node.left:
        collect_used_features(node.left, feature_set)
    if node.right:
        collect_used_features(node.right, feature_set)
