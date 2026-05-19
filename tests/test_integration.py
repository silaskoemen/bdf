import numpy as np
from sklearn.datasets import make_regression

from bdf import _bdf_rs as bdf_rs
from bdf.distributions.distribution_manager import DistributionManager
from bdf.tree_classes.bdf_regressor import BDFClassifier, BDFRegressor


def test_end_to_end_regression():
    """Test the full regression pipeline"""
    # Generate synthetic data — fewer informative features for a clearer signal
    X, y = make_regression(n_samples=200, n_features=5, n_informative=3, noise=10.0, random_state=42)

    # Train model with auto params so priors adapt to data scale
    regressor = BDFRegressor(
        dist="NormalMuNormal",
        params={"mu_mu": "auto", "sigma_mu": "auto", "sigma_mu_auto_scale": 1.0},
        n_trees=50,
        min_samples_leaf=5,
    )
    regressor.fit(X, y)

    # Make predictions
    preds = regressor.predict(X)

    # Basic sanity checks
    assert preds.shape == y.shape
    assert np.corrcoef(preds, y)[0, 1] > 0.5  # Moderate correlation with true values

    # Check parameter passing
    assert len(regressor.trees) == 50

    # Test serialization/deserialization
    import pickle

    serialized = pickle.dumps(regressor)
    loaded = pickle.loads(serialized)

    loaded_preds = loaded.predict(X)
    assert np.allclose(preds, loaded_preds)


def test_fit_returns_self_for_sklearn_style_api():
    X, y = make_regression(n_samples=80, n_features=4, n_informative=2, noise=5.0, random_state=42)

    regressor = BDFRegressor(
        dist="NormalMuNormal",
        params={"mu_mu": "auto", "sigma_mu": "auto", "sigma_mu_auto_scale": 1.0},
        n_trees=3,
        min_samples_leaf=5,
        n_jobs=1,
    )
    assert regressor.fit(X, y) is regressor

    y_binary = (y > np.median(y)).astype(float)
    classifier = BDFClassifier(
        dist="BetaMVBernoulli",
        params={"mean_p": "auto", "var_p": 0.05},
        n_trees=3,
        min_samples_leaf=5,
        n_jobs=1,
    )
    assert classifier.fit(X, y_binary) is classifier
    proba = classifier.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_distribution_parameters():
    """Test that distribution parameters are correctly passed through the system"""
    # Create a distribution with non-default parameters
    dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 3.5, "sigma_mu": 4.2})

    # Convert to Rust spec
    rust_spec = DistributionManager.to_rust_spec(dist)

    # Check parameters are preserved
    assert rust_spec["mu_mu"] == 3.5
    assert rust_spec["sigma_mu"] == 4.2

    # Use in a simple split finding test
    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    # Call Rust function (updated signature: X, y, min_samples_leaf, min_child_weight, spec, eta, gamma, col_idcs, split_gain_method)
    result = bdf_rs.find_best_split(X, y, 1, 0.0, rust_spec, 0.1, 0.0, None, "map")  # type: ignore

    # Just check it runs without error - actual values tested elsewhere
    assert result is not None


def test_subsample_colsample():
    """Test that subsampling and column sampling work correctly"""
    X = np.random.rand(100, 20)
    y = np.random.rand(100)

    # Test with very low subsampling and column sampling
    regressor = BDFRegressor(
        dist="NormalMuNormal",
        params={"mu_mu": 0, "sigma_mu": 1},
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
    dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": mean, "sigma_mu": std})
    py_nll = dist.nll(data)

    # Calculate in Rust
    rust_spec = DistributionManager.to_rust_spec(dist)
    rust_nll = bdf_rs.calculate_nll(data, rust_spec)  # type: ignore

    # Should be very close
    assert abs(py_nll - rust_nll) < 1e-10


# Helper function for feature usage tracking
def collect_used_features(node, feature_set):
    if node is None:
        return

    # Use best_feature attribute (renamed from feature_idx)
    if hasattr(node, "best_feature") and node.best_feature is not None:
        feature_set.add(node.best_feature)

    if node.left_node:
        collect_used_features(node.left_node, feature_set)
    if node.right_node:
        collect_used_features(node.right_node, feature_set)
