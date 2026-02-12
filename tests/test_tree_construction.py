import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from bdf.tree_classes.bdf_regressor import BDFRegressor


def test_tree_building():
    """Test tree building process"""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    # Create regressor with limited depth
    regressor = BDFRegressor(
        dist="NormalMuNormal", n_trees=1, max_depth=3, params={"mu_mu": y.mean(), "sigma_mu": y.std()}
    )
    regressor.fit(X, y)

    # Check tree structure
    tree = regressor.trees[0]

    # The root node should have a split if not a leaf
    # best_feature/best_threshold are only set when split_node() is called
    if not tree.root._is_leaf():
        assert hasattr(tree.root, "best_feature") and tree.root.best_feature is not None
        assert hasattr(tree.root, "best_threshold") and tree.root.best_threshold is not None
    else:
        # If it's a leaf, check the tree has at least the root (may happen with small datasets)
        assert tree.root.count_nodes() >= 1

    # Traverse to verify structure
    depth = max_depth_traverse(tree.root)
    assert depth <= 3  # Max depth should be respected


def test_forest_ensemble():
    """Test forest ensemble predictions"""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train with different tree counts
    regressor1 = BDFRegressor(
        dist="NormalMuNormal", n_trees=1, max_depth=3, params={"mu_mu": y.mean(), "sigma_mu": y.std()}
    )
    regressor1.fit(X_train, y_train)

    regressor10 = BDFRegressor(
        dist="NormalMuNormal", n_trees=10, max_depth=3, params={"mu_mu": y.mean(), "sigma_mu": y.std()}
    )
    regressor10.fit(X_train, y_train)

    regressor50 = BDFRegressor(
        dist="NormalMuNormal", n_trees=50, max_depth=3, params={"mu_mu": y.mean(), "sigma_mu": y.std()}
    )
    regressor50.fit(X_train, y_train)

    # Calculate RMSE
    rmse1 = np.sqrt(np.mean((regressor1.predict(X_test) - y_test) ** 2))
    rmse10 = np.sqrt(np.mean((regressor10.predict(X_test) - y_test) ** 2))
    rmse50 = np.sqrt(np.mean((regressor50.predict(X_test) - y_test) ** 2))

    # More trees should give better performance
    assert rmse10 < rmse1
    assert rmse50 < rmse10


def test_distribution_specific_methods():
    """Test distribution-specific functionality"""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)

    # For NormalMuNormal, we should be able to get predictive variance
    regressor = BDFRegressor(dist="NormalMuNormal", n_trees=10, params={"mu_mu": y.mean(), "sigma_mu": y.std()})
    regressor.fit(X, y)

    # predict_params returns (n_obs, n_trees) array of param dicts
    params_array = regressor.predict(X, method="params")

    # Check results
    assert params_array.shape[0] == 100
    assert params_array.shape[1] == regressor.n_trees
    # Each element should be a dict with posterior parameters
    assert isinstance(params_array[0, 0], dict)


def test_categorical_features():
    """Test handling of categorical features"""
    # Create data with categorical feature
    X = np.random.rand(100, 5)
    X[:, 0] = np.random.choice([0, 1, 2], size=100)  # Categorical feature
    y = 2 * X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1  # Response depends on categorical

    regressor = BDFRegressor(
        dist="NormalMuNormal", n_trees=10, max_depth=3, params={"mu_mu": y.mean(), "sigma_mu": y.std()}
    )
    regressor.fit(X, y)

    # Create test data with categorical values
    X_test = np.random.rand(10, 5)
    X_test[:5, 0] = 0  # Category 0
    X_test[5:, 0] = 2  # Category 2

    preds = regressor.predict(X_test)

    # Check that category 0 and 2 predictions are different
    assert abs(np.mean(preds[:5]) - np.mean(preds[5:])) > 0.5


# Helper function for tree traversal
def max_depth_traverse(node, current_depth=0):
    if node is None:
        return current_depth - 1

    # Use left_node/right_node (renamed from left/right)
    if node.left_node is None and node.right_node is None:
        return current_depth

    left_depth = max_depth_traverse(node.left_node, current_depth + 1) if node.left_node else current_depth
    right_depth = max_depth_traverse(node.right_node, current_depth + 1) if node.right_node else current_depth

    return max(left_depth, right_depth)
