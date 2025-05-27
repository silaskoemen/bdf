import numpy as np
import pytest
from bdf.tree_classes.bdf_regressor import BDFRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

def test_tree_building():
    """Test tree building process"""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)  # type: ignore
    
    # Create regressor with limited depth
    regressor = BDFRegressor(dist="normal", n_trees=1, max_depth=3, prior_params={"mean": y.mean(), "std": y.std()})
    regressor.fit(X, y)
    
    # Check tree structure
    tree = regressor.trees[0]
    
    # The root node should have a split
    assert tree.root.feature_idx is not None
    assert tree.root.threshold is not None
    
    # Traverse to verify structure
    depth = max_depth_traverse(tree.root)
    assert depth <= 3  # Max depth should be respected

def test_forest_ensemble():
    """Test forest ensemble predictions"""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)  # type: ignore
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with different tree counts
    regressor1 = BDFRegressor(dist="normal", n_trees=1, max_depth=3, prior_params={"mean": y.mean(), "std": y.std()})
    regressor1.fit(X_train, y_train)
    
    regressor10 = BDFRegressor(dist="normal", n_trees=10, max_depth=3, prior_params={"mean": y.mean(), "std": y.std()})
    regressor10.fit(X_train, y_train)
    
    regressor50 = BDFRegressor(dist="normal", n_trees=50, max_depth=3, prior_params={"mean": y.mean(), "std": y.std()})
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
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)  # type: ignore
    
    # For normal_normal, we should be able to get predictive variance
    regressor = BDFRegressor(dist="normal", n_trees=10, prior_params={"mean": y.mean(), "std": y.std()})
    regressor.fit(X, y)
    
    # Call method to get variance estimates
    means, variances = regressor.predict(X, method='params')
    
    # Check results
    assert means.shape == (100,)
    assert variances.shape == (100,)
    assert np.all(variances > 0)  # Variances should be positive

def test_categorical_features():
    """Test handling of categorical features"""
    # Create data with categorical feature
    X = np.random.rand(100, 5)
    X[:, 0] = np.random.choice([0, 1, 2], size=100)  # Categorical feature
    y = 2 * X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1  # Response depends on categorical
    
    regressor = BDFRegressor(dist="normal", n_trees=10, max_depth=3, prior_params={"mean": y.mean(), "std": y.std()})
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
    
    if node.left is None and node.right is None:
        return current_depth
    
    left_depth = max_depth_traverse(node.left, current_depth + 1) if node.left else current_depth
    right_depth = max_depth_traverse(node.right, current_depth + 1) if node.right else current_depth
    
    return max(left_depth, right_depth)