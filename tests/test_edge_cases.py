import numpy as np
import pytest

from bdf import _bdf_rs as bdf_rs
from bdf.tree_classes.bdf_regressor import BDFRegressor


def test_empty_dataset():
    """Test handling of empty datasets"""
    X = np.array([], dtype=np.float64).reshape(0, 2)
    y = np.array([], dtype=np.float64)

    regressor = BDFRegressor(dist="NormalMuNormal", params={"mu_mu": 0.0, "sigma_mu": 1.0})

    # Should raise ValueError for empty dataset
    with pytest.raises(ValueError):
        regressor.fit(X, y)


def test_constant_response():
    """Test with constant response variable"""
    X = np.random.rand(100, 5)
    y = np.ones(100) * 3.0  # Constant response

    regressor = BDFRegressor(dist="NormalMuNormal", max_depth=5, params={"mu_mu": 3.0, "sigma_mu": 0.1})
    regressor.fit(X, y)

    # Predictions should all be close to 3.0
    preds = regressor.predict(X)
    assert np.all(np.abs(preds - 3.0) < 0.1)


def test_nan_values():
    """Test robustness against NaN values"""
    X = np.random.rand(100, 5)
    X[10, 2] = np.nan  # Add a NaN value
    y = np.random.rand(100)

    regressor = BDFRegressor(dist="NormalMuNormal", params={"mu_mu": y.mean(), "sigma_mu": y.std()})

    # Should handle NaNs without errors if we preprocess
    X_clean = np.nan_to_num(X)
    regressor.fit(X_clean, y)

    # Should also predict correctly for data with NaNs after preprocessing
    X_test = np.random.rand(10, 5)
    X_test[2, 3] = np.nan
    X_test_clean = np.nan_to_num(X_test)
    preds = regressor.predict(X_test_clean)
    assert len(preds) == 10


def test_extreme_values():
    """Test with extreme values"""
    X = np.random.rand(100, 5)
    y = np.random.rand(100) * 1e9  # Very large values

    regressor = BDFRegressor(dist="NormalMuNormal", params={"mu_mu": y.mean(), "sigma_mu": y.std()})

    # Should handle large values without numerical issues
    regressor.fit(X, y)
    preds = regressor.predict(X)
    assert not np.any(np.isnan(preds))
    assert not np.any(np.isinf(preds))


def test_duplicate_features():
    """Test with duplicate features"""
    X = np.random.rand(100, 3)
    X = np.hstack([X, X])  # Duplicate all features
    y = np.random.rand(100)

    regressor = BDFRegressor(dist="NormalMuNormal", params={"mu_mu": y.mean(), "sigma_mu": y.std()})
    regressor.fit(X, y)

    # Should still predict without errors
    preds = regressor.predict(X)
    assert len(preds) == 100


def test_imbalanced_split():
    """Test with highly imbalanced split points"""
    np.random.seed(42)
    n = 200
    X = np.random.rand(n, 5)
    # Create a clear signal: top 10% of feature 0 have high y
    y = np.where(X[:, 0] > 0.9, 50.0, 1.0)

    regressor = BDFRegressor(
        dist="NormalMuNormal",
        params={"mu_mu": "auto", "sigma_mu": "auto", "sigma_mu_auto_scale": 1.0},
        n_trees=50,
        min_samples_leaf=5,
    )
    regressor.fit(X, y)

    # Predictions for high-X0 samples should be higher than low-X0
    high_mask = X[:, 0] > 0.9
    low_mask = X[:, 0] < 0.5
    high_pred = regressor.predict(X[high_mask]).mean()
    low_pred = regressor.predict(X[low_mask]).mean()

    assert high_pred > low_pred + 1.0
