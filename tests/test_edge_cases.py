import bdf_rs
import numpy as np
import pytest

from bdf.tree_classes.bdf_regressor import BDFRegressor


def test_empty_dataset():
    """Test handling of empty datasets"""
    X = np.array([], dtype=np.float64).reshape(0, 2)
    y = np.array([], dtype=np.float64)

    regressor = BDFRegressor(dist="normal", prior_params={"mean": 0.0, "std": 1.0})

    # Should raise ValueError for empty dataset
    with pytest.raises(ValueError):
        regressor.fit(X, y)


def test_single_sample():
    """Test handling of single sample dataset"""
    X = np.array([[1.0, 2.0]])
    y = np.array([5.0])

    regressor = BDFRegressor(dist="normal", prior_params={"mean": 5.0, "std": 0.1})

    # Should fit without errors but not create any splits
    regressor.fit(X, y)

    # Prediction should be close to the input value
    pred = regressor.predict(X)
    assert np.abs(pred[0] - y[0]) < 0.1


def test_constant_response():
    """Test with constant response variable"""
    X = np.random.rand(100, 5)
    y = np.ones(100) * 3.0  # Constant response

    regressor = BDFRegressor(dist="normal", max_depth=5, prior_params={"mean": 3.0, "std": 0.1})
    regressor.fit(X, y)

    # Predictions should all be close to 3.0
    preds = regressor.predict(X)
    assert np.all(np.abs(preds - 3.0) < 0.1)


def test_nan_values():
    """Test robustness against NaN values"""
    X = np.random.rand(100, 5)
    X[10, 2] = np.nan  # Add a NaN value
    y = np.random.rand(100)

    regressor = BDFRegressor(dist="normal", prior_params={"mean": y.mean(), "std": y.std()})

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

    regressor = BDFRegressor(dist="normal", prior_params={"mean": y.mean(), "std": y.std()})

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

    regressor = BDFRegressor(dist="normal", prior_params={"mean": y.mean(), "std": y.std()})
    regressor.fit(X, y)

    # Should still predict without errors
    preds = regressor.predict(X)
    assert len(preds) == 100


def test_imbalanced_split():
    """Test with highly imbalanced split points"""
    X = np.random.rand(100, 5)
    # Create a very imbalanced step function (99:1 split)
    y = np.ones(100)
    y[0] = 100.0

    regressor = BDFRegressor(dist="normal", prior_params={"mean": 0, "std": 5})
    regressor.fit(X, y)

    # Should be able to identify the extreme value
    outlier_pred = regressor.predict(X[[0]])
    regular_pred = regressor.predict(X[[1]])

    # Prediction for outlier should be different
    assert abs(outlier_pred[0] - regular_pred[0]) > 1.0
