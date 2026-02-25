import numpy as np
import pytest
from sklearn.datasets import make_regression

from bdf.tree_classes.bdf_regressor import BDFRegressor


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(n_samples=500, n_features=5, noise=10.0, random_state=42)
    return X, y


def test_oob_weights_basic(regression_data):
    """OOB weights are computed and normalized correctly."""
    X, y = regression_data
    model = BDFRegressor(n_trees=10, subsample=0.8, oob_weights=True, random_state=42, min_samples_leaf=5)
    model.fit(X, y)

    assert model.tree_weights_ is not None
    assert model.oob_scores_ is not None
    assert model.tree_weights_.shape == (10,)
    assert model.oob_scores_.shape == (10,)
    np.testing.assert_almost_equal(model.tree_weights_.sum(), 1.0)
    assert np.all(model.tree_weights_ >= 0)


def test_oob_weights_disabled(regression_data):
    """Without oob_weights, tree_weights_ is None."""
    X, y = regression_data
    model = BDFRegressor(n_trees=5, subsample=0.8, oob_weights=False, random_state=42, min_samples_leaf=5)
    model.fit(X, y)

    assert model.tree_weights_ is None
    assert model.oob_scores_ is None


def test_oob_weights_fallback_high_subsample():
    """oob_weights=True with subsample=1.0 warns and falls back to uniform weights."""
    model = BDFRegressor(n_trees=5, subsample=1.0, oob_weights=True, random_state=42, min_samples_leaf=5)
    X, y = make_regression(n_samples=50, n_features=3, random_state=42)
    with pytest.warns(UserWarning, match="no effect when subsample >= 1.0"):
        model.fit(X, y)
    assert model.tree_weights_ is None
    assert model.oob_scores_ is None
    # Predictions should still work (uniform weights)
    assert model.predict_mean(X).shape == (50,)


def test_oob_predictions_work(regression_data):
    """All prediction methods work with OOB weights enabled."""
    X, y = regression_data
    model = BDFRegressor(n_trees=10, subsample=0.8, oob_weights=True, random_state=42, min_samples_leaf=5)
    model.fit(X, y)

    mean = model.predict_mean(X[:5])
    assert mean.shape == (5,)

    var = model.predict_variance(X[:5])
    assert var.shape == (5,)
    assert np.all(var >= 0)

    samples = model.predict_samples(X[:5], n_samples=100)
    assert samples.shape == (5, 100)

    wmean = model.predict_weighted_mean(X[:5])
    assert wmean.shape == (5,)


def test_uniform_weights_match_no_oob(regression_data):
    """With high temperature (near-uniform weights), results approximate no-OOB."""
    X, y = regression_data
    model_oob = BDFRegressor(
        n_trees=10, subsample=0.8, oob_weights=True, oob_temperature=1e6, random_state=42, min_samples_leaf=5
    )
    model_oob.fit(X, y)

    model_no_oob = BDFRegressor(n_trees=10, subsample=0.8, oob_weights=False, random_state=42, min_samples_leaf=5)
    model_no_oob.fit(X, y)

    mean_oob = model_oob.predict_mean(X[:10])
    mean_no = model_no_oob.predict_mean(X[:10])
    np.testing.assert_allclose(mean_oob, mean_no, rtol=1e-2)
