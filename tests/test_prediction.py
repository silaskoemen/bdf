import numpy as np
import pytest
from sklearn.datasets import make_regression

from bdf.tree_classes.bdf_regressor import BDFRegressor

# Define model parameters for tests
N_SAMPLES = 100
N_FEATURES = 5
N_TREES = 10
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def regression_data():
    """Generate synthetic regression data."""
    X, y = make_regression(  # type: ignore[reportAssignmentType]
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=10.0,
        random_state=RANDOM_STATE,
    )
    return X, y


@pytest.fixture(scope="module")
def fitted_model(regression_data):
    """Fixture for a fitted BDFRegressor model."""
    X, y = regression_data
    model = BDFRegressor(n_trees=N_TREES, random_state=RANDOM_STATE, min_samples_leaf=5)
    model.fit(X, y, standardize_y=True)
    return model


def test_predict_mean(fitted_model, regression_data):
    """Test the predict_mean method for correct shape, type, and aggregation logic."""
    X, _ = regression_data
    predictions = fitted_model.predict_mean(X)

    # Test shape and type
    assert predictions.shape == (N_SAMPLES,)
    assert predictions.dtype == np.float64

    # Manually verify aggregation for the first observation
    first_obs = X[[0], :]
    tree_means = np.array([tree.predict_mean(first_obs)[0] for tree in fitted_model.trees])
    manual_mean = np.mean(tree_means)
    manual_mean_unstandardized = fitted_model._unstandardize_y(manual_mean)

    assert np.isclose(predictions[0], manual_mean_unstandardized)


def test_predict_variance(fitted_model, regression_data):
    """Test the predict_variance method for correct shape, type, and aggregation logic."""
    X, _ = regression_data
    variances = fitted_model.predict_variance(X)

    # Test shape, type, and non-negativity
    assert variances.shape == (N_SAMPLES,)
    assert variances.dtype == np.float64
    assert np.all(variances >= 0)

    # Manually verify Law of Total Variance for the first observation
    first_obs = X[[0], :]
    tree_means = np.array([tree.predict_mean(first_obs)[0] for tree in fitted_model.trees])
    tree_vars = np.array([tree.predict_variance(first_obs)[0] for tree in fitted_model.trees])

    expected_variance = np.mean(tree_vars)
    variance_of_expectation = np.var(tree_means)
    manual_variance = expected_variance + variance_of_expectation
    manual_variance_unstandardized = manual_variance * fitted_model.y_std**2

    assert np.isclose(variances[0], manual_variance_unstandardized)


def test_predict_median(fitted_model, regression_data):
    """Test the predict_median method."""
    X, _ = regression_data
    n_samples = 500
    medians = fitted_model.predict_median(X, n_samples=n_samples)

    # Test shape and type
    assert medians.shape == (N_SAMPLES,)
    assert medians.dtype == np.float64

    # Manually verify for the first observation
    first_obs = X[[0], :]
    manual_samples = fitted_model._get_pooled_samples(first_obs, n_samples=n_samples)
    manual_median = np.median(manual_samples)
    manual_median_unstandardized = fitted_model._unstandardize_y(manual_median)

    assert np.isclose(medians[0], manual_median_unstandardized, atol=0.1)


def test_predict_quantiles(fitted_model, regression_data):
    """Test the predict_quantiles method for single and multiple quantiles."""
    X, _ = regression_data
    n_samples = 500

    # Test single quantile
    q_single = 0.75
    quantiles_single = fitted_model.predict_quantiles(X, q=q_single, n_samples=n_samples)
    assert quantiles_single.shape == (N_SAMPLES,)
    assert quantiles_single.dtype == np.float64

    # Test multiple quantiles
    q_multi = [0.25, 0.5, 0.75]
    quantiles_multi = fitted_model.predict_quantiles(X, q=q_multi, n_samples=n_samples)
    assert quantiles_multi.shape == (N_SAMPLES, len(q_multi))
    assert quantiles_multi.dtype == np.float64

    # Manually verify for the first observation
    first_obs = X[[0], :]
    manual_samples = fitted_model._get_pooled_samples(first_obs, n_samples=n_samples)
    manual_quantiles = np.quantile(manual_samples, q=q_multi)
    manual_quantiles_unstandardized = fitted_model._unstandardize_y(manual_quantiles)

    assert np.allclose(quantiles_multi[0], manual_quantiles_unstandardized, atol=0.1)


def test_predict_samples(fitted_model, regression_data):
    """Test the predict_samples method for correct shape and type."""
    X, _ = regression_data
    n_samples = 50
    samples = fitted_model.predict_samples(X, n_samples=n_samples)

    assert samples.shape == (N_SAMPLES, n_samples)
    assert samples.dtype == np.float64


def test_predict_params(fitted_model, regression_data):
    """Test the predict_params method for correct shape and type."""
    X, _ = regression_data
    params = fitted_model.predict_params(X)

    # Shape should be (n_obs, n_trees)
    assert params.shape == (N_SAMPLES, N_TREES)
    # Each element should be a dictionary
    assert isinstance(params[0, 0], dict)
    # Check for expected keys (for NormalMuNormal distribution)
    assert "posterior_mu" in params[0, 0]
    assert "posterior_sigma_mu" in params[0, 0]


def test_predict_weighted_mean(fitted_model, regression_data):
    """Test the predict_weighted_mean method."""
    X, _ = regression_data
    weighted_means = fitted_model.predict_weighted_mean(X)

    # Test shape and type
    assert weighted_means.shape == (N_SAMPLES,)
    assert weighted_means.dtype == np.float64

    # Manually verify for the first observation
    first_obs = X[[0], :]
    tree_means = np.array([tree.predict_mean(first_obs)[0] for tree in fitted_model.trees])
    tree_vars = np.array([tree.predict_variance(first_obs)[0] for tree in fitted_model.trees])
    weights = 1.0 / tree_vars
    manual_weighted_mean = np.sum(tree_means * weights) / np.sum(weights)
    manual_weighted_mean_unstandardized = fitted_model._unstandardize_y(manual_weighted_mean)

    assert np.isclose(weighted_means[0], manual_weighted_mean_unstandardized)


def test_prediction_input_validation(regression_data):
    """Test that prediction methods raise errors for invalid input."""
    X, y = regression_data
    unfitted_model = BDFRegressor(n_trees=N_TREES, random_state=RANDOM_STATE)

    # Test prediction on unfitted model
    with pytest.raises(RuntimeError, match="instance is not fitted yet"):
        unfitted_model.predict(X)

    # Test prediction with wrong number of features
    fitted_model = BDFRegressor(n_trees=N_TREES, random_state=RANDOM_STATE)
    fitted_model.fit(X, y)
    X_wrong_features = X[:, : N_FEATURES - 1]
    with pytest.raises(ValueError, match="features, but BDFRegressor was fitted with"):
        fitted_model.predict(X_wrong_features)


# ── compact_memory tests ──


def _collect_nodes(node):
    """Collect all nodes in a tree into (leaves, internals) lists."""
    leaves, internals = [], []
    stack = [node]
    while stack:
        n = stack.pop()
        if n._is_leaf():
            leaves.append(n)
        else:
            internals.append(n)
            if n.left_node is not None:
                stack.append(n.left_node)
            if n.right_node is not None:
                stack.append(n.right_node)
    return leaves, internals


def test_compact_memory_strips_internal_posteriors(regression_data):
    """compact_memory=True removes posterior_params from internal nodes."""
    X, y = regression_data
    model = BDFRegressor(n_trees=5, random_state=RANDOM_STATE, min_samples_leaf=5, compact_memory=True)
    model.fit(X, y)

    for tree in model.trees:
        leaves, internals = _collect_nodes(tree.root)
        assert len(internals) > 0, "Tree should have internal nodes"
        for node in leaves:
            assert hasattr(node, "posterior_params"), "Leaf nodes must retain posterior_params"
        for node in internals:
            assert not hasattr(node, "posterior_params"), "Internal nodes should have posterior_params stripped"


def test_no_compact_memory_retains_internal_posteriors(regression_data):
    """compact_memory=False keeps posterior_params on all nodes."""
    X, y = regression_data
    model = BDFRegressor(n_trees=5, random_state=RANDOM_STATE, min_samples_leaf=5, compact_memory=False)
    model.fit(X, y)

    for tree in model.trees:
        leaves, internals = _collect_nodes(tree.root)
        for node in leaves + internals:
            assert hasattr(node, "posterior_params"), "All nodes should retain posterior_params"


def test_compact_memory_predictions_match(regression_data):
    """Predictions are identical regardless of compact_memory setting."""
    X, y = regression_data
    kwargs = dict(n_trees=10, random_state=RANDOM_STATE, min_samples_leaf=5)

    model_compact = BDFRegressor(**kwargs, compact_memory=True)
    model_full = BDFRegressor(**kwargs, compact_memory=False)
    model_compact.fit(X, y)
    model_full.fit(X, y)

    np.testing.assert_array_equal(model_compact.predict_mean(X), model_full.predict_mean(X))
    np.testing.assert_array_equal(model_compact.predict_variance(X), model_full.predict_variance(X))
    np.testing.assert_array_equal(
        model_compact.predict_samples(X, n_samples=10), model_full.predict_samples(X, n_samples=10)
    )

    # Also verify tree-level log-likelihood (not exposed on regressor)
    for t_compact, t_full in zip(model_compact.trees, model_full.trees):
        np.testing.assert_array_equal(t_compact.predict_log_likelihood(X, y), t_full.predict_log_likelihood(X, y))
