import os

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def test_xgboostlss_gaussian_mixture_smoke():
    pytest.importorskip("catboost")
    pytest.importorskip("xgboostlss")

    from benchmarks.metrics.regression import crps_wrapper
    from benchmarks.models.wrappers import XGBoostLSSRegressorWrapper

    rng = np.random.default_rng(123)
    X = rng.normal(size=(48, 3))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.2, size=48)

    model = XGBoostLSSRegressorWrapper(
        dist_name="GaussianMixture",
        response_fn="softplus",
        mixture_components=2,
        n_estimators=2,
        max_depth=1,
        eta=0.1,
        nthread=1,
        random_state=123,
        verbosity=0,
    )
    model.fit(X, y)

    point_pred = model.predict(X[:5])
    sample_pred = model.predict_samples(X[:5], n_samples=7)

    assert point_pred.shape == (5,)
    assert sample_pred.shape == (5, 7)
    assert np.all(np.isfinite(point_pred))
    assert np.all(np.isfinite(sample_pred))
    assert np.isfinite(crps_wrapper(y[:5], sample_pred))
