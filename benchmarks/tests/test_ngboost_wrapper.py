import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


def _require_real_ngboost() -> None:
    """Skip runtime tests when another test installed a lightweight stub."""
    try:
        spec = importlib.util.find_spec("ngboost")
    except ValueError:
        spec = None
    if spec is None:
        pytest.skip("NGBoost is not installed in this environment")


@pytest.mark.parametrize(
    ("dist_name", "target", "support_check"),
    [
        ("Normal", np.linspace(-1.0, 1.0, 30), lambda x: np.isfinite(x).all()),
        ("Laplace", np.linspace(-1.0, 1.0, 30), lambda x: np.isfinite(x).all()),
        ("LogNormal", np.linspace(0.1, 3.0, 30), lambda x: np.all(x > 0)),
        ("Exponential", np.linspace(0.0, 3.0, 30), lambda x: np.all(x >= 0)),
        ("Poisson", np.arange(30) % 5, lambda x: np.all(x >= 0) and np.equal(np.mod(x, 1), 0).all()),
    ],
)
def test_ngboost_distribution_smoke(dist_name, target, support_check):
    _require_real_ngboost()

    from benchmarks.models.wrappers import NGBRegressorWrapper

    X = np.arange(60, dtype=float).reshape(-1, 2)
    model = NGBRegressorWrapper(
        dist_name=dist_name,
        n_estimators=2,
        learning_rate=0.05,
        verbose=False,
        random_state=123,
    )
    model.fit(X, target)

    point_pred = model.predict(X[:4])
    sample_pred = model.predict_samples(X[:4], n_samples=7)

    assert point_pred.shape == (4,)
    assert sample_pred.shape == (4, 7)
    assert np.isfinite(point_pred).all()
    assert np.isfinite(sample_pred).all()
    assert support_check(sample_pred)


@pytest.mark.parametrize(
    ("dist_name", "target", "message"),
    [
        ("LogNormal", np.array([0.0, 1.0, 2.0]), "strictly positive"),
        ("Exponential", np.array([-1.0, 1.0, 2.0]), "nonnegative"),
        ("Poisson", np.array([0.0, 1.5, 2.0]), "nonnegative integers"),
    ],
)
def test_ngboost_rejects_targets_outside_support(dist_name, target, message):
    _require_real_ngboost()

    from benchmarks.models.wrappers import NGBRegressorWrapper

    X = np.arange(6, dtype=float).reshape(-1, 2)
    model = NGBRegressorWrapper(dist_name=dist_name, n_estimators=1, verbose=False)
    with pytest.raises(ValueError, match=message):
        model.fit(X, target)


def test_ngboost_config_domains_match_distribution_support():
    config_dir = Path(__file__).parents[1] / "configs" / "model"

    expected = {
        "ngboost_normal.yaml": {
            "real",
            "positive_real",
            "nonnegative_real",
            "integer",
            "nonnegative_integer",
            "positive_integer",
        },
        "ngboost_laplace.yaml": {
            "real",
            "positive_real",
            "nonnegative_real",
            "integer",
            "nonnegative_integer",
            "positive_integer",
        },
        "ngboost_lognormal.yaml": {"positive_real", "positive_integer"},
        "ngboost_exponential.yaml": {"positive_real", "nonnegative_real", "positive_integer", "nonnegative_integer"},
        "ngboost_poisson.yaml": {"positive_integer", "nonnegative_integer"},
    }

    for filename, domains in expected.items():
        with (config_dir / filename).open() as stream:
            config = yaml.safe_load(stream)
        assert set(config["compatible_target_domains"]) == domains
        assert "dist_name" in config["fixed_init_kwargs"]
        assert "dist_name" not in config["tunable_init_kwargs"]
