import importlib.util
import sys
import types

import numpy as np


class _OptionalDependencyStub:
    def __init__(self, *args, **kwargs):
        pass


def _install_optional_benchmark_stubs():
    if importlib.util.find_spec("catboost") is None:
        catboost = types.ModuleType("catboost")
        catboost.CatBoostRegressor = _OptionalDependencyStub
        sys.modules["catboost"] = catboost

    if importlib.util.find_spec("ngboost") is None:
        ngboost = types.ModuleType("ngboost")
        ngboost.NGBClassifier = _OptionalDependencyStub
        ngboost.NGBRegressor = _OptionalDependencyStub
        sys.modules["ngboost"] = ngboost

        distns = types.ModuleType("ngboost.distns")
        for name in ("Bernoulli", "Exponential", "LogNormal", "Normal", "Poisson"):
            setattr(distns, name, _OptionalDependencyStub)
        sys.modules["ngboost.distns"] = distns

        scores = types.ModuleType("ngboost.scores")

        class LogScore:
            def grad(self, Y, natural=True):
                raise NotImplementedError

        scores.LogScore = LogScore
        sys.modules["ngboost.scores"] = scores


_install_optional_benchmark_stubs()

from benchmarks.models.wrappers import DRFWrapper


class DummyDRFWrapper(DRFWrapper):
    def __init__(self, **kwargs):
        kwargs.setdefault("backend", "inprocess")
        super().__init__(**kwargs)
        self.batch_shapes = []

    def _predict_weights_batch(self, X_arr: np.ndarray) -> np.ndarray:
        self.batch_shapes.append(X_arr.shape)
        return np.ones((len(X_arr), len(self._y_train)), dtype=float)


def test_drf_predict_weights_batches_and_normalizes():
    model = DummyDRFWrapper(predict_batch_size=2)
    model._fit_obj = object()
    model._y_train = np.array([1.0, 2.0, 3.0])

    weights = model._predict_weights(np.arange(10, dtype=float).reshape(5, 2))

    assert model.batch_shapes == [(2, 2), (2, 2), (1, 2)]
    assert weights.shape == (5, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(weights, np.full((5, 3), 1.0 / 3.0))


def test_drf_predict_weights_handles_empty_input():
    model = DummyDRFWrapper(predict_batch_size=2)
    model._fit_obj = object()
    model._y_train = np.array([1.0, 2.0, 3.0])

    weights = model._predict_weights(np.empty((0, 2), dtype=float))

    assert weights.shape == (0, 3)
    assert model.batch_shapes == []


def test_drf_get_set_params_includes_predict_batch_size():
    model = DRFWrapper(predict_batch_size=17, backend="inprocess")

    assert model.get_params()["predict_batch_size"] == 17
    assert model.get_params()["backend"] == "inprocess"

    model.set_params(predict_batch_size=9, backend="subprocess")

    assert model.predict_batch_size == 9
    assert model.backend == "subprocess"
