"""BDF-specific wrappers that only depend on bdf and sklearn (no bench-model packages).

Kept separate from wrappers.py so the default pixi environment can import them
without CatBoost / LightGBM / NGBoost being installed.
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split as TTS

PredictionType = Literal["samples", "quantiles"]

_CAL_N_SAMPLES = 500  # samples used to compute CQR scores during fit


class ConformalizedSamplesWrapper(RegressorMixin, BaseEstimator):
    """CQR-style split-conformal wrapper for distributional models with predict_samples().

    Uses Conformalized Quantile Regression (CQR) scoring so that the conformal
    correction is applied *relative to the model's own predictive intervals*, not
    on top of them.  For a well-calibrated base model q_hat ≈ 0 and the
    correction is negligible; for an under-covering model the distribution is
    expanded just enough to reach the nominal level.

    Contrast with the naive "add point-prediction residuals to samples" approach,
    which doubles the variance because the base samples already encode the model's
    own uncertainty.

    Usage:
        wrapper = ConformalizedSamplesWrapper(
            base_model_class=BDFRegressor,
            base_model_init_kwargs={"dist": "NormalMuNormal", "random_state": 42},
            base_model_params={"mu_mu": "auto", "score_correction": "bic"},
            cal_coverage=0.90,
        )
        wrapper.fit(X_train, y_train)
        samples = wrapper.predict_samples(X_test, n_samples=500)
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(
        self,
        base_model_class,
        base_model_init_kwargs=None,
        base_model_params=None,
        cal_fraction: float = 0.2,
        cal_coverage: float = 0.90,
        random_state=None,
    ):
        self.base_model_class = base_model_class
        self.base_model_init_kwargs = base_model_init_kwargs or {}
        self.base_model_params = base_model_params or {}
        self.cal_fraction = cal_fraction
        self.cal_coverage = cal_coverage
        self.random_state = random_state

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_np = (y.values if hasattr(y, "values") else np.asarray(y)).ravel()

        X_train, X_calib, y_train, y_calib = TTS(
            X_np, y_np, test_size=self.cal_fraction, random_state=self.random_state or 1234
        )

        if self.base_model_params:
            self.estimator_ = self.base_model_class(**self.base_model_init_kwargs, params=self.base_model_params)
        else:
            self.estimator_ = self.base_model_class(**self.base_model_init_kwargs)

        self.estimator_.fit(X_train, y_train)

        # CQR conformality scores: max(q_lo - y, y - q_hi), negative = covered
        cal_samples = self.estimator_.predict_samples(X_calib, n_samples=_CAL_N_SAMPLES)
        alpha = 1.0 - self.cal_coverage
        q_lo = np.percentile(cal_samples, 100 * alpha / 2, axis=1)
        q_hi = np.percentile(cal_samples, 100 * (1 - alpha / 2), axis=1)
        scores = np.maximum(q_lo - y_calib, y_calib - q_hi)

        # Finite-sample adjusted quantile (append +inf for formal guarantee)
        n = len(scores)
        level = np.ceil((n + 1) * (1 - alpha)) / n
        level = min(level, 1.0)
        scores_inf = np.append(scores, np.inf)
        self.q_hat_ = float(np.quantile(scores_inf, level, method="higher"))
        self.n_calib_ = n
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_samples(self, X, n_samples: int) -> np.ndarray:
        base_samples = self.estimator_.predict_samples(X, n_samples=n_samples)
        if self.q_hat_ == 0.0:
            return base_samples
        # Expand each sample distribution around its median by q_hat:
        # samples above the median are shifted up, samples below are shifted down.
        # This is equivalent to widening the CQR interval by q_hat on each side.
        centers = np.median(base_samples, axis=1, keepdims=True)
        return base_samples + np.sign(base_samples - centers) * self.q_hat_

    def get_params(self, deep=True):
        return {
            "base_model_class": self.base_model_class,
            "base_model_init_kwargs": self.base_model_init_kwargs,
            "base_model_params": self.base_model_params,
            "cal_fraction": self.cal_fraction,
            "cal_coverage": self.cal_coverage,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
