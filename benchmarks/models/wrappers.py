# pyright: reportMissingImports=false
import gc
import multiprocessing as mp
import queue
import warnings
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Exponential, LogNormal, Normal, Poisson
from ngboost.scores import LogScore
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel as Ker
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split as TTS
from sklearn.neighbors import KNeighborsRegressor

# Type alias for prediction type
PredictionType = Literal["samples", "quantiles"]


def configure_kernel(kernel_comb: str) -> Ker:
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

    rbf = RBF()
    rq = RationalQuadratic()
    matern = Matern()

    if kernel_comb == "RBF+RQ":
        kernel = rbf + rq
    elif kernel_comb == "RBF*RQ":
        kernel = rbf * rq
    elif kernel_comb == "RBF+Matern":
        kernel = rbf + matern
    elif kernel_comb == "RBF*Matern":
        kernel = rbf * matern
    elif kernel_comb == "RBF":
        kernel = rbf
    else:
        raise ValueError(f"Unknown kernel combination: {kernel_comb}")

    return C(1.0) * kernel


class GPRegressorWrapper(GaussianProcessRegressor):
    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, kernel_comb="RBF", random_state=None, normalize_y=True, max_samples=1000, **kwargs):
        self.kernel_comb = kernel_comb  # Store the string representation
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.max_samples = max_samples
        self.kwargs = kwargs

        # Configure the actual kernel object
        kernel_obj = configure_kernel(kernel_comb)
        init_kwargs = {k: v for k, v in kwargs.items() if k != "kernel_comb"}
        # Initialize parent
        # Note: GaussianProcessRegressor takes normalize_y in __init__
        super().__init__(kernel=kernel_obj, normalize_y=normalize_y, random_state=random_state, **init_kwargs)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        n_samples = len(X)
        if n_samples > self.max_samples:
            raise ValueError(
                f"GP training data too large ({n_samples} > {self.max_samples}). "
                f"Gaussian processes scale poorly beyond ~1000 samples due to O(n³) complexity."
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            super().fit(X, y)
        return self

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int, batch_size: int = 50) -> np.ndarray:
        """Generate samples from the predictive distribution at inputs X.

        Uses batching to avoid memory issues with large datasets.

        Returns:
            np.ndarray of shape (n_observations, n_samples)
        """
        n_obs = len(X)
        samples = np.empty((n_obs, n_samples))

        # Process in batches to avoid OOM
        for start_idx in range(0, n_obs, batch_size):
            end_idx = min(start_idx + batch_size, n_obs)
            X_batch = X[start_idx:end_idx]

            y_mean, y_std = self.predict(X_batch, return_std=True)  # pyright: ignore[reportAssignmentType]
            y_std = np.maximum(y_std, 1e-10)

            samples[start_idx:end_idx] = np.random.normal(
                loc=y_mean[:, np.newaxis], scale=y_std[:, np.newaxis], size=(end_idx - start_idx, n_samples)
            )

        return samples


class GPClassifierWrapper(GaussianProcessClassifier):
    def __init__(self, kernel_comb="RBF", random_state=None, max_samples=1000, **kwargs):
        self.kernel_comb = kernel_comb
        self.random_state = random_state
        self.max_samples = max_samples
        self.kwargs = kwargs

        # Configure the actual kernel object
        kernel_obj = configure_kernel(kernel_comb)

        # Initialize parent
        super().__init__(kernel=kernel_obj, random_state=random_state, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = len(X)
        if n_samples > self.max_samples:
            raise ValueError(
                f"GP training data too large ({n_samples} > {self.max_samples}). "
                f"Gaussian processes scale poorly beyond ~1000 samples due to O(n³) complexity."
            )
        super().fit(X, y)
        return self


# =============================================================================
# MONKEY PATCH: Fix NGBoost gradient shape for NumPy compatibility
# =============================================================================
# The issue is that for 1-parameter distributions, the gradient is (N,),
# but the metric is (N, 1, 1). np.linalg.solve expects (N, 1) for the RHS.

_original_grad = LogScore.grad


def _patched_grad(self, Y, natural=True):
    # Ensure Y is 1D to avoid (N, 1) vs (N,) broadcasting -> (N, N)
    if hasattr(Y, "ndim") and Y.ndim > 1:
        Y = Y.ravel()

    # Calculate metric handling both signatures: metric(Y) and metric()
    try:
        metric = self.metric(Y)
    except TypeError:
        metric = self.metric()

    grad = self.d_score(Y)

    if natural:
        # Handle the shape mismatch for np.linalg.solve
        # metric is (N, P, P) where P is the number of parameters
        # grad should be (N, P) for solve to work correctly

        if metric.ndim == 3:
            n_samples = metric.shape[0]
            n_params = metric.shape[1]

            # Force grad to be (N, P) shape
            grad = np.asarray(grad)
            if grad.ndim == 1:
                # 1D grad of shape (N,) -> (N, 1) for single-param distributions
                grad = grad.reshape(n_samples, n_params)
            elif grad.ndim == 2:
                # Check for broadcasting error (N, N) instead of (N, P)
                if grad.shape[0] == grad.shape[1] == n_samples and n_params == 1:
                    grad = np.diag(grad).reshape(n_samples, n_params)
                elif grad.shape != (n_samples, n_params):
                    grad = grad.reshape(n_samples, n_params)

            # For np.linalg.solve with batched inputs:
            # metric: (N, P, P), grad needs to be (N, P, 1) for proper broadcasting
            # Then squeeze the result back to (N, P)
            grad = grad[..., np.newaxis]  # (N, P) -> (N, P, 1)

        # At this point grad should be (N, P, 1) and metric should be (N, P, P)
        try:
            grad = np.linalg.solve(metric, grad)
            # Squeeze back to (N, P) if we added the extra dimension
            if grad.ndim == 3:
                grad = grad.squeeze(-1)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices (rare but possible)
            reg = np.eye(metric.shape[1]) * 1e-6
            grad = np.linalg.solve(metric + reg, grad)
            if grad.ndim == 3:
                grad = grad.squeeze(-1)

    return grad


# Apply the patch
LogScore.grad = _patched_grad

# =============================================================================
# NGBoost Wrappers
# =============================================================================

# Mapping from NGBoost distribution names to scipy.stats distributions
NGBOOST_TO_SCIPY = {
    "Normal": lambda params, n_samples: stats.norm.rvs(
        loc=params["loc"], scale=params["scale"], size=(n_samples, len(params["loc"]))
    ).T,  # pyright: ignore[reportAttributeAccessIssue]
    "LogNormal": lambda params, n_samples: stats.lognorm.rvs(
        s=params["s"], scale=params["scale"], size=(n_samples, len(params["s"]))
    ).T,  # pyright: ignore[reportAttributeAccessIssue]
    "Exponential": lambda params, n_samples: stats.expon.rvs(
        scale=params["scale"], size=(n_samples, len(params["scale"]))
    ).T,  # pyright: ignore[reportAttributeAccessIssue]
    "Poisson": lambda params, n_samples: stats.poisson.rvs(
        mu=params["mu"], size=(n_samples, len(params["mu"]))
    ).T,  # pyright: ignore[reportAttributeAccessIssue]
    "Bernoulli": lambda params, n_samples: stats.bernoulli.rvs(
        p=params["p1"], size=(n_samples, len(params["p1"]))
    ).T,  # pyright: ignore[reportAttributeAccessIssue]
}


def return_dist_by_name(dist_name: str):
    dist_map = {
        "Normal": Normal,
        "LogNormal": LogNormal,
        "Exponential": Exponential,
        "Poisson": Poisson,
        "Bernoulli": Bernoulli,
    }
    if dist_name not in dist_map:
        raise ValueError(f"Distribution '{dist_name}' not recognized.")
    return dist_map[dist_name]


class NGBRegressorWrapper(NGBRegressor):
    """Uses the monkey patched NGBoost for regression tasks.

    NOTE: Internal use of np.bool was deprecated in numpy 1.20.0, but we need a higher numpy
    for python minor compatibility. Change the following two lines in ngboost/helpers.py (ll 21-22) if needed:
    ```python
    Y = np.empty(dtype=[("Event", np.bool_), ("Time", np.float64)], shape=T.shape[0])
    Y["Event"] = E.astype(np.bool_)
    ```
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(
        self,
        dist_name="Normal",
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        natural_gradient=True,
        verbose=False,
        random_state=None,
        **kwargs,
    ):
        self.dist_name = dist_name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.natural_gradient = natural_gradient
        self.verbose = verbose
        self.random_state = random_state
        self.kwargs = kwargs

        self.dist = return_dist_by_name(dist_name)
        super().__init__(
            Dist=self.dist,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            natural_gradient=natural_gradient,
            verbose=verbose,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, **kwargs):
        X_np: np.ndarray = (
            X.values if hasattr(X, "values") else X
        )  # pyright: ignore[reportAttributeAccessIssue] # ty:ignore[invalid-assignment]
        y_np: np.ndarray = (
            y.values if hasattr(y, "values") else y
        )  # pyright: ignore[reportAttributeAccessIssue] # ty:ignore[invalid-assignment]

        if y_np.ndim > 1:
            y_np = y_np.ravel()
        super().fit(X_np, y_np, **kwargs)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_np = X.values if hasattr(X, "values") else X  # pyright: ignore[reportAttributeAccessIssue]
        preds = super().predict(X_np)
        preds[np.isinf(preds)] = np.finfo(np.float64).max  # sklearn raises error on inf
        preds[np.isneginf(preds)] = np.finfo(np.float64).min
        return preds

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int) -> np.ndarray:
        """Generate samples from the predictive distribution at inputs X.

        Returns:
            np.ndarray of shape (n_observations, n_samples)
        """
        X_np = X.values if hasattr(X, "values") else X  # pyright: ignore[reportAttributeAccessIssue]
        y_dists = self.pred_dist(X_np)
        params = y_dists.params

        # Use the scipy sampling function for this distribution
        sampler = NGBOOST_TO_SCIPY[self.dist_name]
        samples = sampler(params, n_samples)
        # Set infinite samples to large finite values
        samples[np.isinf(samples)] = np.finfo(np.float64).max
        samples[np.isneginf(samples)] = np.finfo(np.float64).min
        return samples


class NGBClassifierWrapper(NGBClassifier):
    def __init__(
        self,
        dist_name="Bernoulli",
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        natural_gradient=True,
        verbose=False,
        random_state=None,
        **kwargs,
    ):
        self.dist_name = dist_name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.natural_gradient = natural_gradient
        self.verbose = verbose
        self.random_state = random_state
        self.kwargs = kwargs

        self.dist = return_dist_by_name(dist_name)
        super().__init__(
            Dist=self.dist,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            natural_gradient=natural_gradient,
            verbose=verbose,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, **kwargs):
        X_np: np.ndarray = (
            X.values if hasattr(X, "values") else X
        )  # pyright: ignore[reportAttributeAccessIssue]  # ty:ignore[invalid-assignment]
        y_np: np.ndarray = (
            y.values if hasattr(y, "values") else y
        )  # pyright: ignore[reportAttributeAccessIssue]  # ty:ignore[invalid-assignment]

        # Ensure y is 1D integer array for classification
        if y_np.ndim > 1:
            y_np = y_np.ravel()
        y_np = y_np.astype(int)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)
            super().fit(X_np, y_np, **kwargs)
        return self

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int) -> np.ndarray:
        """Generate samples from the predictive distribution at inputs X.

        Returns:
            np.ndarray of shape (n_observations, n_samples)
        """
        X_np = X.values if hasattr(X, "values") else X  # pyright: ignore[reportAttributeAccessIssue]
        y_dists = self.pred_dist(X_np)
        params = y_dists.params

        # Use the scipy sampling function for this distribution
        sampler = NGBOOST_TO_SCIPY[self.dist_name]
        samples = sampler(params, n_samples)
        return samples


class XGBoostLSSRegressorWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for XGBoostLSS univariate regression.

    XGBoostLSS follows the native XGBoost API rather than sklearn's estimator
    API. This wrapper keeps the benchmark interface consistent by exposing
    fit(), predict(), and predict_samples().
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(
        self,
        dist_name="Gaussian",
        stabilization="None",
        response_fn="exp",
        loss_fn="nll",
        mixture_components=2,
        mixture_tau=1.0,
        mixture_hessian_mode="individual",
        n_estimators=100,
        eta=0.05,
        max_depth=3,
        gamma=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1.0,
        booster="gbtree",
        verbosity=0,
        nthread=None,
        random_state=None,
    ):
        self.dist_name = dist_name
        self.stabilization = stabilization
        self.response_fn = response_fn
        self.loss_fn = loss_fn
        self.mixture_components = mixture_components
        self.mixture_tau = mixture_tau
        self.mixture_hessian_mode = mixture_hessian_mode
        self.n_estimators = n_estimators
        self.eta = eta
        self.max_depth = max_depth
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.booster = booster
        self.verbosity = verbosity
        self.nthread = nthread
        self.random_state = random_state
        self.model_ = None

    def _make_distribution(self):
        if self.dist_name == "Gaussian":
            from xgboostlss.distributions.Gaussian import Gaussian

            return Gaussian(
                stabilization=self.stabilization,
                response_fn=self.response_fn,
                loss_fn=self.loss_fn,
            )
        if self.dist_name == "StudentT":
            from xgboostlss.distributions.StudentT import StudentT

            return StudentT(
                stabilization=self.stabilization,
                response_fn=self.response_fn,
                loss_fn=self.loss_fn,
            )
        if self.dist_name == "Laplace":
            from xgboostlss.distributions.Laplace import Laplace

            return Laplace(
                stabilization=self.stabilization,
                response_fn=self.response_fn,
                loss_fn=self.loss_fn,
            )
        if self.dist_name in {"GaussianMixture", "MixtureGaussian"}:
            from xgboostlss.distributions.Gaussian import Gaussian
            from xgboostlss.distributions.Mixture import Mixture

            return Mixture(
                Gaussian(
                    stabilization=self.stabilization,
                    response_fn=self.response_fn,
                    loss_fn=self.loss_fn,
                ),
                M=int(self.mixture_components),
                tau=float(self.mixture_tau),
                hessian_mode=self.mixture_hessian_mode,
            )
        raise ValueError(f"Unsupported XGBoostLSS distribution: {self.dist_name}")

    def _xgb_params(self) -> dict:
        params = {
            "eta": self.eta,
            "max_depth": int(self.max_depth),
            "gamma": self.gamma,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "booster": self.booster,
            "verbosity": self.verbosity,
        }
        if self.random_state is not None:
            params["seed"] = int(self.random_state)
        if self.nthread is not None:
            params["nthread"] = int(self.nthread)
        return params

    def fit(self, X, y):
        import xgboost as xgb
        from xgboostlss.model import XGBoostLSS

        self.model_ = XGBoostLSS(self._make_distribution())
        dtrain = xgb.DMatrix(X, label=np.asarray(y), nthread=self.nthread)
        self.model_.train(
            self._xgb_params(),
            dtrain,
            num_boost_round=int(self.n_estimators),
            verbose_eval=False,
        )
        return self

    def _dmatrix(self, X):
        import xgboost as xgb

        return xgb.DMatrix(X, nthread=self.nthread)

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model has not been fitted.")
        params = self.model_.predict(self._dmatrix(X), pred_type="parameters")
        loc_cols = sorted(
            [c for c in params.columns if str(c).startswith("loc_")],
            key=lambda c: int(str(c).split("_", 1)[1]),
        )
        mix_cols = sorted(
            [c for c in params.columns if str(c).startswith("mix_prob_")],
            key=lambda c: int(str(c).split("_", 2)[2]),
        )
        if loc_cols and mix_cols and len(loc_cols) == len(mix_cols):
            loc = params[loc_cols].to_numpy(dtype=float)
            weights = params[mix_cols].to_numpy(dtype=float)
            weight_sums = np.sum(weights, axis=1, keepdims=True)
            weights = np.divide(weights, weight_sums, out=np.zeros_like(weights), where=weight_sums > 0)
            return np.sum(weights * loc, axis=1)
        if "loc" in params:
            return np.asarray(params["loc"], dtype=float)
        if "location" in params:
            return np.asarray(params["location"], dtype=float)
        samples = self.predict_samples(X, n_samples=200)
        return np.mean(samples, axis=1)

    def predict_samples(self, X, n_samples=100):
        if self.model_ is None:
            raise ValueError("Model has not been fitted.")
        samples = self.model_.predict(
            self._dmatrix(X),
            pred_type="samples",
            n_samples=n_samples,
            seed=self.random_state if self.random_state is not None else 1234,
        )
        return np.asarray(samples, dtype=float)


# =============================================================================
# LightGBM Quantile Regression Wrapper
# =============================================================================


class LGBMQuantileRegressorWrapper:
    """LightGBM wrapper that trains multiple quantile regressors for probabilistic predictions."""

    PREDICTION_TYPE: PredictionType = "quantiles"
    DEFAULT_QUANTILES = (0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975)

    def __init__(
        self,
        quantiles: tuple[float, ...] | None = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = None,
        verbose: int = -1,
        **kwargs,
    ):
        self.quantiles = quantiles if quantiles is not None else self.DEFAULT_QUANTILES
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        # Store fitted models for each quantile
        self.models_: dict[float, lgb.LGBMRegressor] = {}

    def _get_base_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "verbose": self.verbose,
            **self.kwargs,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):

        base_params = self._get_base_params()

        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,  # alpha is the quantile level
                **base_params,
            )
            model.fit(X, y, **kwargs)
            self.models_[q] = model

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return median prediction (quantile 0.5)."""
        # Use median if available, otherwise closest to 0.5
        if 0.5 in self.models_:
            return np.array(self.models_[0.5].predict(X))
        else:
            closest_q = min(self.quantiles, key=lambda q: abs(q - 0.5))
            return np.array(self.models_[closest_q].predict(X))

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Predict all quantiles.

        Returns:
            np.ndarray of shape (n_observations, n_quantiles)
        """
        n_obs = X.shape[0]
        n_quantiles = len(self.quantiles)

        predictions = np.empty((n_obs, n_quantiles))
        for i, q in enumerate(sorted(self.quantiles)):
            predictions[:, i] = np.array(self.models_[q].predict(X))

        return predictions

    # sklearn-compatible interface
    def get_params(self, deep=True):
        return {
            "quantiles": self.quantiles,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "verbose": self.verbose,
            **self.kwargs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CatBoostUncertaintyWrapper(CatBoostRegressor):
    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, **kwargs):
        # Force the loss function to be probabilistic
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        return super().fit(X, y, **kwargs)

    def predict(self, X):
        if hasattr(X, "values"):
            X = X.values
        return super().predict(X)[:, 0]  # Return only the mean predictions

    def predict_mean_std(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) from CatBoost's RMSEWithUncertainty output."""
        if hasattr(X, "values"):
            X = X.values
        preds = super().predict(X)
        return preds[:, 0], np.sqrt(np.maximum(preds[:, 1], 0.0))

    def predict_samples(self, X, n_samples=100):
        if hasattr(X, "values"):
            X = X.values

        # CatBoost returns [mean, variance] for RMSEWithUncertainty
        preds = super().predict(X)
        mean = preds[:, 0]
        variance = preds[:, 1]
        std = np.sqrt(variance)
        return np.random.normal(loc=mean[:, np.newaxis], scale=std[:, np.newaxis], size=(X.shape[0], n_samples))


class BayesianRidgeWrapper(BayesianRidge):
    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, random_state=None, **kwargs):
        # Store random_state before calling super().__init__
        # BayesianRidge doesn't accept random_state, but sklearn expects it as an attribute
        self.random_state = random_state
        super().__init__(**kwargs)

    def predict_samples(self, X, n_samples=100) -> np.ndarray:
        # BayesianRidge returns mean and std
        mean, std = self.predict(X, return_std=True)  # pyright: ignore[reportGeneralTypeIssues]
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)

        rng = np.random.default_rng(self.random_state)
        # Sample from Normal(mean, std) -> shape (n_obs, n_samples)
        return rng.normal(
            loc=mean[:, None],
            scale=std[:, None],
            size=(mean.shape[0], n_samples),
        )


class QuantileForestWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for Quantile Regression Forests.
    Requires: pip install quantile-forest
    """

    PREDICTION_TYPE: PredictionType = "quantiles"

    # Default quantiles covering all levels needed by quantile-compatible metrics:
    # - CRPS (any quantiles work)
    # - PICA (needs alpha/2 and 1-alpha/2 for levels 0.5, 0.8, 0.9, 0.95, 0.975, 0.99)
    # - interval_score_50/90/95 (needs 0.25/0.75, 0.05/0.95, 0.025/0.975)
    # - weighted_interval_score (needs 0.5 and pairs for alphas 0.01, 0.025, 0.05, 0.1, 0.2, 0.5)
    # - ci_width_50/90/95 (needs 0.25/0.75, 0.05/0.95, 0.025/0.975)
    # - coverage_curve (needs pairs for levels 0.1 to 0.99)
    DEFAULT_QUANTILES = [
        0.005,
        0.0125,
        0.025,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.975,
        0.9875,
        0.995,
    ]

    def __init__(self, n_estimators=100, quantiles=None, random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.quantiles = quantiles if quantiles is not None else self.DEFAULT_QUANTILES
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y):
        try:
            from quantile_forest import RandomForestQuantileRegressor
        except ImportError:
            raise ImportError("Please install quantile-forest: pip install quantile-forest")

        self.model_ = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, **self.kwargs
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X, quantiles="mean")  # pyright: ignore[reportOptionalMemberAccess]

    def predict_quantiles(self, X) -> np.ndarray:
        """Returns shape (n_obs, n_quantiles).

        QRF can produce non-monotonic quantile predictions (quantile crossings) because each
        quantile level is read off the empirical distribution of leaf samples independently.
        scoringrules.crps_quantile requires monotonic predictions per row, so we sort row-wise
        as a post-hoc isotonic correction (standard practice for QRF).
        """
        sorted_quantiles = sorted(self.quantiles)
        preds = self.model_.predict(X, quantiles=sorted_quantiles)  # pyright: ignore[reportOptionalMemberAccess]
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return np.sort(preds, axis=1)

    def predict_samples(self, X, n_samples=100):
        """Generate approximate predictive samples from the predicted quantile function.

        The underlying quantile forest exposes quantiles rather than analytic sampling.
        For diagnostics that expect samples, draw uniforms and invert the row-wise
        empirical quantile curve by linear interpolation.
        """
        quantile_levels = np.asarray(sorted(self.quantiles), dtype=float)
        quantile_preds = self.predict_quantiles(X)
        rng = np.random.default_rng(self.random_state)
        uniforms = rng.uniform(size=(quantile_preds.shape[0], n_samples))
        samples = np.empty_like(uniforms)
        for i, row in enumerate(quantile_preds):
            samples[i] = np.interp(uniforms[i], quantile_levels, row, left=row[0], right=row[-1])
        return samples

    def get_params(self, deep=True):
        params = {
            "n_estimators": self.n_estimators,
            "quantiles": self.quantiles,
            "random_state": self.random_state,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key in ("n_estimators", "quantiles", "random_state"):
            if key in params:
                setattr(self, key, params.pop(key))
        self.kwargs.update(params)
        return self


def _drf_worker_loop(request_queue, response_queue):
    """Own one embedded R session for a subprocess-backed DRFWrapper."""
    model = None
    try:
        while True:
            command, args = request_queue.get()
            if command == "close":
                break

            try:
                if command == "fit":
                    params, X_arr, y_arr = args
                    params["backend"] = "inprocess"
                    model = DRFWrapper(**params)
                    model.fit(X_arr, y_arr)
                    response_queue.put(("ok", None))
                    continue

                if model is None:
                    raise RuntimeError("DRF worker received predict before fit.")

                if command == "predict":
                    result = model.predict(args[0])
                elif command == "predict_quantiles":
                    result = model.predict_quantiles(args[0])
                elif command == "predict_samples":
                    result = model.predict_samples(args[0], n_samples=args[1])
                else:
                    raise ValueError(f"Unknown DRF worker command: {command}")
                response_queue.put(("ok", result))
            except BaseException as exc:
                import traceback

                response_queue.put(("error", repr(exc), traceback.format_exc()))
    finally:
        try:
            if model is not None:
                model._r_gc()
        except Exception:
            pass


class DRFWrapper(BaseEstimator, RegressorMixin):
    """Distributional Random Forests (Cevid et al., 2022) via rpy2 + the R drf package.

    The deprecated python-package shipped with the upstream drf repo
    (https://github.com/lorismichel/drf) is unmaintained; this wrapper re-implements
    the same call surface against a system R installation. drf returns, for each test
    point, a set of weights over the training responses; quantiles, samples, and the
    predictive mean are derived from those weights without imposing a parametric form.
    """

    PREDICTION_TYPE: PredictionType = "quantiles"

    # Matches QuantileForestWrapper so the benchmark harness sees a consistent quantile grid.
    DEFAULT_QUANTILES = QuantileForestWrapper.DEFAULT_QUANTILES

    def __init__(
        self,
        num_trees: int = 500,
        splitting_rule: str = "FourierMMD",
        num_features: int = 10,
        min_node_size: int = 15,
        mtry: int | None = None,
        sample_fraction: float = 0.5,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        ci_group_size: int = 1,
        predict_batch_size: int = 256,
        backend: str = "subprocess",
        quantiles: list[float] | None = None,
        random_state: int | None = None,
        **kwargs,
    ):
        self.num_trees = num_trees
        self.splitting_rule = splitting_rule
        self.num_features = num_features
        self.min_node_size = min_node_size
        self.mtry = mtry
        self.sample_fraction = sample_fraction
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        # ci.group.size > 1 enables DRF's paired-tree variance estimation, which forces
        # sample.fraction < 0.5. We only consume the weights matrix (not DRF's CIs), so set
        # this to 1 by default to free the sample_fraction tuning range.
        self.ci_group_size = ci_group_size
        self.predict_batch_size = predict_batch_size
        self.backend = backend
        self.quantiles = quantiles if quantiles is not None else list(self.DEFAULT_QUANTILES)
        self.random_state = random_state
        self.kwargs = kwargs
        self._fit_obj = None
        self._y_train = None
        self._worker_process = None
        self._request_queue = None
        self._response_queue = None

    @staticmethod
    def _configure_r_runtime():
        # libomp on macOS is loaded both by conda-numpy and R/BLAS; allow duplicate init.
        import os

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        # rpy2 embeds a single R runtime in the Python process. The DRF benchmark calls into
        # R repeatedly during Optuna and CV evaluation; disabling R's byte-code JIT avoids a
        # known fatal failure mode where R cannot initialize the JIT after its session is stressed.
        os.environ.setdefault("R_ENABLE_JIT", "0")

        import rpy2.robjects as ro

        try:
            ro.r("compiler::enableJIT(0)")
        except Exception:
            pass

    @staticmethod
    def _r_gc():
        try:
            import rpy2.robjects as ro

            ro.r("gc(FALSE)")
        except Exception:
            pass
        gc.collect()

    @classmethod
    def _load_drf(cls):
        cls._configure_r_runtime()

        from rpy2.robjects.packages import importr  # local import keeps module load cheap

        try:
            return importr("drf")
        except Exception as exc:
            raise ImportError(
                "The R package 'drf' is not installed in the R that rpy2 binds to. "
                "Install it with:\n"
                '  Rscript -e \'install.packages("drf", repos="https://cloud.r-project.org")\'\n'
                "(use the Rscript on PATH inside the pixi benchmark env)."
            ) from exc

    def _drf_kwargs(self) -> dict:
        # Map snake_case to drf's dot-named R args. None values are dropped so drf picks defaults.
        out = {
            "num.trees": self.num_trees,
            "splitting.rule": self.splitting_rule,
            "num.features": self.num_features,
            "min.node.size": self.min_node_size,
            "sample.fraction": self.sample_fraction,
            "honesty": self.honesty,
            "honesty.fraction": self.honesty_fraction,
            "ci.group.size": self.ci_group_size,
        }
        if self.mtry is not None:
            out["mtry"] = self.mtry
        if self.random_state is not None:
            out["seed"] = int(self.random_state)
        out.update(self.kwargs)
        return out

    def fit(self, X, y):
        if self.backend == "subprocess":
            return self._fit_subprocess(X, y)
        if self.backend != "inprocess":
            raise ValueError(f"Unknown DRF backend: {self.backend}")

        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter

        drf_pkg = self._load_drf()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self._y_train = y_arr.ravel()

        if self.random_state is not None:
            ro.r(f"set.seed({int(self.random_state)})")

        # rpy2 needs `**{"num.trees": ...}` to pass dot-named args through.
        with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
            self._fit_obj = drf_pkg.drf(X_arr, y_arr, **self._drf_kwargs())
        self._r_gc()
        return self

    def _fit_subprocess(self, X, y):
        self.close()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self._y_train = y_arr.ravel()

        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._worker_process = ctx.Process(target=_drf_worker_loop, args=(self._request_queue, self._response_queue))
        self._worker_process.start()

        params = self.get_params(deep=False)
        params["backend"] = "inprocess"
        try:
            self._worker_call("fit", params, X_arr, y_arr)
            self._fit_obj = "subprocess"
        except Exception:
            self.close()
            raise
        return self

    def _worker_call(self, command: str, *args):
        if self._worker_process is None or self._request_queue is None or self._response_queue is None:
            raise RuntimeError("DRFWrapper: call fit() before predict.")
        if not self._worker_process.is_alive():
            raise RuntimeError(f"DRF R worker is not running (exitcode={self._worker_process.exitcode}).")

        self._request_queue.put((command, args))
        while True:
            try:
                response = self._response_queue.get(timeout=1)
                break
            except queue.Empty:
                if not self._worker_process.is_alive():
                    raise RuntimeError(f"DRF R worker exited with code {self._worker_process.exitcode}.")

        status = response[0]
        if status == "ok":
            return response[1]
        if status == "error":
            message, tb = response[1], response[2]
            raise RuntimeError(f"DRF R worker failed: {message}\n{tb}")
        raise RuntimeError(f"DRF R worker returned unknown status: {status}")

    def close(self):
        if self._worker_process is None:
            return

        process = self._worker_process
        request_queue = self._request_queue
        response_queue = self._response_queue
        try:
            if process.is_alive() and request_queue is not None:
                request_queue.put(("close", ()))
                process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        finally:
            if request_queue is not None:
                request_queue.close()
                request_queue.join_thread()
            if response_queue is not None:
                response_queue.close()
                response_queue.join_thread()
            process.close()
            self._worker_process = None
            self._request_queue = None
            self._response_queue = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _predict_weights_batch(self, X_arr: np.ndarray) -> np.ndarray:
        """Return a dense weight matrix for one batch of test points."""
        from rpy2.robjects import default_converter, numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        if self._fit_obj is None:
            raise RuntimeError("DRFWrapper: call fit() before predict.")

        drf_pkg = self._load_drf()
        base_r = importr("base")

        try:
            with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
                r_out = drf_pkg.predict_drf(self._fit_obj, newdata=X_arr)
                # predict.drf returns a NamedList with $weights as a dgCMatrix (sparse). Densify in R,
                # then copy to Python before requesting R garbage collection.
                weights_dense = base_r.as_matrix(r_out.getbyname("weights"))
                weights = np.array(weights_dense, dtype=float, copy=True)
            del r_out, weights_dense
        finally:
            self._r_gc()

        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        return weights

    def _predict_weights(self, X) -> np.ndarray:
        """Return the n_test x n_train weight matrix for the given test points."""
        if self._fit_obj is None:
            raise RuntimeError("DRFWrapper: call fit() before predict.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if len(X_arr) == 0:
            return np.empty((0, len(self._y_train)), dtype=float)

        batch_size = int(self.predict_batch_size) if self.predict_batch_size else len(X_arr)
        batch_size = max(1, batch_size)
        if len(X_arr) <= batch_size:
            weights = self._predict_weights_batch(X_arr)
        else:
            batches = [
                self._predict_weights_batch(X_arr[start : start + batch_size])
                for start in range(0, len(X_arr), batch_size)
            ]
            weights = np.vstack(batches)

        # weights rows should already sum to 1; renormalize defensively against numerical drift.
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return weights / row_sums

    def predict(self, X):
        if self.backend == "subprocess":
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return self._worker_call("predict", X_arr)
        weights = self._predict_weights(X)
        return weights @ self._y_train  # weighted mean per row

    def predict_quantiles(self, X) -> np.ndarray:
        """Return shape (n_obs, n_quantiles) weighted empirical quantiles from leaf weights."""
        if self.backend == "subprocess":
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return self._worker_call("predict_quantiles", X_arr)
        weights = self._predict_weights(X)
        y_train = np.asarray(self._y_train, dtype=float)
        order = np.argsort(y_train)
        y_sorted = y_train[order]
        weights_sorted = weights[:, order]
        # Use the same weighted-quantile rule as QRF wrapper (Type-7-like via interp on cum weights).
        cum = np.cumsum(weights_sorted, axis=1) - 0.5 * weights_sorted
        cum /= cum[:, -1:].clip(min=1e-12)
        quantiles = np.asarray(sorted(self.quantiles), dtype=float)
        out = np.empty((weights.shape[0], quantiles.size), dtype=float)
        for i in range(weights.shape[0]):
            out[i] = np.interp(quantiles, cum[i], y_sorted)
        return out

    def predict_samples(self, X, n_samples: int = 100) -> np.ndarray:
        if self.backend == "subprocess":
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return self._worker_call("predict_samples", X_arr, n_samples)
        weights = self._predict_weights(X)
        y_train = np.asarray(self._y_train, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_test = weights.shape[0]
        n_train = weights.shape[1]
        samples = np.empty((n_test, n_samples), dtype=float)
        for i in range(n_test):
            idx = rng.choice(n_train, size=n_samples, replace=True, p=weights[i])
            samples[i] = y_train[idx]
        return samples

    def get_params(self, deep=True):
        params = {
            "num_trees": self.num_trees,
            "splitting_rule": self.splitting_rule,
            "num_features": self.num_features,
            "min_node_size": self.min_node_size,
            "mtry": self.mtry,
            "sample_fraction": self.sample_fraction,
            "honesty": self.honesty,
            "honesty_fraction": self.honesty_fraction,
            "ci_group_size": self.ci_group_size,
            "predict_batch_size": self.predict_batch_size,
            "backend": self.backend,
            "quantiles": self.quantiles,
            "random_state": self.random_state,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        explicit_keys = {
            "num_trees",
            "splitting_rule",
            "num_features",
            "min_node_size",
            "mtry",
            "sample_fraction",
            "honesty",
            "honesty_fraction",
            "ci_group_size",
            "predict_batch_size",
            "backend",
            "quantiles",
            "random_state",
        }
        for key in list(params.keys()):
            if key in explicit_keys:
                setattr(self, key, params.pop(key))
        self.kwargs.update(params)
        return self


class KNNKDE(BaseEstimator, RegressorMixin):
    """
    KNN with KDE-based sampling from neighbors.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, n_neighbors=50, bandwidth="scott", **kwargs):
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.kwargs = kwargs
        self.knn_ = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
        self.y_train_ = None

    def fit(self, X, y):
        self.knn_.fit(X, y)
        self.y_train_ = np.array(y)
        return self

    def predict(self, X):
        return self.knn_.predict(X)

    def predict_samples(self, X, n_samples=100):
        # Find indices of k nearest neighbors
        neigh_ind = self.knn_.kneighbors(X, return_distance=False)

        neighbor_values = self.y_train_[neigh_ind]  # pyright: ignore[reportOptionalSubscript]

        n_obs = X.shape[0]
        samples = np.empty((n_obs, n_samples))

        for i in range(n_obs):
            values = np.asarray(neighbor_values[i], dtype=float)
            finite_values = values[np.isfinite(values)]

            if finite_values.size < 2 or np.allclose(finite_values, finite_values[0]):
                samples[i] = np.random.choice(finite_values, size=n_samples, replace=True)
                continue

            try:
                kde = stats.gaussian_kde(finite_values, bw_method=self.bandwidth)
                samples[i] = kde.resample(n_samples).flatten()
            except (np.linalg.LinAlgError, ValueError):
                samples[i] = np.random.choice(finite_values, size=n_samples, replace=True)

        return samples


class ConformalizedLGBMWrapper(RegressorMixin, BaseEstimator):
    """
    Split Conformal Prediction wrapper for LightGBM with proper probabilistic sampling.

    Implements the algorithm from "A Gentle Introduction to Conformal Prediction and
    Distribution-Free Uncertainty Quantification" (Angelopoulos & Bates, 2022).

    For prediction at a new point x:
    - Point prediction: f(x) from base model
    - Distribution: f(x) + R, where R is sampled from calibration residuals

    This provides valid finite-sample marginal coverage under exchangeability.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, **lgbm_kwargs):
        self.lgbm_kwargs = lgbm_kwargs

    def fit(self, X, y):
        # Convert to numpy to ensure consistent feature handling
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_np = y.values if hasattr(y, "values") else np.asarray(y)
        y_np = y_np.ravel()

        # Split data for conformal prediction
        X_train, X_calib, y_train, y_calib = TTS(
            X_np, y_np, test_size=0.2, random_state=self.lgbm_kwargs.get("random_state", 1234)
        )

        # Create and fit base estimator
        self.estimator_ = lgb.LGBMRegressor(**self.lgbm_kwargs)

        # Fit on training set
        self.estimator_.fit(X_train, y_train)

        # Compute residuals on calibration set
        y_calib_pred = self.estimator_.predict(X_calib)
        self.residuals_ = y_calib - y_calib_pred

        # Store for reproducibility
        self.n_calib_ = len(self.residuals_)

        return self

    def predict(self, X):
        """Point prediction (mean of predictive distribution)."""
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.estimator_.predict(X_np)

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int) -> np.ndarray:
        """
        Generate samples from the conformal predictive distribution.

        For each test point x, the predictive distribution is:
            Y_new | X=x ~ f(x) + R
        where R is uniformly sampled from calibration residuals.

        This is the exact conformal predictive distribution (no interpolation/approximation).

        Args:
            X: Test inputs of shape (n_test, n_features)
            n_samples: Number of samples to generate per test point
        Returns:
            samples: Array of shape (n_test, n_samples)
        """

        # Convert to numpy
        X_np = X.values if hasattr(X, "values") else np.asarray(X)

        # Get point predictions
        y_pred = self.estimator_.predict(X_np)
        n_test = len(y_pred)

        # Sample residuals with replacement
        # Shape: (n_test, n_samples)
        rng = np.random.default_rng(self.lgbm_kwargs.get("random_state", None))
        residual_samples = rng.choice(
            self.residuals_,
            size=(n_test, n_samples),
            replace=True,
        )

        # Conformal predictive distribution: f(x) + sampled_residual
        samples = y_pred[:, np.newaxis] + residual_samples

        return samples


class ConformalizedRFWrapper(RegressorMixin, BaseEstimator):
    """
    Split Conformal Prediction wrapper for RandomForestRegressor with proper probabilistic sampling.

    Implements the algorithm from "A Gentle Introduction to Conformal Prediction and
    Distribution-Free Uncertainty Quantification" (Angelopoulos & Bates, 2022).

    For prediction at a new point x:
    - Point prediction: f(x) from base model
    - Distribution: f(x) + R, where R is sampled from calibration residuals

    This provides valid finite-sample marginal coverage under exchangeability.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, **rf_kwargs):
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y):
        # Split data for conformal prediction
        X_train, X_calib, y_train, y_calib = TTS(
            X, y, test_size=0.2, random_state=self.rf_kwargs.get("random_state", 1234)
        )

        # Create and fit base estimator
        self.estimator_ = RandomForestRegressor(**self.rf_kwargs)

        # Fit on training set
        self.estimator_.fit(X_train, y_train)

        # Compute residuals on calibration set
        y_calib_pred = self.estimator_.predict(X_calib)
        self.residuals_ = y_calib - y_calib_pred

        # Store for reproducibility
        self.n_calib_ = len(self.residuals_)

        return self

    def predict(self, X):
        """Point prediction (mean of predictive distribution)."""
        return self.estimator_.predict(X)

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int) -> np.ndarray:
        """
        Generate samples from the conformal predictive distribution.

        For each test point x, the predictive distribution is:
            Y_new | X=x ~ f(x) + R
        where R is uniformly sampled from calibration residuals.

        This is the exact conformal predictive distribution (no interpolation/approximation).

        Args:
            X: Test inputs of shape (n_test, n_features)
            n_samples: Number of samples to generate per test point
        Returns:
            samples: Array of shape (n_test, n_samples)
        """

        # Get point predictions
        y_pred = self.estimator_.predict(X)
        n_test = len(y_pred)

        # Sample residuals with replacement
        # Shape: (n_test, n_samples)
        rng = np.random.default_rng(self.rf_kwargs.get("random_state", None))
        residual_samples = rng.choice(
            self.residuals_,
            size=(n_test, n_samples),
            replace=True,
        )

        # Conformal predictive distribution: f(x) + sampled_residual
        samples = y_pred[:, np.newaxis] + residual_samples

        return samples


class ConformalizedCatBoostWrapper(RegressorMixin, BaseEstimator):
    """Split conformal prediction wrapper around CatBoostUncertaintyWrapper.

    Guarantees marginal coverage via additive residual correction.  Local
    adaptivity comes solely from the model's point predictions, not from
    CatBoost's uncertainty estimates.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, random_state=None, **catboost_kwargs):
        self.random_state = random_state
        self.catboost_kwargs = catboost_kwargs

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_np = (y.values if hasattr(y, "values") else np.asarray(y)).ravel()

        X_train, X_calib, y_train, y_calib = TTS(X_np, y_np, test_size=0.2, random_state=self.random_state or 1234)
        self.estimator_ = CatBoostUncertaintyWrapper(**self.catboost_kwargs)
        self.estimator_.fit(X_train, y_train)

        y_calib_pred = self.estimator_.predict(X_calib)
        self.residuals_ = y_calib - y_calib_pred
        self.n_calib_ = len(self.residuals_)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.estimator_.predict(X_np)

    def predict_samples(self, X, n_samples: int) -> np.ndarray:
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_pred = self.estimator_.predict(X_np)
        rng = np.random.default_rng(self.random_state)
        residual_samples = rng.choice(self.residuals_, size=(len(y_pred), n_samples), replace=True)
        return y_pred[:, np.newaxis] + residual_samples

    def get_params(self, deep=True):
        return {"random_state": self.random_state, **self.catboost_kwargs}

    def set_params(self, **params):
        self.random_state = params.pop("random_state", self.random_state)
        self.catboost_kwargs.update(params)
        return self


class CQRCatBoostWrapper(RegressorMixin, BaseEstimator):
    """Conformalized Quantile Regression on top of CatBoost uncertainty estimates.

    Unlike additive conformal, this method inherits CatBoost's local uncertainty
    structure (wider intervals where CatBoost predicts higher variance) and only
    corrects the global coverage level via a single scalar q_hat shift.

    predict_samples uses a Gaussian approximation of the conformal interval
    [mean - z*std - q_hat, mean + z*std + q_hat] by inflating the standard
    deviation proportionally.  This preserves the local shape for CRPS/coverage
    diagnostics while guaranteeing the target marginal coverage.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, alpha: float = 0.10, random_state=None, **catboost_kwargs):
        self.alpha = alpha
        self.random_state = random_state
        self.catboost_kwargs = catboost_kwargs

    def fit(self, X, y):
        from scipy.stats import norm as scipy_norm

        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_np = (y.values if hasattr(y, "values") else np.asarray(y)).ravel()

        X_train, X_calib, y_train, y_calib = TTS(X_np, y_np, test_size=0.2, random_state=self.random_state or 1234)
        self.estimator_ = CatBoostUncertaintyWrapper(**self.catboost_kwargs)
        self.estimator_.fit(X_train, y_train)

        mean_calib, std_calib = self.estimator_.predict_mean_std(X_calib)
        self.z_ = float(scipy_norm.ppf(1 - self.alpha / 2))

        lower_calib = mean_calib - self.z_ * std_calib
        upper_calib = mean_calib + self.z_ * std_calib

        scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)
        n_cal = len(scores)
        adjusted_level = min(np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal, 1.0)
        self.q_hat_ = float(np.quantile(scores, adjusted_level))
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.estimator_.predict(X_np)

    def predict_samples(self, X, n_samples: int) -> np.ndarray:
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        mean, std = self.estimator_.predict_mean_std(X_np)
        # Inflate std so that the Gaussian 90% interval width matches the CQR interval width
        std_expanded = np.maximum(std + self.q_hat_ / max(self.z_, 1e-6), 1e-10)
        rng = np.random.default_rng(self.random_state)
        return rng.normal(loc=mean[:, None], scale=std_expanded[:, None], size=(len(mean), n_samples))

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "random_state": self.random_state, **self.catboost_kwargs}

    def set_params(self, **params):
        self.alpha = params.pop("alpha", self.alpha)
        self.random_state = params.pop("random_state", self.random_state)
        self.catboost_kwargs.update(params)
        return self


class CalibratedRFWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper that exposes RF init args at top-level, fits a RandomForestClassifier
    inside CalibratedClassifierCV(cv=3), and exposes predict / predict_proba.
    """

    def __init__(
        self, cv: int = 3, method: Literal["sigmoid", "isotonic"] = "sigmoid", random_state=None, **rf_init_kwargs
    ):
        self.cv = cv
        self.method: Literal["sigmoid", "isotonic"] = method
        self.random_state = random_state
        self.rf_init_kwargs = dict(rf_init_kwargs)

    def fit(self, X, y):
        rf_kwargs = dict(self.rf_init_kwargs)
        if self.random_state is not None:
            rf_kwargs.setdefault("random_state", self.random_state)
        base = RandomForestClassifier(**rf_kwargs)
        self.calibrator_ = CalibratedClassifierCV(estimator=base, method=self.method, cv=self.cv)
        self.calibrator_.fit(X, y)
        return self

    def predict(self, X):
        return self.calibrator_.predict(X)

    def predict_proba(self, X):
        return self.calibrator_.predict_proba(X)

    def get_params(self, deep=True):
        params = {"cv": self.cv, "method": self.method, "random_state": self.random_state}
        params.update(self.rf_init_kwargs)
        return params

    def set_params(self, **params):
        # Pull out wrapper params
        for k in ("cv", "method", "random_state"):
            if k in params:
                setattr(self, k, params.pop(k))
        # remaining params are RF init kwargs
        self.rf_init_kwargs.update(params)
        return self


class BARTPyRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Thin sklearn-compatible wrapper for the external BARTPyRegressor class.

    Stores constructor args verbatim (so sklearn.clone works) and only
    instantiates the real BARTPyRegressor inside fit(), using a deep-copy of
    dict-like args to avoid in-place mutations.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, **init_kwargs):
        self.init_kwargs = dict(init_kwargs)
        for k, v in self.init_kwargs.items():
            setattr(self, k, v)
        self.model_ = None

    def fit(self, X, y, **fit_kwargs):
        import copy

        from bartpy.sklearnmodel import SklearnModel

        bart_kwargs = copy.deepcopy(self.init_kwargs)
        self.model_ = SklearnModel(**bart_kwargs)
        # Assume BARTPyRegressor implements .fit(X, y)
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model_.predict(X)  # pyright: ignore[reportOptionalMemberAccess]

    def predict_samples(self, X, n_samples=100):
        """Generally NOT implemented in BARTPy; access internals directly."""
        # unnormalize_y expects a numpy array, not a list
        preds = np.array([x.predict(X) for x in self.model_._model_samples])  # (n_model_samples, n_obs)
        arr = self.model_.data.y.unnormalize_y(
            preds
        ).T  # (n_obs, n_model_samples)  # pyright: ignore[reportOptionalMemberAccess]
        # Could be less or more samples depending on 'fit' samples
        # If more, downsample, if less, upsample with replacement
        if arr.shape[1] >= n_samples:
            return arr[:, :n_samples]
        else:
            rng = np.random.default_rng()
            indices = rng.choice(arr.shape[1], size=n_samples, replace=True)
            return arr[:, indices]

    def get_params(self, deep=True):
        return dict(self.init_kwargs)

    def set_params(self, **params):
        self.init_kwargs.update(params)
        for k, v in params.items():
            setattr(self, k, v)
        return self


class PyMCBARTRegressorWrapper(BaseEstimator, RegressorMixin):
    """PyMC-BART regressor.

    Uses the Particle Gibbs for BART (PGBART) sampler from the PyMC team — much
    faster than classical MCMC BART and actively maintained. Posterior predictive
    samples include both posterior uncertainty on f(x) and the observation noise σ.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(
        self,
        n_trees: int = 50,
        n_draws: int = 500,
        n_tune: int = 500,
        chains: int = 1,
        cores: int = 1,
        alpha: float = 0.95,
        beta: float = 2.0,
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.chains = chains
        self.cores = cores
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

    def fit(self, X, y):
        import logging

        import pymc as pm
        import pymc_bart as pmb

        X_np = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        y_np = np.asarray(y.values if hasattr(y, "values") else y, dtype=np.float64).ravel()

        # BART is sensitive to target scale; standardize internally.
        self.y_mean_ = float(y_np.mean())
        self.y_std_ = float(y_np.std() + 1e-8)
        y_scaled = (y_np - self.y_mean_) / self.y_std_

        pymc_logger = logging.getLogger("pymc")
        prev_level = pymc_logger.level
        pymc_logger.setLevel(logging.ERROR)

        try:
            with pm.Model() as self.model_:
                X_data = pm.Data("X", X_np)
                Y_data = pm.Data("Y", y_scaled)
                mu = pmb.BART(
                    "mu",
                    X=X_data,
                    Y=Y_data,
                    m=self.n_trees,
                    alpha=self.alpha,
                    beta=self.beta,
                )
                sigma = pm.HalfNormal("sigma", sigma=1.0)
                pm.Normal("obs", mu=mu, sigma=sigma, observed=Y_data, shape=mu.shape)

                self.idata_ = pm.sample(
                    draws=self.n_draws,
                    tune=self.n_tune,
                    chains=self.chains,
                    cores=self.cores,
                    random_seed=self.random_state,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
        finally:
            pymc_logger.setLevel(prev_level)
        return self

    def _predict_posterior_samples(self, X) -> np.ndarray:
        """Returns posterior predictive samples of shape (n_obs, n_total_draws) on the original y scale."""
        import pymc as pm

        X_np = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        with self.model_:
            pm.set_data({"X": X_np, "Y": np.zeros(X_np.shape[0], dtype=np.float64)})
            ppc = pm.sample_posterior_predictive(
                self.idata_,
                predictions=True,
                random_seed=self.random_state,
                progressbar=False,
            )
        # shape: (chain, draw, n_obs) -> (n_obs, chain*draw)
        arr = np.asarray(ppc.predictions["obs"].values)
        arr = arr.reshape(-1, arr.shape[-1]).T
        return arr * self.y_std_ + self.y_mean_

    def predict(self, X) -> np.ndarray:
        return self._predict_posterior_samples(X).mean(axis=1)

    def predict_samples(self, X, n_samples: int = 100) -> np.ndarray:
        samples = self._predict_posterior_samples(X)
        n_avail = samples.shape[1]
        if n_avail >= n_samples:
            idx = np.linspace(0, n_avail - 1, n_samples, dtype=int)
            return samples[:, idx]
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n_avail, size=n_samples, replace=True)
        return samples[:, idx]

    def get_params(self, deep=True):
        return {
            "n_trees": self.n_trees,
            "n_draws": self.n_draws,
            "n_tune": self.n_tune,
            "chains": self.chains,
            "cores": self.cores,
            "alpha": self.alpha,
            "beta": self.beta,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TreeffuserWrapper(BaseEstimator, RegressorMixin):
    """
    Thin sklearn-compatible wrapper for the external Treeffuser class.

    Stores constructor args verbatim (so sklearn.clone works) and only
    instantiates the real Treeffuser inside fit(), using a deep-copy of
    dict-like args to avoid in-place mutations.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, **init_kwargs):
        self.init_kwargs = dict(init_kwargs)
        for k, v in self.init_kwargs.items():
            setattr(self, k, v)
        self.model_ = None

    def fit(self, X, y, **fit_kwargs):
        warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
        warnings.filterwarnings("ignore", message="Input array is not float32")
        import copy

        from treeffuser import Treeffuser

        X = X.values if hasattr(X, "values") else X
        y = y.values if hasattr(y, "values") else y

        tf_kwargs = copy.deepcopy(self.init_kwargs)
        self.model_ = Treeffuser(**tf_kwargs)
        # Assume Treeffuser implements .fit(X, y)
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        X = X.values if hasattr(X, "values") else X
        return self.model_.predict(X)  # pyright: ignore[reportOptionalMemberAccess]

    def predict_samples(self, X, n_samples=100):
        X = X.values if hasattr(X, "values") else X
        # TODO: add n_steps as parameter when tuning on crps
        return self.model_.sample(  # pyright: ignore[reportOptionalMemberAccess]
            X, n_samples=n_samples, seed=1234, n_steps=50
        ).T

    def get_params(self, deep=True):
        return dict(self.init_kwargs)

    def set_params(self, **params):
        self.init_kwargs.update(params)
        for k, v in params.items():
            setattr(self, k, v)
        return self


# =============================================================================
# Deep Ensemble with Gaussian Output Heads (Lakshminarayanan et al., 2017)
# =============================================================================


class GaussianDeepEnsembleWrapper(BaseEstimator, RegressorMixin):
    """
    Deep Ensemble with Gaussian output heads (Lakshminarayanan et al., 2017).

    Each network outputs (mu, log_var) and is trained with Gaussian NLL loss.
    The predictive distribution is a mixture of Gaussians.

    Args:
        n_estimators: Number of ensemble members
        hidden_layer_sizes: Tuple of hidden layer sizes, e.g. (100, 100)
        max_epochs: Maximum training epochs per network
        learning_rate: Adam learning rate
        batch_size: Mini-batch size
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs
        val_fraction: Fraction of training data for early stopping validation
        min_var: Minimum variance (numerical stability)
        random_state: Random seed for reproducibility
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(
        self,
        n_estimators: int = 5,
        hidden_layer_dim: int = 100,
        max_epochs: int = 500,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        early_stopping_patience: int = 20,
        val_fraction: float = 0.1,
        min_var: float = 1e-6,
        log_var_min: float = -10.0,
        log_var_max: float = 5.0,
        mu_clip: float = 10.0,
        grad_clip_norm: float = 5.0,
        weight_decay: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.hidden_layer_dim = hidden_layer_dim
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction = val_fraction
        self.min_var = min_var
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.mu_clip = mu_clip
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay
        self.random_state = random_state

        self.networks_: list = []
        self.input_dim_: int | None = None
        self.X_mean_: np.ndarray | None = None
        self.X_std_: np.ndarray | None = None
        self.y_mean_: float | None = None
        self.y_std_: float | None = None

    def _build_network(self, input_dim: int, seed: int):
        """Build a single Gaussian MLP."""
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)

        layers = []
        in_features = input_dim

        for _ in range(2):  # Two hidden layers
            layers.append(nn.Linear(in_features, self.hidden_layer_dim))
            layers.append(nn.ReLU())
            in_features = self.hidden_layer_dim

        # Output layer: 2 outputs (mu, log_var)
        layers.append(nn.Linear(in_features, 2))

        return nn.Sequential(*layers)

    def _gaussian_nll_loss(self, y_true, mu, log_var):
        """Gaussian negative log-likelihood loss."""
        import torch

        if self.mu_clip is not None:
            mu = self.mu_clip * torch.tanh(mu / self.mu_clip)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)
        var = torch.exp(log_var) + self.min_var
        nll = 0.5 * (torch.log(var) + (y_true - mu) ** 2 / var)
        return nll.mean()

    def _train_single_network(self, X: np.ndarray, y: np.ndarray, seed: int):
        """Train a single network with early stopping."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train/val split for early stopping
        n_samples = len(X)
        n_val = max(1, int(n_samples * self.val_fraction))
        indices = np.random.permutation(n_samples)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # DataLoader
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Build network
        network = self._build_network(X.shape[1], seed)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Early stopping state
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            network.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                out = network(X_batch)
                mu, log_var = out[:, 0:1], out[:, 1:2]
                loss = self._gaussian_nll_loss(y_batch, mu, log_var)
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), self.grad_clip_norm)
                optimizer.step()

            # Validation
            network.eval()
            with torch.no_grad():
                out_val = network(X_val_t)
                mu_val, log_var_val = out_val[:, 0:1], out_val[:, 1:2]
                val_loss = self._gaussian_nll_loss(y_val_t, mu_val, log_var_val).item()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in network.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        # Restore best weights
        if best_state is not None:
            network.load_state_dict(best_state)

        network.eval()
        return network

    def fit(self, X, y):
        """Fit the ensemble of Gaussian networks."""

        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        y_np = y.values if hasattr(y, "values") else np.asarray(y)
        y_np = y_np.ravel()

        # Standardize inputs and outputs for stable training
        self.X_mean_ = X_np.mean(axis=0)
        self.X_std_ = X_np.std(axis=0) + 1e-8
        self.y_mean_ = y_np.mean()
        self.y_std_ = y_np.std() + 1e-8

        X_scaled = (X_np - self.X_mean_) / self.X_std_
        y_scaled = (y_np - self.y_mean_) / self.y_std_

        self.input_dim_ = X_scaled.shape[1]

        # Train each ensemble member with different seed
        base_seed = self.random_state if self.random_state is not None else 0
        self.networks_ = []

        for i in range(self.n_estimators):
            seed = base_seed + i * 1000  # Different seed per network
            network = self._train_single_network(X_scaled, y_scaled, seed)
            self.networks_.append(network)

        return self

    def _predict_params(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Get mean and variance from each ensemble member.

        Returns:
            mus: shape (n_estimators, n_obs)
            vars: shape (n_estimators, n_obs)
        """
        import torch

        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        X_scaled = (X_np - self.X_mean_) / self.X_std_
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        mus = []
        vars_ = []

        for network in self.networks_:
            network.eval()
            with torch.no_grad():
                out = network(X_t)
                mu = out[:, 0].numpy()
                log_var = out[:, 1].numpy()
                if self.mu_clip is not None:
                    mu = self.mu_clip * np.tanh(mu / self.mu_clip)
                log_var = np.clip(log_var, self.log_var_min, self.log_var_max)
                var = np.exp(log_var) + self.min_var

            # Un-standardize
            mu_orig = mu * self.y_std_ + self.y_mean_
            var_orig = var * (self.y_std_**2)

            mus.append(mu_orig)
            vars_.append(var_orig)

        return np.array(mus), np.array(vars_)

    def predict(self, X) -> np.ndarray:
        """
        Point prediction: mean of the Gaussian mixture.

        For a mixture of Gaussians with equal weights:
            E[Y] = (1/M) * sum_m mu_m
        """
        mus, _ = self._predict_params(X)
        return mus.mean(axis=0)

    def predict_samples(self, X, n_samples: int = 100) -> np.ndarray:
        """
        Sample from the predictive mixture of Gaussians.

        Algorithm:
        1. For each sample, uniformly pick an ensemble member
        2. Sample from that member's Gaussian(mu_m, var_m)

        Returns:
            samples: shape (n_obs, n_samples)
        """
        mus, vars_ = self._predict_params(X)  # (M, n_obs), (M, n_obs)
        n_obs = mus.shape[1]

        rng = np.random.default_rng(self.random_state)

        # Sample which ensemble member to use for each (obs, sample)
        member_idx = rng.integers(0, self.n_estimators, size=(n_obs, n_samples))

        # Gather the corresponding mu and var
        # mus/vars_ are (M, n_obs), we need (n_obs, n_samples)
        mus_selected = np.take_along_axis(mus.T, member_idx, axis=1)  # (n_obs, M)
        vars_selected = np.take_along_axis(vars_.T, member_idx, axis=1)  # (n_obs, M)

        # Sample from Gaussian
        samples = rng.normal(loc=mus_selected, scale=np.sqrt(vars_selected))

        return samples

    def predict_mean_and_var(self, X) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytical mean and variance of the mixture.

        For equal-weight mixture:
            E[Y] = (1/M) sum_m mu_m
            Var[Y] = (1/M) sum_m (var_m + mu_m^2) - E[Y]^2
                   = mean(var) + mean(mu^2) - mean(mu)^2
                   = mean(var) + var(mu)  [law of total variance]
        """
        mus, vars_ = self._predict_params(X)

        mean = mus.mean(axis=0)
        # Total variance = E[Var] + Var[E] (law of total variance)
        epistemic_var = mus.var(axis=0)  # variance of means (epistemic)
        aleatoric_var = vars_.mean(axis=0)  # mean of variances (aleatoric)
        total_var = epistemic_var + aleatoric_var

        return mean, total_var

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "hidden_layer_dim": self.hidden_layer_dim,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience,
            "val_fraction": self.val_fraction,
            "min_var": self.min_var,
            "log_var_min": self.log_var_min,
            "log_var_max": self.log_var_max,
            "mu_clip": self.mu_clip,
            "grad_clip_norm": self.grad_clip_norm,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ClimatologicalRegressor(BaseEstimator, RegressorMixin):
    """Baseline that predicts the unconditional training distribution for every test point.

    This is the "no-skill" reference for computing CRPS Skill Scores (CRPSS).
    For each test observation, the predictive distribution is a subsample of
    the training targets — equivalent to ignoring all features.
    """

    PREDICTION_TYPE: PredictionType = "samples"

    def __init__(self, n_subsample: int = 500, random_state: int = 42):
        self.n_subsample = n_subsample
        self.random_state = random_state

    def fit(self, X, y):
        self.y_train_ = np.asarray(y).ravel()
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, np.mean(self.y_train_))

    def predict_samples(self, X, n_samples=None):
        n_obs = X.shape[0] if hasattr(X, "shape") else len(X)
        n_samples = n_samples or self.n_subsample
        rng = np.random.RandomState(self.random_state)
        replace = n_samples > len(self.y_train_)
        subsample = rng.choice(self.y_train_, size=n_samples, replace=replace)
        return np.tile(subsample, (n_obs, 1))
