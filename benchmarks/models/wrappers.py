# pyright: reportMissingImports=false
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
        X_np = X.values if hasattr(X, "values") else X  # pyright: ignore[reportAttributeAccessIssue]
        y_np = y.values if hasattr(y, "values") else y  # pyright: ignore[reportAttributeAccessIssue]

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
        X_np = X.values if hasattr(X, "values") else X  # pyright: ignore[reportAttributeAccessIssue]
        y_np = y.values if hasattr(y, "values") else y  # pyright: ignore[reportAttributeAccessIssue]

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
        return self.model_.predict(X, quantiles=0.5)  # pyright: ignore[reportOptionalMemberAccess]

    def predict_quantiles(self, X) -> np.ndarray:
        """Returns shape (n_obs, n_quantiles)"""
        sorted_quantiles = sorted(self.quantiles)
        return self.model_.predict(X, quantiles=sorted_quantiles)  # pyright: ignore[reportOptionalMemberAccess]

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
            kde = stats.gaussian_kde(neighbor_values[i], bw_method=self.bandwidth)
            samples[i] = kde.resample(n_samples).flatten()

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
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)

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
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
