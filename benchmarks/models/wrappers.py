import warnings
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from mapie.regression import SplitConformalRegressor
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
from sklearn.neural_network import MLPRegressor


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
    def __init__(self, kernel_comb="RBF", random_state=None, normalize_y=True, **kwargs):
        self.kernel_comb = kernel_comb  # Store the string representation
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.kwargs = kwargs

        # Configure the actual kernel object
        kernel_obj = configure_kernel(kernel_comb)
        init_kwargs = {k: v for k, v in kwargs.items() if k != "kernel_comb"}
        # Initialize parent
        # Note: GaussianProcessRegressor takes normalize_y in __init__
        super().__init__(kernel=kernel_obj, normalize_y=normalize_y, random_state=random_state, **init_kwargs)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
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

            y_mean, y_std = self.predict(X_batch, return_std=True)
            y_std = np.maximum(y_std, 1e-10)

            samples[start_idx:end_idx] = np.random.normal(
                loc=y_mean[:, np.newaxis], scale=y_std[:, np.newaxis], size=(end_idx - start_idx, n_samples)
            )

        return samples


class GPClassifierWrapper(GaussianProcessClassifier):
    def __init__(self, kernel_comb="RBF", random_state=None, **kwargs):
        self.kernel_comb = kernel_comb
        self.random_state = random_state
        self.kwargs = kwargs

        # Configure the actual kernel object
        kernel_obj = configure_kernel(kernel_comb)

        # Initialize parent
        super().__init__(kernel=kernel_obj, random_state=random_state, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)


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
    ).T,
    "LogNormal": lambda params, n_samples: stats.lognorm.rvs(
        s=params["s"], scale=params["scale"], size=(n_samples, len(params["s"]))
    ).T,
    "Exponential": lambda params, n_samples: stats.expon.rvs(
        scale=params["scale"], size=(n_samples, len(params["scale"]))
    ).T,
    "Poisson": lambda params, n_samples: stats.poisson.rvs(mu=params["mu"], size=(n_samples, len(params["mu"]))).T,
    "Bernoulli": lambda params, n_samples: stats.bernoulli.rvs(p=params["p1"], size=(n_samples, len(params["p1"]))).T,
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
        X_np, y_np = X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y

        if y_np.ndim > 1:
            y_np = y_np.ravel()
        super().fit(X_np, y_np, **kwargs)
        return self

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int) -> np.ndarray:
        """Generate samples from the predictive distribution at inputs X.

        Returns:
            np.ndarray of shape (n_observations, n_samples)
        """
        X_np = X.values if hasattr(X, "values") else X
        y_dists = self.pred_dist(X_np)
        params = y_dists.params

        # Use the scipy sampling function for this distribution
        sampler = NGBOOST_TO_SCIPY[self.dist_name]
        samples = sampler(params, n_samples)
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
        X_np, y_np = X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y

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
        X_np = X.values if hasattr(X, "values") else X
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

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int) -> np.ndarray:
        """Generate samples by interpolating between predicted quantiles.

        Returns:
            np.ndarray of shape (n_observations, n_samples)
        """

        # Get quantile predictions: shape (n_obs, n_quantiles)
        quantile_preds = self.predict_quantiles(X)

        sorted_quantiles = np.array(sorted(self.quantiles))

        # Generate samples by interpolating
        samples = np.empty((X.shape[0], n_samples))

        # Generate uniform random quantile levels
        random_quantiles = np.random.uniform(0, 1, size=(X.shape[0], n_samples))

        for i in range(X.shape[0]):
            # Interpolate: for each random quantile, find the corresponding value
            samples[i] = np.interp(random_quantiles[i], sorted_quantiles, quantile_preds[i])

        return samples

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
        preds = self.predict(X)
        mean = preds[:, 0]
        variance = preds[:, 1]
        std = np.sqrt(variance)
        return np.random.normal(loc=mean[:, np.newaxis], scale=std[:, np.newaxis], size=(X.shape[0], n_samples))


class DeepEnsembleWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=5, hidden_layers=2, hidden_size=100, random_state=None, max_iter=200, **kwargs):
        self.n_estimators = n_estimators
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.hidden_layer_sizes = tuple([self.hidden_size] * self.hidden_layers)
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = {k: v for k, v in kwargs.items() if k != "hidden_layers" and k != "hidden_size"}
        self.estimators_ = []

    def fit(self, X, y):
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Each MLP gets a different seed
            est = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                random_state=self.random_state + i,
                max_iter=self.max_iter,
                **self.kwargs,
            )
            est.fit(X, y)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        # Mean of ensemble predictions
        preds = np.array([est.predict(X) for est in self.estimators_])
        return np.mean(preds, axis=0)

    def predict_samples(self, X, n_samples=100):
        # Sample by randomly selecting an estimator for each sample
        # Or treat as Mixture of Gaussians with fixed variance (simplified)

        # Approach: Empirical distribution of the ensemble
        # Shape: (n_estimators, n_obs)
        preds = np.array([est.predict(X) for est in self.estimators_])

        # Resample from these predictions to get n_samples
        # Shape: (n_obs, n_samples)
        indices = np.random.randint(0, self.n_estimators, size=(X.shape[0], n_samples))
        samples = np.take_along_axis(preds.T, indices, axis=1)

        # Add small aleatoric noise (optional, but helps smoothing)
        samples += np.random.normal(0, 1e-6, size=samples.shape)
        return samples


class BayesianRidgeWrapper(BayesianRidge):
    def predict_samples(self, X, n_samples=100) -> np.ndarray:
        # BayesianRidge returns mean and std
        mean, std = self.predict(X, return_std=True)  # type: ignore[returnedValue]
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)

        rng = np.random.default_rng(self.random_state)  # type: ignore[attr-defined]
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

    def __init__(self, n_estimators=100, quantiles=None, random_state=None, **kwargs):
        self.n_estimators = n_estimators
        # Default quantiles to cover the distribution well
        self.quantiles = quantiles or [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
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
        return self.model_.predict(X, quantiles=0.5)

    def predict_samples(self, X, n_samples=100):
        # Predict quantiles
        sorted_quantiles = sorted(self.quantiles)
        # shape: (n_samples, n_quantiles)
        quantile_preds = self.model_.predict(X, quantiles=sorted_quantiles)

        # Interpolate samples (Inverse Transform Sampling approximation)
        n_obs = X.shape[0]
        samples = np.empty((n_obs, n_samples))

        # Vectorized interpolation is tricky, looping is safer for now
        # We treat the predicted quantiles as the CDF
        random_u = np.random.uniform(0, 1, size=(n_obs, n_samples))

        # Add 0 and 1 bounds for interpolation
        # We assume the distribution doesn't extend much beyond the 1st and 99th quantile
        # A robust way is to use the min/max of the predicted quantiles as bounds

        x_points = np.array(sorted_quantiles)

        for i in range(n_obs):
            # y_points are the predicted values for the quantiles
            y_points = quantile_preds[i]
            samples[i] = np.interp(random_u[i], x_points, y_points)

        return samples


class ProbabilisticKNNWrapper(KNeighborsRegressor):
    """
    Probabilistic KNN: Uses the y-values of the k-nearest neighbors
    as the empirical predictive distribution.
    """

    def __init__(self, n_neighbors=50, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.y_train_ = None

    def fit(self, X, y):
        super().fit(X, y)
        self.y_train_ = np.array(y)
        return self

    def predict_samples(self, X, n_samples=100):
        # Find indices of k nearest neighbors
        # neigh_ind: (n_obs, n_neighbors)
        neigh_dist, neigh_ind = self.kneighbors(X)

        # Retrieve the y values of these neighbors
        # shape: (n_obs, n_neighbors)
        neighbor_values = self.y_train_[neigh_ind]

        # Resample from these neighbors to get exactly n_samples
        # If n_neighbors > n_samples, we downsample. If <, we upsample (bootstrap).

        idx = np.random.randint(0, self.n_neighbors, size=(X.shape[0], n_samples))
        samples = np.take_along_axis(neighbor_values, idx, axis=1)

        # Add tiny jitter to avoid identical samples
        samples += np.random.normal(0, 1e-6, size=samples.shape)

        return samples


class MapieQuantileRegressorWrapper(RegressorMixin):
    """
    Wrapper for MapieQuantileRegressor to provide predict_samples method.
    """

    DEFAULT_CONFIDENCE_LEVELS = (0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.confidence_levels = self.DEFAULT_CONFIDENCE_LEVELS
        self.estimator = RandomForestRegressor(**kwargs)
        self.mapie = SplitConformalRegressor(self.estimator, confidence_level=self.confidence_levels, prefit=False)

    def fit(self, X, y):
        X_train, X_calib, y_train, y_calib = TTS(
            X, y, test_size=0.2, random_state=self.kwargs.get("random_state", 1234)
        )
        self.mapie.fit(X_train, y_train).conformalize(X_calib, y_calib)
        return self

    def get_params(self, deep=True):
        return self.kwargs

    def predict(self, X):
        return self.mapie.predict(X)

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int) -> np.ndarray:
        """
        Generate samples by interpolating between the lower and upper bounds of all conformal intervals.

        Returns:
            np.ndarray of shape (n_observations, n_samples)
        """
        # Get intervals: shape (n_obs, 2, n_intervals)
        _, intervals = self.mapie.predict_interval(X)

        # The confidence_levels correspond to central intervals, e.g. 0.9 -> (0.05, 0.95)
        # For each interval, compute the lower and upper quantile
        lower_quantiles = [(1 - c) / 2 for c in self.confidence_levels]
        upper_quantiles = [1 - (1 - c) / 2 for c in self.confidence_levels]

        # Combine and sort all quantile levels
        all_quantiles = np.array(sorted(set(lower_quantiles + upper_quantiles)))

        # For each observation, collect the corresponding lower and upper bounds
        n_obs = X.shape[0]
        n_q = len(all_quantiles)
        quantile_preds = np.empty((n_obs, n_q))

        # Map quantile levels to interval indices
        quantile_to_interval = {q: i for i, q in enumerate(lower_quantiles)}
        quantile_to_interval.update({q: i for i, q in enumerate(upper_quantiles)})

        for i in range(n_obs):
            # For each quantile, pick the corresponding lower or upper bound
            for j, q in enumerate(all_quantiles):
                if q in quantile_to_interval:
                    idx = quantile_to_interval[q]
                    if q in lower_quantiles:
                        quantile_preds[i, j] = intervals[i, 0, idx]
                    else:
                        quantile_preds[i, j] = intervals[i, 1, idx]
                else:
                    # Should not happen, but just in case
                    quantile_preds[i, j] = np.nan

        # Now interpolate as in the LightGBM wrapper
        samples = np.empty((n_obs, n_samples))
        random_quantiles = np.random.uniform(0, 1, size=(n_obs, n_samples))
        for i in range(n_obs):
            samples[i] = np.interp(random_quantiles[i], all_quantiles, quantile_preds[i])

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


# from bartpy.sklearnmodel import SklearnModel
# class BARTRegressor(SklearnModel):
#     def __init__(self, **kwargs):
#         p_prune = 1 - kwargs['p_grow']
#         kwargs.update({'p_prune': p_prune})
#         self.kwargs = kwargs
