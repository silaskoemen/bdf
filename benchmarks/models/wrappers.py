import warnings

import numpy as np
import pandas as pd
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import Bernoulli, Exponential, LogNormal, Normal, Poisson
from ngboost.scores import LogScore
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel as Ker


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


import lightgbm as lgb

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


from catboost import CatBoostRegressor


class CatBoostUncertaintyWrapper(CatBoostRegressor):
    def __init__(self, iterations=1000, **kwargs):
        # Force the loss function to be probabilistic
        super().__init__(iterations=iterations, loss_function="RMSEWithUncertainty", posterior_sampling=True, **kwargs)

    def fit(self, X, y, **kwargs):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        return super().fit(X, y, **kwargs)

    def predict_samples(self, X, n_samples=100):
        if hasattr(X, "values"):
            X = X.values

        # CatBoost returns [mean, variance] for RMSEWithUncertainty
        preds = self.predict(X)
        mean = preds[:, 0]
        variance = preds[:, 1]
        std = np.sqrt(variance)
        return np.random.normal(loc=mean[:, np.newaxis], scale=std[:, np.newaxis], size=(X.shape[0], n_samples))


from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.neural_network import MLPRegressor


class DeepEnsembleWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=5, hidden_layer_sizes=(100,), random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.kwargs = kwargs
        self.estimators_ = []

    def fit(self, X, y):
        self.estimators_ = []
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            # Each MLP gets a different seed
            est = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes, random_state=rng.randint(0, 10000), **self.kwargs
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


from sklearn.linear_model import BayesianRidge


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
