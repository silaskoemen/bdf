import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

from bdf.distributions.distribution_manager import DistributionManager as DM
from bdf.tree_classes.bdf_tree import BDFTree


class BDFRegressor(BaseEstimator, RegressorMixin):
    """BDFRegressor class for Bayesian Distributional Forests."""

    def __init__(
        self,
        dist: str,
        prior_params: dict,
        n_trees: int = 100,
        reg_beta: float = 0,
        reg_lambda: float = 0,
        max_depth: int = 10,
        min_samples_leaf: int = 10,
        min_samples_split: int = 20,
        min_child_weight: int | float = 10,
        subsample: float = 0.7,
        colsample: float = 1.0,
        eta: float = 0.025,
    ):
        """Initialize the BDFRegressor with prior parameters.
        Args
        ----
        `data_dist` : str | BDFDistribution.BDFDistribution, optional
            Data distribution type or instance, default is 'normal'.
        `prior_params` : str | BDFDistribution.BDFDistribution | dict, optional
            Prior parameters for the distribution, can be string name or 'auto', a BDFDistribution instance, or a dictionary parameters as keys. Default is 'auto'.
        `n_trees` : int, optional
            Number of trees in the forest, default is 100.
        `reg_beta` : float, optional
            Regularization parameter for the beta term, default is 0.
        `reg_lambda` : float, optional
            Regularization parameter for the lambda term, default is 0.
        `max_depth` : int, optional
            Maximum depth of the regression tree, default is 10.
        `min_samples_leaf` : int, optional
            Minimum number of samples required to be at a leaf node, default is 1.
        `min_samples_split` : int, optional
            Minimum number of samples required to split an internal node, default is 2.
        `min_child_weight` : int | float, optional
            Minimum sum of instance weight (hessian) needed in a child, default is 1.
        """
        self.distribution = DM.create_distribution(dist=dist, prior_params=prior_params)
        self.dist, self.prior_params = dist, prior_params
        self._validate_init_params(
            n_trees=n_trees,
            reg_beta=reg_beta,
            reg_lambda=reg_lambda,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample=colsample,
            eta=eta,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False, standardize_y: bool = True) -> "BDFRegressor":
        """Fit the BDFRegressor to the training data.
        Args
        ----
        `X` : np.ndarray | pd.DataFrame
            Training data features.
        `y` : np.ndarray | pd.Series
            Training data target values.
        """
        X, y = self._validate_fit_input(X, y)
        if standardize_y:
            y = self._standardize_y(y.copy())

        # Otherwise regularization depends on size of the dataset (NLL as sum)
        n_features_iter = int(np.ceil(X.shape[1] * self.colsample))

        self.trees = np.empty(self.n_trees, dtype=object)
        # Create a progress bar for tree creation and fitting
        for i in tqdm(range(self.n_trees)):
            # Create and fit a tree
            self.trees[i] = BDFTree(
                distribution=self.distribution,
                reg_beta=self.reg_beta,
                reg_lambda=self.reg_lambda,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                min_child_weight=self.min_child_weight,
            )
            # Subsample rows and columns if specified
            if self.subsample < 1.0:
                n_samples = int(X.shape[0] * self.subsample)
                row_indices = np.random.choice(X.shape[0], n_samples, replace=False)
                X_iter = X[row_indices]
                y_iter = y[row_indices]
            else:
                X_iter = X
                y_iter = y
            col_idcs = np.random.choice(X.shape[1], n_features_iter, replace=False) if self.colsample < 1.0 else None
            self.trees[i].fit(X_iter, y_iter, col_idcs=col_idcs, verbose=verbose, eta=self.eta)
        self.is_fitted_ = True
        return self

    def _standardize_y(self, y: np.ndarray) -> np.ndarray:
        """Standardize the target variable y.

        Args
        ----
        `y` : np.ndarray
            The target variable to standardize.

        Returns
        -------
        np.ndarray
            Standardized target variable.
        """
        mean_y, std_y = np.mean(y), np.std(y)
        if std_y == 0:
            raise ValueError("Standard deviation of y is zero, cannot standardize.")
        standardized_y = (y - mean_y) / std_y
        self.y_mean, self.y_std = mean_y, std_y
        return standardized_y

    def predict(self, X: np.ndarray | pd.DataFrame, method: str = "mean", values: list | None = None) -> np.ndarray:
        X = self._validate_prediction_input(X, method=method, values=values)
        preds = np.empty((X.shape[0],), dtype=float)
        match method:
            case "mean":
                preds = np.mean([tree.predict(X, method="params")[0] for tree in self.trees], axis=0)
            case "params":
                preds = np.mean([tree.predict(X, method="params") for tree in self.trees], axis=0)
            case "samples-ind":
                preds = np.concatenate([tree.predict(X, method="sample") for tree in self.trees])
            # case 'samples-avg':
            #     preds = -1#np.mean([tree.predict(X, method='sample') for tree in self.trees], axis=0)
            # case 'quantiles-ind':
            #     preds = -1#np.concatenate([tree.predict(X, method='quantile', values=values) for tree in self.trees])
            # case 'quantiles-avg':
            #     preds = -1#np.mean([tree.predict(X, method='quantile', values=values) for tree in self.trees], axis=0)
            # case 'confint-ind':
            #     preds = -1#np.concatenate([tree.predict(X, method='confint', values=values) for tree in self.trees])
            # case 'confint-avg':
            #     preds = -1#np.mean([tree.predict(X, method='confint', values=values) for tree in self.trees], axis=0)
        if hasattr(self, "y_mean") and hasattr(self, "y_std"):
            if method in ["mean", "params"]:
                preds = preds * self.y_std + self.y_mean
            elif method in [
                "samples-ind",
                "samples-avg",
                "quantiles-ind",
                "quantiles-avg",
                "confint-ind",
                "confint-avg",
            ]:
                preds = preds * self.y_std + self.y_mean
        return preds

    def _validate_init_params(
        self,
        n_trees: int,
        reg_beta: float,
        reg_lambda: float,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_split: int,
        min_child_weight: int | float,
        subsample: float,
        colsample: float,
        eta: float,
    ):
        """Validate the initialization parameters."""
        assert (
            isinstance(reg_beta, (float, int)) and reg_beta >= 0
        ), f"reg_beta must be float and non-negative, got {reg_beta} of type {type(reg_beta)}"
        assert isinstance(
            reg_lambda, (float, int)
        ), f"reg_lambda must be a float, got {reg_lambda} of type {type(reg_lambda)}"
        assert (
            isinstance(n_trees, int) and n_trees > 0
        ), f"n_trees must be a positive integer, got {n_trees} of type {type(n_trees)}"
        assert (
            isinstance(max_depth, int) and max_depth > 0
        ), f"max_depth must be a positive integer, got {max_depth} of type {type(max_depth)}"
        assert (
            isinstance(min_samples_leaf, int) and min_samples_leaf > 0
        ), f"min_samples_leaf must be a positive integer, got {min_samples_leaf} of type {type(min_samples_leaf)}"
        assert (
            isinstance(min_samples_split, int) and min_samples_split > 0
        ), f"min_samples_split must be a positive integer, got {min_samples_split} of type {type(min_samples_split)}"
        assert (
            isinstance(min_child_weight, (int, float)) and min_child_weight >= 0
        ), f"min_child_weight must be a non-negative integer or float, got {min_child_weight} of type {type(min_child_weight)}"
        assert (
            isinstance(subsample, float) and 0 < subsample <= 1
        ), f"subsample must be a float between 0 and 1, got {subsample} of type {type(subsample)}"
        assert (
            isinstance(colsample, float) and 0 < colsample <= 1
        ), f"colsample must be a float between 0 and 1, got {colsample} of type {type(colsample)}"
        assert (
            isinstance(eta, float) and 0 < eta <= 1
        ), f"eta must be a float between 0 and 1, got {eta} of type {type(eta)}"
        self.eta = eta
        self.reg_beta = reg_beta
        self.reg_lambda = reg_lambda
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample = colsample

    def _validate_prediction_input(
        self, X: np.ndarray | pd.DataFrame, method: str = "mean", values: list | None = None
    ):
        """Validate the input for prediction.

        should allow:
          mean, taking avg mean of all trees
          params (take avg params of all trees)
          samples-ind and samples-avg, returning concat samples from trees vs sampling from params
          quantiles-ind and quantiles-avg, returning avg quantiles from trees vs quantiles from sampling avg params
          confint-ind and confint-avg, returning avg confints from trees vs confints sampled from avg params
        """
        if isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names"):
                assert all(
                    col in X.columns for col in self.feature_names  # type: ignore | pyright sees as np.ndarray
                ), "X must contain all feature names used during fitting"
                X = X[self.feature_names].values  # type: ignore | pyright sees as np.ndarray
            else:
                raise ValueError(
                    "X is a DataFrame but no feature names were stored during fitting. Ensure to fit with a DataFrame to predict on DataFrame or fit on np.ndarray"
                )

        assert isinstance(method, str) and method in [
            "mean",
            "params",
            "samples-ind",
            "samples-avg",
            "quantiles-ind",
            "quantiles-avg",
            "confint-ind",
            "confint-avg",
        ], f"Invalid method '{method}' for prediction. Must be one of ['mean', 'params', 'samples-ind', 'samples-avg', 'quantiles-ind', 'quantiles-avg', 'confint-ind', 'confint-avg']"

        if method in ["quantiles-ind", "quantiles-avg", "confint-ind", "confint-avg"]:
            assert values is not None, "values must be provided for quantile/confidence interval predictions"
            assert isinstance(values, list) and all(
                isinstance(v, (int, float)) for v in values
            ), "values must be a list of numeric quantiles or confidence levels"
        assert X.ndim == 2, f"X must be a 2D array, got {X.ndim}D array"
        assert X.shape[0] > 0, "X must contain at least one sample"
        return X

    def _validate_fit_input(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the input for fitting."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns  # type: ignore | pyright sees as np.ndarray
            X = X.values  # type: ignore | pyright sees as np.ndarray
        if isinstance(y, pd.Series):
            y = y.to_numpy()  # type: ignore | pyright sees as np.ndarray
        assert X.ndim == 2, f"X must be a 2D array, got {X.ndim}D array"
        assert y.ndim == 1, f"y must be a 1D array, got {y.ndim}D array"
        assert (
            X.shape[0] == y.shape[0]
        ), f"Number of samples in X ({X.shape[0]}) must match number of samples in y ({y.shape[0]})"
        return X, y
