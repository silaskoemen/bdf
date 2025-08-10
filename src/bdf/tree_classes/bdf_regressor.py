import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

from bdf.distributions.distribution_manager import DistributionManager as DM
from bdf.tree_classes.bdf_tree import BDFTree
from bdf.utils.constants import RANDOM_SEED


class BDFRegressor(BaseEstimator, RegressorMixin):
    """BDFRegressor class for Bayesian Distributional Forests."""

    def __init__(
        self,
        dist: str = "normal_normal",
        prior_params: dict = {},
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
        random_state: int = RANDOM_SEED,
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
        self.is_fitted_ = False
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
            random_state=random_state,
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
        # Seed for reproducibility of subsample and colsample
        np.random.seed(self.random_state)
        X, y = self._validate_fit_input(X, y)
        self.n_features_in_ = X.shape[1]
        if standardize_y:
            y = self._standardize_y(y.copy())

        # Otherwise regularization depends on size of the dataset (NLL as sum)
        n_features_iter = int(np.ceil(X.shape[1] * self.colsample))
        self.trees: list[BDFTree] = []
        # Create a progress bar for tree creation and fitting
        for i in tqdm(range(self.n_trees)):
            # Create and fit a tree
            iter_tree = BDFTree(
                distribution=self.distribution,
                reg_beta=self.reg_beta,
                reg_lambda=self.reg_lambda,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state,
            )
            # Subsample rows and columns if specified
            if self.subsample < 1.0:
                n_samples = int(X.shape[0] * self.subsample)
                # Could allow kw bootstrap to allow replacement, do replacement below too
                row_indices = np.random.choice(
                    X.shape[0],
                    n_samples,
                    replace=False,
                )
                X_iter = X[row_indices]
                y_iter = y[row_indices]
            else:
                X_iter = X
                y_iter = y
            col_idcs = np.random.choice(X.shape[1], n_features_iter, replace=False) if self.colsample < 1.0 else None
            iter_tree.fit(X_iter, y_iter, col_idcs=col_idcs, verbose=verbose, eta=self.eta)
            self.trees.append(iter_tree)
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
        standardized_y = (y - mean_y) / std_y  # type: ignore
        self.y_mean, self.y_std = mean_y, std_y
        return standardized_y

    def _unstandardize_y(self, y: np.ndarray) -> np.ndarray:
        """Unstandardize the target variable y."""
        if hasattr(self, "y_mean") and hasattr(self, "y_std"):
            return y * self.y_std + self.y_mean
        return y

    def _get_pooled_samples(self, X: np.ndarray, sample_size: int) -> np.ndarray:
        """
        Internal helper to draw and pool samples from all trees.

        Returns an array of shape (n_obs, sample_size).
        """
        # To get a total of `sample_size` samples, we need to draw `ceil(sample_size / n_trees)` from each.
        per_tree_size = int(np.ceil(sample_size / self.n_trees))

        # Shape: (n_trees, n_obs, per_tree_size)
        tree_samples = np.array([tree.predict_samples(X, size=per_tree_size) for tree in self.trees])

        # Transpose and reshape to pool samples across trees
        # Shape: (n_obs, n_trees * per_tree_size)
        pooled_samples = tree_samples.transpose(1, 0, 2).reshape(X.shape[0], -1)

        # Return exactly sample_size samples
        return pooled_samples[:, :sample_size]

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the mean for each observation in X.

        This is an alias for `predict_mean`.
        """
        return self.predict_mean(X)

    def predict_mean(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the mean for each observation in X.

        The forest's mean prediction is the average of the means from each tree.
        """
        X_validated = self._validate_prediction_input(X)
        # Shape: (n_trees, n_obs) -> (n_obs,)
        tree_means = np.array([tree.predict_mean(X_validated) for tree in self.trees])
        forest_mean = np.mean(tree_means, axis=0)
        return self._unstandardize_y(forest_mean)

    def predict_weighted_mean(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the inverse-variance weighted mean for each observation in X.
        """
        X_validated = self._validate_prediction_input(X)
        tree_means = np.array([tree.predict_mean(X_validated) for tree in self.trees])
        tree_vars = np.array([tree.predict_variance(X_validated) for tree in self.trees])

        # Inverse variance weighting
        weights = 1.0 / tree_vars
        weighted_mean = np.sum(tree_means * weights, axis=0) / np.sum(weights, axis=0)

        return self._unstandardize_y(weighted_mean)

    def predict_median(self, X: np.ndarray | pd.DataFrame, sample_size: int = 1000) -> np.ndarray:
        """
        Predicts the median for each observation in X by pooling samples from all trees.
        """
        X_validated = self._validate_prediction_input(X)
        pooled_samples = self._get_pooled_samples(X_validated, sample_size)
        median = np.median(pooled_samples, axis=1)
        return self._unstandardize_y(median)

    def predict_quantiles(
        self, X: np.ndarray | pd.DataFrame, q: float | list[float], sample_size: int = 1000
    ) -> np.ndarray:
        """
        Predicts quantiles for each observation in X by pooling samples from all trees.
        """
        X_validated = self._validate_prediction_input(X)
        pooled_samples = self._get_pooled_samples(X_validated, sample_size)
        quantiles = np.quantile(pooled_samples, q=q, axis=1)
        # If multiple quantiles are requested, the result has shape (n_quantiles, n_obs)
        # We want (n_obs, n_quantiles) to be consistent with other predictors
        if isinstance(q, (list, tuple, np.ndarray)) and len(q) > 1:
            quantiles = quantiles.T
        return self._unstandardize_y(quantiles)

    def predict_variance(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the variance for each observation in X.

        Uses the Law of Total Variance over the trees:
        Var(Y) = E[Var(Y|Tree)] + Var(E[Y|Tree])
        """
        X_validated = self._validate_prediction_input(X)
        # Each has shape (n_trees, n_obs)
        tree_means = np.array([tree.predict_mean(X_validated) for tree in self.trees])
        tree_vars = np.array([tree.predict_variance(X_validated) for tree in self.trees])

        # E[Var(Y|T)]: Mean of variances from each tree. Shape: (n_obs,)
        expected_variance = np.mean(tree_vars, axis=0)
        # Var(E[Y|T]): Variance of means from each tree. Shape: (n_obs,)
        variance_of_expectation = np.var(tree_means, axis=0)

        total_variance = expected_variance + variance_of_expectation

        # Variance is scaled by std^2
        if hasattr(self, "y_std"):
            return total_variance * self.y_std**2
        return total_variance

    def predict_samples(self, X: np.ndarray | pd.DataFrame, sample_size: int = 1) -> np.ndarray:
        """
        Draws samples from the predictive distribution for each observation in X.
        """
        X_validated = self._validate_prediction_input(X)
        pooled_samples = self._get_pooled_samples(X_validated, sample_size)
        return self._unstandardize_y(pooled_samples)

    def predict_params(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Returns the predictive distribution parameters from each tree for each observation.

        Returns:
            np.ndarray: An array of shape (n_obs, n_trees), where each element is a
                        dictionary of parameters.
        """
        X_validated = self._validate_prediction_input(X)
        # Shape: (n_trees, n_obs) -> (n_obs, n_trees)
        params_per_tree = np.array([tree.predict_params(X_validated) for tree in self.trees]).T
        return params_per_tree

    def plot_tree(self, tree_index: int = 0, figsize: tuple[int, int] = (20, 16), dpi: int = 300):
        import io

        import graphviz  # type: ignore
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        if tree_index >= len(self.trees):
            raise ValueError(f"Tree index {tree_index} out of range. Only {len(self.trees)} trees available.")

        # Create digraph with hierarchical layout
        dot = graphviz.Digraph()
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr("node", shape="ellipse")  # Default shape for all nodes
        dot.attr(ranksep="0.6")  # Increase spacing between ranks
        dot.attr(nodesep="2")  # Increase spacing between nodes
        dot.attr(ratio="fill")

        # Make nodes larger with custom fonts
        dot.attr("node", shape="ellipse", style="filled", fontsize="20", width="1.7", height="1.4", margin="0.2,0.1")

        # Make edges thicker and more visible
        dot.attr("edge", fontsize="16", penwidth="2")

        # Use the root of the specified tree
        root = self.trees[tree_index].root

        def traverse_tree(node, node_id=None):
            if node_id is None:
                node_id = str(id(node))

            # Check if it's a leaf node (no children)
            if node.left_node is None and node.right_node is None:
                # Format posterior parameters more readably
                params_list = []
                for k, v in node.posterior_params.items():
                    if isinstance(v, float):
                        params_list.append(f"{k}: {v:.3f}")
                    else:
                        params_list.append(f"{k}: {v}")
                leaf_mean = self.distribution.get_posterior_mean(params=node.posterior_params)
                if hasattr(self, "y_mean") and hasattr(self, "y_std"):
                    leaf_mean = leaf_mean * self.y_std + self.y_mean
                # Join parameters with newlines for better readability
                params_str = "\n".join(params_list)

                dot.node(node_id, label=f"LEAF | {leaf_mean:.2f}\n{params_str}", style="filled", fillcolor="lightgreen")
            else:
                # It's a split node
                dot.node(
                    node_id,
                    label=f"Feature: {node.best_feature}\nThreshold: {node.best_threshold:.3f}",
                    style="filled",
                    fillcolor="lightgrey",
                )

                # Create unique IDs for children
                left_id = f"{node_id}_left"
                right_id = f"{node_id}_right"

                # Connect to children
                dot.edge(node_id, left_id, label="≤")
                dot.edge(node_id, right_id, label=">")

                # Traverse children
                traverse_tree(node.left_node, left_id)
                traverse_tree(node.right_node, right_id)

        # Start traversal
        traverse_tree(root)

        # Set graph rendering options
        dot.attr(dpi=str(dpi))  # Higher resolution

        try:
            # Create a PNG image
            png_str = dot.pipe(format="png")

            # Use BytesIO to read the PNG image
            sio = io.BytesIO(png_str)
            img = mpimg.imread(sio, format="png")

            # Plot the image using Matplotlib
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Tree {tree_index} (of {len(self.trees)})")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error rendering tree: {e}")
            print("Trying alternate rendering approach...")
            try:
                # Try saving to file and opening directly
                dot.render(f"tree_{tree_index}", format="png", cleanup=True)
                print(f"Tree rendered to tree_{tree_index}.png")
            except Exception as e2:
                print(f"Error with alternative rendering: {e2}")
                print("Displaying DOT source instead:")
                print(dot.source)

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
        random_state: int,
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
        assert (
            isinstance(random_state, int) and random_state >= 0
        ), f"random_state must be a non-negative integer, got {random_state} of type {type(random_state)}"
        self.random_state = random_state
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

    def _validate_prediction_input(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Validate the input for prediction."""
        if not self.is_fitted_:
            raise RuntimeError(
                "This BDFRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        if isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names"):
                if not all(col in X.columns for col in self.feature_names):  # type: ignore
                    raise ValueError("X must contain all feature names used during fitting")
                if not all(col in self.feature_names for col in X.columns):
                    warnings.warn(
                        "X contains additional columns not seen during fitting. "
                        "Prediction continues only with columns seen during fitting."
                    )
                X = X[self.feature_names].values  # type: ignore
            else:
                raise ValueError(
                    "X is a DataFrame but no feature names were stored during fitting. "
                    "Ensure to fit with a DataFrame to predict on DataFrame or fit on np.ndarray"
                )
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array or pandas DataFrame, got {type(X)}")

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array or DataFrame, got {X.ndim}D array")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but BDFRegressor was fitted with {self.n_features_in_} features."
            )

        return X

    def _validate_fit_input(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the input for fitting."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns  # type: ignore
            X = X.values  # type: ignore
            self.n_features_in_ = X.shape[1]
        if isinstance(y, pd.Series):
            y = y.to_numpy()  # type: ignore
        assert X.ndim == 2, f"X must be a 2D array, got {X.ndim}D array"
        assert y.ndim == 1, f"y must be a 1D array, got {y.ndim}D array"
        assert (
            X.shape[0] == y.shape[0]
        ), f"Number of samples in X ({X.shape[0]}) must match number of samples in y ({y.shape[0]})"
        return X, y
