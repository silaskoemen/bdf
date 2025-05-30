import warnings

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
            )
            # Subsample rows and columns if specified
            if self.subsample < 1.0:
                n_samples = int(X.shape[0] * self.subsample)
                # Could allow kw bootstrap to allow replacement, do replacement below too
                row_indices = np.random.choice(X.shape[0], n_samples, replace=False)
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

    def predict(self, X: np.ndarray | pd.DataFrame, method: str = "mean", values: dict = {}) -> np.ndarray:  # type: ignore
        X: np.ndarray = self._validate_prediction_input(X, method=method, values=values)
        preds = np.empty((X.shape[0],), dtype=float)
        match method:
            case "mean":
                preds = np.mean(
                    np.concat([np.expand_dims(tree.predict(X, method="mean"), -1) for tree in self.trees], axis=-1),
                    axis=1,
                )
            case "median":
                if "total_size" in values:
                    values = {"size": np.ceil(values["total_size"] / self.n_trees).astype(int)}
                else:
                    size = values.get("size", 100)
                    values = {"size": size}
                preds = np.median(
                    np.concat([tree.predict(X, method="sample", values=values) for tree in self.trees], axis=1), axis=1
                )
            case "params":
                # Initialize an empty dictionary to store aggregated parameters
                preds = np.empty((X.shape[0],), dtype=object)

                # Predict parameters using each tree
                tree_params = [tree.predict(X, method="params", values=values) for tree in self.trees]

                # For each sample index
                for sample_idx in range(X.shape[0]):
                    # Get all parameter dictionaries for this sample from each tree
                    sample_params_list = [tree_param[sample_idx] for tree_param in tree_params]

                    # Get all parameter names from the first tree
                    param_names = sample_params_list[0].keys()

                    # Calculate the mean for each parameter across trees
                    preds[sample_idx] = {
                        param_name: np.mean([params[param_name] for params in sample_params_list])
                        for param_name in param_names
                    }
            case "samples-ind":
                preds = np.concatenate([tree.predict(X, method="sample", values=values) for tree in self.trees])
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
            if method in [
                "mean",  # mean of all trees
                "median",  # median of all trees
                "samples-ind",  # concat samples from trees
                "samples-avg",  # sample from avg params
                "quantiles-ind",  # quantiles from concat samples from trees
                "quantiles-avg",  # quantiles from sampling avg params
                "confint-ind",  # confints from concat samples from trees
                "confint-avg",  # confints from sampling avg params
            ]:
                preds: np.ndarray = preds * self.y_std + self.y_mean
            elif method == "params":  # params
                warnings.warn("Returning parameters fitted on standardized data.")
            else:
                raise ValueError(f"Method '{method}' is not supported for prediction with standardized y.")
        return preds

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
        self, X: np.ndarray | pd.DataFrame, method: str = "mean", values: dict = {}
    ) -> np.ndarray:
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
                    col in X.columns for col in self.feature_names  # type: ignore
                ), "X must contain all feature names used during fitting"
                X = X[self.feature_names].values  # type: ignore
            else:
                raise ValueError(
                    "X is a DataFrame but no feature names were stored during fitting. Ensure to fit with a DataFrame to predict on DataFrame or fit on np.ndarray"
                )
        assert isinstance(X, np.ndarray), f"X must be a numpy array, got {type(X)}"

        assert isinstance(method, str) and method in [
            "mean",
            "median",
            "params",
            "samples-ind",
            "samples-avg",
            "quantiles-ind",
            "quantiles-avg",
            "confint-ind",
            "confint-avg",
        ], f"Invalid method '{method}' for prediction. Must be one of ['mean', 'median', 'params', 'samples-ind', 'samples-avg', 'quantiles-ind', 'quantiles-avg', 'confint-ind', 'confint-avg']"

        if method in ["quantiles-ind", "quantiles-avg", "confint-ind", "confint-avg"]:
            assert values is not None, "values must be provided for quantile/confidence interval predictions"
            assert isinstance(values, dict) and all(
                isinstance(v, (int, float)) for v in values.values()
            ), "values must be a dict of numeric quantiles or confidence levels"
        assert X.ndim == 2, f"X must be a 2D array, got {X.ndim}D array"
        assert X.shape[0] > 0, "X must contain at least one sample"
        return X

    def _validate_fit_input(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the input for fitting."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns  # type: ignore
            X = X.values  # type: ignore
        if isinstance(y, pd.Series):
            y = y.to_numpy()  # type: ignore
        assert X.ndim == 2, f"X must be a 2D array, got {X.ndim}D array"
        assert y.ndim == 1, f"y must be a 1D array, got {y.ndim}D array"
        assert (
            X.shape[0] == y.shape[0]
        ), f"Number of samples in X ({X.shape[0]}) must match number of samples in y ({y.shape[0]})"
        return X, y
