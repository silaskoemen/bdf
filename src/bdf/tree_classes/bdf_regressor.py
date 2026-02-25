import warnings
from numbers import Integral, Real
from typing import Literal, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils._param_validation import Interval, StrOptions

from bdf.distributions.distribution_manager import DistributionManager as DM
from bdf.tree_classes.bdf_tree import BDFTree
from bdf.utils.constants import RANDOM_SEED

from .utils import _fit_single_tree


class BDFModel(BaseEstimator):
    """Base class for Bayesian Distributional Forest models.

    Implements a forest of Bayesian decision trees that provide full predictive
    distributions rather than point estimates. This is the shared base for
    :class:`BDFRegressor` and :class:`BDFClassifier`.

    Parameters
    ----------
    dist : str, default="NormalMuNormal"
        Name of the distribution to use. Must match a registered distribution
        class name (e.g. ``"NormalMuNormal"``, ``"GammaMVLambdaPoisson"``,
        ``"BetaMVBernoulli"``). See :doc:`distributions` for all options.
    params : dict, default={"mu_mu": "auto", "sigma_mu": "auto", "sigma_mu_auto_scale": 1.0}
        Distribution-specific parameters. Values may be ``"auto"`` for
        data-driven initialization at fit time. See the chosen distribution's
        documentation for available parameters.
    n_trees : int, default=50
        Number of trees in the forest.
    alpha : float, default=0.0
        Structural prior strength. For ``tree_prior_mode="linear"`` this is
        unused (set to 0). For ``"defer"`` or ``"bernoulli"`` modes, must be
        in (0, 1).
    gamma : float, default=0.1
        Complexity penalty applied to each split. Higher values produce
        simpler trees. Must be in [0, 1].
    delta : float, default=0.01
        Depth decay parameter. For ``tree_prior_mode="linear"`` controls
        per-depth penalty decay. For ``"defer"``/``"bernoulli"`` modes, must
        be in (0, 1).
    tree_prior_mode : {"linear", "defer", "bernoulli"}, default="linear"
        Tree structure prior mode. ``"linear"`` uses additive penalty terms.
        ``"defer"`` and ``"bernoulli"`` use log-scale priors with stopping
        probabilities.
    max_depth : int, default=50
        Maximum depth of each tree.
    min_samples_leaf : int, default=10
        Minimum number of samples required at a leaf node.
    min_samples_split : int, default=20
        Minimum number of samples required to split an internal node.
    min_child_weight : int or float, default=10
        Minimum sum of instance weight (or sample count) in a child node.
    subsample : float, default=0.9
        Fraction of samples used for fitting each tree. Must be in (0, 1].
    colsample : float, default=0.9
        Fraction of features considered at each split. Must be in (0, 1].
    eta : float, default=0.01
        Quantile step size for generating candidate split thresholds. Smaller
        values produce more candidates. Must be in (0, 1].
    bootstrap : bool, default=True
        Whether to use bootstrap sampling (with replacement) for each tree.
        If False, uses subsampling without replacement.
    n_jobs : int, default=-1
        Number of parallel jobs for tree fitting. ``-1`` uses all available
        cores.
    verbose : int, default=0
        Verbosity level. ``0`` is silent, higher values print progress.
    random_state : int, default=1234
        Random seed for reproducibility.
    """

    _parameter_constraints = {
        "dist": [str],
        "params": [dict],
        "n_trees": [Interval(Integral, 1, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "gamma": [Interval(Real, 0, 1, closed="both")],
        "delta": [Interval(Real, 0, 1, closed="both")],
        "tree_prior_mode": [StrOptions({"linear", "defer", "bernoulli"})],
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "min_samples_split": [Interval(Integral, 2, None, closed="left")],
        "min_child_weight": [Interval(Real, 0, None, closed="left")],
        "subsample": [Interval(Real, 0, 1, closed="right")],
        "colsample": [Interval(Real, 0, 1, closed="right")],
        "eta": [Interval(Real, 0, 1, closed="right")],
        "bootstrap": ["boolean"],
        "oob_weights": ["boolean"],
        "oob_temperature": [Interval(Real, 0, None, closed="neither")],
        "n_jobs": [Integral],
        "verbose": [Interval(Integral, -1, None, closed="left")],
        "random_state": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        dist: str,
        params: dict,
        n_trees: int = 50,
        alpha: float = 0.0,
        gamma: float = 0.01,
        delta: float = 0.001,
        tree_prior_mode: Literal["linear", "defer", "bernoulli"] = "linear",
        max_depth: int = 50,
        min_samples_leaf: int = 10,
        min_samples_split: int = 20,
        min_child_weight: int | float = 10,
        subsample: float = 0.75,
        colsample: float = 0.9,
        eta: float = 0.01,
        bootstrap: bool = True,
        oob_weights: bool = False,
        oob_temperature: float = 1.0,
        n_jobs: int = -1,
        verbose: int = 0,
        random_state: int = RANDOM_SEED,
    ):
        self.is_fitted_ = False
        self.dist = dist
        self.params = params
        self.n_trees = n_trees
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.tree_prior_mode = tree_prior_mode
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample = colsample
        self.eta = eta
        self.bootstrap = bootstrap
        self.oob_weights = oob_weights
        self.oob_temperature = oob_temperature
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def _validate_cross_params(self):
        """Validate cross-parameter constraints that can't be expressed declaratively."""
        if self.tree_prior_mode in ("defer", "bernoulli"):
            if not (0 < self.alpha < 1):
                raise ValueError(
                    f"For tree_prior_mode='{self.tree_prior_mode}', alpha must be in (0, 1), got {self.alpha}"
                )
            if not (0 < self.delta < 1):
                raise ValueError(
                    f"For tree_prior_mode='{self.tree_prior_mode}', delta must be in (0, 1), got {self.delta}"
                )
        if self.oob_weights and self.bootstrap:
            warnings.warn(
                "oob_weights=True with bootstrap=True may produce unreliable OOB scores "
                "because bootstrap duplicates tighten leaf posteriors. "
                "Consider setting bootstrap=False.",
                stacklevel=2,
            )
        if self.oob_weights and self.subsample >= 1.0:
            warnings.warn(
                "oob_weights=True has no effect when subsample >= 1.0 (no out-of-bag samples). "
                "Falling back to uniform weights.",
                stacklevel=2,
            )

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Fit the model to the training data.
        Args
        ----
        `X` : np.ndarray | pd.DataFrame
            Training data features.
        `y` : np.ndarray | pd.Series
            Training data target values.
        """
        self._validate_params()
        self._validate_cross_params()
        # Seed for reproducibility of subsample and colsample, reseed for each fit call
        self.rng = np.random.default_rng(self.random_state)
        np.random.seed(self.random_state)
        X, y = self._validate_fit_input(X, y)
        self.n_features_in_ = X.shape[1]

        # Optimization: Convert to Fortran order for faster column access in Rust
        if not np.isfortran(X):
            X = np.asfortranarray(X)

        self.distribution = DM.create_distribution(dist=self.dist, params=self.params, y=y)
        self.distribution.validate_targets(y)

        # Initialize FFT for KDE if needed
        if self._is_fft_kde():
            self._init_kde_fft(y)

        # Otherwise regularization depends on size of the dataset (NLL as sum)
        # n_features_iter = int(np.ceil(X.shape[1] * self.colsample))
        penalty = self.alpha  # * np.log(np.ceil(X.shape[0] * self.subsample))  # before had n

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_tree)(
                X=X,
                y=y,
                verbose=verbose,
                distribution=self.distribution,
                alpha=self.alpha,
                gamma=self.gamma,
                delta=self.delta,
                tree_prior_mode=self.tree_prior_mode,
                penalty=penalty,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample=self.colsample,
                bootstrap=self.bootstrap,
                eta=self.eta,
                random_state=self.random_state + i,
                return_oob_mask=self.oob_weights,
            )
            for i in range(self.n_trees)
        )

        if self.oob_weights:
            trees_and_masks = cast(list[tuple[BDFTree, np.ndarray | None]], results)
            self.trees = [t for t, _ in trees_and_masks]
            oob_masks = [m for _, m in trees_and_masks]
            self.oob_scores_, self.tree_weights_ = self._compute_oob_weights(X, y, self.trees, oob_masks)
        else:
            self.trees = cast(list[BDFTree], results)
            self.oob_scores_ = None
            self.tree_weights_ = None

        self.is_fitted_ = True

        # temporary debug: print average number of nodes and depth
        if self.verbose > 0:
            avg_nodes = np.mean([tree.count_nodes() for tree in self.trees])
            avg_depth = np.mean([tree.get_max_depth() for tree in self.trees])
            print(f"Fitted {self.n_trees} trees with average nodes: {avg_nodes:.2f}, average depth: {avg_depth:.2f}")

    def _compute_oob_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        trees: list[BDFTree],
        oob_masks: list[np.ndarray | None],
        min_oob_samples: int = 50,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute OOB NLL scores and softmax weights for each tree.

        Falls back to uniform weights (returns None, None) when too few
        trees have enough OOB samples for reliable scoring.
        """
        n_total = X.shape[0]
        if self.bootstrap:
            # With replacement: expected unique fraction is 1-exp(-s), so OOB ≈ n*exp(-s)
            expected_oob = n_total * np.exp(-self.subsample)
        else:
            expected_oob = n_total * (1 - self.subsample)
        if expected_oob < min_oob_samples:
            warnings.warn(
                f"Expected ~{expected_oob:.0f} OOB samples per tree (subsample={self.subsample}), "
                f"which is below {min_oob_samples}. Falling back to uniform weights. "
                f"Consider lowering subsample or increasing the dataset size.",
                stacklevel=2,
            )
            return None, None

        scores = np.full(len(trees), np.inf)
        n_valid = 0
        for i, (tree, mask) in enumerate(zip(trees, oob_masks)):
            if mask is None:
                continue
            n_oob = int(mask.sum())
            if n_oob < min_oob_samples:
                continue
            n_valid += 1
            X_oob, y_oob = X[mask], y[mask]
            log_liks = tree.predict_log_likelihood(X_oob, y_oob)
            # Use median NLL — robust to outlier observations that produce
            # extreme log-likelihoods (e.g. tight-posterior leaf seeing an
            # out-of-distribution OOB point).
            scores[i] = -np.median(log_liks)  # median NLL (lower = better)

        if n_valid < len(trees) // 2:
            warnings.warn(
                f"Only {n_valid}/{len(trees)} trees had >= {min_oob_samples} OOB samples. "
                f"Falling back to uniform weights.",
                stacklevel=2,
            )
            return None, None

        # For trees without enough OOB samples, assign median score (neutral weight)
        valid_mask = np.isfinite(scores)
        if not valid_mask.all():
            scores[~valid_mask] = np.median(scores[valid_mask])

        # Softmax: lower NLL → higher weight
        shifted = -scores / self.oob_temperature
        shifted -= shifted.max()  # numerical stability
        weights = np.exp(shifted)
        weights /= weights.sum()
        return scores, weights

    def _weighted_mean(self, values: np.ndarray) -> np.ndarray:
        """Weighted or uniform mean across trees (axis=0)."""
        if self.tree_weights_ is not None:
            return np.average(values, axis=0, weights=self.tree_weights_)
        return np.mean(values, axis=0)

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

    def _unstandardize_y(self, y: np.ndarray) -> np.ndarray:
        """Unstandardize the target variable y."""
        if hasattr(self, "y_mean") and hasattr(self, "y_std"):
            return y * self.y_std + self.y_mean
        return y

    def _get_pooled_samples(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Internal helper to draw and pool samples from all trees.

        Returns an array of shape (n_obs, n_samples).
        When OOB weights are active, each tree contributes samples proportional
        to its weight.
        """
        if self.tree_weights_ is not None:
            # Draw samples per tree proportional to weight
            counts = self.rng.multinomial(n_samples, self.tree_weights_)
            all_samples = []
            for tree, count in zip(self.trees, counts):
                if count > 0:
                    all_samples.append(tree.predict_samples(X, n_samples=count))
            # Shape: (n_obs, n_samples)
            return np.concatenate(all_samples, axis=1)

        # Uniform: draw equal samples from each tree
        per_tree_size = int(np.ceil(n_samples / self.n_trees))
        tree_samples = np.array([tree.predict_samples(X, n_samples=per_tree_size) for tree in self.trees])
        pooled_samples = tree_samples.transpose(1, 0, 2).reshape(X.shape[0], -1)
        indices = self.rng.choice(pooled_samples.shape[1], size=n_samples, replace=False)
        return pooled_samples[:, indices]

    def predict_mean(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the mean for each observation in X.

        The forest's mean prediction is the average of the means from each tree.
        """
        X_validated = self._validate_prediction_input(X)
        # Shape: (n_trees, n_obs) -> (n_obs,)
        tree_means = np.array([tree.predict_mean(X_validated) for tree in self.trees])
        forest_mean = self._weighted_mean(tree_means)
        return self._unstandardize_y(forest_mean)

    def predict_weighted_mean(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the inverse-variance weighted mean for each observation in X.
        """
        X_validated = self._validate_prediction_input(X)
        tree_means = np.array([tree.predict_mean(X_validated) for tree in self.trees])
        tree_vars = np.array([tree.predict_variance(X_validated) for tree in self.trees])

        # Inverse variance weighting, optionally combined with OOB weights
        weights = 1.0 / tree_vars
        if self.tree_weights_ is not None:
            weights *= self.tree_weights_[:, np.newaxis]
        weighted_mean = np.sum(tree_means * weights, axis=0) / np.sum(weights, axis=0)

        return self._unstandardize_y(weighted_mean)

    def predict_median(self, X: np.ndarray | pd.DataFrame, n_samples: int = 1000) -> np.ndarray:
        """
        Predicts the median for each observation in X by pooling samples from all trees.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Input features for prediction.
        n_samples : int, optional
            Number of samples to pool for computing the median. Default is 1000.

        Returns
        -------
        np.ndarray
            Median predictions of shape (n_samples,).
        """
        X_validated = self._validate_prediction_input(X)
        pooled_samples = self._get_pooled_samples(X_validated, n_samples)
        median = np.median(pooled_samples, axis=1)
        return self._unstandardize_y(median)

    def predict_quantiles(
        self, X: np.ndarray | pd.DataFrame, q: float | list[float], n_samples: int = 1000
    ) -> np.ndarray:
        """
        Predicts quantiles for each observation in X by pooling samples from all trees.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Input features for prediction.
        q : float | list[float]
            Quantile(s) to compute. Must be between 0 and 1.
        n_samples : int, optional
            Number of samples to pool for computing quantiles. Default is 1000.

        Returns
        -------
        np.ndarray
            Quantile predictions. Shape is (n_obs,) for single quantile or
            (n_obs, n_quantiles) for multiple quantiles.
        """
        X_validated = self._validate_prediction_input(X)
        pooled_samples = self._get_pooled_samples(X_validated, n_samples)
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

        # E[Var(Y|T)]: (Weighted) mean of variances from each tree. Shape: (n_obs,)
        expected_variance = self._weighted_mean(tree_vars)
        # Var(E[Y|T]): (Weighted) variance of means from each tree. Shape: (n_obs,)
        if self.tree_weights_ is not None:
            weighted_mu = np.average(tree_means, axis=0, weights=self.tree_weights_)
            variance_of_expectation = np.average((tree_means - weighted_mu) ** 2, axis=0, weights=self.tree_weights_)
        else:
            variance_of_expectation = np.var(tree_means, axis=0)

        total_variance = expected_variance + variance_of_expectation

        # Variance is scaled by std^2
        if hasattr(self, "y_std"):
            return total_variance * self.y_std**2
        return total_variance

    def predict_samples(self, X: np.ndarray | pd.DataFrame, n_samples: int = 1) -> np.ndarray:
        """
        Draws samples from the predictive distribution for each observation in X.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Input features for prediction.
        n_samples : int, optional
            Number of samples to draw from the predictive distribution. Default is 1.

        Returns
        -------
        np.ndarray
            Samples from the predictive distribution of shape (n_obs, n_samples).
        """
        X_validated = self._validate_prediction_input(X)
        pooled_samples = self._get_pooled_samples(X_validated, n_samples)
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

        import graphviz
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

    def _validate_prediction_input(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Validate the input for prediction."""
        if not self.is_fitted_:
            raise RuntimeError(
                "This BDFRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        if isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names"):
                if not all(col in X.columns for col in self.feature_names):
                    raise ValueError("X must contain all feature names used during fitting")
                if not all(col in self.feature_names for col in X.columns):
                    warnings.warn(
                        "X contains additional columns not seen during fitting. "
                        "Prediction continues only with columns seen during fitting."
                    )
                X = X[self.feature_names].values
            else:
                raise ValueError(
                    "X is a DataFrame but no feature names were stored during fitting. "
                    "Ensure to fit with a DataFrame to predict on DataFrame or fit on np.ndarray"
                )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but BDFRegressor was fitted with {self.n_features_in_} features."
            )
        if not isinstance(X, np.ndarray):
            raise ValueError(f"After casting from pandas, X has to be a np.ndarray, got type {type(X)}")

        self._validate_features(X)
        return X

    def validate_array(
        self, data: np.ndarray, allow_nan: bool = False, allow_inf: bool = False, allow_none: bool = False
    ):
        """
        Validates a NumPy array to ensure it is numeric and does not contain unwanted values.

        This function is optimized for performance and can optionally allow None values
        for future sparsity-aware implementations.

        Args:
            data (np.ndarray): The input array to validate.
            allow_nan (bool): If False, raises an error if NaN is found.
            allow_inf (bool): If False, raises an error if Inf or -Inf is found.
            allow_none (bool): If True, allows the array to contain None values.

        Raises:
            ValueError: If the data is not numeric or contains forbidden values.
        """
        # Fast path for purely numeric arrays (cannot contain None)
        if np.issubdtype(data.dtype, np.number):
            if not allow_nan and np.isnan(data).any():
                raise ValueError("Input data contains NaN values.")
            if not allow_inf and np.isinf(data).any():
                raise ValueError("Input data contains infinite values.")
            return

        # Slower path for object arrays, which might contain None or other types
        if data.dtype == "object":
            if not allow_none:
                # If None is not allowed, try a direct conversion which will fail on None.
                try:
                    data.astype(np.float64)
                except (ValueError, TypeError):
                    raise ValueError(
                        "Input data contains non-numeric values or None, and could not be converted to float."
                    )

            # If None is allowed, we must check the non-None elements
            else:
                # Create a mask to isolate non-None elements
                non_none_mask = data is not None
                numeric_subset = data[non_none_mask]

                # If there are any non-None elements, validate them
                if numeric_subset.size > 0:
                    try:
                        # Check if the non-None elements are actually numeric
                        numeric_subset = numeric_subset.astype(np.float64)
                    except (ValueError, TypeError):
                        raise ValueError("Input data contains non-numeric string values alongside None.")

                    # Perform checks on the validated numeric subset
                    if not allow_nan and np.isnan(numeric_subset).any():
                        raise ValueError("Input data's numeric subset contains NaN values.")
                    if not allow_inf and np.isinf(numeric_subset).any():
                        raise ValueError("Input data's numeric subset contains infinite values.")
            return

        # If it's some other non-numeric, non-object dtype, it's invalid.
        raise ValueError(f"Unsupported data type '{data.dtype}'. Data must be numeric or object type.")

    def _validate_fit_input(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the input for fitting."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
            self.n_features_in_ = X.shape[1]
        if isinstance(y, pd.Series):
            y = y.to_numpy().astype(np.float64)
        assert (
            X.shape[0] == y.shape[0]
        ), f"Number of samples in X ({X.shape[0]}) must match number of samples in y ({y.shape[0]})"
        X, y = X.astype(np.float64), y.astype(np.float64)
        self._validate_features(X)
        self._validate_targets(y)  # Passes or raises Assertion-/ValueError
        return X, y

    def _validate_features(self, X: np.ndarray):
        """Validate that input features are all not NaN/None/numerical etc."""
        assert X.ndim == 2, f"X must be a 2D array, got {X.ndim}D array"
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        assert not np.isnan(X).any(), "Input data cannot be NaN."
        assert np.issubdtype(
            X.dtype, np.floating
        ), "Input data has to be subtype of float. If it fails although all features are numeric, consider casting to float/int for all columns."

    def _validate_targets(self, y: np.ndarray):
        """Validation function to check whether targets are allowed under the
        given distribution.

        Args
        ----
        `y` : np.ndarray
            targets used for fit input
        """
        assert y.ndim == 1, f"y must be a 1D array, got {y.ndim}D array"
        assert not np.isnan(y).any(), "Targets cannot be NaN."
        assert np.issubdtype(
            y.dtype, np.floating
        ), "Targets have to be subtype of float. If it fails although all targets are numeric, consider casting to float/int."
        # self.distribution.validate_targets(y)

    def _is_fft_kde(self) -> bool:
        """Check if distribution is FFT-based KDE."""
        from bdf.distributions.kde import KDE, BayesianKDE

        if not isinstance(self.distribution, (KDE, BayesianKDE)):
            return False
        backend = getattr(self.distribution.params, "kde_backend", "pairwise")
        kernel = getattr(self.distribution.params, "kernel", "gaussian")
        policy = getattr(self.distribution.params, "bandwidth_policy", "parent")
        return backend in ("fft", "switch") and kernel == "gaussian" and policy == "parent"

    def _init_kde_fft(self, y: np.ndarray) -> None:
        """Initialize FFT grid and kernel for KDE.

        Computes global grid edges and kernel FFT that will be used
        by all tree nodes for fast density evaluation.

        Parameters
        ----------
        y : np.ndarray
            Target values (after standardization if applicable).
        """
        n_grid = int(self.distribution.params.fft_grid_points)

        # Compute global grid edges with padding.
        # Match Rust's Gaussian cutoff (≈ ±4σ) when possible.
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        if y_range <= 0:
            y_range = 1.0
        try:
            h = float(self.distribution._compute_bandwidth(y))  # type: ignore[attr-defined]
        except Exception:
            h = float("nan")

        padding = 4.0 * h if np.isfinite(h) and h > 0 else (0.2 * y_range)

        grid_min = float(y_min - padding)
        grid_max = float(y_max + padding)
        delta = float((grid_max - grid_min) / n_grid)

        # Update distribution params with FFT precomputation
        # Use model_copy to create new params with FFT data
        updated_params = self.distribution.params.model_copy(
            update={
                "fft_grid_min": grid_min,
                "fft_grid_max": grid_max,
                "fft_grid_delta": delta,
                "fft_grid_points": n_grid,
            }
        )

        # Replace distribution with updated params
        self.distribution = self.distribution.__class__(params=updated_params)


class BDFRegressor(BDFModel, RegressorMixin):
    """BDF Regressor for regression tasks with continuous targets.

    Provides sklearn-compatible interface with flexible prediction methods
    for uncertainty quantification.
    """

    def __init__(
        self,
        dist: str = "NormalMuNormal",
        params: dict = {"mu_mu": "auto", "sigma_mu": "auto", "sigma_mu_auto_scale": 5.0},
        **kwargs,
    ):
        super().__init__(dist=dist, params=params, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False, standardize_y: bool = False):
        """Fit the BDFRegressor to the training data.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Training data features.
        y : np.ndarray | pd.Series
            Training data target values.
        verbose : bool, optional
            Whether to print training progress. Default is False.
        standardize_y : bool, optional
            Whether to standardize the target values. Default is False.

        Returns
        -------
        self
            Fitted regressor.
        """
        # Standardize y if requested (classification doesn't support this)
        if standardize_y:
            # Seed for reproducibility
            self.rng = np.random.default_rng(self.random_state)
            np.random.seed(self.random_state)
            X, y = self._validate_fit_input(X, y)
            y = self._standardize_y(y.copy())
            # Call parent fit without standardize_y parameter
            return super().fit(X, y, verbose=verbose)
        else:
            # Call parent fit directly
            return super().fit(X, y, verbose=verbose)

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        method: Literal[
            "mean",  # mean of means
            "median",  # median of pooled samples
            "var",  # mean of variances
            "var-samples",  # variance of pooled samples
            "interval",  # one or more intervals at confidence levels
            "quantile",  # one or more quantiles
            "samples",  # pooled samples
            "params",  # distribution parameters from each tree
        ] = "mean",
        method_params: dict | None = None,
    ) -> np.ndarray:
        """
        Flexible prediction method supporting multiple prediction types.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Input features for prediction.
        method : str, optional
            Prediction method to use. Default is "mean".
            - "mean": Mean of posterior means from each tree
            - "median": Median of pooled samples
            - "var": Variance via Law of Total Variance
            - "samples": Pooled samples from posterior predictive
            - "params": Distribution parameters from each tree
        method_params : dict | None, optional
            Additional parameters for the chosen method.
            For "samples", use {"n_samples": int}.

        Returns
        -------
        np.ndarray
            Predictions according to the specified method.
        """
        match method:
            case "mean":
                return self.predict_mean(X)
            case "median":
                return self.predict_median(X)
            case "var":
                return self.predict_variance(X)
            case "samples":
                return self.predict_samples(X, method_params.get("n_samples", 1) if method_params else 1)
            case "params":
                return self.predict_params(X)
            case _:
                raise ValueError(f"Unknown prediction method: {method}")


class BDFClassifier(BDFModel, ClassifierMixin):
    """BDF Classifier for binary classification tasks.

    Provides sklearn-compatible interface with predict() returning class labels
    and predict_proba() returning class probabilities. All underlying predict_XXX
    methods operate on the probability of the positive class (class 1).
    """

    def __init__(
        self,
        dist: str = "BetaMVBernoulli",
        params: dict = {"mean_p": "auto", "var_p": 0.05},
        **kwargs,
    ):
        super().__init__(dist=dist, params=params, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """Fit the BDFClassifier to the training data.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Training data features.
        y : np.ndarray | pd.Series
            Binary target values (0 or 1). Will be validated to ensure all
            values are in {0, 1}.
        verbose : bool, optional
            Whether to print training progress. Default is False.

        Raises
        ------
        ValueError
            If targets contain values other than 0 and 1.
        """
        # Validate binary targets before fitting
        if isinstance(y, pd.Series):
            y_check = y.to_numpy()
        else:
            y_check = y

        unique_values = np.unique(y_check)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                f"BDFClassifier requires binary targets with values in {{0, 1}}. "
                f"Found unique values: {unique_values}"
            )

        # Call parent fit (classification never standardizes targets)
        super().fit(X, y, verbose=verbose)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for X.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) with probabilities for each class.
            Column 0 is P(y=0), column 1 is P(y=1).
        """
        p1 = self.predict_mean(X)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels for X.

        Args
        ----
        X : np.ndarray | pd.DataFrame
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1) of shape (n_samples,).
        """
        return (self.predict_mean(X) >= 0.5).astype(int)

    # def _validate_targets(self, y: np.ndarray):
    #     """Validation function to check whether targets are allowed under the
    #     given distribution.

    #     Args
    #     ----
    #     `y` : np.ndarray
    #         targets used for fit input
    #     """
    #     assert y.ndim == 1, f"y must be a 1D array, got {y.ndim}D array"
    #     assert not np.isnan(y).any(), "Targets cannot be NaN."
    #     assert np.issubdtype(
    #         y.dtype, np.floating
    #     ), "Targets have to be subtype of float. If it fails although all targets are numeric, consider casting to float/int."
    #     # self.distribution.validate_targets(y)
