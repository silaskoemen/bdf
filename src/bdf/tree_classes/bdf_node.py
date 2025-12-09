import warnings

import numpy as np

from bdf.distributions.bdf_distribution import BDFDistribution


class BDFNode:
    """Base class for all BDF nodes."""

    def __init__(self, distribution: BDFDistribution, depth: int, random_state: int):
        self.distribution = distribution
        self.depth = depth
        self.random_state = random_state
        self.left_node, self.right_node = None, None

    def estimate_posterior(self, y: np.ndarray, params: dict | None = None):
        if params is not None:
            self.posterior_params: dict = params
        else:
            self.posterior_params: dict = self.distribution.calc_posterior_params(y)  # type: ignore

    def predict_variance(self) -> float:
        """Return the posterior variance of the node's distribution."""
        assert hasattr(self, "posterior_params"), "Posterior params not estimated. Call estimate_posterior() first."
        return self.distribution.get_posterior_variance(params=self.posterior_params)

    def predict_params(self) -> dict:
        """Return the posterior parameters of the node's distribution."""
        assert hasattr(self, "posterior_params"), "Posterior params not estimated. Call estimate_posterior() first."
        return self.posterior_params

    def _is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return self.left_node is None and self.right_node is None

    def split_node(self, y, feat_idx: int, threshold: float, left_idx, right_idx, left_params, right_params):
        """Split the node into left and right children based on the best feature and threshold.

        Parameters
        ----------
        feat_idx : int
            Index of the feature to split on
        threshold : float
            Threshold value to split on
        left_idx : np.ndarray
            Boolean indices for left child
        right_idx : np.ndarray
            Boolean indices for right child
        left_params : dict
            Precomputed parameters for the left child distribution
        right_params : dict
            Precomputed parameters for the right child distribution
        """
        self.best_feature = feat_idx
        self.best_threshold = threshold

        # Create left and right nodes
        self.left_node = BDFNode(distribution=self.distribution, depth=self.depth + 1, random_state=self.random_state)  # type: ignore
        self.right_node = BDFNode(distribution=self.distribution, depth=self.depth + 1, random_state=self.random_state)  # type: ignore

        # Estimate posterior for left and right nodes
        self.left_node.estimate_posterior(y[left_idx], params=left_params)  # type: ignore
        self.right_node.estimate_posterior(y[right_idx], params=right_params)  # type: ignore

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_leaf: int,
        min_child_weight: float,
        col_idcs: list | np.ndarray | None = None,
        eta=0.025,
    ) -> (
        tuple[int, float, float, np.ndarray, np.ndarray, dict | None, dict | None]
        | tuple[None, None, float, None, None, None, None]
    ):
        """Find best split considering constraints directly in the node

        Args
        ----
        `X` : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        `y` : np.ndarray
            Target vector of shape (n_samples,)
        `min_samples_leaf` : int
            Minimum number of samples required to be at a leaf node
        `min_child_weight` : float
            Minimum sum of instance weight (hessian) needed in a child
        `col_idcs` : list | np.ndarray, optional
            Indices of features to consider for splitting. If None, all features are considered.
        `eta` : float, optional
            Step size for quantile thresholds, default is 0.025.

        Returns
        -------
        `best_feature, best_threshold, best_loss_reduction, best_left_indices, best_right_indices` : tuple
            - `best_feature`: Index of the best feature to split on
            - `best_threshold`: Threshold value for the best split
            - `best_loss_reduction`: Reduction in loss from the split
            - `best_left_indices`: Boolean array indicating indices for left child
            - `best_right_indices`: Boolean array indicating indices for right child
        """
        try:
            # Import and use the Rust implementation
            import bdf_rs  # type: ignore[import-untyped]

            # Create distribution spec with native and fallback options
            dist_spec = self.distribution.to_rust_spec()

            # Convert column indices if provided
            if col_idcs is not None:
                col_idcs = np.array(col_idcs, dtype=np.int64)

            # Call the unified Rust implementation
            (
                feature_idx,
                threshold,
                loss_reduction,
                left_indices,
                right_indices,
                left_params,
                right_params,
            ) = bdf_rs.find_best_split(  # type: ignore
                X, y, min_samples_leaf, min_child_weight, dist_spec, eta, col_idcs
            )

            return feature_idx, threshold, loss_reduction, left_indices, right_indices, left_params, right_params

        except (ImportError, Exception) as e:
            warnings.warn(f"Rust implementation not available or failed: {e}. Falling back to Python implementation.")
            return self._find_best_split_python(X, y, min_samples_leaf, min_child_weight, col_idcs, eta)

    def _find_best_split_python(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_leaf: int,
        min_child_weight: float,
        col_idcs: list | np.ndarray | None = None,
        eta=0.025,
    ) -> (
        tuple[int, float, float, np.ndarray, np.ndarray, None, None] | tuple[None, None, float, None, None, None, None]
    ):
        n_samples, n_features = X.shape
        best_feature: int | None = None
        best_threshold: float | None = None
        best_loss_reduction = 0.0
        best_left_indices: np.ndarray | None = None
        best_right_indices: np.ndarray | None = None
        n_thresholds = int(np.ceil(1 / eta))  # Number of thresholds to consider per feature

        # Current node NLL
        current_score = self.distribution.score(y)

        # Try each feature
        feature_idcs = range(n_features) if col_idcs is None else col_idcs
        for feature_idx in feature_idcs:
            # thresholds = np.quantile(X[:, feature_idx], np.linspace(0, 1, n_thresholds+2)[1:-1], method='closest_observation')
            thresholds = self._generate_candidate_thresholds(X[:, feature_idx], n_thresholds)

            # Ensures constant values will return None, always value in between taken as threshold
            for i in range(1, len(thresholds)):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                threshold = (thresholds[i] + thresholds[i - 1]) / 2
                # Split data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices

                # Check min_samples_leaf constraint
                if (
                    np.sum(left_indices) < min_samples_leaf
                    or np.sum(right_indices) < min_samples_leaf
                    or np.sum(left_indices) < min_child_weight
                    or np.sum(right_indices) < min_child_weight
                ):
                    continue

                left_score = self.distribution.score(y[left_indices])
                right_score = self.distribution.score(y[right_indices])

                # Calculate loss reduction (improvement)
                loss_reduction = current_score - (left_score + right_score)
                if loss_reduction > best_loss_reduction:
                    best_loss_reduction = loss_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_feature is None or best_threshold is None or best_left_indices is None or best_right_indices is None:
            return None, None, 0, None, None, None, None

        # No need to estimate params as python distribution call will be done either way
        return best_feature, best_threshold, best_loss_reduction, best_left_indices, best_right_indices, None, None

    def _is_kde_distribution(self) -> bool:
        """Return True if this node's distribution is a KDE-like distribution."""
        from bdf.distributions.kde import KDE, BayesianKDE  # local import to avoid cycles

        return isinstance(self.distribution, (KDE, BayesianKDE))

    def _find_best_split_python_kde(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_leaf: int,
        min_child_weight: float,
        col_idcs: list | np.ndarray | None = None,
        eta=0.025,
    ):
        """Specialized split search for KDE distributions with node-level kernel cache."""
        n_samples, n_features = X.shape
        best_feature: int | None = None
        best_threshold: float | None = None
        best_loss_reduction = 0.0
        best_left_indices: np.ndarray | None = None
        best_right_indices: np.ndarray | None = None
        n_thresholds = int(np.ceil(1 / eta))

        # Precompute node-level kernel once
        kde = self.distribution  # typed as KDE/BayesianKDE
        # Use the node's posterior params if already estimated, else compute
        params = getattr(self, "posterior_params", None)
        log_kernel, _ = kde._precompute_node_log_kernel(y, params=params)  # type: ignore[attr-defined]

        # Current node NLL using entire y (reuse kernel)
        full_mask = np.ones(n_samples, dtype=bool)
        current_score = kde._subset_log_likelihood_from_kernel(full_mask, log_kernel)  # type: ignore[attr-defined]

        feature_idcs = range(n_features) if col_idcs is None else col_idcs
        for feature_idx in feature_idcs:
            thresholds = self._generate_candidate_thresholds(X[:, feature_idx], n_thresholds)

            for i in range(1, len(thresholds)):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                threshold = 0.5 * (thresholds[i] + thresholds[i - 1])

                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices

                n_left = np.sum(left_indices)
                n_right = n_samples - n_left

                if (
                    n_left < min_samples_leaf
                    or n_right < min_samples_leaf
                    or n_left < min_child_weight
                    or n_right < min_child_weight
                ):
                    continue

                left_score = kde._subset_log_likelihood_from_kernel(left_indices, log_kernel)  # type: ignore[attr-defined]
                right_score = kde._subset_log_likelihood_from_kernel(right_indices, log_kernel)  # type: ignore[attr-defined]

                loss_reduction = current_score - (left_score + right_score)
                if loss_reduction > best_loss_reduction:
                    best_loss_reduction = loss_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_feature is None or best_threshold is None:
            return None, None, 0.0, None, None, None, None

        return (
            best_feature,
            best_threshold,
            best_loss_reduction,
            best_left_indices,  # type: ignore[return-value]
            best_right_indices,  # type: ignore[return-value]
            None,
            None,
        )

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predict the mean for each observation in X."""
        if self._is_leaf():
            return np.full(X.shape[0], self.distribution.get_posterior_mean(params=self.posterior_params))
        # Split indices based on the best feature/threshold
        left_mask = X[:, self.best_feature] <= self.best_threshold
        right_mask = ~left_mask

        # Allocate full-size output
        preds = np.empty(X.shape[0], dtype=float)

        # Compute predictions for left subset (or fall back to this node's mean)
        if self.left_node:
            preds[left_mask] = self.left_node.predict_mean(X[left_mask])
        else:
            preds[left_mask] = self.distribution.get_posterior_mean(params=self.posterior_params)

        # Compute predictions for right subset (or fall back to this node's mean)
        if self.right_node:
            preds[right_mask] = self.right_node.predict_mean(X[right_mask])
        else:
            preds[right_mask] = self.distribution.get_posterior_mean(params=self.posterior_params)

        return preds

    def predict_samples(self, X, size: int = 1) -> np.ndarray:
        """Draw samples for each observation in X."""
        if self._is_leaf():
            return self.distribution.sample_posterior(
                size=(X.shape[0], size), params=self.posterior_params, random_state=self.random_state
            )

        # Split indices based on the best feature/threshold
        left_mask = X[:, self.best_feature] <= self.best_threshold
        right_mask = ~left_mask

        # Allocate full-size output
        preds = np.empty((X.shape[0], size), dtype=float)

        # Compute samples for left subset (or fall back to this node's samples)
        if self.left_node:
            preds[left_mask] = self.left_node.predict_samples(X[left_mask], size=size)
        else:
            preds[left_mask] = self.distribution.sample_posterior(
                size=(np.sum(left_mask), size), params=self.posterior_params, random_state=self.random_state
            )

        # Compute samples for right subset (or fall back to this node's samples)
        if self.right_node:
            preds[right_mask] = self.right_node.predict_samples(X[right_mask], size=size)
        else:
            preds[right_mask] = self.distribution.sample_posterior(
                size=(np.sum(right_mask), size), params=self.posterior_params, random_state=self.random_state
            )

        return preds

    def _generate_candidate_thresholds(self, data: np.ndarray, n_thresholds: int) -> np.ndarray:
        """Generate candidate thresholds based on the data and eta value.

        Args
        -----
        `data` : np.ndarray
            1D array of feature values to generate thresholds from
        `n_thresholds` : int
            Number of candidate thresholds to generate

        Returns
        -------
        `thresholds` : np.ndarray
            1D array of candidate thresholds
        """
        if data.ndim != 1:
            raise ValueError("Data must be a 1D array of feature values.")

        return np.quantile(data, np.linspace(0, 1, n_thresholds), method="closest_observation")
