import warnings
from typing import Literal

import numpy as np

from bdf.distributions.bdf_distribution import BDFDistribution

# from bdf.tree_classes.utils import _logsumexp


class BDFNode:
    """Base class for all BDF nodes."""

    __slots__ = (
        "distribution",
        "depth",
        "random_state",
        "left_node",
        "right_node",
        "posterior_params",
        "best_feature",
        "best_threshold",
    )

    def __init__(self, distribution: BDFDistribution, depth: int, random_state: int):
        self.distribution = distribution
        self.depth = depth
        self.random_state = random_state
        self.left_node, self.right_node = None, None

    def estimate_posterior(self, y: np.ndarray, params: dict | None = None):
        if params is not None:
            self.posterior_params: dict = params
        else:
            self.posterior_params: dict = self.distribution.calc_posterior_params(y)

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
        self.left_node = BDFNode(distribution=self.distribution, depth=self.depth + 1, random_state=self.random_state)
        self.right_node = BDFNode(distribution=self.distribution, depth=self.depth + 1, random_state=self.random_state)

        # Estimate posterior for left and right nodes
        self.left_node.estimate_posterior(y[left_idx], params=left_params)
        self.right_node.estimate_posterior(y[right_idx], params=right_params)

    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_leaf: int,
        min_child_weight: float,
        col_idcs: list | np.ndarray | None = None,
        gamma: float = 0.0,
        eta=0.025,
        split_gain_method: Literal["evidence", "map"] = "map",
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
            from bdf import _bdf_rs as bdf_rs

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
                X, y, min_samples_leaf, min_child_weight, dist_spec, eta, gamma, col_idcs, split_gain_method
            )

            return feature_idx, threshold, loss_reduction, left_indices, right_indices, left_params, right_params

        except (ImportError, Exception) as e:
            warnings.warn(f"Rust implementation not available or failed: {e}. Falling back to Python implementation.")
            return self._find_best_split_python(
                X,
                y,
                min_samples_leaf,
                min_child_weight,
                col_idcs,
                eta=eta,
                gamma=gamma,
                split_gain_method=split_gain_method,
            )

    def _find_best_split_python(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_leaf: int,
        min_child_weight: float,
        col_idcs: list | np.ndarray | None = None,
        gamma: float = 1.0,
        eta: float = 0.025,
        split_gain_method: Literal["evidence", "map"] = "evidence",
    ) -> (
        tuple[int, float, float, np.ndarray, np.ndarray, None, None] | tuple[None, None, float, None, None, None, None]
    ):
        """
        Unified, streaming implementation that computes both MAP and Evidence style gains
        in a single pass over features and cutpoints. Returns the same tuple shape as
        existing `_find_best_split_python_map` / `_find_best_split_python_evidence`.

        Notes:
        - Uses streaming log-sum-exp for numerical stability (no per-cut arrays).
        - Uses `tried_thresholds` as multiplicity `m_j` by default (change to `valid_splits`
        if you prefer counting only valid cuts).
        - If this node holds a KDE distribution, it delegates to the specialized KDE path.
        """
        # fast-path for KDE distributions (preserves existing optimized behavior)
        if self._is_kde_distribution():
            return self._find_best_split_python_kde(
                X, y, min_samples_leaf, min_child_weight, col_idcs, gamma=gamma, eta=eta
            )

        n_samples, n_features = X.shape
        current_score = self.distribution.score(y)

        feature_idcs = list(range(n_features)) if col_idcs is None else list(col_idcs)
        k = len(feature_idcs)
        if k == 0:
            return None, None, 0.0, None, None, None, None

        stride = max(int(n_samples * eta), 1)

        # Node-level streaming LSE for evidence (over feature scores g_j)
        node_M = -np.inf
        node_S = 0.0

        # Track best choices for both criteria
        best_feature_map = None
        best_threshold_map = None
        best_left_map = None
        best_right_map = None
        best_map_score = -np.inf  # penalized MAP score (loss_reduction - penalty)

        best_feature_evidence = None
        best_threshold_evidence = None
        best_left_evidence = None
        best_right_evidence = None
        best_feature_log_score = -np.inf  # highest g_j (unpenalized evidence feature score)

        # Loop features: stride-based O(N) scan matching Rust semantics
        for feature_idx in feature_idcs:
            col = X[:, feature_idx]
            sorted_indices = np.argsort(col, kind="mergesort")

            # Per-feature streaming LSE for deltas (for evidence)
            feat_M = -np.inf
            feat_S = 0.0

            # Track MAP-best (max delta) for this feature
            feat_best_delta = -np.inf
            feat_best_threshold = None
            feat_best_left = None
            feat_best_right = None

            num_thresholds_tried = 0

            # Walk sorted data at stride intervals (matching Rust fast path)
            for i in range(n_samples - 1):
                if (i + 1) % stride != 0:
                    continue

                feat_val = col[sorted_indices[i]]
                next_feat_val = col[sorted_indices[i + 1]]

                if feat_val >= next_feat_val:
                    continue

                left_n = i + 1
                right_n = n_samples - left_n

                if left_n < min_samples_leaf or right_n < min_samples_leaf:
                    continue

                num_thresholds_tried += 1
                threshold = 0.5 * (feat_val + next_feat_val)

                left_mask = np.zeros(n_samples, dtype=bool)
                left_mask[sorted_indices[: i + 1]] = True
                right_mask = ~left_mask

                # Single delta computation per cut
                left_score = self.distribution.score(y[left_mask])
                right_score = self.distribution.score(y[right_mask])
                delta = current_score - (left_score + right_score)

                # MAP bookkeeping
                if delta > feat_best_delta:
                    feat_best_delta = delta
                    feat_best_threshold = threshold
                    feat_best_left = left_mask
                    feat_best_right = right_mask

                # Streaming log-sum-exp update for this feature
                if feat_M == -np.inf:
                    feat_M = delta
                    feat_S = 1.0
                elif delta > feat_M:
                    feat_S = feat_S * np.exp(feat_M - delta) + 1.0
                    feat_M = delta
                else:
                    feat_S += np.exp(delta - feat_M)

            # Skip feature if no valid splits
            if num_thresholds_tried == 0:
                continue

            # Per-feature log-sum-exp (Σ_c exp(delta_{j,c}))
            log_sum_exp = feat_M + np.log(feat_S)

            # multiplicity m_j: valid threshold count (matching Rust semantics)
            m_j = num_thresholds_tried

            # Compute g_j (optionally tempered by gamma)
            if gamma != 1.0:
                g_j = log_sum_exp - gamma * np.log(m_j)
            else:
                g_j = log_sum_exp - np.log(m_j)

            # Update node-level streaming LSE over features (for evidence)
            if node_M == -np.inf:
                node_M = g_j
                node_S = 1.0
            elif g_j > node_M:
                node_S = node_S * np.exp(node_M - g_j) + 1.0
                node_M = g_j
            else:
                node_S += np.exp(g_j - node_M)

            # Track best feature by evidence (g_j)
            if g_j > best_feature_log_score and feat_best_threshold is not None:
                best_feature_log_score = g_j
                best_feature_evidence = feature_idx
                best_threshold_evidence = feat_best_threshold
                best_left_evidence = feat_best_left
                best_right_evidence = feat_best_right

            # Compute MAP penalized score for this feature:
            # base = feat_best_delta, penalty = gamma*(ln(k) + ln(m_j)) if gamma>0
            feat_map_score = feat_best_delta
            if gamma > 0.0:
                feat_map_score -= gamma * (np.log(k) + np.log(m_j))

            # Track best feature by MAP
            if feat_map_score > best_map_score and feat_best_threshold is not None:
                best_map_score = feat_map_score
                best_feature_map = feature_idx
                best_threshold_map = feat_best_threshold
                best_left_map = feat_best_left
                best_right_map = feat_best_right

        # No valid split found
        if best_feature_map is None and best_feature_evidence is None:
            return None, None, 0.0, None, None, None, None

        # Return according to requested method
        if split_gain_method == "map":
            if (
                best_feature_map is None
                or best_threshold_map is None
                or best_left_map is None
                or best_right_map is None
            ):
                return None, None, 0.0, None, None, None, None
            # best_map_score is already penalized loss-reduction
            return (
                best_feature_map,
                best_threshold_map,
                float(best_map_score),
                best_left_map,
                best_right_map,
                None,
                None,
            )

        # evidence branch
        # Compute node-level integrated evidence G = log( (1/k) * sum_j exp(g_j) )
        G = node_M + np.log(node_S)
        # Apply feature-prior multiplicity tempering (same semantics as original code)
        if gamma != 1.0:
            G -= gamma * np.log(k)
        else:
            G -= np.log(k)

        if (
            best_feature_evidence is None
            or best_threshold_evidence is None
            or best_left_evidence is None
            or best_right_evidence is None
        ):
            return None, None, 0.0, None, None, None, None

        return (
            best_feature_evidence,
            best_threshold_evidence,
            float(G),
            best_left_evidence,
            best_right_evidence,
            None,
            None,
        )

    def _is_kde_distribution(self) -> bool:
        """Return True if this node's distribution is a KDE-like distribution."""
        from bdf.distributions.kde import KDE, PseudoHKDE  # local import to avoid cycles

        return isinstance(self.distribution, (KDE, PseudoHKDE))

    def _find_best_split_python_kde(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_leaf: int,
        min_child_weight: float,
        col_idcs: list | np.ndarray | None = None,
        gamma: float = 0.0,
        eta=0.025,
    ) -> (
        tuple[int, float, float, np.ndarray, np.ndarray, None, None] | tuple[None, None, float, None, None, None, None]
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
        k = sum(feature_idcs)
        for feature_idx in feature_idcs:
            thresholds = self._generate_candidate_thresholds(X[:, feature_idx], n_thresholds)
            (
                feat_best_loss_reduction,
                feat_best_feature,
                feat_best_threshold,
                feat_best_left_indices,
                feat_best_right_indices,
            ) = (0.0, None, None, None, None)
            tried_thresholds = 0
            for i in range(1, len(thresholds)):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                tried_thresholds += 1
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
                if loss_reduction > feat_best_loss_reduction:
                    feat_best_loss_reduction = loss_reduction
                    feat_best_feature = feature_idx
                    feat_best_threshold = threshold
                    feat_best_left_indices = left_indices
                    feat_best_right_indices = right_indices
            if gamma > 0.0:
                # Apply complexity penalty of gamma*(ln(k) + ln(m_j))
                feat_best_loss_reduction -= gamma * (np.log(k) + np.log(tried_thresholds))
            if feat_best_loss_reduction > best_loss_reduction:
                best_loss_reduction = feat_best_loss_reduction
                best_feature = feat_best_feature
                best_threshold = feat_best_threshold
                best_left_indices = feat_best_left_indices
                best_right_indices = feat_best_right_indices

        if best_feature is None or best_threshold is None or best_left_indices is None or best_right_indices is None:
            return None, None, 0.0, None, None, None, None

        return (
            best_feature,
            best_threshold,
            best_loss_reduction,
            best_left_indices,
            best_right_indices,
            None,
            None,
        )

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predict the mean for each observation in X."""
        if self._is_leaf():
            return np.full(X.shape[0], self.distribution.get_posterior_mean(params=self.posterior_params))
        left_mask = X[:, self.best_feature] <= self.best_threshold
        right_mask = ~left_mask

        assert self.left_node is not None and self.right_node is not None  # guaranteed by split_node

        preds = np.empty(X.shape[0], dtype=float)
        preds[left_mask] = self.left_node.predict_mean(X[left_mask])
        preds[right_mask] = self.right_node.predict_mean(X[right_mask])
        return preds

    def predict_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute log-likelihood of y under the leaf posterior for each observation."""
        if self._is_leaf():
            return self.distribution.log_likelihood(y, self.posterior_params)

        assert self.left_node is not None and self.right_node is not None  # guaranteed by split_node

        left_mask = X[:, self.best_feature] <= self.best_threshold
        right_mask = ~left_mask

        result = np.empty(X.shape[0], dtype=float)
        result[left_mask] = self.left_node.predict_log_likelihood(X[left_mask], y[left_mask])
        result[right_mask] = self.right_node.predict_log_likelihood(X[right_mask], y[right_mask])
        return result

    def predict_samples(self, X, size: int = 1) -> np.ndarray:
        """Draw samples for each observation in X.

        Returns array of shape (n_obs, size) where each row contains `size` i.i.d. samples.
        """
        n_obs = X.shape[0]

        if self._is_leaf():
            # Sample flat and reshape - distribution returns 1D array
            flat_samples = self.distribution.sample_posterior(
                size=n_obs * size, params=self.posterior_params, random_state=self.random_state
            )
            return flat_samples.reshape(n_obs, size)

        assert self.left_node is not None and self.right_node is not None  # guaranteed by split_node

        left_mask = X[:, self.best_feature] <= self.best_threshold
        right_mask = ~left_mask

        preds = np.empty((n_obs, size), dtype=float)
        preds[left_mask] = self.left_node.predict_samples(X[left_mask], size=size)
        preds[right_mask] = self.right_node.predict_samples(X[right_mask], size=size)

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
            1D array of unique candidate thresholds (deduplicated).
            Returns empty array for constant features.
        """
        if data.ndim != 1:
            raise ValueError("Data must be a 1D array of feature values.")

        # Get unique sorted values to avoid duplicates from repeated data
        unique_vals = np.unique(data)

        # Constant feature → no valid split thresholds
        if len(unique_vals) <= 1:
            return np.array([])

        # Generate quantiles from unique values
        quantiles = np.linspace(0, 1, n_thresholds)
        thresholds = np.quantile(unique_vals, quantiles, method="closest_observation")

        # Deduplicate (quantile can still return duplicates at distribution edges)
        return np.unique(thresholds)

    def count_nodes(self) -> int:
        """Count the total number of nodes in the subtree rooted at this node."""
        count = 1  # Count this node
        if self.left_node is not None:
            count += self.left_node.count_nodes()
        if self.right_node is not None:
            count += self.right_node.count_nodes()
        return count

    def get_max_depth(self) -> int:
        """Get the maximum depth of the subtree rooted at this node."""
        if self._is_leaf():
            return self.depth
        left_depth = self.left_node.get_max_depth() if self.left_node is not None else self.depth
        right_depth = self.right_node.get_max_depth() if self.right_node is not None else self.depth
        return max(left_depth, right_depth)


"""
import numpy as np

def _logsumexp(a: np.ndarray) -> float:
    #Stable logsumexp; returns -inf for empty arrays.
    if a.size == 0:
        return -np.inf
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))

def _find_best_split_python(
    self,
    X: np.ndarray,
    y: np.ndarray,
    min_samples_leaf: int,
    min_child_weight: float,
    col_idcs: list | np.ndarray | None = None,
    # gamma kept for backwards compatibility; see note below
    gamma: float = 1.0,
    eta: float = 0.025,
):
    n_samples, n_features = X.shape

    # Current node negative log evidence (or NLL+BIC)
    current_score = self.distribution.score(y)

    feature_idcs = list(range(n_features)) if col_idcs is None else list(col_idcs)
    k = len(feature_idcs)  # <-- FIX (do not use sum)

    # Track: Bayesian feature scores g_j = log(1/m_j * sum exp(delta))
    feature_log_scores: list[float] = []

    # Track the actual split we will apply (you can choose MAP or BMA-based)
    best_feature = None
    best_threshold = None
    best_left_indices = None
    best_right_indices = None

    # Choose feature by integrated score (posterior mass), not by best cut.
    best_feature_log_score = -np.inf
    best_feature_best_delta = -np.inf  # optional, if you still want it

    n_thresholds = int(np.ceil(1 / eta))

    for feature_idx in feature_idcs:
        thresholds = self._generate_candidate_thresholds(X[:, feature_idx], n_thresholds)

        tried_thresholds = 0
        deltas = []  # store delta_{j,c} = loss_reduction for valid splits

        # For producing an actual split rule to execute, keep the best cut too
        feat_best_delta = -np.inf
        feat_best_threshold = None
        feat_best_left_indices = None
        feat_best_right_indices = None

        for i in range(1, len(thresholds)):
            if thresholds[i] == thresholds[i - 1]:
                continue
            tried_thresholds += 1  # depends whether invalid thru min samples should be counted or not!
            threshold = (thresholds[i] + thresholds[i - 1]) / 2.0

            left_indices = X[:, feature_idx] <= threshold
            right_indices = ~left_indices

            nL = int(np.sum(left_indices))
            nR = int(np.sum(right_indices))
            if (
                nL < min_samples_leaf
                or nR < min_samples_leaf
                or nL < min_child_weight
                or nR < min_child_weight
            ):
                continue

            left_score = self.distribution.score(y[left_indices])
            right_score = self.distribution.score(y[right_indices])

            delta = current_score - (left_score + right_score)
            deltas.append(delta)

            if delta > feat_best_delta:
                feat_best_delta = delta
                feat_best_threshold = threshold
                feat_best_left_indices = left_indices
                feat_best_right_indices = right_indices

        if tried_thresholds == 0:
            continue

        # If no valid splits, contribution is effectively zero mass -> log score -inf
        deltas_arr = np.array(deltas, dtype=float)

        # --- Exact Bayesian (uniform prior) uses gamma = 1.0 ---
        # g_j = log( (1/m_j) * sum_c exp(delta_{j,c}) )
        # If you want m_j to be "all proposed cutpoints", use tried_thresholds.
        # If you want m_j to be only valid cutpoints, use len(deltas_arr).
        m_j = tried_thresholds

        log_sum_exp = _logsumexp(deltas_arr)  # log Σ exp(delta)
        g_j = log_sum_exp - np.log(m_j)       # log(1/m_j Σ exp(delta))

        # Optional "tempering": scale the multiplicity charge only (not fully Bayes unless γ=1).
        # If you want exact Bayes, keep gamma=1 and REMOVE the next line.
        if gamma != 1.0:
            g_j = log_sum_exp - gamma * np.log(m_j)

        feature_log_scores.append(g_j)

        # Pick which feature to actually split on:
        # Bayesian mass choice: maximize g_j (equivalently posterior feature probability)
        if g_j > best_feature_log_score and feat_best_threshold is not None:
            best_feature_log_score = g_j
            best_feature_best_delta = feat_best_delta
            best_feature = feature_idx
            best_threshold = feat_best_threshold
            best_left_indices = feat_best_left_indices
            best_right_indices = feat_best_right_indices

    if best_feature is None:
        return None, None, 0.0, None, None, None, None

    # Full node-level integrated log BF over features too:
    # G = log( (1/k) * sum_j exp(g_j) )
    G = _logsumexp(np.array(feature_log_scores, dtype=float)) - np.log(k)

    # Optional tempering of feature prior mass (again, exact Bayes is gamma=1)
    if gamma != 1.0:
        G = _logsumexp(np.array(feature_log_scores, dtype=float)) - gamma * np.log(k)

    return best_feature, best_threshold, float(G), best_left_indices, best_right_indices, None, None
"""
