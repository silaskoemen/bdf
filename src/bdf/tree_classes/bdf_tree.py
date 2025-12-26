import numpy as np

from bdf.distributions.bdf_distribution import BDFDistribution
from bdf.tree_classes.bdf_node import BDFNode


class BDFTree:
    """Base class for all BDF trees."""

    def __init__(
        self,
        distribution: BDFDistribution,
        reg_lambda: float,
        reg_gamma: float,
        reg_nu: float,
        max_depth: int,
        min_samples_leaf: int,
        penalty: float,
        min_samples_split: int,
        min_child_weight: float | int,
        random_state: int,
    ):
        """Initialize the BDFTree with distribution and regularization parameters.

        Args
        ----
        `distribution` : BDFDistribution
            The distribution to use for the tree.
        `reg_gamma` : float
            Regularization parameter for the tree.
        `reg_lambda` : float
            Additional regularization parameter
        `max_depth` : int
            Maximum depth of the tree
        `min_samples_leaf` : int
            Minimum number of samples required to be at a leaf node
        `min_samples_split` : int
            Minimum number of samples required to split an internal node
        `min_child_weight` : int or float
            Minimum sum of instance weight (hessian) needed in a child
        """
        self.distribution = distribution
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.reg_nu = reg_nu
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.penalty = penalty

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        colsample: float = 1.0,
        verbose: int = 0,
        eta: float = 0.025,
    ) -> "BDFTree":
        # NOTE: if reg_lambda is 0, each loss component is fully seperable, meaning each
        # node can be split simply by considering NLL reduction and including reg_gamma; no need for a queue.
        # TODO: Implement separate fit functions given reg_lambda == 0 and reg_lambda > 0.
        # Create root node
        self.root = BDFNode(distribution=self.distribution, depth=0, random_state=self.random_state)
        self.root.estimate_posterior(y)

        # Only try splitting if we have enough samples
        if len(y) >= self.min_samples_split:
            self._grow_node(
                node=self.root,
                X=X,
                y=y,
                penalty=self.penalty,
                reg_nu=self.reg_nu,
                colsample=colsample,
                rng=rng,
                eta=eta,
            )
        return self

    def _grow_node(
        self,
        node: BDFNode,
        X: np.ndarray,
        y: np.ndarray,
        penalty: float,
        reg_nu: float,
        colsample: float,
        rng: np.random.Generator,
        eta: float = 0.025,
    ) -> None:
        """Recursively grow the tree from the given node, using the linear (in |T|) penalty, meaning
        fixed threshold"""
        if node.depth >= self.max_depth:
            return
        n_features_iter = np.ceil(X.shape[1] * colsample).astype(int)
        col_idcs = rng.choice(X.shape[1], n_features_iter, replace=False) if n_features_iter < X.shape[1] else None
        feature_idx, threshold, loss_reduction, left_indices, right_indices, left_params, right_params = (
            node.find_best_split(
                X, y, self.min_samples_leaf, self.min_child_weight, col_idcs=col_idcs, eta=eta, reg_gamma=self.reg_gamma
            )
        )
        if reg_nu > 0.0:
            # Add nu * d penalty for depth d
            depth_penalty = penalty + self.reg_nu * node.depth
        else:
            depth_penalty = penalty

        # Check if valid split found (feature_idx is not None) AND gain > penalty
        if feature_idx is not None and loss_reduction > depth_penalty:
            # Safety check for indices and params (though Rust logic implies they exist if feature_idx exists)
            if left_indices is None or right_indices is None or threshold is None:
                return

            node.split_node(
                y=y,
                feat_idx=feature_idx,
                threshold=threshold,
                left_idx=left_indices,
                right_idx=right_indices,
                left_params=left_params,
                right_params=right_params,
            )

            # Recursively grow left and right child nodes
            if node.left_node is None or node.right_node is None:
                raise ValueError("Node split failed to create child nodes.")
            self._grow_node(
                node.left_node,
                X=X[left_indices],
                y=y[left_indices],
                penalty=penalty,
                reg_nu=reg_nu,
                colsample=colsample,
                rng=rng,
                eta=eta,
            )
            self._grow_node(
                node=node.right_node,
                X=X[right_indices],
                y=y[right_indices],
                penalty=penalty,
                reg_nu=reg_nu,
                colsample=colsample,
                rng=rng,
                eta=eta,
            )

    def _find_leaf_node(self, x: np.ndarray) -> BDFNode:
        """Traverse the tree to find the leaf node for a single observation."""
        node = self.root
        while not node._is_leaf():
            if node.left_node is None or node.right_node is None:
                # This should not happen in a fitted tree, but as a safeguard:
                break
            if x[node.best_feature] <= node.best_threshold:
                node = node.left_node
            else:
                node = node.right_node
        return node

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predict the mean for each observation in X."""
        return self.root.predict_mean(X)
        return np.array([self._find_leaf_node(x).predict_mean() for x in X])

    def predict_variance(self, X: np.ndarray) -> np.ndarray:
        """Predict the variance for each observation in X."""
        return np.array([self._find_leaf_node(x).predict_variance() for x in X])

    def predict_params(self, X: np.ndarray) -> np.ndarray:
        """Predict the parameters for each observation in X."""
        return np.array([self._find_leaf_node(x).predict_params() for x in X], dtype=object)

    def predict_samples(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw samples for each observation in X."""
        return self.root.predict_samples(X, n_samples)
        return np.array([self._find_leaf_node(x).predict_samples(n_samples=n_samples) for x in X])
