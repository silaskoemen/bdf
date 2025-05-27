import numpy as np
import warnings

from bdf.distributions import (
    normal,
    bdf_distribution,
)


class BDFNode:
    """ Base class for all BDF nodes.
    """
    def __init__(self, distribution: bdf_distribution.BDFDistribution, depth: int = 0):
        self.distribution = distribution
        self.depth = depth
        self.left_node, self.right_node = None, None
    
    def estimate_posterior(self, y: np.ndarray):
        self.posterior_mean, self.posterior_std = self.distribution.calc_posterior_params(y)

    def predict(self, X: np.ndarray, method: str = 'params') -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        assert isinstance(method, str) and method in ['params', 'sample'], "Method must be str 'params' or 'sample'"
        if self.left_node is None and self.right_node is None:
            # Leaf node
            assert hasattr(self, 'posterior_mean'), "Posterior mean not estimated. Call estimate_posterior() first."
            assert hasattr(self, 'posterior_std'), "Posterior std not estimated. Call estimate_posterior() first."
            if method == 'params':
                return self.posterior_mean, self.posterior_std
            elif method == 'sample':
                return self.distribution.sample_posterior(size=X.shape[0], params={'mean': self.posterior_mean, 'std': self.posterior_std})  # Sample from the posterior distribution
            else:
                raise ValueError("Method must be 'params' or 'sample'")
        else:
            # Non-leaf node
            assert self.left_node is not None and self.right_node is not None, "Invalid tree structure"
            
            left_mask = X[:, self.best_feature] <= self.best_threshold
            right = X[~left_mask]
            left = X[left_mask]
            
            if method == 'params':
                # For params method, we need to handle two return values
                means = np.zeros(X.shape[0], dtype=float)
                stds = np.zeros(X.shape[0], dtype=float)
                
                if left.shape[0] > 0:
                    left_means, left_stds = self.left_node.predict(left, method=method)
                    means[left_mask] = left_means
                    stds[left_mask] = left_stds
                    
                if right.shape[0] > 0:
                    right_means, right_stds = self.right_node.predict(right, method=method)
                    means[~left_mask] = right_means
                    stds[~left_mask] = right_stds
                    
                return means, stds
            else:  # method == 'sample'
                # For sample method, we handle a single return value
                preds = np.zeros(X.shape[0], dtype=float)
                
                if left.shape[0] > 0:
                    preds[left_mask] = self.left_node.predict(left, method=method)
                    
                if right.shape[0] > 0:
                    preds[~left_mask] = self.right_node.predict(right, method=method)
                    
                return preds
        
    def split_node(self, y, feat_idx: int, threshold: float, left_idx, right_idx):
        """Split the node into left and right children based on the best feature and threshold.
        
        Args:
            feat_idx: Index of the feature to split on
            threshold: Value to split on
            left_idx: Boolean indices for left child
            right_idx: Boolean indices for right child
        """
        self.best_feature = feat_idx
        self.best_threshold = threshold
        
        # Create left and right nodes
        self.left_node = BDFNode(distribution=self.distribution, depth=self.depth + 1)
        self.right_node = BDFNode(distribution=self.distribution, depth=self.depth + 1)
        
        # Estimate posterior for left and right nodes
        self.left_node.estimate_posterior(y[left_idx])
        self.right_node.estimate_posterior(y[right_idx])
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray,
                        min_samples_leaf: int, min_child_weight: float,
                        col_idcs: list | np.ndarray | None = None, eta = 0.025) -> tuple[int, float, float, np.ndarray, np.ndarray] | tuple[None, None, float, None, None]:
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
            import bdf_rust
            from bdf.distributions.distribution_manager import DistributionManager as DM
            
            # Create distribution spec with native and fallback options
            dist_spec = DM.to_rust_spec(self.distribution)
            
            # Always include the Python object as fallback
            dist_spec["_python_object"] = self.distribution
            
            # Convert column indices if provided
            if col_idcs is not None:
                col_idcs = np.array(col_idcs, dtype=np.int64)
                
            # Call the unified Rust implementation
            feature_idx, threshold, loss_reduction, left_indices, right_indices = (
                bdf_rust.find_best_split(  # type: ignore
                    X, y, min_samples_leaf, min_child_weight, 
                    dist_spec, eta, col_idcs
                )
            )
        
            return feature_idx, threshold, loss_reduction, left_indices, right_indices
    
        except (ImportError, Exception) as e:
            warnings.warn(f"Rust implementation not available or failed: {e}. Falling back to Python implementation.")
            return self._find_best_split_python(X, y, min_samples_leaf, min_child_weight, col_idcs, eta)
    
    def _find_best_split_python(self, X: np.ndarray, y: np.ndarray,
                        min_samples_leaf: int, min_child_weight: float,
                        col_idcs: list | np.ndarray | None = None, eta = 0.025) -> tuple[int, float, float, np.ndarray, np.ndarray] | tuple[None, None, float, None, None]:
        n_samples, n_features = X.shape
        best_feature: int | None = None
        best_threshold: float | None = None
        best_loss_reduction = 0.
        best_left_indices: np.ndarray | None = None
        best_right_indices: np.ndarray | None = None
        n_thresholds = int(np.ceil(1/eta))  # Number of thresholds to consider per feature
        
        # Current node NLL
        current_nll = self.distribution.nll(y)
        
        # Try each feature
        feature_idcs = range(n_features) if col_idcs is None else col_idcs
        for feature_idx in feature_idcs:
            # thresholds = np.quantile(X[:, feature_idx], np.linspace(0, 1, n_thresholds+2)[1:-1], method='closest_observation')
            thresholds = self._generate_candidate_thresholds(X[:, feature_idx], n_thresholds)
            
            # Ensures constant values will return None, always value in between taken as threshold
            for i in range(1, len(thresholds)):
                if thresholds[i] == thresholds[i-1]:
                    continue
                threshold = (thresholds[i] + thresholds[i-1]) / 2
                # Split data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Check min_samples_leaf constraint
                if np.sum(left_indices) < min_samples_leaf or np.sum(right_indices) < min_samples_leaf \
                    or np.sum(left_indices) < min_child_weight or np.sum(right_indices) < min_child_weight:
                    continue
                
                left_nll = self.distribution.nll(y[left_indices])
                right_nll = self.distribution.nll(y[right_indices])
                
                # Calculate loss reduction (improvement)
                loss_reduction = current_nll - (left_nll + right_nll)
                
                if loss_reduction > best_loss_reduction:
                    best_loss_reduction = loss_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices
        
        if best_feature is None or best_threshold is None or best_left_indices is None or best_right_indices is None:
            return None, None, 0, None, None
            
        return best_feature, best_threshold, best_loss_reduction, best_left_indices, best_right_indices
    
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
        
        return np.quantile(data, np.linspace(0, 1, n_thresholds), method='closest_observation')
