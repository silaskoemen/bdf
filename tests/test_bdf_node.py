import pytest
import numpy as np

from bdf.tree_classes.bdf_node import BDFNode
from bdf.distributions.distribution_manager import DistributionManager as DM



class TestFindBestSplit:
    def test_all_features(self):
        # Create a toy dataset with 4 observations and 3 features
        # Only feature 1 should vary with the output
        X = np.array([
            [5., 0, 5, 0],
            [2, 0, 6, 5],
            [3, 1, 7, -1],
            [4, 1, 1, 2],
        ], dtype=float)
        print(X)
        y = np.array([1, 2, 11, 12])
        
        dist = DM.create_distribution('normal', prior_params={'mean': 6.5, 'std': 20})
        # Create a BDFNode with this data
        node = BDFNode(distribution=dist, depth=1)
        
        # Find the best split
        feat, thresh, loss, left_idcs, right_idcs = node.find_best_split(
            X, y, col_idcs=None, eta=0.025, min_child_weight=0.0, min_samples_leaf=1
        )
        assert feat == 1
        assert thresh == 0.5
        assert loss > 0

    def test_subset_features(self):
        # Create a toy dataset with 4 observations and 3 features
        # Only feature 1 should vary with the output
        X = np.array([
            [2, 0, 1, 0],
            [1, 0, 2, 5],
            [4, 1, 7, 6],
            [0, 1, 10, 2],
        ], dtype=float)
        y = np.array([1, 2, 11, 12])
        
        dist = DM.create_distribution('normal', prior_params={'mean': 6.5, 'std': 20})
        # Create a BDFNode with this data
        node = BDFNode(distribution=dist, depth=1)
        
        # Find the best split using only feature indices [0, 2]
        feat, thresh, loss, left_idcs, right_idcs = node.find_best_split(
            X, y, col_idcs=[0, 2], eta=0.025, min_child_weight=0.0, min_samples_leaf=2
        )
        assert feat == 2
        assert thresh == 4.5
        assert loss > 0

    def test_constant_values(self):
        # Create a toy dataset with constant values
        X = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=float)
        y = np.array([0, 0, 1, 1])  
        dist = DM.create_distribution('normal', prior_params={'mean': 8, 'std': 1})
        # Create a BDFNode with this data
        node = BDFNode(distribution=dist, depth=1)
        # Find the best split
        feat, thresh, loss, left_idcs, right_idcs = node.find_best_split(
            X, y, col_idcs=None, eta=0.025, min_child_weight=0.0, min_samples_leaf=1
        )
        assert feat is None
        assert thresh is None
        assert loss == 0

    def test_min_samples_leaf(self):
        pass

    def test_min_child_weight(self):
        pass

    def test_no_valid_splits(self):
        pass


class TestSplitNode():
    def test_saving_feat_thresh(self):
        pass

    def test_child_nodes_created(self):
        pass

