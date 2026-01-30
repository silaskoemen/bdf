"""Tests for threshold generation behavior.

These tests verify the behavior of threshold generation, not implementation details.
The key invariants are:
1. Thresholds enable finding valid splits
2. Constant features produce no valid thresholds
3. Duplicates are removed
4. eta controls the granularity of candidate thresholds
"""

import numpy as np
import pytest

from bdf.distributions.distribution_manager import DistributionManager as DM
from bdf.tree_classes.bdf_node import BDFNode


@pytest.fixture
def create_test_node():
    """Factory to create a test node with a simple distribution."""

    def _create(y=None):
        if y is None:
            y = np.ones(10)
        dist = DM.create_distribution("NormalMuNormal", {"mu_mu": 0, "sigma_mu": 1}, y=y)
        return BDFNode(distribution=dist, depth=0, random_state=1234)

    return _create


class TestThresholdGeneration:
    """Tests for the _generate_candidate_thresholds method."""

    def test_threshold_count_respects_n_thresholds(self, create_test_node):
        """Verify n_thresholds controls the maximum number of candidates."""
        data = np.linspace(0, 10, 101)  # 101 unique values
        node = create_test_node()

        thresholds_10 = node._generate_candidate_thresholds(data, n_thresholds=10)
        thresholds_5 = node._generate_candidate_thresholds(data, n_thresholds=5)
        thresholds_20 = node._generate_candidate_thresholds(data, n_thresholds=20)

        # Should return at most n_thresholds unique values
        assert len(thresholds_10) <= 10
        assert len(thresholds_5) <= 5
        assert len(thresholds_20) <= 20

        # With enough unique data, should get close to requested count
        assert len(thresholds_10) >= 8
        assert len(thresholds_20) >= 15

    def test_thresholds_within_data_range(self, create_test_node):
        """Thresholds should be within the data range."""
        data = np.linspace(0, 10, 101)
        node = create_test_node()

        thresholds = node._generate_candidate_thresholds(data, n_thresholds=10)

        assert len(thresholds) > 0
        assert thresholds.min() >= data.min()
        assert thresholds.max() <= data.max()

    def test_thresholds_are_unique(self, create_test_node):
        """All returned thresholds should be unique."""
        data = np.linspace(0, 10, 101)
        node = create_test_node()

        thresholds = node._generate_candidate_thresholds(data, n_thresholds=50)

        assert len(thresholds) == len(np.unique(thresholds))

    def test_constant_feature_returns_empty(self, create_test_node):
        """Constant features should produce no valid thresholds."""
        data = np.full(100, 5.0)
        node = create_test_node()

        thresholds = node._generate_candidate_thresholds(data, n_thresholds=10)

        assert len(thresholds) == 0

    def test_single_value_returns_empty(self, create_test_node):
        """Single value should produce no valid thresholds."""
        data = np.array([5.0])
        node = create_test_node()

        thresholds = node._generate_candidate_thresholds(data, n_thresholds=10)

        assert len(thresholds) == 0

    def test_duplicates_in_data_handled(self, create_test_node):
        """Data with many duplicates should still produce unique thresholds."""
        # Data with repeated values
        data = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0])
        node = create_test_node()

        thresholds = node._generate_candidate_thresholds(data, n_thresholds=20)

        # Should be deduplicated - at most 5 unique values in data
        assert len(thresholds) <= 5
        assert len(thresholds) == len(np.unique(thresholds))

        # All thresholds should be actual data values
        for t in thresholds:
            assert t in data

    def test_two_unique_values(self, create_test_node):
        """Two unique values should produce exactly two thresholds."""
        data = np.array([0.0, 0.0, 0.0, 10.0, 10.0])
        node = create_test_node()

        thresholds = node._generate_candidate_thresholds(data, n_thresholds=10)

        # Should get both unique values
        assert len(thresholds) == 2
        assert 0.0 in thresholds
        assert 10.0 in thresholds

    def test_thresholds_sorted(self, create_test_node):
        """Thresholds should be returned in sorted order."""
        data = np.random.RandomState(42).randn(100)
        node = create_test_node()

        thresholds = node._generate_candidate_thresholds(data, n_thresholds=20)

        assert np.all(np.diff(thresholds) >= 0)

    def test_invalid_input_raises(self, create_test_node):
        """Non-1D input should raise ValueError."""
        data_2d = np.array([[1, 2], [3, 4]])
        node = create_test_node()

        with pytest.raises(ValueError, match="1D array"):
            node._generate_candidate_thresholds(data_2d, n_thresholds=10)


class TestSplitFindingBehavior:
    """Integration tests verifying split finding works correctly with thresholds."""

    def test_clear_split_is_found(self, create_test_node):
        """A clear split in the data should be found."""
        np.random.seed(42)  # Deterministic

        # Create data with obvious split at x=5
        # Second feature is constant to ensure we split on first
        X = np.column_stack([np.linspace(0, 10, 100), np.zeros(100)])
        y = np.where(X[:, 0] < 5, 0.0, 10.0)

        node = create_test_node(y=y)
        node.estimate_posterior(y)

        feat, thresh, gain, left, right, *_ = node.find_best_split(
            X, y, min_samples_leaf=5, min_child_weight=0.0, eta=0.1
        )

        assert feat == 0  # Should split on first feature
        assert 4.0 < thresh < 6.0, f"Threshold {thresh} should be near 5"
        assert gain > 0

    def test_constant_feature_produces_no_split(self, create_test_node):
        """A constant feature should not produce a valid split."""
        # Single constant feature
        X = np.full((100, 1), 5.0)
        y = np.random.randn(100)

        node = create_test_node(y=y)
        node.estimate_posterior(y)

        feat, thresh, gain, left, right, *_ = node.find_best_split(
            X, y, min_samples_leaf=5, min_child_weight=0.0, eta=0.1
        )

        # Should not find a valid split
        assert feat is None
        assert thresh is None

    def test_noisy_feature_vs_informative_feature(self, create_test_node):
        """Informative feature should be preferred over noisy feature."""
        np.random.seed(42)

        # Feature 0: informative (y depends on it)
        # Feature 1: pure noise
        x0 = np.linspace(0, 10, 100)
        x1 = np.random.randn(100)
        X = np.column_stack([x0, x1])
        y = np.where(x0 < 5, 0.0, 10.0) + np.random.randn(100) * 0.1

        node = create_test_node(y=y)
        node.estimate_posterior(y)

        feat, thresh, gain, *_ = node.find_best_split(X, y, min_samples_leaf=5, min_child_weight=0.0, eta=0.1)

        # Should split on informative feature
        assert feat == 0
        assert gain > 0

    def test_min_samples_leaf_respected(self, create_test_node):
        """min_samples_leaf constraint should prevent invalid splits."""
        X = np.linspace(0, 10, 20).reshape(-1, 1)
        y = np.where(X[:, 0] < 5, 0.0, 10.0)

        node = create_test_node(y=y)
        node.estimate_posterior(y)

        # With min_samples_leaf=15, no valid split exists (20 samples total)
        feat, thresh, gain, *_ = node.find_best_split(X, y, min_samples_leaf=15, min_child_weight=0.0, eta=0.1)

        assert feat is None
        assert thresh is None
