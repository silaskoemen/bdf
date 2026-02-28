"""Tests for benchmark statistical tests, particularly Holm-adjusted p-values."""

import numpy as np
import pytest

from benchmarks.utils.statistical_tests import (
    holm_adjusted_p_values,
    holm_correction,
)


class TestHolmAdjustedPValues:
    """Test Holm's step-down adjusted p-values.

    Reference values computed by hand following the procedure:
    1. Sort p-values ascending with original indices
    2. adj_p(i) = p(i) * (m - rank), capped at 1.0
    3. Enforce monotonicity via running max
    """

    def test_basic(self):
        # Sorted: 0.005(3), 0.01(0), 0.03(2), 0.04(1)
        # adj: 0.005*4=0.02, max(0.02, 0.01*3=0.03)=0.03, max(0.03, 0.03*2=0.06)=0.06, max(0.06, 0.04*1=0.04)=0.06
        p = [0.01, 0.04, 0.03, 0.005]
        adj = holm_adjusted_p_values(p)
        assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])

    def test_ties(self):
        # Sorted: 0.01(2), 0.05(0), 0.05(1)
        # adj: 0.01*3=0.03, max(0.03, 0.05*2=0.10)=0.10, max(0.10, 0.05*1=0.05)=0.10
        p = [0.05, 0.05, 0.01]
        adj = holm_adjusted_p_values(p)
        assert adj == pytest.approx([0.10, 0.10, 0.03])

    def test_single(self):
        adj = holm_adjusted_p_values([0.03])
        assert adj == pytest.approx([0.03])

    def test_all_non_significant(self):
        # All capped at 1.0
        p = [0.5, 0.6, 0.7]
        adj = holm_adjusted_p_values(p)
        assert adj == pytest.approx([1.0, 1.0, 1.0])

    def test_all_significant(self):
        # Sorted: 0.001(0), 0.002(1), 0.003(2)
        # adj: 0.001*3=0.003, max(0.003, 0.002*2=0.004)=0.004, max(0.004, 0.003*1=0.003)=0.004
        p = [0.001, 0.002, 0.003]
        adj = holm_adjusted_p_values(p)
        assert adj == pytest.approx([0.003, 0.004, 0.004])

    def test_empty(self):
        assert holm_adjusted_p_values([]) == []

    def test_monotonicity(self):
        """Adjusted p-values must be >= raw p-values."""
        p = [0.01, 0.04, 0.03, 0.005, 0.10]
        adj = holm_adjusted_p_values(p)
        for raw, adjusted in zip(p, adj):
            assert adjusted >= raw

    def test_cap_at_one(self):
        """No adjusted p-value exceeds 1.0."""
        p = [0.3, 0.4, 0.5]
        adj = holm_adjusted_p_values(p)
        assert all(a <= 1.0 for a in adj)

    def test_preserves_order_of_significance(self):
        """If p_i < p_j, then adj_i <= adj_j."""
        p = [0.01, 0.04, 0.03, 0.005]
        adj = holm_adjusted_p_values(p)
        # 0.005 < 0.01 < 0.03 < 0.04, so adj[3] <= adj[0] <= adj[2] <= adj[1]
        assert adj[3] <= adj[0] <= adj[2] <= adj[1]


class TestHolmCorrectionConsistency:
    """Ensure holm_correction (bool) is consistent with holm_adjusted_p_values."""

    @pytest.mark.parametrize(
        "p_values",
        [
            [0.01, 0.04, 0.03, 0.005],
            [0.05, 0.05, 0.01],
            [0.001, 0.002, 0.003],
            [0.5, 0.6, 0.7],
        ],
    )
    def test_bool_matches_adjusted(self, p_values):
        alpha = 0.05
        rejections = holm_correction(p_values, alpha)
        adjusted = holm_adjusted_p_values(p_values)
        for reject, adj_p in zip(rejections, adjusted):
            assert reject == (adj_p <= alpha)
