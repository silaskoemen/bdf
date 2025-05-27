import numpy as np
import pytest
from bdf.tree_classes.bdf_regressor import BDFRegressor
from bdf.tree_classes.bdf_node import BDFNode
from bdf.distributions.distribution_manager import DistributionManager as DM
import bdf_rust

def test_threshold_generation_python():
    """Test threshold generation in Python"""
    # Linear data for predictable thresholds
    data = np.linspace(0, 10, 101)
    
    # Create node
    dist = DM.create_distribution("normal", {"mean": 0, "std": 1})
    node = BDFNode(distribution=dist)
    
    # Expose a method to get candidate thresholds for testing
    def get_thresholds(eta):
        # This function would need to be added to your regressor
        return node._generate_candidate_thresholds(data, 10)
    
    # Test with different eta values
    thresholds_01 = get_thresholds(0.1)
    thresholds_02 = get_thresholds(0.2)
    thresholds_05 = get_thresholds(0.5)
    
    # Check approximate counts (might not be exact due to deduplication)
    assert 8 <= len(thresholds_01) <= 12  # ~1/0.1 = 10
    assert 4 <= len(thresholds_02) <= 6   # ~1/0.2 = 5
    assert 1 <= len(thresholds_05) <= 3   # ~1/0.5 = 2
    
    # Check thresholds are within data range
    for t in thresholds_01:
        assert 0 <= t <= 10
    
    # Check thresholds are properly spaced
    if len(thresholds_01) >= 2:
        min_gap = min(thresholds_01[i+1] - thresholds_01[i] for i in range(len(thresholds_01)-1))
        assert min_gap > 0  # No duplicate thresholds

def test_threshold_generation_rust():
    """Test threshold generation in Rust matches Python"""
    # Linear data for predictable thresholds
    data = np.linspace(0, 10, 101)
    
    # You'll need to expose a function that accesses the Rust threshold generation
    # This is for testing purposes only
    def get_rust_thresholds(data, eta):
        return bdf_rust.generate_thresholds(data, eta)  # type: ignore
    
    # Test with different eta values
    thresholds_01 = get_rust_thresholds(data, 0.1)
    
    # Basic checks
    assert 8 <= len(thresholds_01) <= 12  # ~1/0.1 = 10
    
    # Check thresholds are within data range
    for t in thresholds_01:
        assert 0 <= t <= 10

def test_threshold_deduplication():
    """Test deduplication of thresholds with repeated values"""
    # Data with repeated values
    data = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
    
    dist = DM.create_distribution("normal", {"mean": 0, "std": 1})
    node = BDFNode(distribution=dist)

    thresholds = node._generate_candidate_thresholds(data, 101)
    
    # Check no duplicates
    for i in range(len(thresholds) - 1):
        assert thresholds[i] != thresholds[i+1]
    
    # Check threshold between 1 and 2 exists (midpoint 1.5)
    assert any(1.4 <= t <= 1.6 for t in thresholds)

def test_constant_feature_handling():
    """Test handling of constant features"""
    # Constant data
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    
    dist = DM.create_distribution("normal", {"mean": 0, "std": 1})
    node = BDFNode(distribution=dist)

    thresholds = node._generate_candidate_thresholds(data, 101)
    
    # Constant feature should produce empty thresholds
    assert len(thresholds) == 0