import numpy as np
import pytest
from bdf.distributions.distribution_manager import DistributionManager
from bdf.distributions.normal import NormalNormal
import bdf_rust

class TestDistributionManager:
    def test_create_distribution(self):
        """Test distribution creation from string names with various notations"""
        # Test different ways to specify normal distribution
        normal1 = DistributionManager.create_distribution("normal", {"mean": 1.0, "std": 2.0})
        normal2 = DistributionManager.create_distribution("normal_normal", {"mean": 1.0, "std": 2.0})
        normal3 = DistributionManager.create_distribution("gaussian", {"mean": 1.0, "std": 2.0})
        
        assert isinstance(normal1, NormalNormal)
        assert isinstance(normal2, NormalNormal)
        assert isinstance(normal3, NormalNormal)
        
        # Test default parameters
        normal_default = DistributionManager.create_distribution("normal", {})
        assert normal_default.prior_params["mean"] == 0.0
        assert normal_default.prior_params["std"] == 1.0
        
        # Test invalid distribution name
        with pytest.raises(ValueError):
            DistributionManager.create_distribution("invalid_dist", {})
    
    def test_to_rust_spec(self):
        """Test conversion to Rust spec dictionary"""
        dist = DistributionManager.create_distribution("normal", {"mean": 1.5, "std": 2.5})
        rust_spec = DistributionManager.to_rust_spec(dist)
        
        assert rust_spec["dist_type"] == "NormalNormal"
        assert rust_spec["prior_mean"] == 1.5
        assert rust_spec["prior_std"] == 2.5
        
        # Test with another distribution type (once implemented)
        # e.g., beta, bernoulli, etc.
    
    def test_from_rust_spec(self):
        """Test creation from Rust spec dictionary"""
        rust_spec = {
            "dist_type": "NormalNormal",
            "prior_mean": 3.0,
            "prior_std": 4.0
        }
        
        dist = DistributionManager.from_rust_spec(rust_spec)
        assert isinstance(dist, NormalNormal)
        assert dist.prior_params["mean"] == 3.0
        assert dist.prior_params["std"] == 4.0
    
    def test_nll_calculation(self):
        """Test NLL calculation for distributions"""
        dist = DistributionManager.create_distribution("normal", {"mean": 0.0, "std": 1.0})
        data = np.array([1.0, 2.0, 3.0, 4.0])
        
        nll = dist.nll(data)
        assert nll > 0
        
        # Test NLL is the same for identical data with small numerical differences
        nll2 = dist.nll(data + 1e-10)
        assert np.abs(nll - nll2) < 1e-5
    
    def test_rust_python_nll_equivalence(self):
        """Test that Rust and Python NLL calculations match"""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Python side
        py_dist = DistributionManager.create_distribution("normal", {"mean": 0.0, "std": 1.0})
        py_nll = py_dist.nll(data)
        
        # Rust side via test function we need to expose
        rust_spec = DistributionManager.to_rust_spec(py_dist)
        # You'll need to implement this function that calls the Rust NLL
        rust_nll = bdf_rust.calculate_nll(data, rust_spec)  # type: ignore
        
        assert np.abs(py_nll - rust_nll) < 1e-10
    
    def test_distribution_sampling(self):
        """Test distribution sampling functionality"""
        dist = DistributionManager.create_distribution("normal_normal", {"mean": 0.0, "std": 1.0})
        samples = dist.sample_prior(1000)
        
        # Basic sanity checks
        assert len(samples) == 1000
        assert -4.0 < np.mean(samples) < 4.0  # Should be near 0 with high probability
        assert 0.5 < np.std(samples) < 1.5     # Should be near 1 with high probability