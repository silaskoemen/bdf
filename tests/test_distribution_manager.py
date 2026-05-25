import numpy as np
import pytest

from bdf import _bdf_rs as bdf_rs
from bdf.distributions.distribution_manager import DistributionManager
from bdf.distributions.normal import NormalMuNormal


class TestDistributionManager:
    def test_create_distribution(self):
        """Test distribution creation from string names with various notations"""
        # Test different ways to specify normal distribution
        normal1 = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 1.0, "sigma_mu": 2.0})

        assert isinstance(normal1, NormalMuNormal)

        # Test default parameters
        normal_default = DistributionManager.create_distribution("NormalMuNormal", {})
        assert normal_default.params.mu_mu == 0.0
        assert normal_default.params.sigma_mu == 1.0

        # Test invalid distribution name
        with pytest.raises(ValueError):
            DistributionManager.create_distribution("invalid_dist", {})

    def test_to_rust_spec(self):
        """Test conversion to Rust spec dictionary"""
        dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 1.5, "sigma_mu": 2.5})
        rust_spec = DistributionManager.to_rust_spec(dist)

        assert rust_spec["dist_type"] == "NormalMuNormal"
        assert rust_spec["mu_mu"] == 1.5
        assert rust_spec["sigma_mu"] == 2.5

        # Test with another distribution type (once implemented)
        # e.g., beta, bernoulli, etc.

    def test_from_rust_spec(self):
        """Test creation from Rust spec dictionary"""
        # from_spec expects params nested under "params" key
        rust_spec = {"dist_type": "NormalMuNormal", "params": {"mu_mu": 3.0, "sigma_mu": 4.0}}

        dist = DistributionManager.from_rust_spec(rust_spec)
        assert isinstance(dist, NormalMuNormal)
        assert dist.params.mu_mu == 3.0
        assert dist.params.sigma_mu == 4.0

    def test_nll_calculation(self):
        """Test NLL calculation for distributions"""
        dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0})
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
        py_dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0})
        py_nll = py_dist.nll(data)

        # Rust side via test function we need to expose
        rust_spec = DistributionManager.to_rust_spec(py_dist)
        # You'll need to implement this function that calls the Rust NLL
        rust_nll = bdf_rs.calculate_nll(data, rust_spec)  # type: ignore

        assert np.abs(py_nll - rust_nll) < 1e-10

    def test_distribution_sampling(self):
        """Test distribution sampling functionality"""
        dist = DistributionManager.create_distribution("NormalMuNormal", {"mu_mu": 0.0, "sigma_mu": 1.0})
        samples = dist.sample_prior(1000)

        # Basic sanity checks
        assert len(samples) == 1000
        assert -4.0 < np.mean(samples) < 4.0  # Should be near 0 with high probability
        assert 0.5 < np.std(samples) < 1.5  # Should be near 1 with high probability

    def test_betamvbernoulli_auto_var_p_uses_scale(self):
        """Auto var_p scales the maximum valid variance at the empirical mean."""
        y = np.array([1.0] * 10 + [0.0] * 90)
        dist = DistributionManager.create_distribution(
            "BetaMVBernoulli",
            {"mean_p": "auto", "var_p": "auto", "var_p_auto_scale": 0.25},
            y=y,
        )

        assert dist.params.mean_p == 0.1
        assert dist.params.var_p == pytest.approx(0.1 * 0.9 * 0.25)
