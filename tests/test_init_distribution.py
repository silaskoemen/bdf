import pytest
import numpy as np

from bdf.distributions.initialize import init_distribution
from bdf.distributions import Normal, BDFDistribution


class TestNormalDist():
    def test_init_distribution_str(self):
        """ Test initializing a Normal distribution with a string data distribution. """
        dist = init_distribution('normal', 'auto')
        assert isinstance(dist, Normal.Normal)
        assert dist.prior_params['mean'] == 0
        assert dist.prior_params['std'] == 5

        data = np.array([0, 0, 0, 0, 0])
        posterior_params = dist.get_posterior_params(data)
        assert 'mean' in posterior_params
        assert 'std' in posterior_params
        assert posterior_params['mean'] == 0.
        assert posterior_params['std'] < 5.
    
    def test_init_distribution_dict(self):
        """ Test initializing a Normal distribution with a dictionary prior distribution. """
        prior_params = {'mean': 0, 'std': 2}
        dist = init_distribution('normal', prior_params)
        assert isinstance(dist, Normal.Normal)
        assert dist.prior_params['mean'] == 0
        assert dist.prior_params['std'] == 2

        data = np.array([0, 0, 0, 0, 0])
        posterior_params = dist.get_posterior_params(data)
        assert 'mean' in posterior_params
        assert 'std' in posterior_params
        assert posterior_params['mean'] == 0.
        assert posterior_params['std'] < 5.

        prior_params = {'mean': 1, 'std': 2}
        dist = init_distribution('normal', prior_params)
        assert isinstance(dist, Normal.Normal)
        assert dist.prior_params['mean'] == 1
        assert dist.prior_params['std'] == 2

        data = np.array([0, 0, 0, 0, 0])
        posterior_params = dist.get_posterior_params(data)
        assert 'mean' in posterior_params
        assert 'std' in posterior_params
        assert posterior_params['mean'] < 1.
        assert posterior_params['std'] < 5.

        with pytest.raises(AssertionError):
            # Missing 'std' key in prior_params
            init_distribution('normal', {'mean': 1})
        
        with pytest.raises(AssertionError):
            # Invalid prior distribution type
            init_distribution('normal', {})
