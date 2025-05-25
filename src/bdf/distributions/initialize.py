import numpy as np
from bdf.distributions.BDFDistribution import BDFDistribution
from bdf.distributions import (
    Normal,
    NegativeBinomial,
    Poisson,
    Exponential,
    Bernoulli,
    Beta,
    Gamma,
    Uniform,
    LogNormal,
    ChiSquared,
    Weibull,
)


def init_distribution(data_dist: str | BDFDistribution, prior_dist: str | BDFDistribution | dict = 'auto') -> BDFDistribution:
    """ Initialize a BDF distribution based on the data distribution and prior distribution.
    """
    if isinstance(data_dist, str):
        posterior_dist = init_distribution_str(data_dist, prior_dist)
    elif isinstance(data_dist, BDFDistribution):
        posterior_dist = init_distribution_dist(data_dist, prior_dist)
    else:
        raise ValueError(f"{data_dist = } must be a string or a BDFDistribution instance, got {type(data_dist)}")
    return posterior_dist


def init_distribution_str(data_dist: str, prior_dist: str | BDFDistribution | dict = 'auto') -> BDFDistribution:
    """ Initialize a BDF distribution based on the data distribution and prior distribution.

    For now just supports Normal distribution, including normal prior.
    Posterior can easily be calculated from prior and data mean and variances.
    """
    assert data_dist == 'normal', f"Currently only 'normal' data distribution is supported, got {data_dist}"
    if prior_dist == 'auto':
        return Normal.Normal(prior_params={'mean': 0, 'std': 5})
    elif isinstance(prior_dist, BDFDistribution):
        assert isinstance(prior_dist, Normal.Normal), f"Currently only 'normal' prior distribution is supported, got {prior_dist.__class__.__name__}"
        raise ValueError(f"Prior distribution must be a string or a dictionary, got {prior_dist} of type {type(prior_dist)}")
    elif isinstance(prior_dist, dict):
        assert 'mean' in prior_dist and 'std' in prior_dist, "Prior distribution dictionary must contain 'mean' and 'std' keys"
        return Normal.Normal(prior_params=prior_dist)
    else:
        raise ValueError(f"{prior_dist = } must be a string, BDFDistribution instance or a dictionary, got {prior_dist} of type {type(prior_dist)}")
    



def init_distribution_dist(data_dist: BDFDistribution, prior_dist: str | BDFDistribution | dict = 'auto') -> BDFDistribution:
    """ Initialize a BDF distribution based on the data distribution and prior distribution.
    """
    pass