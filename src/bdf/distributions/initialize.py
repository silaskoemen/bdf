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
    """
    pass


def init_distribution_dist(data_dist: BDFDistribution, prior_dist: str | BDFDistribution | dict = 'auto') -> BDFDistribution:
    """ Initialize a BDF distribution based on the data distribution and prior distribution.
    """
    pass