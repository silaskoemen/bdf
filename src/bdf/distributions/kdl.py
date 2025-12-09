"""Classes for using parametric distributions for split finding (e.g., Normal, Exponential, etc.), and Kernel Density Estimation (KDE)
in the final leaves - Kernel Density Leaves (KDL).

Leverages already implemented BDFDistribution classes, routes for split finding and prediction.
"""

"""Thoughts on how to do this:
- `nll` should use _dist distribution for float return for split finding
- `predict` of Node uses saved params, so {'data': y, 'bandwidth': bw} for kde should be returned
- Conflict of internal `calc_posterior_params` needed for split finding, BUT KDE params needed for prediction
- Need info on whether split finding or prediction is being done during posterior param calculation
- Could also always calculate both & use _dist for `nll` but actually return kde params, but seems inefficient
"""
import warnings

import numpy as np
from pydantic import Field

from bdf.utils.constants import RANDOM_SEED

from .bdf_distribution import BDFDistribution, BDFDistributionParams
from .kde import KDEParams


class KDLParams(BDFDistributionParams):
    """Composite params: parametric for splitting + KDE for leaves."""

    # Parametric (splitting) params
    dist_params: BDFDistributionParams = Field(
        description="Prior parameters for parametric distribution used during splitting"
    )

    kde_params: KDEParams = Field(description="KDE configuration for leaf modeling")

    def __init__(self, **data):
        missing = {}
        for field in ["dist_params", "kde_params", "kde_params"]:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        super().__init__(**data)

        for field, default in missing.items():
            warnings.warn(f"No value for '{field}', using default: {default}", UserWarning, stacklevel=2)


class KDL(BDFDistribution[KDLParams]):
    """Lightweight wrapper for any BDFDistribution + KDE combination.

    Naming convention is f'{<FullDistributionName>}+{<FullKDEName>}', e.g. 'NormalMuNormal+BayesianKDE'.

    During tree growth, `nll`/`log_likelihood` route to the parametric distribution,
    leveraging its priors for regularized split finding.

    After a leaf finalizes, `calc_posterior_params` returns KDE posterior
    (data + bandwidth), and leaf predictions use the non-parametric KDE.
    """

    def __init__(self, dist: BDFDistribution, kde: BDFDistribution, use_kde_for_splitting: bool = False):
        self._dist: BDFDistribution = dist
        self._kde: BDFDistribution = kde
        self.use_kde_for_splitting = use_kde_for_splitting

        # Create for compatibility & bookkeeping
        params = KDLParams.model_validate({"dist_params": dist.params, "kde_params": kde.params})
        # Set public params/params to maintain compatibility
        super().__init__(params=params)

    # ============ SPLITTING PHASE (use parametric) ============

    def nll(self, data: np.ndarray) -> float:
        """Negative log-likelihood for split scoring."""
        if self.use_kde_for_splitting:
            return self._kde.nll(data)
        return self._dist.nll(data)

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Log-likelihood for split scoring."""
        if self.use_kde_for_splitting:
            return self._kde.log_likelihood(data)
        return self._dist.log_likelihood(data)

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Likelihood for split scoring."""
        if self.use_kde_for_splitting:
            return self._kde.likelihood(data)
        return self._dist.likelihood(data)

    # ============ LEAF PHASE (use KDE) ============

    def calc_posterior_params(self, data: np.ndarray) -> dict | tuple:
        """Leaf posterior: always returns KDE (data, bandwidth)."""
        return self._kde.calc_posterior_params(data)

    def get_posterior_mean(self, data: np.ndarray) -> float:
        """Leaf mean from KDE."""
        return self._kde.get_posterior_mean(data=data)

    def get_posterior_variance(self, data: np.ndarray) -> float:
        """Leaf variance from KDE."""
        return self._kde.get_posterior_variance(data=data)

    def sample_prior(self, size: int) -> np.ndarray:
        """Sample from the prior KDE using prior parameters."""
        return self._kde.sample_prior(size)

    def _sample_posterior_params(self, params: dict, n_samples: int = 1) -> np.ndarray:
        """Sample from KDE leaf posterior."""
        return self._kde._sample_posterior_params(params, n_samples, random_state=RANDOM_SEED)

    def validate_targets(self, data: np.ndarray):
        """Validate targets using KDE rules (stricter: needs ≥2 samples)."""
        self._kde.validate_targets(data)
