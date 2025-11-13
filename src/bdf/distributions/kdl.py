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

from .bdf_distribution import BDFDistribution, BDFDistributionParams
from .kde import KDEBaseParams, PenalizedHKDE, PenalizedHKDEParams
from .normal import NormalMuNormal, NormalMuNormalParams


class NormalMuNormalPenalizedHKDLParams(BDFDistributionParams):
    """Composite params: parametric for splitting + KDE for leaves."""

    # Parametric (splitting) params
    dist_prior_params: NormalMuNormalParams = Field(
        description="Prior parameters for Normal distribution used during splitting"
    )

    # KDE (leaf) params
    kde_prior_params: PenalizedHKDEParams = Field(description="Prior parameters for KDE used in leaf predictions")

    kde_params: KDEBaseParams = Field(description="KDE configuration for leaf modeling")

    class Config:
        extra = "forbid"

    def __init__(self, **data):
        missing = {}
        for field in ["normal_prior_params", "kde_prior_params", "kde_params"]:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        super().__init__(**data)

        for field, default in missing.items():
            warnings.warn(f"No value for '{field}', using default: {default}", UserWarning, stacklevel=2)


class NormalMuNormalPenalizedHKDL(BDFDistribution):
    """Hybrid: Normal for splitting, KDE for leaf predictions.

    During tree growth, `nll`/`log_likelihood` route to the parametric Normal
    distribution, leveraging its priors for regularized split finding.

    After a leaf finalizes, `calc_posterior_params` returns KDE posterior
    (data + bandwidth), and leaf predictions use the non-parametric KDE.
    """

    def __init__(
        self,
        prior_params: dict | NormalMuNormalPenalizedHKDLParams,
        params: dict | None = None,  # kept for API consistency
        use_kde_for_splitting: bool = False,
    ):
        if isinstance(prior_params, dict):
            dist_prior_params = NormalMuNormalParams.model_validate(prior_params.get("normal_prior_params", {}))
            kde_prior_params = PenalizedHKDEParams.model_validate(prior_params.get("kde_prior_params", {}))
            kde_params = KDEBaseParams.model_validate(prior_params.get("kde_params", {}))
            prior_params = NormalMuNormalPenalizedHKDLParams.model_validate(
                {"dist_prior_params": dist_prior_params, "kde_prior_params": kde_prior_params, "kde_params": kde_params}
            )
        assert isinstance(
            prior_params, NormalMuNormalPenalizedHKDLParams
        ), "prior_params must be a dict or NormalMuNormalPenalizedHKDLParams instance"
        self.composite_params = prior_params
        self.use_kde_for_splitting = use_kde_for_splitting

        # Initialize parametric component (for splitting)
        self._normal = NormalMuNormal(prior_params=self.composite_params.dist_prior_params, params={})

        # Initialize KDE component (for leaves)
        self._kde = PenalizedHKDE(
            prior_params=self.composite_params.kde_prior_params, params=self.composite_params.kde_params
        )

        # Set public params/prior_params to maintain compatibility
        super().__init__(prior_params=self.composite_params, params=params)

    # ============ SPLITTING PHASE (use parametric) ============

    def nll(self, data: np.ndarray) -> float:
        """Negative log-likelihood for split scoring."""
        if self.use_kde_for_splitting:
            return self._kde.nll(data)
        return self._normal.nll(data)

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Log-likelihood for split scoring."""
        if self.use_kde_for_splitting:
            return self._kde.log_likelihood(data)
        return self._normal.log_likelihood(data)

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Likelihood for split scoring."""
        if self.use_kde_for_splitting:
            return self._kde.likelihood(data)
        return self._normal.likelihood(data)

    # ============ LEAF PHASE (use KDE) ============

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> dict | tuple:
        """Leaf posterior: always returns KDE (data, bandwidth)."""
        return self._kde.calc_posterior_params(data, return_dict=return_dict)

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get KDE posterior parameters."""
        return self._kde.get_posterior_params(data)

    def get_posterior_mean(self, data: np.ndarray) -> float:
        """Leaf mean from KDE."""
        return self._kde.get_posterior_mean(data)

    def get_posterior_variance(self, data: np.ndarray) -> float:
        """Leaf variance from KDE."""
        return self._kde.get_posterior_variance(data)

    def _sample_posterior_data(self, data: np.ndarray, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        """Sample from KDE leaf posterior."""
        return self._kde._sample_posterior_data(data, n_samples, seed)

    def _sample_posterior_params(self, params: dict, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        """Sample from KDE leaf posterior."""
        return self._kde._sample_posterior_params(params, n_samples, seed)

    def validate_targets(self, data: np.ndarray) -> np.ndarray:
        """Validate targets using KDE rules (stricter: needs ≥2 samples)."""
        return self._kde.validate_targets(data)


class KDLParams(BDFDistributionParams):
    """Composite params: parametric for splitting + KDE for leaves."""

    # Parametric (splitting) params
    dist_prior_params: BDFDistributionParams = Field(
        description="Prior parameters for parametric distribution used during splitting"
    )

    # KDE (leaf) params
    kde_prior_params: BDFDistributionParams = Field(description="Prior parameters for KDE used in leaf predictions")

    kde_params: BDFDistributionParams = Field(description="KDE configuration for leaf modeling")

    class Config:
        extra = "forbid"

    def __init__(self, **data):
        missing = {}
        for field in ["dist_prior_params", "kde_prior_params", "kde_params"]:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        super().__init__(**data)

        for field, default in missing.items():
            warnings.warn(f"No value for '{field}', using default: {default}", UserWarning, stacklevel=2)


class KDL(BDFDistribution):
    """Lightweight wrapper for any BDFDistribution + KDE combination.

    Naming convention is f'{<FullDistributionName>}+{<FullKDEName>}', e.g. 'NormalMuNormal+PenalizedHKDE'.

    During tree growth, `nll`/`log_likelihood` route to the parametric distribution,
    leveraging its priors for regularized split finding.

    After a leaf finalizes, `calc_posterior_params` returns KDE posterior
    (data + bandwidth), and leaf predictions use the non-parametric KDE.
    """

    def __init__(self, dist: BDFDistribution, kde: BDFDistribution, use_kde_for_splitting: bool = False):
        self._dist = dist
        self._kde = kde
        self.use_kde_for_splitting = use_kde_for_splitting

        # Create for compatibility & bookkeeping
        prior_params = KDLParams.model_validate(
            {"dist_prior_params": dist.prior_params, "kde_prior_params": kde.prior_params, "kde_params": kde.params}
        )
        # Set public params/prior_params to maintain compatibility
        super().__init__(prior_params=prior_params, params=None)

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

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> dict | tuple:
        """Leaf posterior: always returns KDE (data, bandwidth)."""
        return self._kde.calc_posterior_params(data, return_dict=return_dict)

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get KDE posterior parameters."""
        return self._kde.get_posterior_params(data)

    def get_posterior_mean(self, data: np.ndarray) -> float:
        """Leaf mean from KDE."""
        return self._kde.get_posterior_mean(data)

    def get_posterior_variance(self, data: np.ndarray) -> float:
        """Leaf variance from KDE."""
        return self._kde.get_posterior_variance(data)

    def sample_prior(self, size: int) -> np.ndarray:
        """Sample from the prior KDE using prior parameters."""
        return self._kde.sample_prior(size)

    def _sample_posterior_data(self, data: np.ndarray, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        """Sample from KDE leaf posterior."""
        return self._kde._sample_posterior_data(data, n_samples, seed)

    def _sample_posterior_params(self, params: dict, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        """Sample from KDE leaf posterior."""
        return self._kde._sample_posterior_params(params, n_samples, seed)

    def validate_targets(self, data: np.ndarray) -> np.ndarray:
        """Validate targets using KDE rules (stricter: needs ≥2 samples)."""
        return self._kde.validate_targets(data)
