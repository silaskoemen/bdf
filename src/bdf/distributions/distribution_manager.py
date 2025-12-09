from typing import Any

import numpy as np

from bdf.distributions.bdf_distribution import BDFDistribution
from bdf.distributions.kdl import KDL
from bdf.utils.distribution_helpers import import_all_distributions

# Import other distribution classes as needed
import_all_distributions()


class DistributionManager:
    """Single manager class for handling distribution creation and conversion"""

    @classmethod
    def _registry(cls) -> dict[str, type[BDFDistribution]]:
        return BDFDistribution._registry

    @classmethod
    def create_distribution(cls, dist: str, params: dict[str, Any], y: np.ndarray) -> BDFDistribution:
        registry = cls._registry()

        if "+" in dist:
            base_name, leaf_name = (part.strip() for part in dist.split("+", 1))
            try:
                base_cls = registry[base_name]
                leaf_cls = registry[leaf_name]
            except KeyError as exc:
                raise ValueError(f"Unknown component '{exc.args[0]}' for {dist}") from None
            base = base_cls(params=params.get("dist_params", {}) if params else {})  # type: ignore | params is dict, will be converted in class
            leaf = leaf_cls(
                params=params.get("kde_params", {}) if params else {},  # type: ignore
            )
            return KDL(dist=base, kde=leaf, params=params or {})  # type: ignore

        try:
            DistClass = registry[dist]
        except KeyError:
            raise ValueError(f"Unknown distribution: {dist}") from None

        # Check whether any value in params has value 'auto', then call `resolve_auto_params`
        # on keys with value 'auto' and data y
        for key, value in params.items():
            if value == "auto":
                params[key] = DistClass.resolve_auto_params(key, y)
        return DistClass(params=params)  # type: ignore

    @classmethod
    def to_rust_spec(cls, distribution: BDFDistribution) -> dict[str, Any]:
        return distribution.to_rust_spec()
