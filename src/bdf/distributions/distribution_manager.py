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
    def create_distribution(cls, dist: str, params: dict[str, Any], y: np.ndarray | None = None) -> BDFDistribution:
        registry = cls._registry()

        if "+" in dist:
            base_name, leaf_name = (part.strip() for part in dist.split("+", 1))
            try:
                base_cls = registry[base_name]
                leaf_cls = registry[leaf_name]
            except KeyError as exc:
                raise ValueError(f"Unknown component '{exc.args[0]}' for {dist}") from None
            base = base_cls(params=params.get("dist_params", {}) if params else {})
            leaf = leaf_cls(
                params=params.get("kde_params", {}) if params else {},
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
                if y is None:
                    raise ValueError(f"Parameter '{key}' is set to 'auto' but no data (y) was provided")
                params[key] = DistClass.resolve_auto_params(key, y, params)
        return DistClass(params=params)

    @classmethod
    def to_rust_spec(cls, distribution: BDFDistribution) -> dict[str, Any]:
        return distribution.to_rust_spec()

    @classmethod
    def from_rust_spec(cls, spec: dict[str, Any]) -> BDFDistribution:
        """Create a distribution from a Rust spec dictionary."""
        return BDFDistribution.from_spec(spec)
