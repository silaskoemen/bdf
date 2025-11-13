from typing import Any

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
    def create_distribution(
        cls, dist: str, prior_params: dict[str, Any], params: dict[str, Any] | None
    ) -> BDFDistribution:
        registry = cls._registry()

        if "+" in dist:
            base_name, leaf_name = (part.strip() for part in dist.split("+", 1))
            try:
                base_cls = registry[base_name]
                leaf_cls = registry[leaf_name]
            except KeyError as exc:
                raise ValueError(f"Unknown component '{exc.args[0]}' for {dist}") from None
            base = base_cls(prior_params=prior_params.get("dist_prior_params", {}))
            leaf = leaf_cls(
                prior_params=prior_params.get("kde_prior_params", {}),
                params=params.get("kde_params", {}) if params else {},
            )
            return KDL(dist=base, kde=leaf, params=params or {})

        try:
            DistClass = registry[dist]
        except KeyError:
            raise ValueError(f"Unknown distribution: {dist}") from None
        return DistClass(prior_params=prior_params, params=params)

    @classmethod
    def to_rust_spec(cls, distribution: BDFDistribution) -> dict[str, Any]:
        return distribution.to_spec()
