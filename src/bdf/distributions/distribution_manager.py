from typing import Any, Dict

from bdf.distributions.bdf_distribution import BDFDistribution
from bdf.distributions.normal import NormalNormal, NormalNormalParams
from bdf.distributions.skew_normal import (
    NormalEBSkewNormal,
    NormalEBSkewNormalParams,
    NormalMeanNormalGammaSkewNormal,
    NormalMeanNormalGammaSkewNormalParams,
)

# Import other distribution classes as needed


class DistributionManager:
    """Single manager class for handling distribution creation and conversion"""

    # Registry of available distributions
    DISTRIBUTIONS = {
        "normal_normal": NormalNormal,
        # "zip": ZeroInflatedPoisson,
        # Add other distributions here
    }

    @classmethod
    def create_distribution(cls, dist: str, prior_params: Dict[str, Any]) -> BDFDistribution:
        """Create a distribution instance from name and parameters"""
        # Convert name to lowercase and handle aliases
        name = dist.lower().replace("-", "_").replace(" ", "_")

        # Match pattern for distribution creation
        match name:
            case "normal" | "normalnormal" | "normal_normal" | "gaussian":
                prior_params = NormalNormalParams.model_validate(prior_params)  # type: ignore
                return NormalNormal(prior_params=prior_params)
            case "normalmeanpseudoalphaskewnormal" | "normalmeanpseudoalpha_skewnormal" | "normalpseudoskewnormal" | "normalpseudo_skewnormal" | "npsn" | "np_sn" | "nmpasn" | "nmpa_sn":
                prior_params = NormalEBSkewNormalParams.model_validate(prior_params)  # type: ignore
                return NormalEBSkewNormal(prior_params=prior_params)
            case "normalmeannormalgammaskewnormal" | "normalmeannormalgamma_skewnormal" | "normalnormalskewnormal" | "normalnormal_skewnormal" | "nnsn" | "nn_sn" | "nmngsn" | "nmng_sn":
                prior_params = NormalMeanNormalGammaSkewNormalParams.model_validate(prior_params)  # type: ignore
                return NormalMeanNormalGammaSkewNormal(prior_params=prior_params)
            case _:
                raise ValueError(f"Unknown distribution: {name}")

    @classmethod
    def to_rust_spec(cls, distribution: BDFDistribution) -> Dict[str, Any]:
        """Convert a Python distribution to a spec dict for Rust"""
        # Match pattern for Rust conversion
        match distribution:
            case NormalNormal():
                return {
                    "dist_type": "NormalNormal",
                    "prior_mean": distribution.prior_params.mean,  # type: ignore
                    "prior_std": distribution.prior_params.std,  # type: ignore
                }
            case NormalEBSkewNormal():
                return {
                    "dist_type": "NormalEBSkewNormal",
                    "prior_mu": distribution.prior_params.mu,  # type: ignore
                    "prior_sigma": distribution.prior_params.sigma,  # type: ignore
                    "prior_mean_alpha": distribution.prior_params.mean_alpha,  # type: ignore
                    "prior_m_alpha": distribution.prior_params.m_alpha,  # type: ignore
                }
            # case ZeroInflatedPoisson():
            #     return {
            #         "dist_type": "ZeroInflatedPoisson",
            #         "lambda_prior_alpha": distribution.prior_params.get("lambda_alpha", 1.0),
            #         "lambda_prior_beta": distribution.prior_params.get("lambda_beta", 1.0),
            #         "zero_prob_prior_alpha": distribution.prior_params.get("zero_alpha", 1.0),
            #         "zero_prob_prior_beta": distribution.prior_params.get("zero_beta", 1.0)
            #     }
            case _:
                raise ValueError(f"Unknown distribution type: {type(distribution)}")

    @classmethod
    def from_rust_spec(cls, spec: Dict[str, Any]) -> BDFDistribution:
        """Create a Python distribution from Rust spec dict"""
        match spec.get("dist_type"):
            case "NormalNormal":
                return NormalNormal(
                    prior_params={"mean": spec.get("prior_mean", 0.0), "std": spec.get("prior_std", 1.0)}
                )
            # case "ZeroInflatedPoisson":
            #     return ZeroInflatedPoisson(prior_params={
            #         "lambda_alpha": spec.get("lambda_prior_alpha", 1.0),
            #         "lambda_beta": spec.get("lambda_prior_beta", 1.0),
            #         "zero_alpha": spec.get("zero_prob_prior_alpha", 1.0),
            #         "zero_beta": spec.get("zero_prob_prior_beta", 1.0)
            #     })
            case _:
                raise ValueError(f"Unknown distribution spec: {spec}")
