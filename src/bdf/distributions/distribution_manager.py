from typing import Any, Dict

from bdf.distributions.bdf_distribution import BDFDistribution
from bdf.distributions.normal import NormalMuNormal, NormalMuNormalParams
from bdf.distributions.skew_normal import (
    NormalMeanNormalGammaSkewNormal,
    NormalMeanNormalGammaSkewNormalParams,
    NormalMeanPseudoAlphaSkewNormal,
    NormalMeanPseudoAlphaSkewNormalParams,
    NormalXiNormalAlphaSkewNormalMAP,
    NormalXiNormalAlphaSkewNormalMAPParams,
)

# Import other distribution classes as needed


class DistributionManager:
    """Single manager class for handling distribution creation and conversion"""

    # Registry of available distributions
    DISTRIBUTIONS = {
        "normalmu_normal": NormalMuNormal,
        "normalmeannormalgamma_skewnormal": NormalMeanNormalGammaSkewNormal,
        "normalmeanpseudoalpha_skewnormal": NormalMeanPseudoAlphaSkewNormal,
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
            case "n" | "normal" | "normalnormal" | "normal_normal" | "gaussian":
                prior_params = NormalMuNormalParams.model_validate(prior_params)  # type: ignore
                return NormalMuNormal(prior_params=prior_params)
            case "normalmeanpseudoalphaskewnormal" | "normalmeanpseudoalpha_skewnormal" | "normalpseudoskewnormal" | "normalpseudo_skewnormal" | "npsn" | "np_sn" | "nmpasn" | "nmpa_sn":
                prior_params = NormalMeanPseudoAlphaSkewNormalParams.model_validate(prior_params)  # type: ignore
                return NormalMeanPseudoAlphaSkewNormal(prior_params=prior_params)
            case "normalmeannormalgammaskewnormal" | "normalmeannormalgamma_skewnormal" | "normalnormalskewnormal" | "normalnormal_skewnormal" | "nnsn" | "nn_sn" | "nmngsn" | "nmng_sn":
                prior_params = NormalMeanNormalGammaSkewNormalParams.model_validate(prior_params)  # type: ignore
                return NormalMeanNormalGammaSkewNormal(prior_params=prior_params)
            case "normalxinormalalphaskewnormalmap" | "normalxinormalalpha_skewnormalmap" | "nnsnmap" | "nn_snmap" | "nn_sn_map" | "normalnormal_skewnormalmap" | "normalnormal_skewnormal_map":
                prior_params = NormalXiNormalAlphaSkewNormalMAPParams.model_validate(prior_params)  # type: ignore
                return NormalXiNormalAlphaSkewNormalMAP(prior_params=prior_params)
            case _:
                raise ValueError(f"Unknown distribution: {name}")

    @classmethod
    def to_rust_spec(cls, distribution: BDFDistribution) -> Dict[str, Any]:
        """Convert a Python distribution to a spec dict for Rust"""
        # Match pattern for Rust conversion
        match distribution:
            case NormalMuNormal():
                return {
                    "dist_type": "NormalMuNormal",
                    "mu_zero": distribution.prior_params.mu_zero,  # type: ignore
                    "sigma_zero": distribution.prior_params.sigma_zero,  # type: ignore
                }
            case NormalMeanPseudoAlphaSkewNormal():
                return {
                    "dist_type": "NormalMeanPseudoAlphaSkewNormal",
                    "mu_zero": distribution.prior_params.mu_zero,  # type: ignore
                    "sigma_zero": distribution.prior_params.sigma_zero,  # type: ignore
                    "alpha_zero": distribution.prior_params.alpha_zero,  # type: ignore
                    "m_alpha": distribution.prior_params.m_alpha,  # type: ignore
                }
            case NormalMeanNormalGammaSkewNormal():
                return {
                    "dist_type": "NormalMeanNormalGammaSkewNormal",
                    "mu_zero": distribution.prior_params.mu_zero,  # type: ignore
                    "sigma_zero": distribution.prior_params.sigma_zero,  # type: ignore
                    "mu_gamma": distribution.prior_params.mu_gamma,  # type: ignore
                    "sigma_gamma": distribution.prior_params.sigma_gamma,  # type: ignore
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
            case "NormalMuNormal":
                return NormalMuNormal(
                    prior_params={"mean": spec.get("prior_mean", 0.0), "std": spec.get("prior_std", 1.0)}
                )
            case "NormalMeanPseudoAlphaSkewNormal":
                return NormalMeanPseudoAlphaSkewNormal(
                    prior_params={
                        "mu": spec.get("prior_", 0.0),
                        "sigma": spec.get("prior_sigma", 1.0),
                        "mean_alpha": spec.get("prior_mean_alpha", 0.0),
                        "m_alpha": spec.get("prior_m_alpha", 1.0),
                    }
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
