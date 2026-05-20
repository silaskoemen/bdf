"""Generate comparison plots and tables from synthetic DGP benchmark results.

This script loads the JSON results from synthetic_dgp_benchmark.py and creates:
- Cross-model metric comparisons
- Summary tables (markdown)
- Aggregated performance visualizations

Individual diagnostic plots (predictions, conditional densities, calibration) are
generated during the benchmark run itself in synthetic_dgp_benchmark.py.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from benchmarks.utils.synthetic_plotting import (
    create_synthetic_benchmark_summary,
    order_models,
    plot_conditional_calibration_by_uncertainty,
    plot_spread_skill,
)

# Configuration
RESULTS_DIR = Path("benchmarks/results/synthetic_dgp")
PLOTS_DIR = Path("benchmarks/plots/synthetic_dgp")

CLIMATOLOGICAL_MODEL = "Climatological"

# DGPs to include in comparison
DGPS = [
    "heteroscedastic_sinusoidal",
    "step_function",
    "bimodal_mixture",
    "heavy_tailed",
    "sparse_sampling",
]


def load_results(results_file: Path) -> dict:
    """Load results JSON file."""
    with open(results_file) as f:
        return json.load(f)


def has_aggregated_metrics(model_data: dict) -> bool:
    """Return True when a model result has usable aggregate metrics."""
    metrics = model_data.get("aggregated_metrics", {})
    return isinstance(metrics, dict) and bool(metrics)


def load_dgp_results(dgp_name: str) -> Optional[dict]:
    """Load and merge results for a DGP from full/core/incremental result files.

    This allows incremental benchmarking where you can:
    1. Run core models (BDF, RF) → creates _core.json
    2. Run bench-models separately (NGBoost, etc.) → creates _results.json
    3. Run extra model shards, e.g. QRF/DRF → creates _results_forests.json
    4. Comparison merges all models from all files

    Args:
        dgp_name: Name of the DGP

    Returns:
        Merged results dictionary or None if no results found
    """
    preferred = [
        RESULTS_DIR / f"{dgp_name}_results.json",
        RESULTS_DIR / f"{dgp_name}_results_core.json",
    ]
    shard_paths = sorted(
        p for p in RESULTS_DIR.glob(f"{dgp_name}_results_*.json") if p.name != f"{dgp_name}_results_core.json"
    )
    paths = [p for p in preferred if p.exists()] + shard_paths

    if not paths:
        logger.warning(f"  No results found for {dgp_name}")
        return None

    logger.info(f"  Merging {len(paths)} result file(s) for {dgp_name}")
    merged = load_results(paths[0])
    merged.setdefault("models", {})
    merged["models"] = {
        model_name: model_data
        for model_name, model_data in merged["models"].items()
        if has_aggregated_metrics(model_data)
    }

    for path in paths[1:]:
        data = load_results(path)
        logger.info(f"    Reading {path.name}")
        for model_name, model_data in data.get("models", {}).items():
            if not has_aggregated_metrics(model_data):
                logger.warning(f"    Skipping {model_name} from {path.name}: no aggregated metrics")
                continue
            if model_name in merged["models"]:
                logger.warning(f"    Model {model_name} exists in multiple files - keeping first occurrence")
                continue
            merged["models"][model_name] = model_data
            logger.info(f"    Added {model_name} from {path.name}")

    return merged


def compute_crpss(all_results: dict) -> dict:
    """Compute CRPS Skill Score using the Climatological baseline and inject into results.

    CRPSS = 1 - CRPS_model / CRPS_climatological.
    Higher is better: 0 = no skill, 1 = perfect, negative = worse than climatological.

    Args:
        all_results: Results dict (dgp_name -> {models -> {aggregated_metrics -> ...}}).
            Modified in-place to add "crpss" metric to each model.

    Returns:
        The modified results dict (same reference).
    """
    n_injected = 0

    for dgp_name, dgp_res in all_results.items():
        models_data = dgp_res.get("models", {})
        clim_data = models_data.get(CLIMATOLOGICAL_MODEL, {})
        clim_agg = clim_data.get("aggregated_metrics", {})
        clim_crps_info = clim_agg.get("crps")

        if clim_crps_info is None or "mean" not in clim_crps_info:
            logger.warning(f"  No Climatological CRPS for {dgp_name} — skipping CRPSS")
            continue

        clim_crps_mean = clim_crps_info["mean"]
        if clim_crps_mean <= 0:
            logger.warning(f"  Climatological CRPS <= 0 for {dgp_name} — skipping CRPSS")
            continue

        logger.info(f"  {dgp_name}: Climatological CRPS = {clim_crps_mean:.4f}")

        for model_name, model_data in models_data.items():
            if model_name == CLIMATOLOGICAL_MODEL:
                continue

            agg = model_data.get("aggregated_metrics", {})
            crps_info = agg.get("crps")
            if crps_info is None or "mean" not in crps_info:
                continue

            crps_mean = crps_info["mean"]
            crps_std = crps_info.get("std", 0.0)

            crpss_mean = 1.0 - crps_mean / clim_crps_mean
            # Delta method: std(CRPSS) ≈ std(CRPS) / CRPS_clim
            crpss_std = crps_std / clim_crps_mean

            # Build CRPSS entry matching the aggregated_metrics structure
            crpss_entry = {"mean": crpss_mean, "std": crpss_std}

            # Propagate per-fold values if available
            if "values" in crps_info:
                crpss_entry["values"] = [1.0 - v / clim_crps_mean for v in crps_info["values"]]

            agg["crpss"] = crpss_entry
            n_injected += 1

    logger.info(f"  Injected CRPSS for {n_injected} model-DGP combinations")
    return all_results


def generate_per_dgp_plots(all_results: dict, plots_dir: Path) -> None:
    """Generate combined spread-skill and conditional-calibration plots per DGP.

    Mirrors the conformalization study's per-DGP comparison. The aggregated
    metric keys from `synthetic_dgp_benchmark.py` already match the plotter
    signatures, so we pass them through directly.
    """
    for dgp_name, dgp_data in all_results.items():
        dgp_dir = plots_dir / "summary" / dgp_name
        dgp_dir.mkdir(parents=True, exist_ok=True)

        spread_by_model: dict = {}
        cond_cal_by_model: dict = {}

        for model_name, model_data in dgp_data.get("models", {}).items():
            agg = model_data.get("aggregated_metrics", {})

            if "spread_skill_pred_std" in agg and "spread_skill_rmse" in agg:
                spread_by_model[model_name] = {
                    "spread_skill_pred_std": agg["spread_skill_pred_std"],
                    "spread_skill_rmse": agg["spread_skill_rmse"],
                    "spread_skill_true_std": agg.get("spread_skill_true_std", {"mean": []}),
                }

            if "cond_cal_cov_90" in agg:
                cond_cal_by_model[model_name] = {
                    "cond_cal_cov_90": agg["cond_cal_cov_90"],
                }

        if spread_by_model:
            plot_spread_skill(
                results_by_model=spread_by_model,
                save_path=dgp_dir / "spread_skill_all_models.pdf",
            )

        if cond_cal_by_model:
            plot_conditional_calibration_by_uncertainty(
                results_by_model=cond_cal_by_model,
                save_path=dgp_dir / "cond_calibration_all_models.pdf",
            )


def generate_comparison_plots():
    """Generate cross-DGP comparison plots and tables.

    Automatically detects which models are present in the results.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Generating cross-DGP comparison plots and tables")
    logger.info(f"{'='*80}")

    # Load all results (tries full then core for each DGP)
    all_results = {}
    for dgp_name in DGPS:
        dgp_results = load_dgp_results(dgp_name)
        if dgp_results is not None:
            all_results[dgp_name] = dgp_results

    if not all_results:
        logger.error("No results files found!")
        return

    # Compute CRPSS from Climatological baseline (modifies all_results in-place)
    logger.info("\nComputing CRPS Skill Scores (CRPSS)...")
    compute_crpss(all_results)

    # Extract unique set of models across all DGPs, excluding Climatological
    all_models = set()
    for dgp_results in all_results.values():
        all_models.update(dgp_results.get("models", {}).keys())
    all_models.discard(CLIMATOLOGICAL_MODEL)

    models = order_models(list(all_models))
    logger.info(f"  Found models across all DGPs: {', '.join(models)}")

    if not models:
        logger.warning("  No models found in any results")
        return

    # Remove Climatological from results before plotting (it's a reference, not a competitor)
    for dgp_res in all_results.values():
        dgp_res.get("models", {}).pop(CLIMATOLOGICAL_MODEL, None)

    # Create summary plots and tables
    summary_dir = PLOTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    create_synthetic_benchmark_summary(
        results_dict=all_results,
        models=models,
        output_dir=summary_dir,
    )

    # Combined per-DGP plots: spread-skill + conditional calibration by uncertainty decile
    generate_per_dgp_plots(all_results, PLOTS_DIR)

    logger.info(f"  Saved comparison plots and tables to {summary_dir}")


def main():
    """Main comparison pipeline."""
    logger.info("Starting synthetic DGP comparison pipeline")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Output directory: {PLOTS_DIR}")

    # Create output directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate comparison plots and tables
    try:
        generate_comparison_plots()
    except Exception as e:
        logger.error(f"Failed to generate comparison plots: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("Comparison complete!")
    logger.info("=" * 80)
    logger.info("\nNote: Individual diagnostic plots are generated during benchmark runs.")
    logger.info("See benchmarks/plots/synthetic_dgp/<dgp_name>/ for model-specific visualizations.")


if __name__ == "__main__":
    main()
