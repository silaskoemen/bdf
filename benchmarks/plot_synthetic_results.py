"""Generate publication-quality plots from synthetic DGP benchmark results.

This script loads the JSON results from synthetic_dgp_benchmark.py and creates:
- Summary comparison tables across DGPs
- Variance estimation quality plots

Note: Individual plots (ground truth overlays, conditional densities, local calibration)
are generated during the benchmark run itself (synthetic_dgp_benchmark.py) on fold 1.
"""

import json
from pathlib import Path

from loguru import logger

from benchmarks.utils.synthetic_plotting import create_synthetic_benchmark_summary

# Configuration
RESULTS_DIR = Path("benchmarks/results/synthetic_dgp")
PLOTS_DIR = Path("benchmarks/plots/synthetic_dgp")

# DGPs to include in summary
DGPS = [
    {"name": "heteroscedastic_sinusoidal"},
    {"name": "step_function"},
    {"name": "bimodal_mixture"},
    {"name": "heavy_tailed"},
    {"name": "sparse_sampling"},
]


def load_results(results_file: Path) -> dict:
    """Load results JSON file."""
    with open(results_file) as f:
        return json.load(f)


def has_aggregated_metrics(model_data: dict) -> bool:
    """Return True when a model result has usable aggregate metrics."""
    metrics = model_data.get("aggregated_metrics", {})
    return isinstance(metrics, dict) and bool(metrics)


def load_dgp_results(dgp_name: str) -> dict | None:
    """Load and merge results for a DGP from full/core/incremental result files.

    This allows incremental benchmarking where you can:
    1. Run core models (BDF, RF) → creates _core.json
    2. Run bench-models separately (NGBoost, etc.) → creates _results.json
    3. Run extra model shards, e.g. QRF/DRF → creates _results_forests.json
    4. Plotting merges all models from all files

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


def generate_summary_plots():
    """Generate cross-DGP summary plots.

    Automatically detects which models are present in the results.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Generating cross-DGP summary plots")
    logger.info(f"{'='*80}")

    # Load all results (tries full then core for each DGP)
    all_results = {}
    for dgp_spec in DGPS:
        dgp_name = dgp_spec["name"]
        dgp_results = load_dgp_results(dgp_name)
        if dgp_results is not None:
            all_results[dgp_name] = dgp_results

    if not all_results:
        logger.error("No results files found!")
        return

    # Extract unique set of models across all DGPs
    all_models = set()
    for dgp_results in all_results.values():
        all_models.update(dgp_results.get("models", {}).keys())

    models = sorted(list(all_models))
    logger.info(f"  Found models across all DGPs: {', '.join(models)}")

    if not models:
        logger.warning("  No models found in any results")
        return

    # Create summary plots
    summary_dir = PLOTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    create_synthetic_benchmark_summary(
        results_dict=all_results,
        models=models,
        output_dir=summary_dir,
    )

    logger.info(f"  Saved summary plots to {summary_dir}")


def main():
    """Main plotting pipeline - generates cross-DGP summary plots.

    Individual diagnostic plots are generated during benchmark execution.
    """
    logger.info("Starting synthetic DGP summary plotting")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Output directory: {PLOTS_DIR}/summary")

    # Create output directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate summary plots (collects all available models from results)
    try:
        generate_summary_plots()
    except Exception as e:
        logger.error(f"Failed to generate summary plots: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Summary plotting complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
