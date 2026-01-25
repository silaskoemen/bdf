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

from benchmarks.utils.synthetic_plotting import create_synthetic_benchmark_summary

# Configuration
RESULTS_DIR = Path("benchmarks/results/synthetic_dgp")
PLOTS_DIR = Path("benchmarks/plots/synthetic_dgp")

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


def load_dgp_results(dgp_name: str) -> Optional[dict]:
    """Load and merge results for a DGP from both full and core files.

    This allows incremental benchmarking where you can:
    1. Run core models (BDF, RF) → creates _core.json
    2. Run bench-models separately (NGBoost, etc.) → creates _results.json
    3. Comparison merges all models from both files

    Args:
        dgp_name: Name of the DGP

    Returns:
        Merged results dictionary or None if no results found
    """
    full_path = RESULTS_DIR / f"{dgp_name}_results.json"
    core_path = RESULTS_DIR / f"{dgp_name}_results_core.json"

    full_results = None
    core_results = None

    # Load both if they exist
    if full_path.exists():
        full_results = load_results(full_path)
        logger.info(f"  Found full results for {dgp_name}")

    if core_path.exists():
        core_results = load_results(core_path)
        logger.info(f"  Found core results for {dgp_name}")

    # No results at all
    if full_results is None and core_results is None:
        logger.warning(f"  No results found for {dgp_name}")
        return None

    # Only one type exists - return it
    if full_results is None:
        logger.info(f"  Using core-only results")
        return core_results
    if core_results is None:
        logger.info(f"  Using full results only")
        return full_results

    # Both exist - merge them
    logger.info(f"  Merging full and core results")
    merged = dict(full_results)  # Start with full results

    # Merge models from core into full
    if "models" not in merged:
        merged["models"] = {}

    for model_name, model_data in core_results.get("models", {}).items():
        if model_name in merged["models"]:
            logger.warning(f"    Model {model_name} exists in both files - using full results version")
        else:
            merged["models"][model_name] = model_data
            logger.info(f"    Added {model_name} from core results")

    return merged


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

    # Extract unique set of models across all DGPs
    all_models = set()
    for dgp_results in all_results.values():
        all_models.update(dgp_results.get("models", {}).keys())

    models = sorted(list(all_models))
    logger.info(f"  Found models across all DGPs: {', '.join(models)}")

    if not models:
        logger.warning("  No models found in any results")
        return

    # Create summary plots and tables
    summary_dir = PLOTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    create_synthetic_benchmark_summary(
        results_dict=all_results,
        models=models,
        output_dir=summary_dir,
    )

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
