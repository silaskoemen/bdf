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


def load_dgp_results(dgp_name: str) -> dict | None:
    """Load and merge results for a DGP from both full and core files.

    This allows incremental benchmarking where you can:
    1. Run core models (BDF, RF) → creates _core.json
    2. Run bench-models separately (NGBoost, etc.) → creates _results.json
    3. Plotting merges all models from both files

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
