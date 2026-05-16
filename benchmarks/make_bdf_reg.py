"""Aggregate BDF regression results by selecting best distribution per dataset.

This script creates a unified BDF result by selecting the best-performing
BDF distribution variant for each dataset based on tuning CRPS.

Usage:
    python -m benchmarks.make_bdf_reg
"""

import json
from pathlib import Path

from .utils.yaml_loader import aggregate_bdf_models

RESULTS_DIR = Path("benchmarks/results/regression")

# BDF models to consider for aggregation
# Add new BDF distribution models here as they become available
BDF_MODELS = [
    "bdf_normalmunormal",
    "bdf_kde",
    # "bdf_gammamvlambdaexponential",
    # "bdf_gammamvlambdapoisson",
    # "bdf_normalmeanpseudoalphaskewnormal",
]


def main():
    """Aggregate BDF results and save to JSON."""
    print("Aggregating BDF regression results...")
    print(f"  Models considered: {BDF_MODELS}")

    # Aggregate using tuning CRPS (from fold 0)
    aggregated, selection_map = aggregate_bdf_models(
        results_dir=RESULTS_DIR,
        bdf_models=BDF_MODELS,
        selection_metric="crps",
        use_tuning_value=True,
    )

    print("\n  Distribution selection per dataset:")
    for ds, model in sorted(selection_map.items()):
        print(f"    {ds}: {model}")

    # Convert to JSON-compatible format (aggregate per-fold metrics)
    output_data = {
        "model_config": {
            "name": "bdf_reg",
            "aggregated_from": aggregated["metadata"]["aggregated_from"],
            "selection_metric": aggregated["metadata"]["selection_metric"],
        },
        "datasets": {},
    }

    import numpy as np

    for ds_name, ds_data in aggregated["datasets"].items():
        metrics_raw = ds_data.get("metrics", {})
        metrics_aggregated = {}

        for metric_name, values in metrics_raw.items():
            if isinstance(values, list):
                metrics_aggregated[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n_folds": len(values),
                }
            elif isinstance(values, dict):
                # Skip complex structures like coverage_curve
                continue
            else:
                metrics_aggregated[metric_name] = {"mean": float(values), "std": 0.0, "n_folds": 1}

        output_data["datasets"][ds_name] = {
            "metrics": metrics_aggregated,
            "selected_model": selection_map.get(ds_name, "unknown"),
            "best_params": ds_data.get("best_params", {}),
        }

    # Save to JSON
    output_path = RESULTS_DIR / "bdf_reg.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Saved aggregated results to: {output_path}")


if __name__ == "__main__":
    main()
