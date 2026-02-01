"""Load benchmark results from YAML files and aggregate BDF models."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

RESULTS_DIR = Path("benchmarks/results/custom")


def load_yaml_result(yaml_path: Path) -> dict[str, Any]:
    """Load a single YAML result file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Parsed YAML content as dict.
    """
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_model_results(
    results_dir: Path = RESULTS_DIR,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load results for multiple models from YAML files.

    Args:
        results_dir: Directory containing result YAML files.
        model_names: List of model names to load. If None, loads all res_*.yaml files.

    Returns:
        Dict mapping model_name -> {datasets: {...}, metadata: {...}}
    """
    results = {}

    if model_names is None:
        yaml_files = sorted(results_dir.glob("res_*.yaml"))
        for p in yaml_files:
            # Extract model name: res_bdf_kde.yaml -> bdf_kde
            model_name = p.stem.replace("res_", "")
            try:
                results[model_name] = load_yaml_result(p)
            except Exception as e:
                warnings.warn(f"Failed to load {p}: {e}")
    else:
        for model_name in model_names:
            p = results_dir / f"res_{model_name}.yaml"
            if not p.exists():
                # Try without res_ prefix
                p = results_dir / f"{model_name}.yaml"
            if not p.exists():
                warnings.warn(f"Result file not found for model '{model_name}'")
                continue
            try:
                results[model_name] = load_yaml_result(p)
            except Exception as e:
                warnings.warn(f"Failed to load {p}: {e}")

    return results


def aggregate_bdf_models(
    results_dir: Path = RESULTS_DIR,
    bdf_models: list[str] | None = None,
    selection_metric: str = "crps",
    use_tuning_value: bool = True,
    datasets: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Aggregate multiple BDF models by selecting best variant per dataset.

    For each dataset, selects the BDF model with the best tuning value
    (from fold 0 Optuna tuning) and uses ALL metrics from that model.

    Args:
        results_dir: Directory containing result YAML files.
        bdf_models: List of BDF model names to consider.
            Defaults to ["bdf_normalmunormal", "bdf_kde"].
        selection_metric: Metric used for selection (must match tuning.metric).
        use_tuning_value: If True, use tuning.best_value (fold 0).
            If False, use mean of evaluation folds (less rigorous).
        datasets: List of datasets to include. If None, includes all available.

    Returns:
        Tuple of:
            - Aggregated result dict with same structure as individual models
            - Dict mapping dataset -> selected model name
    """
    if bdf_models is None:
        bdf_models = ["bdf_normalmunormal", "bdf_kde"]

    # Load all BDF model results
    bdf_results = load_model_results(results_dir, bdf_models)

    if not bdf_results:
        raise ValueError(f"No BDF results found for models: {bdf_models}")

    found_models = list(bdf_results.keys())
    missing_models = set(bdf_models) - set(found_models)
    if missing_models:
        warnings.warn(f"BDF models not found: {missing_models}")

    # Get all datasets across all BDF models
    all_datasets = set()
    for model_data in bdf_results.values():
        all_datasets.update(model_data.get("datasets", {}).keys())

    # Filter to requested datasets if specified
    if datasets is not None:
        all_datasets = all_datasets.intersection(set(datasets))

    # For each dataset, select best BDF model
    aggregated_datasets = {}
    selection_map = {}

    for dataset in sorted(all_datasets):
        best_model = None
        best_value = float("inf")
        best_data = None

        for model_name, model_data in bdf_results.items():
            ds_data = model_data.get("datasets", {}).get(dataset)
            if ds_data is None:
                continue

            if use_tuning_value:
                # Use tuning.best_value from fold 0 (methodologically sound)
                tuning = ds_data.get("tuning", {})
                if tuning.get("metric") != selection_metric:
                    # Tuning was done with different metric, use mean of eval folds
                    metrics = ds_data.get("metrics", {})
                    metric_values = metrics.get(selection_metric, [])
                    if metric_values:
                        value = float(np.mean(metric_values))
                    else:
                        continue
                else:
                    value = tuning.get("best_value", float("inf"))
            else:
                # Use mean of evaluation folds
                metrics = ds_data.get("metrics", {})
                metric_values = metrics.get(selection_metric, [])
                if metric_values:
                    value = float(np.mean(metric_values))
                else:
                    continue

            # Lower is better for CRPS, RMSE, etc.
            if value < best_value:
                best_value = value
                best_model = model_name
                best_data = ds_data

        if best_model is not None:
            aggregated_datasets[dataset] = best_data
            selection_map[dataset] = best_model

    # Build aggregated result
    aggregated = {
        "datasets": aggregated_datasets,
        "metadata": {
            "aggregated_from": found_models,
            "selection_metric": selection_metric,
            "use_tuning_value": use_tuning_value,
        },
    }

    return aggregated, selection_map


def build_comparison_dataframe(
    model_results: dict[str, dict[str, Any]],
    metrics: list[str] | None = None,
    datasets: list[str] | None = None,
) -> pl.DataFrame:
    """Build a Polars DataFrame for model comparison.

    Args:
        model_results: Dict mapping model_name -> result dict.
        metrics: List of metrics to include. If None, includes all available.
        datasets: List of datasets to include. If None, includes all.

    Returns:
        DataFrame with columns:
            - model: Model name
            - dataset: Dataset name
            - metric: Metric name
            - fold_values: List of per-fold values
            - mean: Mean across folds
            - std: Std across folds
    """
    rows = []

    for model_name, model_data in model_results.items():
        for ds_name, ds_data in model_data.get("datasets", {}).items():
            if datasets is not None and ds_name not in datasets:
                continue

            ds_metrics = ds_data.get("metrics", {})

            for metric_name, values in ds_metrics.items():
                if metrics is not None and metric_name not in metrics:
                    continue

                # Handle different value types
                if isinstance(values, list):
                    fold_values = [float(v) for v in values]
                    mean_val = float(np.mean(fold_values))
                    std_val = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0
                elif isinstance(values, dict):
                    # Skip complex nested structures (coverage_curve, pit_histogram)
                    continue
                else:
                    fold_values = [float(values)]
                    mean_val = float(values)
                    std_val = 0.0

                rows.append(
                    {
                        "model": model_name,
                        "dataset": ds_name,
                        "metric": metric_name,
                        "fold_values": fold_values,
                        "mean": mean_val,
                        "std": std_val,
                        "n_folds": len(fold_values),
                    }
                )

    return pl.DataFrame(rows)


def get_metric_matrix(
    df: pl.DataFrame,
    metric: str,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract a metric matrix from comparison DataFrame.

    Args:
        df: Comparison DataFrame from build_comparison_dataframe.
        metric: Metric name to extract.
        models: List of models to include (preserves order).
        datasets: List of datasets to include (preserves order).

    Returns:
        Tuple of:
            - 2D numpy array of shape (n_datasets, n_models) with mean values
            - List of dataset names (row order)
            - List of model names (column order)
    """
    filtered = df.filter(pl.col("metric") == metric)

    if models is None:
        models = sorted(filtered.select("model").unique().to_series().to_list())
    if datasets is None:
        datasets = sorted(filtered.select("dataset").unique().to_series().to_list())

    # Build matrix
    matrix = np.full((len(datasets), len(models)), np.nan)
    model_idx = {m: i for i, m in enumerate(models)}
    dataset_idx = {d: i for i, d in enumerate(datasets)}

    for row in filtered.iter_rows(named=True):
        m, d, val = row["model"], row["dataset"], row["mean"]
        if m in model_idx and d in dataset_idx:
            matrix[dataset_idx[d], model_idx[m]] = val

    return matrix, datasets, models


def get_fold_values_matrix(
    df: pl.DataFrame,
    metric: str,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> tuple[dict[tuple[str, str], list[float]], list[str], list[str]]:
    """Extract per-fold values for statistical testing.

    Args:
        df: Comparison DataFrame from build_comparison_dataframe.
        metric: Metric name to extract.
        models: List of models to include.
        datasets: List of datasets to include.

    Returns:
        Tuple of:
            - Dict mapping (dataset, model) -> list of fold values
            - List of dataset names
            - List of model names
    """
    filtered = df.filter(pl.col("metric") == metric)

    if models is None:
        models = sorted(filtered.select("model").unique().to_series().to_list())
    if datasets is None:
        datasets = sorted(filtered.select("dataset").unique().to_series().to_list())

    fold_dict = {}
    for row in filtered.iter_rows(named=True):
        m, d, vals = row["model"], row["dataset"], row["fold_values"]
        if m in models and d in datasets:
            fold_dict[(d, m)] = vals

    return fold_dict, datasets, models
