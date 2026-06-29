"""Load benchmark results from YAML files and aggregate BDF models."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

RESULTS_DIR = Path("benchmarks/results/regression")
CLASSIFICATION_RESULTS_DIR = Path("benchmarks/results/classification")


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


def aggregate_model_variants(
    results_dir: Path = RESULTS_DIR,
    model_variants: list[str] | None = None,
    selection_metric: str = "crps",
    use_tuning_value: bool = True,
    datasets: list[str] | None = None,
    family_name: str = "model",
    expected_eval_folds: int | None = None,
    required_eval_metrics: list[str] | None = None,
    require_all_variants: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Aggregate model variants by selecting the best variant per dataset.

    For each dataset, selects the variant with the best tuning value
    (from fold 0 Optuna tuning) and uses all metrics from that variant.

    Args:
        results_dir: Directory containing result YAML files.
        model_variants: List of model variant names to consider.
        selection_metric: Metric used for selection (must match tuning.metric).
        use_tuning_value: If True, use tuning.best_value (fold 0).
            If False, use mean of evaluation folds (less rigorous).
        datasets: List of datasets to include. If None, includes all available.
        family_name: Human-readable model-family name for warnings/metadata.
        expected_eval_folds: If set, variants are eligible for a dataset only when
            every required evaluation metric has exactly this many fold values.
        required_eval_metrics: Metrics that must be complete and finite for a
            variant to be eligible. Defaults to the selection metric only.
        require_all_variants: If True, fail unless every requested variant result
            file is present. This prevents silently constructing a fused baseline
            from only a partially completed family sweep.

    Returns:
        Tuple of:
            - Aggregated result dict with same structure as individual models
            - Dict mapping dataset -> selected model name
    """
    if model_variants is None:
        raise ValueError("model_variants must be provided")

    variant_results = load_model_results(results_dir, model_variants)

    if not variant_results:
        raise ValueError(f"No {family_name} results found for models: {model_variants}")

    found_models = list(variant_results.keys())
    missing_models = set(model_variants) - set(found_models)
    if missing_models:
        if require_all_variants:
            raise ValueError(f"Missing required {family_name} models: {sorted(missing_models)}")
        warnings.warn(f"{family_name} models not found: {missing_models}")

    # Get all datasets across all variants
    all_datasets = set()
    for model_data in variant_results.values():
        all_datasets.update(model_data.get("datasets", {}).keys())

    # Filter to requested datasets if specified
    if datasets is not None:
        all_datasets = all_datasets.intersection(set(datasets))

    # For each dataset, select best variant
    aggregated_datasets = {}
    selection_map = {}
    exclusions: dict[str, dict[str, list[str]]] = {}
    eligibility_metrics = list(dict.fromkeys(required_eval_metrics or [selection_metric]))

    for dataset in sorted(all_datasets):
        best_model = None
        best_value = float("inf")
        best_data = None

        for model_name, model_data in variant_results.items():
            ds_data = model_data.get("datasets", {}).get(dataset)
            if ds_data is None:
                continue

            metrics = ds_data.get("metrics", {})
            ineligibility_reasons = []
            for metric_name in eligibility_metrics:
                values = metrics.get(metric_name, [])
                if expected_eval_folds is not None and len(values) != expected_eval_folds:
                    ineligibility_reasons.append(
                        f"{metric_name} has {len(values)} folds, expected {expected_eval_folds}"
                    )
                elif not values:
                    ineligibility_reasons.append(f"{metric_name} has no fold values")
                elif not np.isfinite(np.asarray(values, dtype=float)).all():
                    ineligibility_reasons.append(f"{metric_name} contains non-finite values")

            if ineligibility_reasons:
                exclusions.setdefault(dataset, {})[model_name] = ineligibility_reasons
                warnings.warn(
                    f"Skipping {family_name} variant {model_name} on {dataset}: " + "; ".join(ineligibility_reasons)
                )
                continue

            metric_values = metrics.get(selection_metric, [])

            if use_tuning_value:
                # Use tuning.best_value from fold 0 (methodologically sound)
                tuning = ds_data.get("tuning", {})
                if tuning.get("metric") != selection_metric:
                    # Tuning was done with different metric, use mean of eval folds
                    if metric_values:
                        value = float(np.mean(metric_values))
                    else:
                        continue
                else:
                    value = tuning.get("best_value", float("inf"))
            else:
                # Use mean of evaluation folds
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
            "family_name": family_name,
            "expected_eval_folds": expected_eval_folds,
            "required_eval_metrics": eligibility_metrics,
            "require_all_variants": require_all_variants,
            "excluded_variants": exclusions,
        },
    }

    return aggregated, selection_map


def aggregate_bdf_models(
    results_dir: Path = RESULTS_DIR,
    bdf_models: list[str] | None = None,
    selection_metric: str = "crps",
    use_tuning_value: bool = True,
    datasets: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Aggregate multiple BDF models by selecting best variant per dataset."""
    if bdf_models is None:
        bdf_models = ["bdf_normalmunormal", "bdf_kde"]

    return aggregate_model_variants(
        results_dir=results_dir,
        model_variants=bdf_models,
        selection_metric=selection_metric,
        use_tuning_value=use_tuning_value,
        datasets=datasets,
        family_name="BDF",
    )


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

            # Extract coverage metrics from coverage_curve
            coverage_curve = ds_metrics.get("coverage_curve", {})
            if coverage_curve:
                empirical = coverage_curve.get("empirical", {})
                for level, fold_vals in empirical.items():
                    level_float = float(level)
                    level_int = int(level_float * 100)
                    cov_metric = f"coverage_{level_int}"
                    if metrics is not None and cov_metric not in metrics:
                        continue
                    fold_values = [float(v) for v in fold_vals]
                    mean_val = float(np.mean(fold_values))
                    std_val = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0
                    rows.append(
                        {
                            "model": model_name,
                            "dataset": ds_name,
                            "metric": cov_metric,
                            "fold_values": fold_values,
                            "mean": mean_val,
                            "std": std_val,
                            "n_folds": len(fold_values),
                        }
                    )

            for metric_name, values in ds_metrics.items():
                if metrics is not None and metric_name not in metrics:
                    continue

                # Handle different value types
                if isinstance(values, list):
                    fold_values = [float(v) for v in values]
                    mean_val = float(np.mean(fold_values))
                    std_val = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0
                elif isinstance(values, dict):
                    # Skip nested structures (coverage_curve, pit_histogram) - handled above
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


def extract_coverage_curves(
    model_results: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Extract coverage curve data from model results.

    Args:
        model_results: Dict mapping model_name -> result dict.
        datasets: List of datasets to include. If None, includes all.

    Returns:
        Nested dict: model -> dataset -> {levels: [...], empirical: [[fold1], [fold2], ...]}
    """
    coverage_data = {}

    for model_name, model_data in model_results.items():
        coverage_data[model_name] = {}

        for ds_name, ds_data in model_data.get("datasets", {}).items():
            if datasets is not None and ds_name not in datasets:
                continue

            ds_metrics = ds_data.get("metrics", {})
            cov_curve = ds_metrics.get("coverage_curve", {})

            if not cov_curve:
                continue

            levels = cov_curve.get("levels", [])
            empirical = cov_curve.get("empirical", {})

            if not levels or not empirical:
                continue

            # Convert empirical from {level: [fold_values]} to list of per-fold values
            # Structure: empirical[level_str] = [fold0_val, fold1_val, ...]
            n_folds = len(next(iter(empirical.values()))) if empirical else 0
            fold_curves = []

            for fold_idx in range(n_folds):
                fold_curve = []
                for level in levels:
                    level_key = level if isinstance(level, str) else level
                    level_vals = empirical.get(level_key, [])
                    if fold_idx < len(level_vals):
                        fold_curve.append(float(level_vals[fold_idx]))
                    else:
                        fold_curve.append(np.nan)
                fold_curves.append(fold_curve)

            coverage_data[model_name][ds_name] = {
                "levels": [float(lev) for lev in levels],
                "fold_curves": fold_curves,  # List of [n_folds][n_levels]
                "mean_curve": [float(np.nanmean([fc[i] for fc in fold_curves])) for i in range(len(levels))],
            }

    return coverage_data


def extract_pit_histograms(
    model_results: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract PIT histogram data from model results.

    Args:
        model_results: Dict mapping model_name -> result dict.
        datasets: List of datasets to include. If None, includes all.

    Returns:
        Nested dict: model -> dataset -> {bin_counts: [...], n_bins: int, n_samples: int}
    """
    pit_data = {}

    for model_name, model_data in model_results.items():
        pit_data[model_name] = {}

        for ds_name, ds_data in model_data.get("datasets", {}).items():
            if datasets is not None and ds_name not in datasets:
                continue

            ds_metrics = ds_data.get("metrics", {})
            pit_hist = ds_metrics.get("pit_histogram", {})

            if not pit_hist:
                continue

            bin_counts = pit_hist.get("bin_counts", [])
            n_bins = pit_hist.get("n_bins", 20)
            n_samples = pit_hist.get("n_samples", sum(bin_counts) if bin_counts else 0)

            if bin_counts:
                pit_data[model_name][ds_name] = {
                    "bin_counts": [int(c) for c in bin_counts],
                    "n_bins": n_bins,
                    "n_samples": n_samples,
                    "bin_proportions": [c / n_samples for c in bin_counts] if n_samples > 0 else [],
                }

    return pit_data


def extract_timing_data(
    model_results: dict[str, dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract fitting, tuning, and prediction times from model results.

    Args:
        model_results: Dict mapping model_name -> result dict.
        datasets: List of datasets to include. If None, includes all.

    Returns:
        Nested dict: model -> dataset -> {"mean_fit_time": ..., "tuning_time": ...,
        "mean_prediction_time": ...}

    Raises:
        TypeError: If fitting_times is not a list or tuning_time_seconds is not numeric.
    """
    timing_data = {}

    for model_name, model_data in model_results.items():
        timing_data[model_name] = {}

        for ds_name, ds_data in model_data.get("datasets", {}).items():
            if datasets is not None and ds_name not in datasets:
                continue

            entry = {}

            fitting_times = ds_data.get("fitting_times")
            if fitting_times is not None:
                if not isinstance(fitting_times, list):
                    raise TypeError(
                        f"fitting_times for {model_name}/{ds_name} must be a list, "
                        f"got {type(fitting_times).__name__}"
                    )
                entry["mean_fit_time"] = float(np.mean(fitting_times))

            tuning_time = ds_data.get("tuning_time_seconds")
            if tuning_time is not None:
                if not isinstance(tuning_time, (int, float)):
                    raise TypeError(
                        f"tuning_time_seconds for {model_name}/{ds_name} must be numeric, "
                        f"got {type(tuning_time).__name__}"
                    )
                entry["tuning_time"] = float(tuning_time)

            prediction_times = ds_data.get("prediction_times")
            if prediction_times is not None:
                if not isinstance(prediction_times, list):
                    raise TypeError(
                        f"prediction_times for {model_name}/{ds_name} must be a list, "
                        f"got {type(prediction_times).__name__}"
                    )
                fold_totals = []
                for fold in prediction_times:
                    if not isinstance(fold, dict):
                        raise TypeError(
                            f"prediction_times entries for {model_name}/{ds_name} must be dicts, "
                            f"got {type(fold).__name__}"
                        )
                    fold_totals.append(
                        float(fold.get("point_seconds", 0.0)) + float(fold.get("probabilistic_seconds", 0.0))
                    )
                if fold_totals:
                    entry["mean_prediction_time"] = float(np.mean(fold_totals))

            if entry:
                timing_data[model_name][ds_name] = entry

    return timing_data


def compute_speedup_table(
    timing_data: dict[str, dict[str, dict[str, float]]],
    control_model: str = "BDF",
) -> dict[str, dict[str, float]]:
    """Compute geometric mean speedup of control vs each other model.

    For each dataset where both the control and challenger have timing data,
    computes the ratio challenger_time / control_time. Then aggregates across
    datasets using the geometric mean.

    A speedup > 1 means the control is faster.

    Args:
        timing_data: Output of extract_timing_data.
        control_model: Name of the control model.

    Returns:
        Dict mapping model -> {"fit_speedup": ..., "tune_speedup": ...,
        "n_datasets_fit": ..., "n_datasets_tune": ...}
    """
    from scipy.stats import gmean

    control_timing = timing_data.get(control_model, {})
    if not control_timing:
        return {}

    results = {}

    for model_name, model_timing in timing_data.items():
        if model_name == control_model:
            continue

        fit_ratios = []
        tune_ratios = []

        for ds_name in control_timing:
            if ds_name not in model_timing:
                continue

            ctrl = control_timing[ds_name]
            chal = model_timing[ds_name]

            if "mean_fit_time" in ctrl and "mean_fit_time" in chal:
                if ctrl["mean_fit_time"] > 0:
                    fit_ratios.append(chal["mean_fit_time"] / ctrl["mean_fit_time"])

            if "tuning_time" in ctrl and "tuning_time" in chal:
                if ctrl["tuning_time"] > 0:
                    tune_ratios.append(chal["tuning_time"] / ctrl["tuning_time"])

        entry = {"n_datasets_fit": len(fit_ratios), "n_datasets_tune": len(tune_ratios)}
        prediction_ratios = []

        if fit_ratios:
            entry["fit_speedup"] = float(gmean(fit_ratios))
        if tune_ratios:
            entry["tune_speedup"] = float(gmean(tune_ratios))
        for ds_name in control_timing:
            if ds_name not in model_timing:
                continue
            ctrl = control_timing[ds_name]
            chal = model_timing[ds_name]
            if "mean_prediction_time" in ctrl and "mean_prediction_time" in chal:
                if ctrl["mean_prediction_time"] > 0:
                    prediction_ratios.append(chal["mean_prediction_time"] / ctrl["mean_prediction_time"])
        entry["n_datasets_prediction"] = len(prediction_ratios)
        if prediction_ratios:
            entry["prediction_speedup"] = float(gmean(prediction_ratios))

        results[model_name] = entry

    return results


def build_coverage_level_dataframe(
    model_results: dict[str, dict[str, Any]],
    levels: list[float] | None = None,
    datasets: list[str] | None = None,
) -> pl.DataFrame:
    """Build a DataFrame with coverage at specified levels for all models.

    Args:
        model_results: Dict mapping model_name -> result dict.
        levels: Coverage levels to extract. Defaults to [0.5, 0.9, 0.95].
        datasets: List of datasets to include. If None, includes all.

    Returns:
        DataFrame with columns: model, dataset, level, empirical_coverage, coverage_error
    """
    if levels is None:
        levels = [0.5, 0.9, 0.95]

    coverage_curves = extract_coverage_curves(model_results, datasets)
    rows = []

    for model_name, model_ds in coverage_curves.items():
        for ds_name, ds_data in model_ds.items():
            curve_levels = ds_data["levels"]
            mean_curve = ds_data["mean_curve"]

            for target_level in levels:
                # Find closest level in curve
                if target_level in curve_levels:
                    idx = curve_levels.index(target_level)
                    emp_cov = mean_curve[idx]
                else:
                    # Interpolate or find nearest
                    diffs = [abs(lev - target_level) for lev in curve_levels]
                    idx = diffs.index(min(diffs))
                    emp_cov = mean_curve[idx]

                rows.append(
                    {
                        "model": model_name,
                        "dataset": ds_name,
                        "level": target_level,
                        "empirical_coverage": emp_cov,
                        "coverage_error": abs(emp_cov - target_level),
                    }
                )

    return pl.DataFrame(rows)
