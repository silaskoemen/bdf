import json

# benchmarks/utils/results_io.py
from pathlib import Path
from typing import List

import polars as pl

from ..metrics.regression import REG_POINT_METRICS, REG_PROB_METRICS

ALL_METRIC_NAMES = set(REG_POINT_METRICS.keys()).union(REG_PROB_METRICS.keys())

RESULTS_DIR = Path("benchmarks/results")


def load_all_results(results_dir: Path = RESULTS_DIR, models: List[str] | None = None) -> pl.DataFrame:
    """
    Load all JSON files under `results_dir` into a single polars DataFrame
    with columns: ['model', 'dataset', <metric columns ...>].
    """
    rows: List[dict] = []
    if models is None:
        for p in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            model_name = data.get("model_config", {}).get("name", p.stem)
            datasets = data.get("datasets", {}) or {}
            for ds_name, ds_obj in datasets.items():
                metrics = ds_obj.get("metrics", {}) or {}
                row = {"model": model_name, "dataset": ds_name}
                # copy metrics (flatten)
                for k, v in metrics.items():
                    row[k] = v
                rows.append(row)
    else:
        for model in models:
            p = results_dir / f"{model}.json"
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            datasets = data.get("datasets", {}) or {}
            for ds_name, ds_obj in datasets.items():
                metrics = ds_obj.get("metrics", {}) or {}
                row = {"model": model, "dataset": ds_name}
                # copy metrics (flatten)
                for k, v in metrics.items():
                    row[k] = v
                rows.append(row)

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    # order columns consistently: model, dataset, then metrics
    metric_cols = [c for c in df.columns if c not in ("model", "dataset")]
    cols = ["model", "dataset"] + sorted(metric_cols)
    return df.select(cols)


def per_dataset_df(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Return a DataFrame for `dataset` with rows per model and metric columns."""
    return df.filter(pl.col("dataset") == dataset).drop("dataset").sort("model")


def per_model_df(df: pl.DataFrame, model: str) -> pl.DataFrame:
    """Return a DataFrame for `model` with rows per dataset and metric columns."""
    return df.filter(pl.col("model") == model).drop("model").sort("dataset")


def as_long(df: pl.DataFrame) -> pl.DataFrame:
    """Return tidy long-form DataFrame: model, dataset, metric, value."""
    value_vars = [c for c in df.columns if c not in ("model", "dataset")]
    if not value_vars:
        return df.lazy().collect()
    return df.melt(id_vars=["model", "dataset"], value_vars=value_vars, variable_name="metric", value_name="value")


def average_rmse_rank(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute average RMSE rank across datasets (1 = best/lowest RMSE).
    Returns a DataFrame with columns ['model', 'avg_rmse_rank'] sorted ascending.
    """
    # filter rows with rmse present
    df2 = df.filter(pl.col("rmse").is_not_null())
    if df2.is_empty():
        return pl.DataFrame(schema={"model": pl.Utf8, "avg_rmse_rank": pl.Float64})

    # compute rank per dataset (ascending => lower rmse gets lower rank number)
    df2 = df2.with_columns(pl.col("rmse").rank("average").over("dataset").alias("rmse_rank"))
    out = df2.group_by("model").agg(pl.col("rmse_rank").mean().alias("avg_rmse_rank"))
    return out.sort("avg_rmse_rank")


def rel_to_best(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    """
    Compute relative-to-best RMSE per model (averaging over datasets).
    Returns a DataFrame with columns ['model', 'rel_to_best_rmse'].
    """
    df2 = df.filter(pl.col(metric).is_not_null())
    if df2.is_empty():
        return pl.DataFrame(schema={"model": pl.Utf8, "dataset": pl.Utf8, f"rel_to_best_{metric}": pl.Float64})
    # compute best rmse per dataset
    best_rmse_df = df2.group_by("dataset").agg(pl.col(metric).min().alias(f"best_{metric}"))
    df2 = df2.join(best_rmse_df, on="dataset")
    df2 = df2.with_columns(((pl.col(metric) / pl.col(f"best_{metric}")).alias(f"rel_to_best_{metric}")))
    out = df2.group_by("model").agg(pl.col(f"rel_to_best_{metric}").mean().alias(f"rel_to_best_{metric}"))
    return out.sort(f"rel_to_best_{metric}")


def make_bdf_reg(
    results_dir: Path = RESULTS_DIR,
    models: List[str] | None = None,
    datasets: List[str] | None = None,
    tuning_metric: str = "rmse",
) -> Path:
    """
    Summarize BDF regression results by selecting, for each dataset, the best
    model (from `models` or all files matching 'bdf_*.json') according to
    `tuning_metric` (lower is better). The output JSON uses the same layout
    as the per-model result files: top-level 'model_config' and a 'datasets'
    mapping with each dataset mapping to its 'metrics' dict from the selected
    model.

    Returns the path of the written JSON file.
    """
    # Collect candidate files
    if models is None:
        model_files = sorted(results_dir.glob("bdf_*.json"))
    else:
        model_files = [results_dir / f"{m}.json" for m in models]

    # Map dataset -> best entry {'model': str, 'metrics': dict, 'score': float}
    best_per_dataset: dict[str, dict] = {}

    for p in model_files:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            # skip unreadable files
            continue

        model_name = data.get("model_config", {}).get("name", p.stem)
        datasets_obj = data.get("datasets", {}) or {}

        for ds_name, ds_obj in datasets_obj.items():
            if datasets is not None and ds_name not in datasets:
                continue
            metrics = ds_obj.get("metrics", {}) or {}
            if tuning_metric not in metrics:
                # skip if tuning metric absent for this model-dataset
                continue
            try:
                score = float(metrics[tuning_metric])
            except Exception:
                # non-numeric tuning metric -> skip
                continue

            current = best_per_dataset.get(ds_name)
            if current is None or score < current["score"]:
                # lower is better => replace if strictly lower
                best_per_dataset[ds_name] = {"model": model_name, "metrics": metrics, "score": score}

    if not best_per_dataset:
        raise RuntimeError("No BDF results found (or no tuning metric present).")

    # Build output JSON: keep the same 'datasets' -> {'metrics': {...}} layout
    out_data = {"model_config": {"name": "bdf_reg"}, "datasets": {}}
    for ds_name, entry in sorted(best_per_dataset.items()):
        out_data["datasets"][ds_name] = {"metrics": entry["metrics"]}

    out_path = results_dir / f"bdf_reg_{tuning_metric}.json"
    out_path.write_text(json.dumps(out_data, indent=4))
    print(f"Wrote BDF regression summary to {out_path}")
    return out_path
