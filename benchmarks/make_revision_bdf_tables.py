"""Generate BDF revision tables for the TMLR framing update.

This script consumes existing benchmark result YAMLs. It does not run models or
change the benchmark protocol. It expects result files named
``benchmarks/results/regression/res_<model>.yaml`` for the fixed BDF ablation
configs.

Usage:
    python -m benchmarks.make_revision_bdf_tables
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmarks.calc_plot_regression_metrics import DATASETS
from benchmarks.utils.style import MODEL_DISPLAY_NAMES
from benchmarks.utils.yaml_loader import aggregate_bdf_models, load_model_results

RESULTS_DIR = Path("benchmarks/results/regression")
TABLES_DIR = Path("benchmarks/results/regression/tables")

BDF_EXACT_NORMAL_MODEL = "bdf_normalmunormal_nle"
BDF_FULL_MODEL = "BDF-Full"

BDF_FULL_SELECTION_MODELS = [
    "bdf_normalmunormal",
    "bdf_kde",
    "bdf_gammamvlambdapoisson",
    "bdf_freqstudentt",
]

BDF_EXACT_NORMAL_COMPARISON_MODELS = [
    BDF_EXACT_NORMAL_MODEL,
    BDF_FULL_MODEL,
    "qrf",
    "drf",
    "ngboost_reg",
    "xgboostlss_studentt",
    "catbunc_reg",
]

BDF_SCORE_ABLATION_MODELS = [
    "bdf_normalmunormal_nle",
    "bdf_normalmunormal_nll",
    "bdf_normalmunormal_nll_bic",
    "bdf_freqstudentt_nll",
    "bdf_freqstudentt_nll_bic",
]

BDF_SCORE_ABLATION_META = {
    "bdf_normalmunormal_nle": ("Normal--Normal", "NLE", "Yes"),
    "bdf_normalmunormal_nll": ("Normal--Normal", "NLL", "No"),
    "bdf_normalmunormal_nll_bic": ("Normal--Normal", "NLL+BIC", "No"),
    "bdf_freqstudentt_nll": ("Student-$t$", "NLL", "No"),
    "bdf_freqstudentt_nll_bic": ("Student-$t$", "NLL+BIC", "No"),
}


def _display_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model.replace("_", r"\_"))


def _mean_metric(model_data: dict[str, Any], dataset: str, metric: str) -> float | None:
    ds_data = model_data.get("datasets", {}).get(dataset)
    if not ds_data:
        return None
    metrics = ds_data.get("metrics", {})

    if metric in metrics:
        values = metrics[metric]
        if isinstance(values, list) and values:
            arr = np.asarray(values, dtype=float)
            return float(np.nanmean(arr))
        if isinstance(values, (int, float)):
            return float(values)

    # Some result files store coverage only inside coverage_curve.
    if metric.startswith("coverage_"):
        level = int(metric.split("_", maxsplit=1)[1]) / 100
        curve = metrics.get("coverage_curve", {})
        empirical = curve.get("empirical", {}) if isinstance(curve, dict) else {}
        values = empirical.get(str(level)) or empirical.get(level)
        if values:
            arr = np.asarray(values, dtype=float)
            return float(np.nanmean(arr))

    return None


def _datasets_with_metrics(
    model_results: dict[str, dict[str, Any]],
    models: list[str],
    metrics: list[str],
) -> tuple[list[str], list[str]]:
    datasets = []
    dropped = []
    for dataset in DATASETS:
        if all(
            _mean_metric(model_results[model], dataset, metric) is not None for model in models for metric in metrics
        ):
            datasets.append(dataset)
        else:
            dropped.append(dataset)
    return datasets, dropped


def _geomean_relative_to_best(
    model_results: dict[str, dict[str, Any]],
    models: list[str],
    datasets: list[str],
    metric: str,
) -> dict[str, float]:
    rel: dict[str, list[float]] = {model: [] for model in models}
    for dataset in datasets:
        values = {model: _mean_metric(model_results[model], dataset, metric) for model in models}
        finite = [v for v in values.values() if v is not None and math.isfinite(v)]
        if not finite:
            continue
        best = min(finite)
        if best <= 0:
            continue
        for model, value in values.items():
            if value is not None and math.isfinite(value):
                rel[model].append(value / best)
    return {
        model: float(math.exp(np.mean(np.log(values))))
        for model, values in rel.items()
        if values and all(value > 0 for value in values)
    }


def _average_metric_ranks(
    model_results: dict[str, dict[str, Any]],
    models: list[str],
    datasets: list[str],
    metric: str,
) -> dict[str, float]:
    ranks: dict[str, list[float]] = {model: [] for model in models}
    for dataset in datasets:
        values = []
        for model in models:
            value = _mean_metric(model_results[model], dataset, metric)
            if value is not None and math.isfinite(value):
                values.append((model, value))
        if len(values) < 2:
            continue
        for rank, (model, _value) in enumerate(sorted(values, key=lambda item: item[1]), start=1):
            ranks[model].append(float(rank))
    return {model: float(np.mean(values)) for model, values in ranks.items() if values}


def _median_gap_vs_full(
    model_results: dict[str, dict[str, Any]],
    model: str,
    datasets: list[str],
) -> float | None:
    if model == BDF_FULL_MODEL:
        return 0.0
    gaps = []
    for dataset in datasets:
        value = _mean_metric(model_results[model], dataset, "crps")
        full = _mean_metric(model_results[BDF_FULL_MODEL], dataset, "crps")
        if value is None or full is None or not math.isfinite(value) or not math.isfinite(full) or full == 0:
            continue
        gaps.append(100.0 * (value - full) / abs(full))
    return median(gaps) if gaps else None


def _format_float(value: float | None, precision: int = 3, suffix: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "---"
    return f"{value:.{precision}f}{suffix}"


def _bold_if_best(value: str, is_best: bool) -> str:
    return rf"\textbf{{{value}}}" if is_best and value != "---" else value


def _mean_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _write_table(
    path: Path,
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    footnote: str,
) -> None:
    col_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\par\smallskip\footnotesize{{{footnote}}}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def _metric_fold_count(model_data: dict[str, Any], dataset: str, metric: str) -> int | None:
    ds_data = model_data.get("datasets", {}).get(dataset)
    if not ds_data:
        return None
    values = ds_data.get("metrics", {}).get(metric)
    if isinstance(values, list):
        return len(values)
    return None


def _assert_protocol_consistency(
    model_results: dict[str, dict[str, Any]],
    models: list[str],
    datasets: list[str],
    context: str,
) -> None:
    n_trials: set[int] = set()
    fold_counts: set[int] = set()
    for model in models:
        for dataset in datasets:
            ds_data = model_results.get(model, {}).get("datasets", {}).get(dataset)
            if not ds_data:
                continue
            tuning = ds_data.get("tuning", {})
            if "n_trials" in tuning:
                n_trials.add(int(tuning["n_trials"]))
            fold_count = _metric_fold_count(model_results[model], dataset, "crps")
            if fold_count is not None:
                fold_counts.add(fold_count)

    if len(n_trials) > 1:
        raise ValueError(f"{context}: inconsistent tuning.n_trials values: {sorted(n_trials)}")
    if len(fold_counts) > 1:
        raise ValueError(f"{context}: inconsistent CRPS evaluation-fold counts: {sorted(fold_counts)}")


def _format_dropped(dropped: list[str]) -> str:
    if not dropped:
        return "No datasets were dropped by the common-metric filter."
    escaped = [dataset.replace("_", r"\_") for dataset in dropped]
    return "Dropped datasets under the common-metric filter: " + ", ".join(escaped) + "."


def _load_revision_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    bdf_full, selection_map = aggregate_bdf_models(
        results_dir=results_dir,
        bdf_models=BDF_FULL_SELECTION_MODELS,
        selection_metric="crps",
        use_tuning_value=True,
        datasets=DATASETS,
    )
    model_names = sorted(
        set(BDF_EXACT_NORMAL_COMPARISON_MODELS + BDF_SCORE_ABLATION_MODELS + BDF_FULL_SELECTION_MODELS)
    )
    model_names.remove(BDF_FULL_MODEL)
    model_results = load_model_results(results_dir, model_names)
    model_results[BDF_FULL_MODEL] = bdf_full
    print(f"Loaded BDF-Full selections for {len(selection_map)} datasets")
    return model_results


def make_core_vs_full_table(model_results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    models = [model for model in BDF_EXACT_NORMAL_COMPARISON_MODELS if model in model_results]
    missing = sorted(set(BDF_EXACT_NORMAL_COMPARISON_MODELS) - set(models))
    if missing:
        print(f"Skipping missing core-comparison models: {', '.join(missing)}")
    if BDF_EXACT_NORMAL_MODEL not in model_results:
        print("Normal--Normal NLE result is missing; not writing exact-normal-vs-full table")
        return
    if len(models) < 2:
        print("Not enough core-comparison results to write table")
        return

    datasets, dropped = _datasets_with_metrics(model_results, models, ["crps", "interval_score_90", "coverage_90"])
    if not datasets:
        print("No common datasets for exact-normal-vs-full table")
        return

    rel_crps = _geomean_relative_to_best(model_results, models, datasets, "crps")
    is_rank = _average_metric_ranks(model_results, models, datasets, "interval_score_90")
    best_rel = min(rel_crps.values()) if rel_crps else None
    best_rank = min(is_rank.values()) if is_rank else None
    rows = []
    for model in models:
        coverage_values = [_mean_metric(model_results[model], dataset, "coverage_90") for dataset in datasets]
        coverage = _mean_optional(coverage_values)
        rel_value = rel_crps.get(model)
        rank_value = is_rank.get(model)
        rows.append(
            [
                _display_name(model),
                _bold_if_best(_format_float(rel_value, 3), rel_value == best_rel),
                _format_float(_median_gap_vs_full(model_results, model, datasets), 2, r"\%"),
                _bold_if_best(_format_float(rank_value, 2), rank_value == best_rank),
                _format_float(coverage, 3),
                str(len(datasets)),
            ]
        )

    _write_table(
        output_dir / "bdf_normal_nle_vs_full.tex",
        ["Model", "Table-set rel. CRPS", "Median gap vs Full", "IS90 rank", "Cov@90", "$n$"],
        rows,
        "Exact Normal--Normal NLE BDF versus BDF-Full and selected regression baselines.",
        "tab:bdf-normal-nle-vs-full",
        "Geometric mean relative CRPS is computed relative to the best model per dataset within this table "
        "(lower is better). IS90 rank is the average rank of the 90\\% interval score "
        "(lower is better). BDF-Full selects among canonical per-family BDF configs with one tuning "
        "budget per family; fixed-score ablation configs are not included as separate Full candidates. "
        rf"{_format_dropped(dropped)}",
    )


def make_score_ablation_table(model_results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    models = [model for model in BDF_SCORE_ABLATION_MODELS if model in model_results]
    missing = sorted(set(BDF_SCORE_ABLATION_MODELS) - set(models))
    if missing:
        print(f"Skipping missing score-ablation models: {', '.join(missing)}")
    if len(models) < 2:
        print("Not enough score-ablation results to write table")
        return

    datasets, dropped = _datasets_with_metrics(model_results, models, ["crps", "interval_score_90", "coverage_90"])
    if not datasets:
        print("No common datasets for score-ablation table")
        return

    _assert_protocol_consistency(model_results, models, datasets, "score ablation")

    rel_crps = _geomean_relative_to_best(model_results, models, datasets, "crps")
    is_rank = _average_metric_ranks(model_results, models, datasets, "interval_score_90")
    best_rel = min(rel_crps.values()) if rel_crps else None
    best_rank = min(is_rank.values()) if is_rank else None
    rows = []
    for model in models:
        family, score, exact = BDF_SCORE_ABLATION_META[model]
        coverage_values = [_mean_metric(model_results[model], dataset, "coverage_90") for dataset in datasets]
        coverage = _mean_optional(coverage_values)
        rel_value = rel_crps.get(model)
        rank_value = is_rank.get(model)
        rows.append(
            [
                family,
                score,
                exact,
                _bold_if_best(_format_float(rel_value, 3), rel_value == best_rel),
                _bold_if_best(_format_float(rank_value, 2), rank_value == best_rank),
                _format_float(coverage, 3),
                str(len(datasets)),
            ]
        )

    _write_table(
        output_dir / "bdf_leaf_score_ablation.tex",
        ["Leaf family", "Split score", "Exact Bayes?", "Ablation-set rel. CRPS", "IS90 rank", "Cov@90", "$n$"],
        rows,
        "Leaf-family and split-score ablation for fixed BDF configurations.",
        "tab:bdf-leaf-score-ablation",
        "Student-$t$ rows use MoM-fitted plug-in leaves. The BIC row is therefore a BIC-style "
        rf"complexity correction rather than an exact Laplace evidence calculation. {_format_dropped(dropped)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    model_results = _load_revision_results(args.results_dir)
    make_core_vs_full_table(model_results, args.output_dir)
    make_score_ablation_table(model_results, args.output_dir)


if __name__ == "__main__":
    main()
