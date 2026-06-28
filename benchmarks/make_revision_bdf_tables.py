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
from benchmarks.utils.yaml_loader import aggregate_bdf_models, aggregate_model_variants, load_model_results

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
    "NGBoost",
    "XGBoostLSS",
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

XGBOOSTLSS_SELECTION_MODELS = [
    "xgboostlss_gaussian",
    "xgboostlss_studentt",
    "xgboostlss_laplace",
    "xgboostlss_gaussian_mixture",
]

NGBOOST_SELECTION_MODELS = [
    "ngboost_normal",
    "ngboost_laplace",
    "ngboost_lognormal",
    "ngboost_exponential",
    "ngboost_poisson",
]

UNIFIED_ABLATION_MODELS = [
    "bdf_normalmunormal_nle",
    "bdf_normalmunormal_nll_bic",
    "bdf_freqstudentt_nll",
    "bdf_freqstudentt_nll_bic",
    BDF_FULL_MODEL,
    "qrf",
    "drf",
    "NGBoost",
    "XGBoostLSS",
]

UNIFIED_ABLATION_DISPLAY = {
    "bdf_normalmunormal_nle": "BDF N--N NLE",
    "bdf_normalmunormal_nll_bic": "BDF N--N NLL+BIC",
    "bdf_freqstudentt_nll": "BDF Student-$t$ NLL",
    "bdf_freqstudentt_nll_bic": "BDF Student-$t$ NLL+BIC",
    BDF_FULL_MODEL: "BDF-Full",
    "XGBoostLSS": "XGBoostLSS",
    "NGBoost": "NGBoost",
}

TIMING_MODELS = [
    BDF_FULL_MODEL,
    "bdf_normalmunormal",
    "qrf",
    "drf",
    "NGBoost",
    "XGBoostLSS",
    "catbunc_reg",
    "conflgbm",
]

TIMING_DISPLAY = {
    BDF_FULL_MODEL: "BDF-Full",
    "bdf_normalmunormal": "BDF-Normal",
    "XGBoostLSS": "XGBoostLSS",
    "NGBoost": "NGBoost",
}


def _display_name(model: str) -> str:
    if model in UNIFIED_ABLATION_DISPLAY:
        return UNIFIED_ABLATION_DISPLAY[model]
    if model in TIMING_DISPLAY:
        return TIMING_DISPLAY[model]
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
    xgboostlss_full, _ = aggregate_model_variants(
        results_dir=results_dir,
        model_variants=XGBOOSTLSS_SELECTION_MODELS,
        selection_metric="crps",
        use_tuning_value=True,
        datasets=DATASETS,
        family_name="XGBoostLSS",
        expected_eval_folds=9,
        require_all_variants=True,
    )
    try:
        ngboost_full, _ = aggregate_model_variants(
            results_dir=results_dir,
            model_variants=NGBOOST_SELECTION_MODELS,
            selection_metric="crps",
            use_tuning_value=True,
            datasets=DATASETS,
            family_name="NGBoost",
            expected_eval_folds=9,
            require_all_variants=True,
        )
    except ValueError as exc:
        print(f"Fused NGBoost unavailable ({exc}); using legacy Normal-only result if present")
        ngboost_full = None
    model_names = sorted(
        set(
            BDF_EXACT_NORMAL_COMPARISON_MODELS
            + BDF_SCORE_ABLATION_MODELS
            + BDF_FULL_SELECTION_MODELS
            + XGBOOSTLSS_SELECTION_MODELS
            + NGBOOST_SELECTION_MODELS
            + ["ngboost_reg"]
            + TIMING_MODELS
        )
    )
    model_names.remove(BDF_FULL_MODEL)
    if "XGBoostLSS" in model_names:
        model_names.remove("XGBoostLSS")
    if "NGBoost" in model_names:
        model_names.remove("NGBoost")
    model_results = load_model_results(results_dir, model_names)
    model_results[BDF_FULL_MODEL] = bdf_full
    model_results["XGBoostLSS"] = xgboostlss_full
    if ngboost_full is not None:
        model_results["NGBoost"] = ngboost_full
    elif "ngboost_reg" in model_results:
        model_results["NGBoost"] = model_results["ngboost_reg"]
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


def make_unified_ablation_table(model_results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    models = [model for model in UNIFIED_ABLATION_MODELS if model in model_results]
    missing = sorted(set(UNIFIED_ABLATION_MODELS) - set(models))
    if missing:
        print(f"Skipping missing unified-ablation models: {', '.join(missing)}")
    if len(models) < 2:
        print("Not enough models to write unified ablation table")
        return

    datasets, dropped = _datasets_with_metrics(model_results, models, ["crps", "interval_score_90", "coverage_90"])
    if not datasets:
        print("No common datasets for unified ablation table")
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
        output_dir / "bdf_unified_ablation.tex",
        ["Model", "Unified rel. CRPS", "Median gap vs Full", "IS90 rank", "Cov@90", "$n$"],
        rows,
        "Unified regression ablation and baseline comparison under one normalization.",
        "tab:bdf-unified-ablation",
        "Geometric mean relative CRPS is computed relative to the best model per dataset within this table "
        "(lower is better). Fixed BDF rows use locked per-family score regimes; BDF-Full, NGBoost, and XGBoostLSS "
        "are fold-0 validation-selected aggregates over their submitted variant sets. "
        rf"{_format_dropped(dropped)}",
    )


def _mean_timing(model_data: dict[str, Any], dataset: str, key: str) -> float | None:
    ds_data = model_data.get("datasets", {}).get(dataset)
    if not ds_data:
        return None
    if key == "fit":
        values = ds_data.get("fitting_times")
        if isinstance(values, list) and values:
            return float(np.mean(values))
    if key == "tune":
        value = ds_data.get("tuning_time_seconds")
        if isinstance(value, (int, float)):
            return float(value)
    if key == "predict":
        values = ds_data.get("prediction_times")
        if isinstance(values, list) and values:
            totals = []
            for fold in values:
                if isinstance(fold, dict):
                    totals.append(float(fold.get("point_seconds", 0.0)) + float(fold.get("probabilistic_seconds", 0.0)))
            if totals:
                return float(np.mean(totals))
    return None


def _format_seconds(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "---"
    if value < 1:
        return f"{value:.3f}"
    if value < 100:
        return f"{value:.2f}"
    return f"{value:.1f}"


def make_timing_table(model_results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    models = [model for model in TIMING_MODELS if model in model_results]
    datasets, dropped = _datasets_with_metrics(model_results, models, ["crps"])
    if not datasets:
        print("No common datasets for timing table")
        return

    rows = []
    any_prediction_times = False
    for model in models:
        fit_values = [_mean_timing(model_results[model], dataset, "fit") for dataset in datasets]
        tune_values = [_mean_timing(model_results[model], dataset, "tune") for dataset in datasets]
        predict_values = [_mean_timing(model_results[model], dataset, "predict") for dataset in datasets]
        fit_finite = [value for value in fit_values if value is not None and math.isfinite(value)]
        tune_finite = [value for value in tune_values if value is not None and math.isfinite(value)]
        predict_finite = [value for value in predict_values if value is not None and math.isfinite(value)]
        any_prediction_times = any_prediction_times or bool(predict_finite)
        rows.append(
            [
                _display_name(model),
                _format_seconds(float(np.mean(fit_finite)) if fit_finite else None),
                _format_seconds(float(np.median(fit_finite)) if fit_finite else None),
                _format_seconds(float(np.mean(tune_finite)) if tune_finite else None),
                _format_seconds(float(np.median(tune_finite)) if tune_finite else None),
                _format_seconds(float(np.mean(predict_finite)) if predict_finite else None),
                str(len(fit_finite)),
            ]
        )

    prediction_note = (
        "Mean prediction seconds are included for result files produced after prediction-time instrumentation."
        if any_prediction_times
        else "Locked real-data YAMLs predate prediction-time instrumentation, so prediction seconds are unavailable here; "
        "controlled prediction overhead is reported in Table~\\ref{tab:complexity}."
    )

    _write_table(
        output_dir / "real_benchmark_timing.tex",
        ["Model", "Mean fit s", "Median fit s", "Mean tune s", "Median tune s", "Mean pred. s", "$n$"],
        rows,
        "Absolute real-benchmark wall-clock times from locked YAML artifacts.",
        "tab:real-benchmark-timing",
        "Fit time is the mean over fold-1--9 refits per dataset; tuning time is the fold-0 Optuna study duration. "
        f"{prediction_note} {_format_dropped(dropped)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    model_results = _load_revision_results(args.results_dir)
    make_core_vs_full_table(model_results, args.output_dir)
    make_score_ablation_table(model_results, args.output_dir)
    make_unified_ablation_table(model_results, args.output_dir)
    make_timing_table(model_results, args.output_dir)


if __name__ == "__main__":
    main()
