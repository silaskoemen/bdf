"""Classification benchmark analysis and visualization.

This script produces publication-ready outputs for JMLR:
- Statistical significance tests (Friedman, Nemenyi, Wilcoxon)
- Critical difference diagrams
- LaTeX tables with significance markers
- Win/Tie/Loss analysis
- Reliability diagrams (calibration)
- Timing speedup tables

Usage:
    python -m benchmarks.calc_plot_classification_metrics
"""

from pathlib import Path

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing YAML result files
RESULTS_DIR = Path("benchmarks/results/classification")

# Output directories
PLOTS_DIR = Path("benchmarks/plots/classification")
TABLES_DIR = Path("benchmarks/results/classification/tables")

# Plot format: "pdf" for vector (best for LaTeX), "png" for raster, or "both"
PLOT_FORMAT = "pdf"

# BDF classification models
BDF_MODELS = [
    "bdf_betamvbernoulli",
]

BDF_CORE_MODEL = "bdf_betamvbernoulli_nle"

# Baseline models
BASELINE_MODELS = [
    "rf_clas",
    "lgbm_clas",
    "callgbm_clas",
    "ngboost_clas",
    "calrf_clas",
    "knn_clas",
]

# Model display names and colors from shared style
from benchmarks.utils.style import MODEL_COLORS, MODEL_DISPLAY_NAMES

# LaTeX tables call the fused aggregate "BDF-Full" to match the manuscript prose;
# plots keep the shorter "BDF" legend label.
TABLE_DISPLAY_NAMES = {**MODEL_DISPLAY_NAMES, "BDF": "BDF-Full"}

# Datasets are discovered at runtime from data/processed/*.meta.json (task=classification).
# Explicit list preserved for reproducibility / deterministic ordering.
DATASETS = [
    "ionosphere",
    "breast_cancer_wisconsin",
    "credit_approval",
    "heart_disease",
    "pima_diabetes",
    "titanic",
    "german_credit",
    "aids_ctg_175",
    "spambase",
    "default_credit_card",
    "bank_marketing",
    "adult_income",
]

# Metrics
POINT_METRICS = ["accuracy", "f1", "auroc", "auprc"]
PROB_METRICS = ["log_loss", "brier", "ece"]

# Metric display names
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "log_loss": "Log Loss",
    "bss": "BSS",
    "brier": "Brier",
    "ece": "ECE",
}

# Lower is better for these metrics
LOWER_IS_BETTER = {
    "accuracy": False,
    "f1": False,
    "auroc": False,
    "auprc": False,
    "log_loss": True,
    "brier": True,
    "ece": True,
    "bss": False,  # Higher Brier Skill Score is better
}


# =============================================================================
# RELIABILITY DIAGRAM
# =============================================================================


def plot_reliability_diagram(
    calibration_data: dict[str, dict],
    models: list[str],
    model_colors: dict[str, str] | None = None,
    model_display_names: dict[str, str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot reliability diagram (predicted probability vs observed frequency).

    Aggregates calibration curves across datasets per model using weighted averaging.

    Args:
        calibration_data: model -> dataset -> {prob_true, prob_pred, bin_counts}
        models: List of model names to include.
        model_colors: Color per model.
        model_display_names: Display name per model.
        save_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    if model_colors is None:
        model_colors = {}
    if model_display_names is None:
        model_display_names = {}

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    for model in models:
        model_ds = calibration_data.get(model, {})
        if not model_ds:
            continue

        # Weighted average across datasets
        all_prob_true = []
        all_prob_pred = []
        all_counts = []

        for ds_data in model_ds.values():
            prob_true = ds_data.get("prob_true", [])
            prob_pred = ds_data.get("prob_pred", [])
            bin_counts = ds_data.get("bin_counts", [])

            if not prob_true or not prob_pred or not bin_counts:
                continue

            all_prob_true.append(np.array(prob_true, dtype=float))
            all_prob_pred.append(np.array(prob_pred, dtype=float))
            all_counts.append(np.array(bin_counts, dtype=float))

        if not all_prob_true:
            continue

        n_bins = len(all_prob_true[0])
        total_counts = np.zeros(n_bins)
        weighted_true = np.zeros(n_bins)
        weighted_pred = np.zeros(n_bins)

        for pt, pp, bc in zip(all_prob_true, all_prob_pred, all_counts):
            for i in range(n_bins):
                if not np.isnan(pt[i]) and not np.isnan(pp[i]):
                    weighted_true[i] += pt[i] * bc[i]
                    weighted_pred[i] += pp[i] * bc[i]
                    total_counts[i] += bc[i]

        mask = total_counts > 0
        avg_true = np.full(n_bins, np.nan)
        avg_pred = np.full(n_bins, np.nan)
        avg_true[mask] = weighted_true[mask] / total_counts[mask]
        avg_pred[mask] = weighted_pred[mask] / total_counts[mask]

        valid = ~np.isnan(avg_true) & ~np.isnan(avg_pred)
        display = model_display_names.get(model, model)
        color = model_colors.get(model, None)
        ax.plot(avg_pred[valid], avg_true[valid], "o-", color=color, label=display, markersize=5)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def save_plot(fig_func, base_path: Path, **kwargs):
    """Save plot in configured format(s)."""
    if PLOT_FORMAT in ("pdf", "both"):
        fig_func(save_path=base_path.with_suffix(".pdf"), **kwargs)
    if PLOT_FORMAT in ("png", "both"):
        fig_func(save_path=base_path.with_suffix(".png"), **kwargs)


def main():
    """Run complete classification benchmark analysis."""

    import polars as pl

    from .utils.latex_tables import (
        generate_main_results_table,
        generate_per_dataset_table,
        generate_ranking_table,
        generate_rel_to_best_table,
        generate_speedup_table,
        generate_win_tie_loss_table,
        save_latex_table,
    )
    from .utils.plotting import (
        plot_auroc_and_ece,
        plot_critical_difference_diagram,
        plot_metric_comparison_bars,
        plot_metric_scatter,
        plot_rel_to_best,
    )
    from .utils.statistical_tests import (
        friedman_test,
        nemenyi_test,
        pairwise_wilcoxon_tests,
        pairwise_win_tie_loss,
    )
    from .utils.yaml_loader import (
        build_comparison_dataframe,
        compute_speedup_table,
        extract_timing_data,
        get_metric_matrix,
        load_model_results,
    )

    # Create output directories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLASSIFICATION BENCHMARK ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load BDF classification results
    # -------------------------------------------------------------------------
    print("\n[1] Loading BDF classification model...")

    bdf_results = load_model_results(RESULTS_DIR, BDF_MODELS)

    if not bdf_results:
        print("    ERROR: No BDF classification results found. Run the benchmark suite first.")
        return

    # For classification there is only one BDF distribution (Bernoulli),
    # so no aggregation needed — just rename to "BDF"
    bdf_model_key = next(iter(bdf_results))
    bdf_data = bdf_results[bdf_model_key]

    # Filter to configured datasets
    if bdf_data.get("datasets"):
        available_ds = set(bdf_data["datasets"].keys())
        configured_ds = set(DATASETS)
        active_datasets = sorted(available_ds & configured_ds)
        missing_ds = configured_ds - available_ds
        if missing_ds:
            print(f"    Warning: datasets not found in BDF results: {missing_ds}")
    else:
        print("    ERROR: BDF results contain no datasets.")
        return

    print(f"    Loaded BDF ({bdf_model_key}) with {len(active_datasets)} datasets: {active_datasets}")

    # -------------------------------------------------------------------------
    # 2. Load baseline model results
    # -------------------------------------------------------------------------
    print("\n[2] Loading baseline model results...")

    baseline_results = load_model_results(RESULTS_DIR, BASELINE_MODELS)
    print(f"    Loaded {len(baseline_results)} baseline models: {list(baseline_results.keys())}")

    # Combine BDF with baselines
    all_results = {"BDF": bdf_data, **baseline_results}

    # -------------------------------------------------------------------------
    # 3. Build comparison DataFrame
    # -------------------------------------------------------------------------
    print("\n[3] Building comparison DataFrame...")

    all_metrics = POINT_METRICS + PROB_METRICS
    df = build_comparison_dataframe(all_results, metrics=all_metrics, datasets=active_datasets)

    datasets = sorted(df.select("dataset").unique().to_series().to_list())
    models = ["BDF"] + [m for m in baseline_results.keys()]

    print(f"    Datasets ({len(datasets)}): {datasets}")
    print(f"    Models ({len(models)}): {models}")

    if len(datasets) < 3:
        print("    Warning: fewer than 3 datasets — statistical tests will have low power.")

    # Compute BSS (Brier Skill Score): BSS = 1 − Brier / p(1−p)
    # where p = class prevalence.  Brier_clim = p*(1−p) for a constant classifier.
    from .pipeline.data import load as load_dataset

    clim_brier = {}
    for ds in datasets:
        _, _, y = load_dataset(ds)
        p = float(y.mean())
        clim_brier[ds] = p * (1.0 - p)

    bss_rows = []
    for model in models:
        for ds, clim_val in clim_brier.items():
            row = df.filter((df["model"] == model) & (df["dataset"] == ds) & (df["metric"] == "brier"))
            brier_mean = row.select("mean").to_series()[0]
            brier_std = row.select("std").to_series()[0]
            bss_rows.append(
                {
                    "model": model,
                    "dataset": ds,
                    "metric": "bss",
                    "fold_values": [],
                    "mean": float(1.0 - brier_mean / clim_val),
                    "std": float(brier_std / clim_val),
                    "n_folds": 0,
                }
            )

    df = pl.concat([df, pl.DataFrame(bss_rows)])
    print(f"    Computed BSS for {len(bss_rows)} model-dataset pairs")

    # -------------------------------------------------------------------------
    # 4. Statistical tests for key metrics
    # -------------------------------------------------------------------------
    print("\n[4] Running statistical significance tests...")

    key_metrics = ["log_loss", "brier", "bss"]
    friedman_results = {}
    nemenyi_results = {}

    for metric in key_metrics:
        print(f"\n    --- {METRIC_DISPLAY.get(metric, metric)} ---")

        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)

        nan_count = np.isnan(metric_matrix).sum()
        if nan_count > 0:
            print(f"    Warning: {nan_count} NaN values in {metric} matrix")

        if len(ds_list) < 3:
            print(f"    Skipping Friedman test: need >= 3 datasets, have {len(ds_list)}")
            continue

        try:
            friedman_res = friedman_test(metric_matrix, model_list, lower_is_better=LOWER_IS_BETTER.get(metric, True))
            friedman_results[metric] = friedman_res

            print(f"    Friedman test: chi2 = {friedman_res.statistic:.3f}, p = {friedman_res.p_value:.4f}")
            print(
                f"    Iman-Davenport: F = {friedman_res.iman_davenport_statistic:.3f}, "
                f"p = {friedman_res.iman_davenport_p_value:.4f}"
            )
            print(f"    Reject H0: {friedman_res.reject_null}")

            print("    Average ranks:")
            for name, rank in sorted(friedman_res.avg_ranks.items(), key=lambda x: x[1]):
                print(f"      {name}: {rank:.2f}")

            if friedman_res.reject_null:
                nemenyi_res = nemenyi_test(metric_matrix, model_list, lower_is_better=LOWER_IS_BETTER.get(metric, True))
                nemenyi_results[metric] = nemenyi_res
                print(f"    Nemenyi CD = {nemenyi_res.critical_difference:.3f}")
                print(f"    Significant pairs: {nemenyi_res.significant_pairs}")

        except Exception as e:
            print(f"    Error in Friedman test: {e}")

    # -------------------------------------------------------------------------
    # 5. Pairwise Wilcoxon tests (BDF vs each baseline)
    # -------------------------------------------------------------------------
    print("\n[5] Pairwise Wilcoxon tests (BDF vs baselines)...")

    wilcoxon_results = {}

    for metric in key_metrics:
        print(f"\n    --- {METRIC_DISPLAY.get(metric, metric)} ---")

        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)

        if len(ds_list) < 3:
            print(f"    Skipping: need >= 3 datasets, have {len(ds_list)}")
            continue

        control_idx = model_list.index("BDF") if "BDF" in model_list else 0

        wilcoxon_res = pairwise_wilcoxon_tests(
            metric_matrix, model_list, control_idx=control_idx, lower_is_better=LOWER_IS_BETTER.get(metric, True)
        )
        wilcoxon_results[metric] = wilcoxon_res

        for challenger, res in wilcoxon_res.items():
            sig = "*" if res.reject_null else ""
            print(
                f"    BDF vs {challenger}: p = {res.p_value:.4f}, "
                f"p_adj = {res.adjusted_p_value:.4f}{sig}, A12 = {res.a12:.3f}"
            )

    # -------------------------------------------------------------------------
    # 6. Win/Tie/Loss analysis
    # -------------------------------------------------------------------------
    print("\n[6] Win/Tie/Loss analysis (BDF vs baselines)...")

    wtl_results = {}

    for metric in ["log_loss"]:
        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)

        if len(ds_list) < 2:
            print(f"    Skipping: need >= 2 datasets, have {len(ds_list)}")
            continue

        control_idx = model_list.index("BDF") if "BDF" in model_list else 0

        wtl_res = pairwise_win_tie_loss(metric_matrix, model_list, control_idx=control_idx, lower_is_better=True)
        wtl_results[metric] = wtl_res

        print(f"\n    {METRIC_DISPLAY.get(metric, metric)} Win/Tie/Loss (BDF vs others):")
        for challenger, res in wtl_res.items():
            print(
                f"      vs {challenger}: W={res.wins}, T={res.ties}, L={res.losses} "
                f"(sign test p={res.sign_test_p_value:.3f})"
            )

    # -------------------------------------------------------------------------
    # 7. Timing analysis
    # -------------------------------------------------------------------------
    print("\n[7] Computing timing speedup table...")

    timing_data = extract_timing_data(all_results, datasets=datasets)

    for model_name, model_timing in sorted(timing_data.items()):
        ds_count = len(model_timing)
        if ds_count == 0:
            continue
        fit_times = [v["mean_fit_time"] for v in model_timing.values() if "mean_fit_time" in v]
        tune_times = [v["tuning_time"] for v in model_timing.values() if "tuning_time" in v]
        fit_str = f"fit={np.mean(fit_times):.2f}s" if fit_times else "fit=N/A"
        tune_str = f"tune={np.mean(tune_times):.1f}s" if tune_times else "tune=N/A"
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        print(f"    {display}: {ds_count} datasets, {fit_str}, {tune_str}")

    speedup_data = compute_speedup_table(timing_data, control_model="BDF")

    if speedup_data:
        print("\n    Geometric mean speedup (BDF vs baselines):")
        for model, data in sorted(speedup_data.items(), key=lambda x: x[1].get("fit_speedup", 0), reverse=True):
            display = MODEL_DISPLAY_NAMES.get(model, model)
            fit_spd = data.get("fit_speedup")
            tune_spd = data.get("tune_speedup")
            fit_str = f"{fit_spd:.2f}x" if fit_spd is not None else "N/A"
            tune_str = f"{tune_spd:.2f}x" if tune_spd is not None else "N/A"
            print(f"      {display}: fit={fit_str}, tune={tune_str}")

    # -------------------------------------------------------------------------
    # 8. Generate plots
    # -------------------------------------------------------------------------
    print("\n[8] Generating plots...")

    def save_fig(plot_func, base_name: str, **kwargs):
        base_path = PLOTS_DIR / base_name
        if PLOT_FORMAT in ("pdf", "both"):
            plot_func(save_path=base_path.with_suffix(".pdf"), **kwargs)
        if PLOT_FORMAT in ("png", "both"):
            plot_func(save_path=base_path.with_suffix(".png"), **kwargs)

    # 8a. Critical Difference Diagrams
    for metric in key_metrics:
        if metric in friedman_results:
            friedman_res = friedman_results[metric]
            nemenyi_res = nemenyi_results.get(metric)
            cd = nemenyi_res.critical_difference if nemenyi_res else None

            save_fig(
                plot_critical_difference_diagram,
                f"classification_cd_{metric}",
                avg_ranks=friedman_res.avg_ranks,
                n_datasets=friedman_res.n_datasets,
                cd=cd,
                title=f"Critical Difference Diagram ({METRIC_DISPLAY.get(metric, metric)})",
                model_display_names=MODEL_DISPLAY_NAMES,
            )

    # 8b. Metric scatter plots
    for metric in ["log_loss", "brier", "ece", "auroc", "auprc"]:
        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)
        if metric_matrix.size == 0:
            continue
        save_fig(
            plot_metric_scatter,
            f"classification_scatter_{metric}",
            metric_matrix=metric_matrix,
            models=model_list,
            datasets=ds_list,
            metric_name=METRIC_DISPLAY.get(metric, metric),
            lower_is_better=LOWER_IS_BETTER.get(metric, True),
            model_display_names=MODEL_DISPLAY_NAMES,
        )

    # 8c. Relative-to-best plots
    for metric in ["log_loss", "brier"]:
        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)

        rel_to_best_data = {}
        for j, model in enumerate(model_list):
            col = metric_matrix[:, j]
            ratios = []
            for i in range(len(ds_list)):
                row = metric_matrix[i, :]
                row_valid = ~np.isnan(row)
                if row_valid.any() and not np.isnan(col[i]):
                    best_in_row = np.nanmin(row) if LOWER_IS_BETTER.get(metric, True) else np.nanmax(row)
                    if best_in_row != 0:
                        ratios.append(col[i] / best_in_row)
            if ratios:
                rel_to_best_data[model] = np.mean(ratios)

        if rel_to_best_data:
            save_fig(
                plot_rel_to_best,
                f"classification_rel_to_best_{metric}",
                rel_to_best_data=rel_to_best_data,
                metric=metric,
                model_colors=MODEL_COLORS,
                model_display_names=MODEL_DISPLAY_NAMES,
            )

    # 8d. AUROC + AUPRC + ECE scatter (wide-format pivot required by plot_auroc_and_ece)
    df_wide_auroc_ece = (
        df.filter(pl.col("metric").is_in(["auroc", "auprc", "ece"]))
        .select(["model", "dataset", "metric", "mean"])
        .pivot(on="metric", index=["model", "dataset"], values="mean")
    )
    if not df_wide_auroc_ece.is_empty():
        save_fig(
            plot_auroc_and_ece,
            "classification_auroc_ece",
            df=df_wide_auroc_ece,
            models=models,
            datasets=datasets,
        )

    # 8e. Bar charts for AUROC and AUPRC
    for disc_metric in ["auroc", "auprc"]:
        disc_data = {}
        for model in models:
            model_df = df.filter((df["model"] == model) & (df["metric"] == disc_metric))
            if not model_df.is_empty():
                means = model_df.select("mean").to_series().to_list()
                stds = model_df.select("std").to_series().to_list()
                disc_data[model] = (np.mean(means), np.mean(stds))
        if disc_data:
            save_fig(
                plot_metric_comparison_bars,
                f"classification_{disc_metric}_comparison",
                metric_data=disc_data,
                metric_name=METRIC_DISPLAY.get(disc_metric, disc_metric.upper()),
                lower_is_better=False,
                model_colors=MODEL_COLORS,
                model_display_names=MODEL_DISPLAY_NAMES,
            )

    # 8f. Bar chart for Log Loss
    log_loss_data = {}
    for model in models:
        model_df = df.filter((df["model"] == model) & (df["metric"] == "log_loss"))
        if not model_df.is_empty():
            means = model_df.select("mean").to_series().to_list()
            stds = model_df.select("std").to_series().to_list()
            log_loss_data[model] = (np.mean(means), np.mean(stds))

    if log_loss_data:
        save_fig(
            plot_metric_comparison_bars,
            "classification_log_loss_comparison",
            metric_data=log_loss_data,
            metric_name="Log Loss",
            lower_is_better=True,
            model_colors=MODEL_COLORS,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

    # 8g. BSS scatter and bar chart
    bss_matrix, bss_ds_list, bss_model_list = get_metric_matrix(df, "bss", models=models, datasets=datasets)
    save_fig(
        plot_metric_scatter,
        "classification_scatter_bss",
        metric_matrix=bss_matrix,
        models=bss_model_list,
        datasets=bss_ds_list,
        metric_name="BSS",
        lower_is_better=False,
        model_display_names=MODEL_DISPLAY_NAMES,
    )
    bss_bar_data = {}
    for model in models:
        mdf = df.filter((df["model"] == model) & (df["metric"] == "bss"))
        means = mdf.select("mean").to_series().to_list()
        stds = mdf.select("std").to_series().to_list()
        bss_bar_data[model] = (
            float(np.mean(means)),
            float(np.mean(stds)),
        )
    save_fig(
        plot_metric_comparison_bars,
        "classification_bss_comparison",
        metric_data=bss_bar_data,
        metric_name="BSS",
        lower_is_better=False,
        model_colors=MODEL_COLORS,
        model_display_names=MODEL_DISPLAY_NAMES,
    )

    # 8h. Reliability diagram
    calibration_data = {}
    for model_name, model_data in all_results.items():
        calibration_data[model_name] = {}
        for ds_name, ds_data in model_data.get("datasets", {}).items():
            if ds_name not in datasets:
                continue
            cal_curve = ds_data.get("metrics", {}).get("calibration_curve")
            if cal_curve and isinstance(cal_curve, dict):
                calibration_data[model_name][ds_name] = cal_curve

    if any(calibration_data.values()):
        save_fig(
            plot_reliability_diagram,
            "classification_reliability_diagram",
            calibration_data=calibration_data,
            models=models,
            model_colors=MODEL_COLORS,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

    # -------------------------------------------------------------------------
    # 9. Generate LaTeX tables
    # -------------------------------------------------------------------------
    print("\n[9] Generating LaTeX tables...")

    # Main results table
    main_metrics = ["log_loss", "brier", "ece", "auroc"]
    metric_data = {}

    for metric in main_metrics:
        metric_data[metric] = {}
        for model in models:
            model_df = df.filter((df["model"] == model) & (df["metric"] == metric))
            if not model_df.is_empty():
                mean_val = model_df.select("mean").to_series().mean()
                std_val = model_df.select("std").to_series().mean()
                metric_data[metric][model] = (mean_val, std_val)
            else:
                metric_data[metric][model] = (np.nan, np.nan)

    # Find best per metric
    best_per_metric = {}
    for metric in main_metrics:
        lower = LOWER_IS_BETTER.get(metric, True)
        best_val = float("inf") if lower else float("-inf")
        best_model = None
        for model, (mean, _) in metric_data[metric].items():
            if np.isnan(mean):
                continue
            if (lower and mean < best_val) or (not lower and mean > best_val):
                best_val = mean
                best_model = model
        best_per_metric[metric] = best_model

    main_table = generate_main_results_table(
        models=models,
        datasets=datasets,
        metric_data=metric_data,
        best_per_metric=best_per_metric,
        wilcoxon_results=wilcoxon_results,
        metrics_display=METRIC_DISPLAY,
        lower_is_better=LOWER_IS_BETTER,
        model_display_names=TABLE_DISPLAY_NAMES,
        caption="Classification benchmark results across datasets",
        label="tab:classification-main",
    )
    save_latex_table(main_table, TABLES_DIR / "classification_main_results.tex")

    # Per-dataset table for Log Loss
    ll_matrix, ds_list, model_list = get_metric_matrix(df, "log_loss", models=models, datasets=datasets)
    std_rows = []
    for ds in ds_list:
        row_stds = []
        for model in model_list:
            model_df = df.filter((df["model"] == model) & (df["dataset"] == ds) & (df["metric"] == "log_loss"))
            if not model_df.is_empty():
                row_stds.append(model_df.select("std").to_series()[0])
            else:
                row_stds.append(np.nan)
        std_rows.append(row_stds)
    std_matrix = np.array(std_rows)

    per_dataset_table = generate_per_dataset_table(
        models=model_list,
        datasets=ds_list,
        metric_matrix=ll_matrix,
        std_matrix=std_matrix,
        metric_name="Log Loss",
        caption="Log Loss per dataset",
        label="tab:log-loss-per-dataset",
        lower_is_better=True,
        model_display_names=TABLE_DISPLAY_NAMES,
    )
    save_latex_table(per_dataset_table, TABLES_DIR / "log_loss_per_dataset.tex")

    # Win/Tie/Loss table
    if "log_loss" in wtl_results:
        wtl_table = generate_win_tie_loss_table(
            control_name="BDF",
            challengers=[m for m in models if m != "BDF"],
            wtl_results=wtl_results["log_loss"],
            caption="Win/Tie/Loss for Log Loss (BDF-Full vs baselines)",
            label="tab:classification-win-tie-loss",
            model_display_names=TABLE_DISPLAY_NAMES,
        )
        save_latex_table(wtl_table, TABLES_DIR / "classification_win_tie_loss.tex")

    # Ranking tables (Log Loss and Brier)
    if "log_loss" in friedman_results:
        ranking_table = generate_ranking_table(
            avg_ranks=friedman_results["log_loss"].avg_ranks,
            metric_name="Log Loss",
            friedman_p=friedman_results["log_loss"].iman_davenport_p_value,
            caption="Algorithm rankings by Log Loss",
            label="tab:rankings-log-loss",
            model_display_names=TABLE_DISPLAY_NAMES,
        )
        save_latex_table(ranking_table, TABLES_DIR / "rankings_log_loss.tex")

    if "brier" in friedman_results:
        brier_ranking_table = generate_ranking_table(
            avg_ranks=friedman_results["brier"].avg_ranks,
            metric_name="Brier",
            friedman_p=friedman_results["brier"].iman_davenport_p_value,
            caption="Algorithm rankings by Brier Score",
            label="tab:rankings-brier",
            model_display_names=TABLE_DISPLAY_NAMES,
        )
        save_latex_table(brier_ranking_table, TABLES_DIR / "rankings_brier.tex")

    # Relative-to-best table for Log Loss
    ll_matrix_full, ds_list_full, model_list_full = get_metric_matrix(df, "log_loss", models=models, datasets=datasets)
    rel_to_best_ll: dict[str, float] = {}
    for j, model in enumerate(model_list_full):
        ratios = []
        for i in range(len(ds_list_full)):
            row = ll_matrix_full[i, :]
            if not np.isnan(ll_matrix_full[i, j]) and np.any(~np.isnan(row)):
                best = np.nanmin(row)
                if best > 0:
                    ratios.append(ll_matrix_full[i, j] / best)
        if ratios:
            rel_to_best_ll[model] = float(np.mean(ratios))

    if rel_to_best_ll:
        rel_to_best_table = generate_rel_to_best_table(
            rel_to_best=rel_to_best_ll,
            metric_name="Log Loss",
            model_display_names=TABLE_DISPLAY_NAMES,
            caption="Average Log Loss relative to best model per dataset (lower is better; 1.0 = always best)",
            label="tab:rel-to-best-log-loss",
        )
        save_latex_table(rel_to_best_table, TABLES_DIR / "rel_to_best_log_loss.tex")

    # BDF-Core versus BDF-Full table.  The main benchmark keeps only the
    # canonical tuned BDF; this table reports the fixed-NLE ablation separately.
    core_results = load_model_results(RESULTS_DIR, [BDF_CORE_MODEL])
    if BDF_CORE_MODEL in core_results:
        bdf_core_label = "BDF-Core"
        bdf_full_label = "BDF-Full"
        core_comparison_results = {
            bdf_core_label: core_results[BDF_CORE_MODEL],
            bdf_full_label: bdf_data,
            **baseline_results,
        }
        core_datasets = sorted(
            set(datasets) & set(core_results[BDF_CORE_MODEL].get("datasets", {})) & set(bdf_data.get("datasets", {}))
        )
        core_df = build_comparison_dataframe(
            core_comparison_results,
            metrics=main_metrics,
            datasets=core_datasets,
        )

        def _metric_mean(model: str, metric: str) -> float:
            rows = core_df.filter((core_df["model"] == model) & (core_df["metric"] == metric))
            return float(rows.select("mean").to_series().mean())

        def _relative_to_best(metric: str, model: str) -> float:
            ratios = []
            for ds in core_datasets:
                rows = core_df.filter((core_df["dataset"] == ds) & (core_df["metric"] == metric))
                values = rows.select(["model", "mean"]).to_dicts()
                model_value = next(row["mean"] for row in values if row["model"] == model)
                best_value = min(row["mean"] for row in values)
                if best_value > 0:
                    ratios.append(model_value / best_value)
            return float(np.mean(ratios))

        core_log_loss = [
            core_df.filter(
                (core_df["dataset"] == ds) & (core_df["model"] == bdf_core_label) & (core_df["metric"] == "log_loss")
            )
            .select("mean")
            .to_series()[0]
            for ds in core_datasets
        ]
        full_log_loss = [
            core_df.filter(
                (core_df["dataset"] == ds) & (core_df["model"] == bdf_full_label) & (core_df["metric"] == "log_loss")
            )
            .select("mean")
            .to_series()[0]
            for ds in core_datasets
        ]
        core_gap = 100.0 * np.median((np.array(core_log_loss) - np.array(full_log_loss)) / np.array(full_log_loss))

        core_full_table = "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Fixed-NLE Beta--Bernoulli BDF-Core versus tuned classification BDF-Full.}",
                r"\label{tab:bdf-core-vs-full-classification}",
                r"\begin{tabular}{lccccc}",
                r"\toprule",
                r"Model & Rel. log loss $\downarrow$ & Rel. Brier $\downarrow$ & ECE $\downarrow$ & AUROC $\uparrow$ & Gap vs Full $\downarrow$ \\",
                r"\midrule",
                (
                    rf"BDF-Core (NLE) & {_relative_to_best('log_loss', bdf_core_label):.3f} "
                    rf"& {_relative_to_best('brier', bdf_core_label):.3f} "
                    rf"& {_metric_mean(bdf_core_label, 'ece'):.3f} "
                    rf"& {_metric_mean(bdf_core_label, 'auroc'):.3f} "
                    rf"& {core_gap:+.2f}\% \\"
                ),
                (
                    rf"BDF-Full (tuned) & {_relative_to_best('log_loss', bdf_full_label):.3f} "
                    rf"& {_relative_to_best('brier', bdf_full_label):.3f} "
                    rf"& {_metric_mean(bdf_full_label, 'ece'):.3f} "
                    rf"& {_metric_mean(bdf_full_label, 'auroc'):.3f} "
                    r"& 0.00\% \\"
                ),
                r"\bottomrule",
                r"\end{tabular}",
                (
                    rf"\par\smallskip\footnotesize{{Relative scores are averaged over {len(core_datasets)} "
                    r"datasets against the best model per dataset in the classification comparison set "
                    r"(lower is better). The gap column is the median log-loss difference relative to "
                    r"BDF-Full. BDF-Core fixes the conjugate NLE score, while BDF-Full tunes the canonical "
                    r"Beta--Bernoulli score regime.}"
                ),
                r"\end{table}",
            ]
        )
        save_latex_table(core_full_table, TABLES_DIR / "bdf_core_vs_full_classification.tex")

    # Per-dataset and ranking tables for BSS
    bss_matrix_full, bss_ds_list, bss_model_list = get_metric_matrix(df, "bss", models=models, datasets=datasets)
    bss_std_rows = [
        [
            df.filter((df["model"] == model) & (df["dataset"] == ds) & (df["metric"] == "bss"))
            .select("std")
            .to_series()[0]
            for model in bss_model_list
        ]
        for ds in bss_ds_list
    ]
    save_latex_table(
        generate_per_dataset_table(
            models=bss_model_list,
            datasets=bss_ds_list,
            metric_matrix=bss_matrix_full,
            std_matrix=np.array(bss_std_rows),
            metric_name="BSS",
            caption="Brier Skill Score per dataset (higher is better; baseline = train-prior classifier)",
            label="tab:bss-per-dataset",
            lower_is_better=False,
            model_display_names=TABLE_DISPLAY_NAMES,
        ),
        TABLES_DIR / "bss_per_dataset.tex",
    )
    if "bss" in friedman_results:
        save_latex_table(
            generate_ranking_table(
                avg_ranks=friedman_results["bss"].avg_ranks,
                metric_name="BSS",
                friedman_p=friedman_results["bss"].iman_davenport_p_value,
                caption="Algorithm rankings by Brier Skill Score (higher is better)",
                label="tab:rankings-bss",
                model_display_names=TABLE_DISPLAY_NAMES,
            ),
            TABLES_DIR / "rankings_bss.tex",
        )

    # Speedup table
    if speedup_data:
        speedup_tex = generate_speedup_table(
            speedup_data=speedup_data,
            control_name="BDF",
            model_display_names=MODEL_DISPLAY_NAMES,
            caption="Geometric mean speedup of BDF relative to classification baselines",
            label="tab:classification-speedup",
        )
        save_latex_table(speedup_tex, TABLES_DIR / "classification_speedup.tex")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("\nGenerated files:")
    for f in sorted(PLOTS_DIR.glob("classification_*.*")):
        if f.suffix in (".pdf", ".png"):
            print(f"  - {f}")
    for f in sorted(TABLES_DIR.glob("*.tex")):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
