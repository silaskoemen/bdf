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
RESULTS_DIR = Path("benchmarks/results/custom")

# Output directories
PLOTS_DIR = Path("benchmarks/plots/classification")
TABLES_DIR = Path("benchmarks/results/custom/tables_classification")

# Plot format: "pdf" for vector (best for LaTeX), "png" for raster, or "both"
PLOT_FORMAT = "pdf"

# BDF classification models
BDF_MODELS = [
    "bdf_betamvbernoulli",
]

# Baseline models
BASELINE_MODELS = [
    "rf_clas",
    "lgbm_clas",
    "ngboost_clas",
    "calrf_clas",
    "knn_clas",
    "gp_clas",
]

# Model display names
MODEL_DISPLAY_NAMES = {
    "BDF": "BDF",
    "bdf_betamvbernoulli": "BDF",
    "rf_clas": "RF",
    "lgbm_clas": "LightGBM",
    "ngboost_clas": "NGBoost",
    "calrf_clas": "CalRF",
    "knn_clas": "KNN",
    "gp_clas": "GP",
}

# Fixed colors for each model
MODEL_COLORS = {
    "BDF": "dodgerblue",
    "rf_clas": "forestgreen",
    "lgbm_clas": "limegreen",
    "ngboost_clas": "goldenrod",
    "calrf_clas": "mediumpurple",
    "knn_clas": "orange",
    "gp_clas": "teal",
}

# Datasets (will be expanded as more are added)
DATASETS = [
    "breast_cancer_wisconsin",
    "boston_housing_classification",
    "titanic",
]

# Metrics
POINT_METRICS = ["accuracy", "f1", "auroc"]
PROB_METRICS = ["log_loss", "brier", "ece"]

# Metric display names
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "auroc": "AUROC",
    "log_loss": "Log Loss",
    "brier": "Brier",
    "ece": "ECE",
}

# Lower is better for these metrics
LOWER_IS_BETTER = {
    "accuracy": False,
    "f1": False,
    "auroc": False,
    "log_loss": True,
    "brier": True,
    "ece": True,
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

    from .utils.latex_tables import (
        generate_main_results_table,
        generate_per_dataset_table,
        generate_ranking_table,
        generate_speedup_table,
        generate_win_tie_loss_table,
        save_latex_table,
    )
    from .utils.plotting import (
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

    # -------------------------------------------------------------------------
    # 4. Statistical tests for key metrics
    # -------------------------------------------------------------------------
    print("\n[4] Running statistical significance tests...")

    key_metrics = ["log_loss", "brier"]
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
            print(f"    BDF vs {challenger}: p = {res.p_value:.4f}{sig}, A12 = {res.a12:.3f}")

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
    for metric in ["log_loss", "brier", "ece"]:
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

    # 8d. Bar chart for Log Loss
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

    # 8e. Reliability diagram
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
    )
    save_latex_table(per_dataset_table, TABLES_DIR / "log_loss_per_dataset.tex")

    # Win/Tie/Loss table
    if "log_loss" in wtl_results:
        wtl_table = generate_win_tie_loss_table(
            control_name="BDF",
            challengers=[m for m in models if m != "BDF"],
            wtl_results=wtl_results["log_loss"],
            caption="Win/Tie/Loss for Log Loss (BDF vs baselines)",
            label="tab:classification-win-tie-loss",
        )
        save_latex_table(wtl_table, TABLES_DIR / "classification_win_tie_loss.tex")

    # Ranking table
    if "log_loss" in friedman_results:
        ranking_table = generate_ranking_table(
            avg_ranks=friedman_results["log_loss"].avg_ranks,
            metric_name="Log Loss",
            friedman_p=friedman_results["log_loss"].iman_davenport_p_value,
            caption="Algorithm rankings by Log Loss",
            label="tab:rankings-log-loss",
        )
        save_latex_table(ranking_table, TABLES_DIR / "rankings_log_loss.tex")

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
