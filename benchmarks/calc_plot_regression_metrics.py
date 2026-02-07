"""Main regression benchmark analysis and visualization.

This script produces publication-ready outputs for JMLR:
- Statistical significance tests (Friedman, Nemenyi, Wilcoxon)
- Critical difference diagrams
- LaTeX tables with significance markers
- Win/Tie/Loss analysis
- Per-dataset breakdown tables

Usage:
    python -m benchmarks.calc_plot_regression_metrics
"""

from pathlib import Path

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing YAML result files
RESULTS_DIR = Path("benchmarks/results/custom")

# Output directories
PLOTS_DIR = Path("benchmarks/plots/custom")
TABLES_DIR = Path("benchmarks/results/custom/tables")

# Plot format: "pdf" for vector (best for LaTeX), "png" for raster, or "both"
PLOT_FORMAT = "pdf"

# BDF models to aggregate (best per dataset selected by tuning CRPS)
# Add/remove models here as they become available
BDF_MODELS = [
    "bdf_normalmunormal",
    "bdf_kde",
    "bdf_gammamvlambdapoisson",
    # "bdf_skewnormal",  # uncomment when available
    "bdf_freqstudentt",
]

# Baseline models to compare against
BASELINE_MODELS = [
    "bayesridge_reg",
    "conflgbm",
    "confrf",
    "gaussian_de",
    # "gp_reg",
    "bartpy",  # Only 2 datasets - excluded
    "ngboost_reg",  # add when available
    "qrf",
    "catbunc_reg",
    "knnkde",
]

# Model display names for plots and tables
MODEL_DISPLAY_NAMES = {
    "BDF": "BDF",
    "bayesridge_reg": "BayesRidge",
    "conflgbm": "ConfLGBM",
    "confrf": "ConfRF",
    "gaussian_de": "GaussianDE",
    "bartpy": "BART",
    "ngboost_reg": "NGBoost",
    "gp_reg": "GP",
    "qrf": "QRF",
    "catbunc_reg": "CatBoostUnc",
    "knnkde": "KNNKDE",
}

# Fixed colors for each model (for consistent styling across plots)
MODEL_COLORS = {
    "BDF": "dodgerblue",
    "bayesridge_reg": "orange",
    "conflgbm": "forestgreen",
    "confrf": "limegreen",
    "gaussian_de": "mediumpurple",
    "bartpy": "crimson",
    "ngboost_reg": "goldenrod",
    "gp_reg": "teal",
    "qrf": "darkorange",
    "catbunc_reg": "purple",
    "knnkde": "brown",
}

# Datasets to include (None = all available)
# Set to a list for a representative subset, e.g.:
DATASETS = [
    "abalone_age",
    "bike_sharing",
    "boston_housing",
    "combined_cycle_power_plant",
    "concrete_strength",
    "energy_efficiency",
    "kin8nm",
    "parkinsons_updrs",
    "realestate",
    "superconductor",
    "wine_quality",
]

# Metrics configuration
POINT_METRICS = ["rmse", "mae", "r2"]
PROB_METRICS = ["crps", "nll", "weighted_interval_score", "dawid_sebastiani_score"]
CALIBRATION_METRICS = ["pica", "pit_ks_statistic", "coverage_50", "coverage_90", "coverage_95"]
INTERVAL_METRICS = [
    "ci_width_50",
    "ci_width_90",
    "ci_width_95",
    "interval_score_50",
    "interval_score_90",
    "interval_score_95",
]

# Metric display names for tables
METRIC_DISPLAY = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "crps": "CRPS",
    "nll": "NLL",
    "weighted_interval_score": "WIS",
    "dawid_sebastiani_score": "DSS",
    "pica": "PICA",
    "pit_ks_statistic": "PIT-KS",
    "coverage_90": "Cov@90",
    "coverage_95": "Cov@95",
    "ci_width_90": "Width@90",
    "interval_score_90": "IS@90",
}

# Lower is better for these metrics
LOWER_IS_BETTER = {
    "rmse": True,
    "mae": True,
    "r2": False,  # Higher R² is better
    "crps": True,
    "nll": True,
    "weighted_interval_score": True,
    "dawid_sebastiani_score": True,
    "pica": True,  # Lower PICA = better calibration
    "pit_ks_statistic": True,  # Lower KS = more uniform
    "coverage_90": False,  # Want close to 0.90, but higher is "safer"
    "coverage_95": False,
    "ci_width_90": True,  # Narrower is better (given good coverage)
    "interval_score_90": True,
}


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
    """Run complete regression benchmark analysis."""
    import polars as pl

    from .utils.latex_tables import (
        generate_bdf_selection_table,
        generate_main_results_table,
        generate_per_dataset_table,
        generate_ranking_table,
        generate_win_tie_loss_table,
        save_latex_table,
    )
    from .utils.plotting import (
        plot_calibration_curve,
        plot_coverage_vs_interval_score_grid,
        plot_coverage_vs_sharpness,
        plot_coverage_vs_width_rank,
        plot_critical_difference_diagram,
        plot_metric_comparison_bars,
        plot_metric_scatter,
        plot_pit_histograms,
        plot_rel_to_best,
    )
    from .utils.statistical_tests import (
        friedman_test,
        nemenyi_test,
        pairwise_wilcoxon_tests,
        pairwise_win_tie_loss,
    )
    from .utils.yaml_loader import (
        aggregate_bdf_models,
        build_comparison_dataframe,
        extract_coverage_curves,
        extract_pit_histograms,
        get_metric_matrix,
        load_model_results,
    )

    # Create output directories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REGRESSION BENCHMARK ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load and aggregate BDF results
    # -------------------------------------------------------------------------
    print("\n[1] Loading BDF models and selecting best per dataset...")

    bdf_aggregated, bdf_selection_map = aggregate_bdf_models(
        results_dir=RESULTS_DIR,
        bdf_models=BDF_MODELS,
        selection_metric="crps",
        use_tuning_value=True,
        datasets=DATASETS,
    )

    print("    BDF distribution selection per dataset:")
    for ds, model in sorted(bdf_selection_map.items()):
        print(f"      {ds}: {model}")

    # Save BDF selection table
    bdf_selection_tex = generate_bdf_selection_table(bdf_selection_map)
    save_latex_table(bdf_selection_tex, TABLES_DIR / "bdf_distribution_selection.tex")

    # -------------------------------------------------------------------------
    # 2. Load baseline model results
    # -------------------------------------------------------------------------
    print("\n[2] Loading baseline model results...")

    baseline_results = load_model_results(RESULTS_DIR, BASELINE_MODELS)
    print(f"    Loaded {len(baseline_results)} baseline models: {list(baseline_results.keys())}")

    # Combine BDF (aggregated) with baselines
    all_results = {"BDF": bdf_aggregated, **baseline_results}

    # -------------------------------------------------------------------------
    # 3. Build comparison DataFrame
    # -------------------------------------------------------------------------
    print("\n[3] Building comparison DataFrame...")

    all_metrics = POINT_METRICS + PROB_METRICS + CALIBRATION_METRICS + INTERVAL_METRICS
    df = build_comparison_dataframe(all_results, metrics=all_metrics, datasets=DATASETS)

    # Get actual datasets from data
    datasets = sorted(df.select("dataset").unique().to_series().to_list())
    models = ["BDF"] + [m for m in baseline_results.keys()]

    print(f"    Datasets ({len(datasets)}): {datasets}")
    print(f"    Models ({len(models)}): {models}")

    # -------------------------------------------------------------------------
    # 4. Statistical tests for each key metric
    # -------------------------------------------------------------------------
    print("\n[4] Running statistical significance tests...")

    key_metrics = ["rmse", "crps"]
    friedman_results = {}
    nemenyi_results = {}

    for metric in key_metrics:
        print(f"\n    --- {metric.upper()} ---")

        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)

        # Check for NaN coverage
        nan_count = np.isnan(metric_matrix).sum()
        if nan_count > 0:
            print(f"    Warning: {nan_count} NaN values in {metric} matrix")

        # Friedman test
        try:
            friedman_res = friedman_test(metric_matrix, model_list, lower_is_better=LOWER_IS_BETTER.get(metric, True))
            friedman_results[metric] = friedman_res

            print(f"    Friedman test: χ² = {friedman_res.statistic:.3f}, p = {friedman_res.p_value:.4f}")
            print(
                f"    Iman-Davenport: F = {friedman_res.iman_davenport_statistic:.3f}, "
                f"p = {friedman_res.iman_davenport_p_value:.4f}"
            )
            print(f"    Reject H0: {friedman_res.reject_null}")

            # Average ranks
            print("    Average ranks:")
            for name, rank in sorted(friedman_res.avg_ranks.items(), key=lambda x: x[1]):
                print(f"      {name}: {rank:.2f}")

            # Nemenyi post-hoc test
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
        print(f"\n    --- {metric.upper()} ---")

        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)

        # BDF is index 0
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

    for metric in ["crps"]:  # Focus on CRPS for W/T/L
        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)
        control_idx = model_list.index("BDF") if "BDF" in model_list else 0

        wtl_res = pairwise_win_tie_loss(metric_matrix, model_list, control_idx=control_idx, lower_is_better=True)
        wtl_results[metric] = wtl_res

        print(f"\n    {metric.upper()} Win/Tie/Loss (BDF vs others):")
        for challenger, res in wtl_res.items():
            print(
                f"      vs {challenger}: W={res.wins}, T={res.ties}, L={res.losses} "
                f"(sign test p={res.sign_test_p_value:.3f})"
            )

    # -------------------------------------------------------------------------
    # 7. Generate plots
    # -------------------------------------------------------------------------
    print("\n[7] Generating plots...")

    # Helper to save in both formats if configured
    def save_fig(plot_func, base_name: str, **kwargs):
        base_path = PLOTS_DIR / base_name
        if PLOT_FORMAT in ("pdf", "both"):
            plot_func(save_path=base_path.with_suffix(".pdf"), **kwargs)
        if PLOT_FORMAT in ("png", "both"):
            plot_func(save_path=base_path.with_suffix(".png"), **kwargs)

    # 7a. Critical Difference Diagrams
    for metric in key_metrics:
        if metric in friedman_results:
            friedman_res = friedman_results[metric]
            nemenyi_res = nemenyi_results.get(metric)

            cd = nemenyi_res.critical_difference if nemenyi_res else None

            save_fig(
                plot_critical_difference_diagram,
                f"regression_cd_{metric}",
                avg_ranks=friedman_res.avg_ranks,
                n_datasets=friedman_res.n_datasets,
                cd=cd,
                title=f"Critical Difference Diagram ({metric.upper()})",
                model_display_names=MODEL_DISPLAY_NAMES,
            )

    # 7b. Metric scatter plots (dots per dataset, diamond for mean)

    for metric in ["crps", "rmse"]:
        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)
        save_fig(
            plot_metric_scatter,
            f"regression_scatter_{metric}",
            metric_matrix=metric_matrix,
            models=model_list,
            datasets=ds_list,
            metric_name=metric.upper(),
            lower_is_better=LOWER_IS_BETTER.get(metric, True),
            model_display_names=MODEL_DISPLAY_NAMES,
        )

    # 7c. Relative-to-best plots
    for metric in ["crps", "rmse"]:
        metric_matrix, ds_list, model_list = get_metric_matrix(df, metric, models=models, datasets=datasets)

        # Compute relative to best per dataset
        rel_to_best_data = {}
        for j, model in enumerate(model_list):
            col = metric_matrix[:, j]
            valid = ~np.isnan(col)
            if not valid.any():
                continue

            # Best per dataset (min for each row)
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
                f"regression_rel_to_best_{metric}",
                rel_to_best_data=rel_to_best_data,
                metric=metric,
                model_colors=MODEL_COLORS,
                model_display_names=MODEL_DISPLAY_NAMES,
            )

    # 7d. Coverage vs Interval Score grid
    # Build DataFrame for this plot
    coverage_df_rows = []
    for model in models:
        for ds in datasets:
            model_df = df.filter((df["model"] == model) & (df["dataset"] == ds))
            if model_df.is_empty():
                continue

            cov_row = model_df.filter(pl.col("metric") == "coverage_90")
            iscore_row = model_df.filter(pl.col("metric") == "interval_score_90")

            if not cov_row.is_empty() and not iscore_row.is_empty():
                coverage_df_rows.append(
                    {
                        "model": model,
                        "dataset": ds,
                        "coverage_90": cov_row.select("mean").to_series()[0],
                        "interval_score_90": iscore_row.select("mean").to_series()[0],
                    }
                )

    if coverage_df_rows:
        coverage_df = pl.DataFrame(coverage_df_rows)
        # Only include datasets that have data for most models
        datasets_with_coverage = [ds for ds in datasets if coverage_df.filter(pl.col("dataset") == ds).height >= 3]

        if len(datasets_with_coverage) >= 3:
            save_fig(
                plot_coverage_vs_interval_score_grid,
                "regression_coverage_vs_interval_score",
                df=coverage_df,
                datasets=datasets_with_coverage[:6],  # Max 6 for 2x3 grid
                models=models,
            )

    # 7e. Bar chart for CRPS (overall comparison)
    crps_data = {}
    for model in models:
        model_df = df.filter((df["model"] == model) & (df["metric"] == "crps"))
        if not model_df.is_empty():
            means = model_df.select("mean").to_series().to_list()
            stds = model_df.select("std").to_series().to_list()
            crps_data[model] = (np.mean(means), np.mean(stds))

    if crps_data:
        save_fig(
            plot_metric_comparison_bars,
            "regression_crps_comparison",
            metric_data=crps_data,
            metric_name="CRPS",
            lower_is_better=True,
            model_colors=MODEL_COLORS,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

    # 7f. Calibration curve (empirical vs nominal coverage)
    coverage_curves = extract_coverage_curves(all_results, datasets=datasets)
    if coverage_curves:
        save_fig(
            plot_calibration_curve,
            "regression_calibration_curve",
            coverage_data=coverage_curves,
            models=models,
            model_colors=MODEL_COLORS,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

    # 7g. PIT histograms (one per model, averaged across datasets)
    pit_histograms = extract_pit_histograms(all_results, datasets=datasets)
    if pit_histograms:
        save_fig(
            plot_pit_histograms,
            "regression_pit_histograms",
            pit_data=pit_histograms,
            models=models,
            model_colors=MODEL_COLORS,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

    # 7h. Coverage vs Interval Score Rank at 50/90/95% levels
    save_fig(
        plot_coverage_vs_sharpness,
        "regression_coverage_vs_iscore_rank",
        df=df,
        models=models,
        levels=[50, 90, 95],
        model_colors=MODEL_COLORS,
        model_display_names=MODEL_DISPLAY_NAMES,
    )

    # 7i. Coverage vs Width Rank at 50/90% levels
    save_fig(
        plot_coverage_vs_width_rank,
        "regression_coverage_vs_width_rank",
        df=df,
        models=models,
        levels=[50, 90],
        model_colors=MODEL_COLORS,
        model_display_names=MODEL_DISPLAY_NAMES,
    )

    # -------------------------------------------------------------------------
    # 8. Generate LaTeX tables
    # -------------------------------------------------------------------------
    print("\n[8] Generating LaTeX tables...")

    # Main results table
    main_metrics = ["rmse", "crps", "coverage_90", "pica"]
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
        caption="Regression benchmark results across datasets",
        label="tab:regression-main",
    )
    save_latex_table(main_table, TABLES_DIR / "regression_main_results.tex")

    # Per-dataset table for CRPS
    crps_matrix, ds_list, model_list = get_metric_matrix(df, "crps", models=models, datasets=datasets)
    std_rows = []
    for ds in ds_list:
        row_stds = []
        for model in model_list:
            model_df = df.filter((df["model"] == model) & (df["dataset"] == ds) & (df["metric"] == "crps"))
            if not model_df.is_empty():
                row_stds.append(model_df.select("std").to_series()[0])
            else:
                row_stds.append(np.nan)
        std_rows.append(row_stds)
    std_matrix = np.array(std_rows)

    per_dataset_table = generate_per_dataset_table(
        models=model_list,
        datasets=ds_list,
        metric_matrix=crps_matrix,
        std_matrix=std_matrix,
        metric_name="CRPS",
        caption="CRPS per dataset",
        label="tab:crps-per-dataset",
        lower_is_better=True,
    )
    save_latex_table(per_dataset_table, TABLES_DIR / "crps_per_dataset.tex")

    # Win/Tie/Loss table
    if "crps" in wtl_results:
        wtl_table = generate_win_tie_loss_table(
            control_name="BDF",
            challengers=[m for m in models if m != "BDF"],
            wtl_results=wtl_results["crps"],
            caption="Win/Tie/Loss for CRPS (BDF vs baselines)",
            label="tab:win-tie-loss",
        )
        save_latex_table(wtl_table, TABLES_DIR / "win_tie_loss.tex")

    # Ranking table
    if "crps" in friedman_results:
        ranking_table = generate_ranking_table(
            avg_ranks=friedman_results["crps"].avg_ranks,
            metric_name="CRPS",
            friedman_p=friedman_results["crps"].iman_davenport_p_value,
            caption="Algorithm rankings by CRPS",
            label="tab:rankings-crps",
        )
        save_latex_table(ranking_table, TABLES_DIR / "rankings_crps.tex")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("\nGenerated files:")
    for f in sorted(PLOTS_DIR.glob("regression_*.*")):
        if f.suffix in (".pdf", ".png"):
            print(f"  - {f}")
    for f in sorted(TABLES_DIR.glob("*.tex")):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
