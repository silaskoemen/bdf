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
RESULTS_DIR = Path("benchmarks/results/regression")

# Output directories
PLOTS_DIR = Path("benchmarks/plots/regression")
TABLES_DIR = Path("benchmarks/results/regression/tables")

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
    "bdf_normalmeanstudentt",
]

# Baseline models to compare against
BASELINE_MODELS = [
    "bayesridge_reg",
    "conflgbm",
    "confrf",
    "gaussian_de",
    # "gp_reg",
    "pymc_bart",
    "ngboost_reg",  # add when available
    # "qrf",
    "catbunc_reg",
    "knnkde",
]

# Model display names and colors from shared style
from benchmarks.utils.style import MODEL_COLORS, MODEL_DISPLAY_NAMES

# BDF default distribution for timing comparison (loaded separately alongside BDF selected)
BDF_DEFAULT_DIST = "bdf_normalmunormal"

# Climatological baseline model for CRPSS normalization
CLIMATOLOGICAL_MODEL = "climatological"

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
PROB_METRICS = ["crps", "weighted_interval_score", "dawid_sebastiani_score"]
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
    "weighted_interval_score": True,
    "dawid_sebastiani_score": True,
    "pica": True,  # Lower PICA = better calibration
    "pit_ks_statistic": True,  # Lower KS = more uniform
    "coverage_90": False,  # Want close to 0.90, but higher is "safer"
    "coverage_95": False,
    "ci_width_90": True,  # Narrower is better (given good coverage)
    "interval_score_90": True,
    "crpss": False,  # Higher skill score is better
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
        generate_rel_to_best_table,
        generate_speedup_table,
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
        compute_speedup_table,
        extract_coverage_curves,
        extract_pit_histograms,
        extract_timing_data,
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

    # Load climatological baseline for CRPSS normalization
    climatological_results = load_model_results(RESULTS_DIR, [CLIMATOLOGICAL_MODEL])
    climatological_data = climatological_results.get(CLIMATOLOGICAL_MODEL)
    if climatological_data is None:
        print("    Warning: Climatological baseline not found. CRPSS will not be computed.")
        print(f"    Run: pixi run bench-models model={CLIMATOLOGICAL_MODEL} n_trials=1")
    else:
        print("    Loaded climatological baseline for CRPSS normalization")

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

    # Compute CRPSS (CRPS Skill Score) if climatological baseline available
    crpss_df = None
    if climatological_data is not None:
        # Build climatological CRPS per dataset (mean across folds)
        clim_crps = {}
        for ds_name, ds_data in climatological_data.get("datasets", {}).items():
            crps_vals = ds_data.get("metrics", {}).get("crps", [])
            if crps_vals:
                clim_crps[ds_name] = float(np.mean(crps_vals))

        if clim_crps:
            # Add CRPSS as a derived metric to df: CRPSS = 1 - CRPS / CRPS_clim
            crpss_rows = []
            for model in models:
                for ds in datasets:
                    if ds not in clim_crps:
                        continue
                    clim_val = clim_crps[ds]
                    if np.isclose(clim_val, 0.0):
                        print(
                            f"    Warning: CRPSS undefined for dataset={ds} "
                            f"(near-zero climatological CRPS={clim_val:.6f}). Skipping."
                        )
                        continue
                    model_df = df.filter((df["model"] == model) & (df["dataset"] == ds) & (df["metric"] == "crps"))
                    if model_df.is_empty():
                        continue
                    crps_mean = model_df.select("mean").to_series()[0]
                    crps_std = model_df.select("std").to_series()[0]
                    crpss_mean = 1.0 - crps_mean / clim_val
                    # Propagate std via delta method: std(CRPSS) ≈ std(CRPS) / CRPS_clim
                    crpss_std = crps_std / clim_val
                    crpss_rows.append(
                        {
                            "model": model,
                            "dataset": ds,
                            "metric": "crpss",
                            "fold_values": [],
                            "mean": crpss_mean,
                            "std": crpss_std,
                            "n_folds": 0,
                        }
                    )

            if crpss_rows:
                crpss_df = pl.DataFrame(crpss_rows)
                df = pl.concat([df, crpss_df])
                print(f"    Computed CRPSS for {len(crpss_rows)} model-dataset pairs")

    # -------------------------------------------------------------------------
    # 4. Statistical tests for each key metric
    # -------------------------------------------------------------------------
    print("\n[4] Running statistical significance tests...")

    key_metrics = ["rmse", "crps"] + (["crpss"] if crpss_df is not None else [])
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
    # 5. Pairwise Wilcoxon tests (BDF vs each baseline) with Holm correction
    # -------------------------------------------------------------------------
    print("\n[5] Pairwise Wilcoxon tests (BDF vs baselines, Holm-corrected)...")

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
            print(
                f"    BDF vs {challenger}: p = {res.p_value:.4f}, "
                f"p_adj = {res.adjusted_p_value:.4f}{sig}, A12 = {res.a12:.3f}"
            )

    # CRPSS-based Wilcoxon test (scale-free comparison)
    if crpss_df is not None:
        print("\n    --- CRPSS (Holm-corrected) ---")
        crpss_matrix, ds_list, model_list = get_metric_matrix(df, "crpss", models=models, datasets=datasets)
        control_idx = model_list.index("BDF") if "BDF" in model_list else 0

        wilcoxon_crpss = pairwise_wilcoxon_tests(
            crpss_matrix, model_list, control_idx=control_idx, lower_is_better=False  # Higher CRPSS is better
        )
        wilcoxon_results["crpss"] = wilcoxon_crpss

        for challenger, res in wilcoxon_crpss.items():
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
    # 7. Timing analysis (geometric mean speedup)
    # -------------------------------------------------------------------------
    print("\n[7] Computing timing speedup table...")

    # Load default distribution separately for timing comparison
    bdf_default_results = load_model_results(RESULTS_DIR, [BDF_DEFAULT_DIST])
    bdf_default_data = bdf_default_results.get(BDF_DEFAULT_DIST)
    if bdf_default_data is None:
        raise FileNotFoundError(f"Default BDF distribution '{BDF_DEFAULT_DIST}' not found in {RESULTS_DIR}")

    timing_results = {
        "BDF": bdf_aggregated,
        "BDF_default": bdf_default_data,
        **baseline_results,
    }

    timing_data = extract_timing_data(timing_results, datasets=datasets)

    # Print raw timing summary
    timing_display = {**MODEL_DISPLAY_NAMES, "BDF_default": "BDF (Normal)"}
    for model_name, model_timing in sorted(timing_data.items()):
        ds_count = len(model_timing)
        if ds_count == 0:
            continue
        fit_times = [v["mean_fit_time"] for v in model_timing.values() if "mean_fit_time" in v]
        tune_times = [v["tuning_time"] for v in model_timing.values() if "tuning_time" in v]
        fit_str = f"fit={np.mean(fit_times):.2f}s" if fit_times else "fit=N/A"
        tune_str = f"tune={np.mean(tune_times):.1f}s" if tune_times else "tune=N/A"
        display = timing_display.get(model_name, model_name)
        print(f"    {display}: {ds_count} datasets, {fit_str}, {tune_str}")

    # Compute speedup relative to both BDF variants
    speedup_selected = compute_speedup_table(timing_data, control_model="BDF")
    speedup_normal = compute_speedup_table(timing_data, control_model="BDF_default")

    for label, spd_data in [("BDF (selected)", speedup_selected), ("BDF (Normal)", speedup_normal)]:
        if not spd_data:
            continue
        print(f"\n    Geometric mean speedup ({label} vs baselines):")
        for model, data in sorted(spd_data.items(), key=lambda x: x[1].get("fit_speedup", 0), reverse=True):
            display = timing_display.get(model, model)
            fit_spd = data.get("fit_speedup")
            tune_spd = data.get("tune_speedup")
            fit_str = f"{fit_spd:.2f}x" if fit_spd is not None else "N/A"
            tune_str = f"{tune_spd:.2f}x" if tune_spd is not None else "N/A"
            print(f"      {display}: fit={fit_str}, tune={tune_str}")

    # -------------------------------------------------------------------------
    # 8. Generate plots
    # -------------------------------------------------------------------------
    print("\n[8] Generating plots...")

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

    # 7f. CRPSS scatter and bar chart (skill score vs climatological baseline)
    if crpss_df is not None:
        crpss_matrix, crpss_ds_list, crpss_model_list = get_metric_matrix(df, "crpss", models=models, datasets=datasets)
        save_fig(
            plot_metric_scatter,
            "regression_scatter_crpss",
            metric_matrix=crpss_matrix,
            models=crpss_model_list,
            datasets=crpss_ds_list,
            metric_name="CRPSS",
            lower_is_better=False,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

        crpss_bar_data = {}
        for model in models:
            model_df = df.filter((df["model"] == model) & (df["metric"] == "crpss"))
            if not model_df.is_empty():
                means = model_df.select("mean").to_series().to_list()
                stds = model_df.select("std").to_series().to_list()
                crpss_bar_data[model] = (float(np.mean(means)), float(np.mean(stds)))
        if crpss_bar_data:
            save_fig(
                plot_metric_comparison_bars,
                "regression_crpss_comparison",
                metric_data=crpss_bar_data,
                metric_name="CRPSS",
                lower_is_better=False,
                model_colors=MODEL_COLORS,
                model_display_names=MODEL_DISPLAY_NAMES,
            )

    # 7g. Calibration curve (empirical vs nominal coverage)
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
    # 9. Generate LaTeX tables
    # -------------------------------------------------------------------------
    print("\n[9] Generating LaTeX tables...")

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

    # Per-dataset table for CRPSS
    if crpss_df is not None:
        crpss_matrix, crpss_ds_list, crpss_model_list = get_metric_matrix(df, "crpss", models=models, datasets=datasets)
        crpss_std_rows = []
        for ds in crpss_ds_list:
            row_stds = []
            for model in crpss_model_list:
                mdf = df.filter((df["model"] == model) & (df["dataset"] == ds) & (df["metric"] == "crpss"))
                row_stds.append(mdf.select("std").to_series()[0] if not mdf.is_empty() else np.nan)
            crpss_std_rows.append(row_stds)

        crpss_per_dataset_table = generate_per_dataset_table(
            models=crpss_model_list,
            datasets=crpss_ds_list,
            metric_matrix=crpss_matrix,
            std_matrix=np.array(crpss_std_rows),
            metric_name="CRPSS",
            caption="CRPS Skill Score per dataset (higher is better; baseline = climatological model)",
            label="tab:crpss-per-dataset",
            lower_is_better=False,
        )
        save_latex_table(crpss_per_dataset_table, TABLES_DIR / "crpss_per_dataset.tex")

    # Relative-to-best table for CRPS
    crps_matrix_full, ds_list_full, model_list_full = get_metric_matrix(df, "crps", models=models, datasets=datasets)
    rel_to_best_crps: dict[str, float] = {}
    for j, model in enumerate(model_list_full):
        ratios = []
        for i in range(len(ds_list_full)):
            row = crps_matrix_full[i, :]
            if not np.isnan(crps_matrix_full[i, j]) and np.any(~np.isnan(row)):
                best = np.nanmin(row)
                if best > 0:
                    ratios.append(crps_matrix_full[i, j] / best)
        if ratios:
            rel_to_best_crps[model] = float(np.mean(ratios))

    if rel_to_best_crps:
        rel_to_best_table = generate_rel_to_best_table(
            rel_to_best=rel_to_best_crps,
            metric_name="CRPS",
            model_display_names=MODEL_DISPLAY_NAMES,
            caption="Average CRPS relative to best model per dataset (lower is better; 1.0 = always best)",
            label="tab:rel-to-best-crps",
        )
        save_latex_table(rel_to_best_table, TABLES_DIR / "rel_to_best_crps.tex")

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

    # Ranking tables (CRPS and CRPSS)
    if "crps" in friedman_results:
        ranking_table = generate_ranking_table(
            avg_ranks=friedman_results["crps"].avg_ranks,
            metric_name="CRPS",
            friedman_p=friedman_results["crps"].iman_davenport_p_value,
            caption="Algorithm rankings by CRPS",
            label="tab:rankings-crps",
        )
        save_latex_table(ranking_table, TABLES_DIR / "rankings_crps.tex")

    if "crpss" in friedman_results:
        crpss_ranking_table = generate_ranking_table(
            avg_ranks=friedman_results["crpss"].avg_ranks,
            metric_name="CRPSS",
            friedman_p=friedman_results["crpss"].iman_davenport_p_value,
            caption="Algorithm rankings by CRPS Skill Score (higher is better)",
            label="tab:rankings-crpss",
        )
        save_latex_table(crpss_ranking_table, TABLES_DIR / "rankings_crpss.tex")

    # Speedup tables (selected distribution and default Normal)
    if speedup_selected:
        speedup_sel_tex = generate_speedup_table(
            speedup_data=speedup_selected,
            control_name="BDF",
            model_display_names=timing_display,
            caption="Geometric mean speedup of BDF (best distribution per dataset) relative to baselines",
            label="tab:speedup-selected",
        )
        save_latex_table(speedup_sel_tex, TABLES_DIR / "speedup_selected.tex")

    if speedup_normal:
        speedup_norm_tex = generate_speedup_table(
            speedup_data=speedup_normal,
            control_name="BDF_default",
            model_display_names=timing_display,
            caption="Geometric mean speedup of BDF (Normal) relative to baselines",
            label="tab:speedup-normal",
        )
        save_latex_table(speedup_norm_tex, TABLES_DIR / "speedup_normal.tex")

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
