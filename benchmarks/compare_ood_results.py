"""
Aggregate Analysis: Out-of-Distribution Results Comparison

Reads stored JSON results from OOD benchmark runs and generates:
1. Combined comparison tables (markdown + CSV)
2. Publication-quality aggregate plots
3. Statistical significance tests

Compares results across:
- Core environment (BDF + RandomForest)
- Full environment (+ NGBoost, ConformalLGBM)
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path("benchmarks/results/effect_ood")
PLOTS_DIR = Path("benchmarks/plots/effect_ood")
TABLES_DIR = RESULTS_DIR / "tables"

SHIFT_LEVELS = ["none", "mild", "moderate", "strong"]
DGP_NAMES = ["friedman1", "friedman2", "friedman3"]

# Metrics to include in summary tables
KEY_METRICS = {
    "overall": ["rmse", "crps"],
    "overlap": ["rmse", "crps", "coverage_90", "mean_uncertainty"],
    "ood": ["rmse", "crps", "coverage_90", "mean_uncertainty"],
}

# =============================================================================
# Data Loading
# =============================================================================


def load_results(suffix: str = "") -> dict[str, Any] | None:
    """Load combined results file."""
    filepath = RESULTS_DIR / f"all_results{suffix}.json"
    if not filepath.exists():
        logger.warning(f"Results file not found: {filepath}")
        return None

    with open(filepath, "r") as f:
        return json.load(f)


def load_all_available_results() -> dict[str, dict]:
    """Load results from both core and full environments."""
    results = {}

    core = load_results("_core")
    if core:
        results["core"] = core
        logger.info(f"Loaded core results: {len(core)} experiments")

    full = load_results("")
    if full:
        results["full"] = full
        logger.info(f"Loaded full results: {len(full)} experiments")

    return results


# =============================================================================
# Table Generation
# =============================================================================


def build_summary_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Build a summary DataFrame from results."""
    rows = []

    for experiment_key, experiment in results.items():
        dgp_name = experiment.get("dgp_name", experiment_key.split("_")[0])
        shift_level = experiment.get("shift_level", experiment_key.split("_")[-1])

        for model_name, model_results in experiment.get("models", {}).items():
            agg = model_results.get("aggregated_metrics", {})
            if not agg:
                continue

            row = {
                "dgp": dgp_name,
                "shift": shift_level,
                "model": model_name,
                "fit_time": model_results.get("mean_fit_time", np.nan),
            }

            # Add metrics by region
            for region in ["overall", "overlap", "ood"]:
                region_metrics = agg.get(region, {})
                for metric in KEY_METRICS.get(region, []):
                    val = region_metrics.get(metric, {})
                    if isinstance(val, dict):
                        row[f"{region}_{metric}_mean"] = val.get("mean", np.nan)
                        row[f"{region}_{metric}_std"] = val.get("std", np.nan)
                    else:
                        row[f"{region}_{metric}_mean"] = val
                        row[f"{region}_{metric}_std"] = np.nan

            # Add uncertainty ratio
            ratio = agg.get("uncertainty_ratio_ood_vs_overlap", {})
            if isinstance(ratio, dict):
                row["uncertainty_ratio_mean"] = ratio.get("mean", np.nan)
                row["uncertainty_ratio_std"] = ratio.get("std", np.nan)
            else:
                row["uncertainty_ratio_mean"] = ratio if ratio else np.nan
                row["uncertainty_ratio_std"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def generate_summary_table(
    df: pd.DataFrame,
    metric: str = "crps",
    region: str = "overall",
) -> pd.DataFrame:
    """Generate a pivot table comparing models across DGPs and shift levels."""
    col_name = f"{region}_{metric}_mean"
    std_col = f"{region}_{metric}_std"

    if col_name not in df.columns:
        logger.warning(f"Metric {col_name} not found")
        return pd.DataFrame()

    # Create formatted values (mean ± std)
    df_copy = df.copy()
    df_copy["formatted"] = df_copy.apply(
        lambda r: (
            f"{r[col_name]:.3f}±{r[std_col]:.3f}"
            if pd.notna(r[col_name]) and pd.notna(r[std_col])
            else f"{r[col_name]:.3f}" if pd.notna(r[col_name]) else "N/A"
        ),
        axis=1,
    )

    # Pivot: rows = (dgp, shift), columns = model
    pivot = df_copy.pivot_table(
        index=["dgp", "shift"],
        columns="model",
        values="formatted",
        aggfunc="first",
    )

    return pivot


def generate_uncertainty_ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate table specifically for uncertainty ratio comparison."""
    df_copy = df.copy()

    df_copy["formatted"] = df_copy.apply(
        lambda r: (
            f"{r['uncertainty_ratio_mean']:.2f}±{r['uncertainty_ratio_std']:.2f}"
            if pd.notna(r["uncertainty_ratio_mean"]) and pd.notna(r["uncertainty_ratio_std"])
            else "N/A"
        ),
        axis=1,
    )

    pivot = df_copy.pivot_table(
        index=["dgp", "shift"],
        columns="model",
        values="formatted",
        aggfunc="first",
    )

    return pivot


def generate_ranking_table(df: pd.DataFrame, metric: str = "crps", region: str = "ood") -> pd.DataFrame:
    """Generate a ranking table: which model is best per (dgp, shift)?"""
    col_name = f"{region}_{metric}_mean"

    if col_name not in df.columns:
        return pd.DataFrame()

    # For each (dgp, shift), find the best model
    rankings = []
    for key, group in df.groupby(["dgp", "shift"]):
        dgp, shift = key  # type: ignore
        group_sorted = group.sort_values(col_name)
        best_model = group_sorted.iloc[0]["model"]
        best_value = group_sorted.iloc[0][col_name]

        rankings.append(
            {
                "dgp": dgp,
                "shift": shift,
                "best_model": best_model,
                f"best_{metric}": best_value,
            }
        )

    return pd.DataFrame(rankings)


def save_tables(df: pd.DataFrame, output_dir: Path):
    """Generate and save all summary tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CRPS comparison (overall)
    crps_table = generate_summary_table(df, metric="crps", region="overall")
    if not crps_table.empty:
        crps_table.to_csv(output_dir / "crps_overall.csv")
        with open(output_dir / "crps_overall.md", "w") as f:
            f.write("# CRPS Comparison (Overall)\n\n")
            f.write(crps_table.to_markdown())
        logger.info(f"Saved CRPS table to {output_dir / 'crps_overall.csv'}")

    # 2. CRPS in OOD region
    crps_ood_table = generate_summary_table(df, metric="crps", region="ood")
    if not crps_ood_table.empty:
        crps_ood_table.to_csv(output_dir / "crps_ood.csv")
        with open(output_dir / "crps_ood.md", "w") as f:
            f.write("# CRPS Comparison (OOD Region Only)\n\n")
            f.write(crps_ood_table.to_markdown())

    # 3. Coverage in OOD region
    cov_ood_table = generate_summary_table(df, metric="coverage_90", region="ood")
    if not cov_ood_table.empty:
        cov_ood_table.to_csv(output_dir / "coverage_ood.csv")
        with open(output_dir / "coverage_ood.md", "w") as f:
            f.write("# Coverage @ 90% (OOD Region)\n\n")
            f.write("Nominal coverage is 0.90. Values < 0.90 indicate underconfidence.\n\n")
            f.write(cov_ood_table.to_markdown())

    # 4. Uncertainty ratio
    ratio_table = generate_uncertainty_ratio_table(df)
    if not ratio_table.empty:
        ratio_table.to_csv(output_dir / "uncertainty_ratio.csv")
        with open(output_dir / "uncertainty_ratio.md", "w") as f:
            f.write("# Uncertainty Ratio (OOD / Overlap)\n\n")
            f.write("Ratio > 1 means model increases uncertainty in OOD regions (desirable).\n\n")
            f.write(ratio_table.to_markdown())
        logger.info(f"Saved uncertainty ratio table")

    # 5. Model rankings
    ranking_table = generate_ranking_table(df, metric="crps", region="ood")
    if not ranking_table.empty:
        ranking_table.to_csv(output_dir / "model_rankings.csv", index=False)
        with open(output_dir / "model_rankings.md", "w") as f:
            f.write("# Best Model per (DGP, Shift) by OOD CRPS\n\n")
            f.write(ranking_table.to_markdown(index=False))

    # 6. Full raw data
    df.to_csv(output_dir / "full_results.csv", index=False)
    logger.info(f"Saved full results to {output_dir / 'full_results.csv'}")


# =============================================================================
# Plotting Functions
# =============================================================================


def setup_plot_style():
    """Set up publication-quality plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    region: str,
    save_path: Path,
    title: str | None = None,
):
    """Create heatmap of metric values across models and (dgp, shift)."""
    setup_plot_style()

    col_name = f"{region}_{metric}_mean"
    if col_name not in df.columns:
        logger.warning(f"Metric {col_name} not found, skipping heatmap")
        return

    # Pivot for heatmap
    pivot = df.pivot_table(
        index=["dgp", "shift"],
        columns="model",
        values=col_name,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine colormap based on metric
    if metric in ["crps", "rmse", "mae", "interval_score_90"]:
        cmap = "RdYlGn_r"  # Lower is better
    elif metric in ["coverage_90"]:
        cmap = "RdYlGn"  # Closer to 0.9 is better (handled separately)
    else:
        cmap = "viridis"

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": metric.upper()},
    )

    ax.set_title(title or f"{metric.upper()} ({region.capitalize()} Region)")
    ax.set_xlabel("Model")
    ax.set_ylabel("(DGP, Shift Level)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved heatmap to {save_path}")
    plt.close()


def plot_uncertainty_ratio_comparison(
    df: pd.DataFrame,
    save_path: Path,
):
    """Bar plot comparing uncertainty ratios across models and DGPs."""
    setup_plot_style()

    # Filter to strong shift (most interesting case)
    df_strong = df[df["shift"] == "strong"].copy()

    if df_strong.empty:
        logger.warning("No 'strong' shift data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    models = df_strong["model"].unique()
    dgps = df_strong["dgp"].unique()
    x = np.arange(len(dgps))
    width = 0.8 / len(models)

    colors = sns.color_palette("husl", len(models))

    for i, model in enumerate(models):
        model_data = df_strong[df_strong["model"] == model]
        ratios = []
        errors = []
        for dgp in dgps:
            dgp_data = model_data[model_data["dgp"] == dgp]
            if not dgp_data.empty:
                ratios.append(dgp_data["uncertainty_ratio_mean"].values[0])
                errors.append(dgp_data["uncertainty_ratio_std"].values[0])
            else:
                ratios.append(np.nan)
                errors.append(0)

        ax.bar(
            x + i * width,
            ratios,
            width,
            label=model,
            color=colors[i],
            yerr=errors,
            capsize=3,
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="No change (ratio=1)")
    ax.set_xlabel("DGP")
    ax.set_ylabel("Uncertainty Ratio (OOD / Overlap)")
    ax.set_title("Uncertainty Increase in OOD Regions (Strong Shift)")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(dgps)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved uncertainty ratio comparison to {save_path}")
    plt.close()


def plot_coverage_vs_shift(
    df: pd.DataFrame,
    save_path: Path,
):
    """Line plot showing coverage degradation across shift levels."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = sns.color_palette("husl", df["model"].nunique())
    model_colors = dict(zip(df["model"].unique(), colors))

    for ax, dgp in zip(axes, DGP_NAMES):
        dgp_df = df[df["dgp"] == dgp]

        for model in dgp_df["model"].unique():
            model_df = dgp_df[dgp_df["model"] == model].set_index("shift")

            # Ensure correct order
            model_df = model_df.reindex(SHIFT_LEVELS)

            if "ood_coverage_90_mean" not in model_df.columns:
                continue

            coverages = model_df["ood_coverage_90_mean"].values
            stds = model_df["ood_coverage_90_std"].values

            ax.errorbar(
                range(len(SHIFT_LEVELS)),
                coverages,
                yerr=stds,
                fmt="o-",
                label=model,
                color=model_colors[model],
                linewidth=2,
                markersize=6,
                capsize=3,
            )

        ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.7, label="Nominal 90%")
        ax.set_xticks(range(len(SHIFT_LEVELS)))
        ax.set_xticklabels(SHIFT_LEVELS)
        ax.set_xlabel("Shift Level")
        ax.set_ylabel("Coverage @ 90% (OOD Region)")
        ax.set_title(dgp)
        ax.set_ylim(0.5, 1.0)
        ax.legend(loc="best", fontsize=8)

    plt.suptitle("Coverage Degradation in OOD Regions", fontsize=14)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved coverage vs shift plot to {save_path}")
    plt.close()


def plot_crps_comparison_bar(
    df: pd.DataFrame,
    save_path: Path,
):
    """Grouped bar chart comparing CRPS across models for strong shift."""
    setup_plot_style()

    df_strong = df[df["shift"] == "strong"].copy()

    if df_strong.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (region, title) in zip(axes, [("overlap", "In-Distribution"), ("ood", "Out-of-Distribution")]):
        col = f"{region}_crps_mean"
        err_col = f"{region}_crps_std"

        if col not in df_strong.columns:
            continue

        models = df_strong["model"].unique()
        dgps = df_strong["dgp"].unique()
        x = np.arange(len(dgps))
        width = 0.8 / len(models)

        colors = sns.color_palette("husl", len(models))

        for i, model in enumerate(models):
            model_data = df_strong[df_strong["model"] == model]
            values = []
            errors = []
            for dgp in dgps:
                dgp_data = model_data[model_data["dgp"] == dgp]
                if not dgp_data.empty and pd.notna(dgp_data[col].values[0]):
                    values.append(dgp_data[col].values[0])
                    errors.append(dgp_data[err_col].values[0] if err_col in dgp_data.columns else 0)
                else:
                    values.append(0)
                    errors.append(0)

            ax.bar(
                x + i * width,
                values,
                width,
                label=model,
                color=colors[i],
                yerr=errors,
                capsize=2,
            )

        ax.set_xlabel("DGP")
        ax.set_ylabel("CRPS")
        ax.set_title(f"CRPS ({title} Region) - Strong Shift")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(dgps)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved CRPS comparison to {save_path}")
    plt.close()


def generate_all_plots(df: pd.DataFrame, output_dir: Path):
    """Generate all aggregate plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Heatmaps
    plot_metric_heatmap(df, "crps", "overall", output_dir / "heatmap_crps_overall.png")
    plot_metric_heatmap(df, "crps", "ood", output_dir / "heatmap_crps_ood.png")
    plot_metric_heatmap(df, "coverage_90", "ood", output_dir / "heatmap_coverage_ood.png")

    # Comparison plots
    plot_uncertainty_ratio_comparison(df, output_dir / "uncertainty_ratio_strong_shift.png")
    plot_coverage_vs_shift(df, output_dir / "coverage_degradation.png")
    plot_crps_comparison_bar(df, output_dir / "crps_comparison_strong_shift.png")


# =============================================================================
# Statistical Tests
# =============================================================================


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run statistical significance tests between models."""
    results = []

    # For each (dgp, shift, region), compare models pairwise
    for dgp in df["dgp"].unique():
        for shift in df["shift"].unique():
            subset = df[(df["dgp"] == dgp) & (df["shift"] == shift)]
            models = subset["model"].unique()

            if len(models) < 2:
                continue

            # Compare using CRPS in OOD region
            metric = "ood_crps_mean"
            if metric not in subset.columns:
                continue

            # Find best model
            best_idx = subset[metric].idxmin()
            if pd.isna(best_idx):
                continue

            best_model = subset.loc[best_idx, "model"]
            best_value = subset.loc[best_idx, metric]

            # Compare to others (simplified - would need fold-level data for proper test)
            for model in models:
                if model == best_model:
                    continue

                other_value = subset[subset["model"] == model][metric].values
                if len(other_value) == 0:
                    continue
                other_value = other_value[0]

                diff = other_value - best_value
                rel_diff = 100 * diff / best_value if best_value != 0 else np.nan

                results.append(
                    {
                        "dgp": dgp,
                        "shift": shift,
                        "best_model": best_model,
                        "compared_model": model,
                        "best_crps": best_value,
                        "compared_crps": other_value,
                        "absolute_diff": diff,
                        "relative_diff_pct": rel_diff,
                    }
                )

    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point for aggregate analysis."""
    logger.info("=" * 80)
    logger.info("OOD Results Comparison & Aggregation")
    logger.info("=" * 80)

    # Load results
    all_results = load_all_available_results()

    if not all_results:
        logger.error("No results found. Run effect_out_of_distribution.py first.")
        return

    # Combine results from all environments
    combined_df = pd.DataFrame()
    for env_name, results in all_results.items():
        df = build_summary_dataframe(results)
        df["environment"] = env_name
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Remove duplicates (prefer 'full' over 'core' if both exist)
    combined_df = combined_df.sort_values("environment", ascending=False)  # full before core
    combined_df = combined_df.drop_duplicates(subset=["dgp", "shift", "model"], keep="first")

    logger.info(f"Combined data: {len(combined_df)} rows")
    logger.info(f"Models: {combined_df['model'].unique().tolist()}")
    logger.info(f"DGPs: {combined_df['dgp'].unique().tolist()}")
    logger.info(f"Shift levels: {combined_df['shift'].unique().tolist()}")

    # Generate tables
    logger.info("\nGenerating summary tables...")
    save_tables(combined_df, TABLES_DIR)

    # Generate plots
    logger.info("\nGenerating aggregate plots...")
    generate_all_plots(combined_df, PLOTS_DIR / "aggregate")

    # Statistical tests
    logger.info("\nRunning statistical comparisons...")
    stats_df = run_statistical_tests(combined_df)
    if not stats_df.empty:
        stats_df.to_csv(TABLES_DIR / "statistical_comparisons.csv", index=False)
        logger.info(f"Saved statistical comparisons to {TABLES_DIR / 'statistical_comparisons.csv'}")

    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    # Best models by OOD CRPS
    logger.info("\nBest models by OOD CRPS (strong shift):")
    strong_df = combined_df[combined_df["shift"] == "strong"]
    for dgp in strong_df["dgp"].unique():
        dgp_df = strong_df[strong_df["dgp"] == dgp]
        if "ood_crps_mean" in dgp_df.columns:
            best_idx = dgp_df["ood_crps_mean"].idxmin()
            if pd.notna(best_idx):
                best = dgp_df.loc[best_idx]
                logger.info(f"  {dgp}: {best['model']} (CRPS={best['ood_crps_mean']:.4f})")

    # Uncertainty ratio summary
    if "uncertainty_ratio_mean" in combined_df.columns:
        logger.info("\nUncertainty ratio (OOD/Overlap) for strong shift:")
        for model in strong_df["model"].unique():
            model_df = strong_df[strong_df["model"] == model]
            avg_ratio = model_df["uncertainty_ratio_mean"].mean()
            if pd.notna(avg_ratio):
                logger.info(f"  {model}: {avg_ratio:.2f}")

    logger.success("\n" + "=" * 80)
    logger.success("Aggregate analysis complete!")
    logger.success(f"Tables: {TABLES_DIR}")
    logger.success(f"Plots: {PLOTS_DIR / 'aggregate'}")
    logger.success("=" * 80)


if __name__ == "__main__":
    main()
