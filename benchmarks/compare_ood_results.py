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

from benchmarks.utils.statistical_tests import (
    friedman_test,
    holm_correction,
    nemenyi_test,
    wilcoxon_signed_rank_test,
)

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
        logger.info("Saved uncertainty ratio table")

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
    from benchmarks.utils.style import apply_paper_style

    apply_paper_style()


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


def extract_fold_level_data(all_results: dict[str, Any]) -> pd.DataFrame:
    """Extract fold-level metrics from raw JSON results into a tidy DataFrame.

    Each row is one (dgp, shift, model, fold) observation with all metrics.
    """
    rows = []
    for experiment_key, experiment in all_results.items():
        dgp = experiment["dgp_name"]
        shift = experiment["shift_level"]

        for model_name, model_data in experiment["models"].items():
            for fold_idx, fold_metrics in enumerate(model_data["fold_metrics"]):
                row = {
                    "dgp": dgp,
                    "shift": shift,
                    "model": model_name,
                    "fold": fold_idx,
                }
                for region in ["overall", "overlap", "ood"]:
                    region_data = fold_metrics[region]
                    for metric_key, value in region_data.items():
                        if metric_key == "n_samples":
                            continue
                        row[f"{region}_{metric_key}"] = value

                # uncertainty_ratio may be absent for shift=none (no OOD region)
                ratio = fold_metrics.get("uncertainty_ratio_ood_vs_overlap")
                row["uncertainty_ratio"] = ratio if ratio is not None else np.nan

                rows.append(row)

    return pd.DataFrame(rows)


def run_statistical_tests(
    all_results: dict[str, Any],
    fold_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run proper statistical significance tests using fold-level data.

    Tests performed:
    1. Friedman test per (DGP, shift): are models significantly different on OOD CRPS?
    2. Nemenyi post-hoc: which model pairs differ?
    3. Pairwise Wilcoxon signed-rank: best model vs each other on OOD CRPS
    4. One-sample Wilcoxon: is uncertainty ratio significantly > 1.0?
    """
    if fold_df is None:
        fold_df = extract_fold_level_data(all_results)

    assert not fold_df.empty, "No fold-level data available for statistical tests"

    results: dict[str, Any] = {
        "friedman_tests": [],
        "nemenyi_tests": [],
        "pairwise_wilcoxon": [],
        "uncertainty_ratio_tests": [],
    }

    # -------------------------------------------------------------------------
    # 1 & 2. Friedman + Nemenyi per (DGP, shift) on OOD CRPS
    # -------------------------------------------------------------------------
    for dgp in fold_df["dgp"].unique():
        for shift in fold_df["shift"].unique():
            subset = fold_df[(fold_df["dgp"] == dgp) & (fold_df["shift"] == shift)]
            if subset.empty:
                continue

            # Only include models with OOD CRPS data (probabilistic models)
            all_models = sorted(subset["model"].unique())
            models = []
            for m in all_models:
                m_data = subset[subset["model"] == m]
                if m_data["ood_crps"].notna().sum() > 0:
                    models.append(m)
            if len(models) < 2:
                continue

            # Build metric matrix: rows = folds, columns = models
            folds = sorted(subset["fold"].unique())
            metric_matrix = np.full((len(folds), len(models)), np.nan)
            for j, model in enumerate(models):
                model_data = subset[subset["model"] == model].set_index("fold")
                for i, fold in enumerate(folds):
                    if fold in model_data.index:
                        metric_matrix[i, j] = model_data.loc[fold, "ood_crps"]

            # Skip if too few valid rows
            valid_rows = ~np.isnan(metric_matrix).any(axis=1)
            if valid_rows.sum() < 3:
                continue

            # Friedman test
            friedman_result = friedman_test(metric_matrix, models, lower_is_better=True)
            results["friedman_tests"].append(
                {
                    "dgp": dgp,
                    "shift": shift,
                    "statistic": friedman_result.statistic,
                    "p_value": friedman_result.p_value,
                    "iman_davenport_p": friedman_result.iman_davenport_p_value,
                    "avg_ranks": friedman_result.avg_ranks,
                    "n_folds": friedman_result.n_datasets,
                    "reject_null": friedman_result.reject_null,
                }
            )

            # Nemenyi post-hoc (only if Friedman rejects)
            if friedman_result.reject_null:
                nemenyi_result = nemenyi_test(metric_matrix, models, lower_is_better=True)
                results["nemenyi_tests"].append(
                    {
                        "dgp": dgp,
                        "shift": shift,
                        "critical_difference": nemenyi_result.critical_difference,
                        "significant_pairs": [list(p) for p in nemenyi_result.significant_pairs],
                        "non_significant_pairs": [list(p) for p in nemenyi_result.non_significant_pairs],
                        "avg_ranks": nemenyi_result.avg_ranks,
                    }
                )

            # -----------------------------------------------------------------
            # 3. Pairwise Wilcoxon: best model vs each other
            # -----------------------------------------------------------------
            clean_matrix = metric_matrix[valid_rows]
            mean_crps = np.nanmean(clean_matrix, axis=0)
            best_idx = int(np.argmin(mean_crps))
            best_model = models[best_idx]
            best_values = clean_matrix[:, best_idx]

            p_values_for_holm = []
            wilcoxon_entries = []

            for j, model in enumerate(models):
                if j == best_idx:
                    continue
                other_values = clean_matrix[:, j]
                wresult = wilcoxon_signed_rank_test(best_values, other_values)
                entry = {
                    "dgp": dgp,
                    "shift": shift,
                    "best_model": best_model,
                    "compared_model": model,
                    "best_mean_crps": float(np.mean(best_values)),
                    "other_mean_crps": float(np.mean(other_values)),
                    "wilcoxon_statistic": wresult.statistic,
                    "p_value": wresult.p_value,
                    "effect_size_r": wresult.effect_size_r,
                    "a12": wresult.a12,
                    "n_folds": wresult.n_samples,
                }
                p_values_for_holm.append(wresult.p_value)
                wilcoxon_entries.append(entry)

            # Apply Holm correction for multiple comparisons
            if p_values_for_holm:
                rejections = holm_correction(p_values_for_holm)
                for entry, reject in zip(wilcoxon_entries, rejections):
                    entry["significant_holm"] = reject
                results["pairwise_wilcoxon"].extend(wilcoxon_entries)

    # -------------------------------------------------------------------------
    # 4. One-sample Wilcoxon: uncertainty ratio > 1.0
    # -------------------------------------------------------------------------
    for dgp in fold_df["dgp"].unique():
        for shift in fold_df["shift"].unique():
            if shift == "none":
                continue  # No meaningful OOD region

            subset = fold_df[(fold_df["dgp"] == dgp) & (fold_df["shift"] == shift)]

            for model in sorted(subset["model"].unique()):
                model_data = subset[subset["model"] == model]
                ratios = model_data["uncertainty_ratio"].dropna().values

                if len(ratios) < 5:
                    continue

                # One-sample Wilcoxon: test if median > 1.0
                shifted = ratios - 1.0
                n_positive = int((shifted > 0).sum())
                nonzero = shifted[shifted != 0]

                if len(nonzero) < 2:
                    stat, p_value = np.nan, 1.0
                else:
                    stat, p_value = stats.wilcoxon(nonzero, alternative="greater")

                results["uncertainty_ratio_tests"].append(
                    {
                        "dgp": dgp,
                        "shift": shift,
                        "model": model,
                        "mean_ratio": float(np.mean(ratios)),
                        "median_ratio": float(np.median(ratios)),
                        "std_ratio": float(np.std(ratios)),
                        "n_folds": len(ratios),
                        "n_above_1": n_positive,
                        "wilcoxon_statistic": float(stat) if not np.isnan(stat) else None,
                        "p_value": float(p_value),
                        "significant_at_05": bool(p_value < 0.05),
                    }
                )

    return results


def format_statistical_tests_summary(test_results: dict[str, Any]) -> str:
    """Format statistical test results as a readable markdown summary."""
    lines = ["# Statistical Significance Tests — OOD Study\n"]

    # Friedman
    lines.append("## Friedman Tests (OOD CRPS, per DGP x shift)\n")
    lines.append("Tests whether models differ significantly.\n")
    for ft in test_results["friedman_tests"]:
        sig = "**SIGNIFICANT**" if ft["reject_null"] else "not significant"
        lines.append(
            f"- **{ft['dgp']} / {ft['shift']}**: " f"p={ft['iman_davenport_p']:.4f} ({sig}, n={ft['n_folds']} folds)"
        )
        ranks_str = ", ".join(f"{k}: {v:.2f}" for k, v in ft["avg_ranks"].items())
        lines.append(f"  - Avg ranks: {ranks_str}")

    # Nemenyi
    if test_results["nemenyi_tests"]:
        lines.append("\n## Nemenyi Post-Hoc Tests\n")
        for nt in test_results["nemenyi_tests"]:
            lines.append(f"- **{nt['dgp']} / {nt['shift']}**: " f"CD={nt['critical_difference']:.3f}")
            if nt["significant_pairs"]:
                pairs = [f"{a} vs {b}" for a, b in nt["significant_pairs"]]
                lines.append(f"  - Significant: {', '.join(pairs)}")
            if nt["non_significant_pairs"]:
                pairs = [f"{a} vs {b}" for a, b in nt["non_significant_pairs"]]
                lines.append(f"  - Non-significant: {', '.join(pairs)}")

    # Pairwise Wilcoxon
    if test_results["pairwise_wilcoxon"]:
        lines.append("\n## Pairwise Wilcoxon (best model vs others, Holm-corrected)\n")
        for pw in test_results["pairwise_wilcoxon"]:
            sig = "sig" if pw["significant_holm"] else "n.s."
            lines.append(
                f"- **{pw['dgp']} / {pw['shift']}**: "
                f"{pw['best_model']} (CRPS={pw['best_mean_crps']:.4f}) vs "
                f"{pw['compared_model']} (CRPS={pw['other_mean_crps']:.4f}), "
                f"p={pw['p_value']:.4f}, A12={pw['a12']:.3f} [{sig}]"
            )

    # Uncertainty ratio
    if test_results["uncertainty_ratio_tests"]:
        lines.append("\n## Uncertainty Ratio > 1.0 (one-sided Wilcoxon)\n")
        lines.append("Tests whether models significantly increase uncertainty in OOD regions.\n")
        for ur in test_results["uncertainty_ratio_tests"]:
            sig = "sig" if ur["significant_at_05"] else "n.s."
            lines.append(
                f"- **{ur['dgp']} / {ur['shift']} / {ur['model']}**: "
                f"ratio={ur['mean_ratio']:.3f}+/-{ur['std_ratio']:.3f}, "
                f"{ur['n_above_1']}/{ur['n_folds']} folds > 1, "
                f"p={ur['p_value']:.4f} [{sig}]"
            )

    return "\n".join(lines)


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

    # Statistical tests (using fold-level data)
    logger.info("\nRunning statistical significance tests...")
    # Merge raw results from all environments, combining models from both
    merged_raw = {}
    for env_name, raw_results in all_results.items():
        for key, experiment in raw_results.items():
            if key not in merged_raw:
                merged_raw[key] = experiment
            else:
                # Merge models from this env into existing experiment
                for model_name, model_data in experiment["models"].items():
                    if model_name not in merged_raw[key]["models"]:
                        merged_raw[key]["models"][model_name] = model_data

    fold_df = extract_fold_level_data(merged_raw)
    logger.info(f"Extracted {len(fold_df)} fold-level observations")

    if not fold_df.empty:
        # Save fold-level data for downstream use
        fold_df.to_csv(TABLES_DIR / "fold_level_data.csv", index=False)

        test_results = run_statistical_tests(merged_raw, fold_df)

        # Save as JSON
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        with open(TABLES_DIR / "statistical_tests.json", "w") as f:
            json.dump(test_results, f, indent=2)

        # Save readable summary
        summary_md = format_statistical_tests_summary(test_results)
        (TABLES_DIR / "statistical_tests.md").write_text(summary_md)

        logger.info(f"Saved statistical tests to {TABLES_DIR / 'statistical_tests.json'}")
        logger.info(f"Saved readable summary to {TABLES_DIR / 'statistical_tests.md'}")

        # Log highlights
        n_friedman_sig = sum(1 for ft in test_results["friedman_tests"] if ft["reject_null"])
        n_friedman = len(test_results["friedman_tests"])
        logger.info(f"Friedman: {n_friedman_sig}/{n_friedman} (DGP, shift) cells significant")

        n_ratio_sig = sum(1 for ur in test_results["uncertainty_ratio_tests"] if ur["significant_at_05"])
        n_ratio = len(test_results["uncertainty_ratio_tests"])
        logger.info(f"Uncertainty ratio > 1: {n_ratio_sig}/{n_ratio} cells significant")

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
