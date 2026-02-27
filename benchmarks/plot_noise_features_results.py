"""Generate publication-quality plots and tables from noise feature experiment results.

Generates:
1. CRPS degradation curves (2x2 faceted line plot, one panel per dataset)
2. Relative CRPS degradation (normalized to zero-noise baseline)
3. Feature selection rates for all models
4. Per-feature split counts (bar chart for a selected noise level)
5. Coverage degradation (90% coverage vs noise features)
6. Fit time scaling (wall-clock time vs noise features)
7. LaTeX summary table (CRPS across all conditions)

Usage:
    pixi run noise-output
    # or: PYTHONPATH=src python -m benchmarks.plot_noise_features_results
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from benchmarks.utils.latex_tables import format_metric_value
from benchmarks.utils.style import (
    MODEL_MARKERS,
    apply_paper_style,
    get_color,
    get_display_name,
)

apply_paper_style()

RESULTS_DIR = Path("benchmarks/results/effect_noise_features")
PLOTS_DIR = Path("benchmarks/plots/effect_noise_features")

DATASET_DISPLAY_NAMES = {
    "friedman1": "Friedman 1",
    "friedman2": "Friedman 2",
    "friedman3": "Friedman 3",
    "make_regression": "Linear Regression",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_results(filepath: Path = RESULTS_DIR / "effect_noise_features.json") -> dict:
    """Load experiment results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert nested results dict to flat DataFrame.

    Returns DataFrame with columns: dataset, n_noise, model, and metric columns
    (crps_mean, crps_std, mse_mean, mse_std, etc.) plus feature selection columns.
    """
    rows = []
    for exp_key, exp_data in results["experiments"].items():
        if "error" in exp_data:
            continue

        dataset = exp_data["dataset"]
        n_noise = exp_data["n_noise_features"]

        for model_name, model_data in exp_data["models"].items():
            agg = model_data["aggregated_metrics"]
            feat = model_data["aggregated_feature_stats"]

            row = {
                "dataset": dataset,
                "n_noise": n_noise,
                "model": model_name,
                "mean_fit_time": model_data["mean_fit_time"],
                "mean_tuning_time": model_data["tuning_time"],
            }

            # Add all aggregated metrics
            for metric_name, values in agg.items():
                row[f"{metric_name}_mean"] = values["mean"]
                row[f"{metric_name}_std"] = values["std"]

            # Add feature selection stats
            for stat_name in ["informative_selection_rate", "base_selection_rate", "noise_selection_rate"]:
                if stat_name in feat and isinstance(feat[stat_name], dict):
                    row[f"{stat_name}_mean"] = feat[stat_name]["mean"]
                    row[f"{stat_name}_std"] = feat[stat_name]["std"]

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Plot 1: CRPS Degradation
# =============================================================================


def plot_crps_degradation(
    df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot CRPS as a function of noise features, one panel per dataset."""
    datasets = df["dataset"].unique()
    models = df["model"].unique()

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    for idx, dataset in enumerate(sorted(datasets)):
        ax = axes[idx // ncols, idx % ncols]
        ds_df = df[df["dataset"] == dataset]

        for model in sorted(models):
            model_df = ds_df[ds_df["model"] == model].sort_values("n_noise")
            if model_df.empty:
                continue

            noise_counts = model_df["n_noise"].values
            crps_mean = model_df["crps_mean"].values
            crps_std = model_df["crps_std"].values

            color = get_color(model)
            marker = MODEL_MARKERS.get(model, "o")
            label = get_display_name(model)

            ax.plot(noise_counts, crps_mean, color=color, marker=marker, label=label, markersize=5, zorder=3)
            ax.fill_between(noise_counts, crps_mean - crps_std, crps_mean + crps_std, alpha=0.15, color=color)

        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11)
        ax.set_ylabel("CRPS")

    # Hide unused panels
    for idx in range(n_datasets, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # Shared x-label
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Number of Noise Features")

    # Single shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 5), bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("CRPS Degradation with Increasing Noise Features", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 2: Relative CRPS Degradation
# =============================================================================


def plot_crps_relative_degradation(
    df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot CRPS relative to zero-noise baseline: CRPS(d_noise) / CRPS(0)."""
    datasets = df["dataset"].unique()
    models = df["model"].unique()

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    for idx, dataset in enumerate(sorted(datasets)):
        ax = axes[idx // ncols, idx % ncols]
        ds_df = df[df["dataset"] == dataset]

        for model in sorted(models):
            model_df = ds_df[ds_df["model"] == model].sort_values("n_noise")
            if model_df.empty:
                continue

            noise_counts = model_df["n_noise"].values
            crps_mean = model_df["crps_mean"].values

            # Normalize by zero-noise baseline
            baseline_row = model_df[model_df["n_noise"] == 0]
            if baseline_row.empty:
                continue
            baseline_crps = baseline_row["crps_mean"].values[0]
            if baseline_crps == 0:
                continue

            relative_crps = crps_mean / baseline_crps

            color = get_color(model)
            marker = MODEL_MARKERS.get(model, "o")
            label = get_display_name(model)

            ax.plot(noise_counts, relative_crps, color=color, marker=marker, label=label, markersize=5, zorder=3)

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11)
        ax.set_ylabel("Relative CRPS (vs. 0 noise)")

    for idx in range(n_datasets, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Number of Noise Features")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 5), bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Relative CRPS Degradation (Normalized to Zero-Noise Baseline)", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 3: Feature Selection Rates
# =============================================================================


def plot_feature_selection_rates(
    df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Plot informative and noise feature selection rates for all models with split data."""
    if "informative_selection_rate_mean" not in df.columns:
        logger.warning("No feature selection columns in data, skipping plot")
        return

    feature_df = df.dropna(subset=["informative_selection_rate_mean"]).copy()
    # Only keep rows where the model actually produced split counts
    feature_df = feature_df[feature_df["informative_selection_rate_mean"] > 0]

    if feature_df.empty:
        logger.warning("No feature selection data available, skipping plot")
        return

    models = sorted(feature_df["model"].unique())
    datasets = sorted(feature_df["dataset"].unique())

    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 2, figsize=figsize, sharex=True)
    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    for row_idx, dataset in enumerate(datasets):
        ax_info = axes[row_idx, 0]
        ax_noise = axes[row_idx, 1]
        ds_df = feature_df[feature_df["dataset"] == dataset]

        for model in models:
            model_df = ds_df[ds_df["model"] == model].sort_values("n_noise")
            if model_df.empty:
                continue

            noise_counts = model_df["n_noise"].values
            color = get_color(model)
            marker = MODEL_MARKERS.get(model, "o")
            label = get_display_name(model)

            # Informative selection rate
            info_mean = model_df["informative_selection_rate_mean"].values
            info_std = model_df["informative_selection_rate_std"].values
            ax_info.plot(noise_counts, info_mean, color=color, marker=marker, label=label, markersize=5)
            ax_info.fill_between(noise_counts, info_mean - info_std, info_mean + info_std, alpha=0.15, color=color)

            # Noise selection rate (skip n_noise=0)
            noise_mean = model_df["noise_selection_rate_mean"].values
            noise_std = model_df["noise_selection_rate_std"].values
            ax_noise.plot(noise_counts, noise_mean, color=color, marker=marker, label=label, markersize=5)
            ax_noise.fill_between(noise_counts, noise_mean - noise_std, noise_mean + noise_std, alpha=0.15, color=color)

        ds_display = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        ax_info.set_ylabel(ds_display, fontsize=10)
        ax_info.set_ylim(-0.05, 1.05)
        ax_noise.set_ylim(-0.05, 1.05)

    axes[0, 0].set_title("Informative Feature Selection Rate", fontsize=11)
    axes[0, 1].set_title("Noise Feature Selection Rate", fontsize=11)

    for ax in axes[-1]:
        ax.set_xlabel("Number of Noise Features")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Feature Selection Rates", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 4: Per-Feature Split Counts
# =============================================================================


def plot_per_feature_split_counts(
    results: dict,
    noise_level: int = 100,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """Bar chart of per-feature split counts at a specific noise level.

    Shows which specific features (informative vs noise) each model selects.
    One panel per dataset, grouped bars per model.
    """
    datasets = sorted(results["config"]["datasets"])
    informative_features = results["config"]["informative_features"]

    # Collect models that have per_feature_counts at this noise level
    sample_key = f"{datasets[0]}_noise{noise_level}"
    if sample_key not in results["experiments"]:
        logger.warning(f"No data for noise_level={noise_level}, skipping per-feature plot")
        return

    available_models = []
    for model_name, model_data in results["experiments"][sample_key]["models"].items():
        counts = model_data["fold_feature_stats"][0].get("per_feature_counts", [])
        if sum(counts) > 0:
            available_models.append(model_name)

    if not available_models:
        logger.warning("No models with feature selection data, skipping per-feature plot")
        return

    n_datasets = len(datasets)
    n_models = len(available_models)
    fig, axes = plt.subplots(n_datasets, n_models, figsize=figsize, sharey="row")
    if n_datasets == 1:
        axes = axes[np.newaxis, :]
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for row_idx, dataset in enumerate(datasets):
        exp_key = f"{dataset}_noise{noise_level}"
        if exp_key not in results["experiments"]:
            continue

        exp = results["experiments"][exp_key]
        n_informative = informative_features[dataset]
        n_base = exp["n_base_features"]
        n_total = exp["n_total_features"]

        for col_idx, model_name in enumerate(available_models):
            ax = axes[row_idx, col_idx]

            if model_name not in exp["models"]:
                ax.set_visible(False)
                continue

            # Average per_feature_counts across folds
            fold_counts = []
            for fold_stats in exp["models"][model_name]["fold_feature_stats"]:
                pfc = fold_stats.get("per_feature_counts", [])
                if pfc:
                    fold_counts.append(pfc)

            if not fold_counts:
                ax.set_visible(False)
                continue

            mean_counts = np.mean(fold_counts, axis=0)

            # Truncate display: show all informative + first few noise features
            max_display = min(n_total, n_base + 20)
            display_counts = mean_counts[:max_display]
            x = np.arange(max_display)

            # Color: informative=green, other base=gray, noise=red
            colors = []
            for i in range(max_display):
                if i < n_informative:
                    colors.append("#2D7D46")  # green — informative
                elif i < n_base:
                    colors.append("#999999")  # gray — non-informative base
                else:
                    colors.append("#C03028")  # red — noise
            ax.bar(x, display_counts, color=colors, width=0.8, edgecolor="none")

            if max_display < n_total:
                # Show "..." and the remaining noise total
                remaining = mean_counts[max_display:].sum()
                ax.annotate(
                    f"... +{n_total - max_display} noise\n(avg {remaining:.0f} total)",
                    xy=(max_display - 1, 0),
                    fontsize=7,
                    ha="right",
                    va="bottom",
                    color="#999999",
                )

            if row_idx == 0:
                ax.set_title(get_display_name(model_name), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=9)
            ax.tick_params(axis="x", labelsize=6)

    fig.suptitle(f"Per-Feature Split Counts ($d_{{\\mathrm{{noise}}}}={noise_level}$)", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 5: Coverage Degradation
# =============================================================================


def plot_coverage_degradation(
    df: pd.DataFrame,
    coverage_level: int = 90,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot empirical coverage at a given nominal level vs noise features."""
    col = f"coverage_{coverage_level}_mean"
    if col not in df.columns:
        logger.warning(f"Column {col} not found, skipping coverage plot")
        return

    datasets = df["dataset"].unique()
    models = df["model"].unique()

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    nominal = coverage_level / 100.0

    for idx, dataset in enumerate(sorted(datasets)):
        ax = axes[idx // ncols, idx % ncols]
        ds_df = df[df["dataset"] == dataset]

        for model in sorted(models):
            model_df = ds_df[ds_df["model"] == model].sort_values("n_noise")
            if model_df.empty or model_df[col].isna().all():
                continue

            noise_counts = model_df["n_noise"].values
            cov_mean = model_df[col].values

            color = get_color(model)
            marker = MODEL_MARKERS.get(model, "o")
            label = get_display_name(model)

            ax.plot(noise_counts, cov_mean, color=color, marker=marker, label=label, markersize=5, zorder=3)

            # Std band if available
            std_col = f"coverage_{coverage_level}_std"
            if std_col in df.columns:
                cov_std = model_df[std_col].values
                ax.fill_between(noise_counts, cov_mean - cov_std, cov_mean + cov_std, alpha=0.15, color=color)

        ax.axhline(y=nominal, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label=f"Nominal ({nominal:.0%})")
        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11)
        ax.set_ylabel(f"Empirical Coverage ({coverage_level}%)")

    for idx in range(n_datasets, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Number of Noise Features")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models) + 1, 6), bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{coverage_level}% Prediction Interval Coverage vs. Noise Features", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 6: Fit Time Scaling
# =============================================================================


def plot_fit_time_scaling(
    df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Plot mean fit time as a function of noise features, one panel per dataset."""
    datasets = df["dataset"].unique()
    models = df["model"].unique()

    n_datasets = len(datasets)
    ncols = 2
    nrows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    for idx, dataset in enumerate(sorted(datasets)):
        ax = axes[idx // ncols, idx % ncols]
        ds_df = df[df["dataset"] == dataset]

        for model in sorted(models):
            model_df = ds_df[ds_df["model"] == model].sort_values("n_noise")
            if model_df.empty:
                continue

            noise_counts = model_df["n_noise"].values
            fit_time = model_df["mean_fit_time"].values

            color = get_color(model)
            marker = MODEL_MARKERS.get(model, "o")
            label = get_display_name(model)

            ax.plot(noise_counts, fit_time, color=color, marker=marker, label=label, markersize=5, zorder=3)

        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11)
        ax.set_ylabel("Fit Time (s)")

    for idx in range(n_datasets, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Number of Noise Features")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 5), bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fit Time Scaling with Noise Features", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# LaTeX Summary Table
# =============================================================================


def generate_summary_table(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> str:
    """Generate LaTeX table: CRPS mean +/- std, grouped by dataset, rows = noise levels.

    Compact table suitable for appendix, with one sub-table per dataset.
    """
    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())
    noise_levels = sorted(df["n_noise"].unique())

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{CRPS ($\\downarrow$) across datasets and noise feature counts.}")
    lines.append("\\label{tab:noise-features-crps}")
    lines.append("\\resizebox{\\textwidth}{!}{%")

    # Column spec: noise level + one column per model
    col_spec = "l" + "c" * len(models)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    model_headers = " & ".join(get_display_name(m) for m in models)
    lines.append(f"$d_{{\\text{{noise}}}}$ & {model_headers} \\\\")
    lines.append("\\midrule")

    for ds_idx, dataset in enumerate(datasets):
        ds_display = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        lines.append(f"\\multicolumn{{{len(models) + 1}}}{{l}}{{\\textit{{{ds_display}}}}} \\\\")

        ds_df = df[df["dataset"] == dataset]

        for n_noise in noise_levels:
            row_df = ds_df[ds_df["n_noise"] == n_noise]
            if row_df.empty:
                continue

            # Find best model for this row
            best_crps = float("inf")
            best_model = None
            for model in models:
                model_row = row_df[row_df["model"] == model]
                if not model_row.empty:
                    crps_val = model_row["crps_mean"].values[0]
                    if crps_val < best_crps:
                        best_crps = crps_val
                        best_model = model

            # Format cells
            cells = [str(n_noise)]
            for model in models:
                model_row = row_df[row_df["model"] == model]
                if model_row.empty:
                    cells.append("---")
                else:
                    mean = model_row["crps_mean"].values[0]
                    std = model_row["crps_std"].values[0]
                    is_best = model == best_model
                    cells.append(format_metric_value(mean, std, is_best=is_best, precision=3))

            lines.append(" & ".join(cells) + " \\\\")

        if ds_idx < len(datasets) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}%")
    lines.append(
        "\\par\\smallskip\\footnotesize{Bold indicates best CRPS per row. Values are mean ± std over 9 evaluation folds (tuned on fold 0).}"
    )
    lines.append("\\end{table}")

    table_str = "\n".join(lines)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(table_str)
        logger.info(f"Saved {save_path}")

    return table_str


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all plots and tables from noise feature results."""
    logger.info("Loading noise feature experiment results...")
    results = load_results()
    df = results_to_dataframe(results)

    if df.empty:
        logger.error("No results found!")
        return

    logger.info(f"Loaded {len(df)} rows: {df['model'].nunique()} models, {df['dataset'].nunique()} datasets")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: CRPS degradation
    plot_crps_degradation(df, save_path=PLOTS_DIR / "crps_degradation.pdf")

    # Plot 2: Relative CRPS degradation
    plot_crps_relative_degradation(df, save_path=PLOTS_DIR / "crps_relative_degradation.pdf")

    # Plot 3: Feature selection rates
    plot_feature_selection_rates(df, save_path=PLOTS_DIR / "feature_selection_rates.pdf")

    # Plot 4: Per-feature split counts at d_noise=100
    plot_per_feature_split_counts(results, noise_level=100, save_path=PLOTS_DIR / "per_feature_counts_noise100.pdf")

    # Plot 5: Coverage degradation (90% PI)
    plot_coverage_degradation(df, coverage_level=90, save_path=PLOTS_DIR / "coverage_degradation_90.pdf")

    # Plot 6: Fit time scaling
    plot_fit_time_scaling(df, save_path=PLOTS_DIR / "fit_time_scaling.pdf")

    # LaTeX table
    generate_summary_table(df, save_path=PLOTS_DIR / "crps_summary_table.tex")

    logger.success("All plots and tables generated.")


if __name__ == "__main__":
    main()
