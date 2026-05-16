"""Generate publication-quality plots and tables from n_trees experiment results.

Generates:
1. CRPS vs n_trees (2x2 faceted, one panel per dataset)
2. Fit time vs n_trees (2x2 faceted, log-log)
3. Coverage vs n_trees (2x2 faceted, 90% level, nominal reference line)
4. Combined summary (CRPS left, fit time right, all datasets overlaid)
5. LaTeX summary table (CRPS mean +/- std, bold best, plateau annotation)

Usage:
    PYTHONPATH=src python -m benchmarks.plot_n_trees_results
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from benchmarks.utils.latex_tables import format_metric_value
from benchmarks.utils.style import apply_paper_style, get_color

apply_paper_style()

RESULTS_DIR = Path("benchmarks/results/effect_n_trees")
PLOTS_DIR = Path("benchmarks/plots/effect_n_trees")

DATASET_DISPLAY_NAMES = {
    "friedman1": "Friedman 1",
    "friedman2": "Friedman 2",
    "friedman3": "Friedman 3",
    "make_regression": "Linear Regression",
}

DATASET_COLORS = {
    "friedman1": get_color("BDFNormal"),
    "friedman2": get_color("NGBoost"),
    "friedman3": get_color("ConformalRF"),
    "make_regression": get_color("BDFKDE"),
}

DATASET_MARKERS = {
    "friedman1": "o",
    "friedman2": "s",
    "friedman3": "^",
    "make_regression": "D",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_results(filepath: Path = RESULTS_DIR / "effect_n_trees.json") -> dict:
    """Load experiment results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def extract_metric_series(results: dict, dataset: str, metric: str) -> tuple[list[int], list[float], list[float]]:
    """Extract n_trees grid, metric means, and stds for a dataset."""
    ds = results["experiments"][dataset]
    n_trees_grid = ds["n_trees_grid"]
    means = []
    stds = []
    for nt in n_trees_grid:
        key = str(nt) if str(nt) in ds["n_trees_results"] else nt
        agg = ds["n_trees_results"][key]["aggregated_metrics"]
        means.append(agg[metric]["mean"])
        stds.append(agg[metric]["std"])
    return n_trees_grid, means, stds


def extract_fit_time_series(results: dict, dataset: str) -> tuple[list[int], list[float], list[float]]:
    """Extract n_trees grid, mean fit times, and std fit times."""
    ds = results["experiments"][dataset]
    n_trees_grid = ds["n_trees_grid"]
    means = []
    stds = []
    for nt in n_trees_grid:
        key = str(nt) if str(nt) in ds["n_trees_results"] else nt
        means.append(ds["n_trees_results"][key]["mean_fit_time"])
        stds.append(ds["n_trees_results"][key]["std_fit_time"])
    return n_trees_grid, means, stds


# =============================================================================
# Plot 1: CRPS vs n_trees (2x2 faceted)
# =============================================================================


def plot_crps_vs_n_trees(
    results: dict,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """2x2 faceted plot: CRPS vs n_trees, one panel per dataset."""
    datasets = list(results["experiments"].keys())

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes_flat = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        n_trees, means, stds = extract_metric_series(results, dataset, "crps")
        means = np.array(means)
        stds = np.array(stds)

        color = DATASET_COLORS.get(dataset, "#333333")
        marker = DATASET_MARKERS.get(dataset, "o")

        ax.errorbar(n_trees, means, yerr=stds, fmt=f"{marker}-", color=color, markersize=5, capsize=3, zorder=3)
        ax.fill_between(n_trees, means - stds, means + stds, alpha=0.15, color=color)

        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11)
        ax.set_ylabel("CRPS")
        ax.set_xscale("log")

    for idx in range(len(datasets), 4):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Number of Trees")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 2: Fit time vs n_trees (2x2 faceted, log-log)
# =============================================================================


def plot_fit_time_vs_n_trees(
    results: dict,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """2x2 faceted plot: fit time vs n_trees (log-log)."""
    datasets = list(results["experiments"].keys())

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes_flat = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        n_trees, means, stds = extract_fit_time_series(results, dataset)
        means = np.array(means)
        stds = np.array(stds)

        color = DATASET_COLORS.get(dataset, "#333333")
        marker = DATASET_MARKERS.get(dataset, "o")

        ax.errorbar(n_trees, means, yerr=stds, fmt=f"{marker}-", color=color, markersize=5, capsize=3, zorder=3)
        ax.fill_between(n_trees, means - stds, means + stds, alpha=0.15, color=color)

        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11)
        ax.set_ylabel("Fit Time (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")

    for idx in range(len(datasets), 4):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Number of Trees")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 3: Coverage vs n_trees (2x2 faceted, 90% nominal line)
# =============================================================================


def plot_coverage_vs_n_trees(
    results: dict,
    coverage_level: int = 90,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """2x2 faceted plot: empirical coverage vs n_trees with nominal reference line."""
    datasets = list(results["experiments"].keys())
    metric = f"coverage_{coverage_level}"
    nominal = coverage_level / 100.0

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes_flat = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        n_trees, means, stds = extract_metric_series(results, dataset, metric)
        means = np.array(means)
        stds = np.array(stds)

        color = DATASET_COLORS.get(dataset, "#333333")
        marker = DATASET_MARKERS.get(dataset, "o")

        ax.errorbar(n_trees, means, yerr=stds, fmt=f"{marker}-", color=color, markersize=5, capsize=3, zorder=3)
        ax.fill_between(n_trees, means - stds, means + stds, alpha=0.15, color=color)
        ax.axhline(y=nominal, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label=f"Nominal ({nominal:.0%})")

        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), fontsize=11)
        ax.set_ylabel(f"Empirical Coverage ({coverage_level}%)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)

    for idx in range(len(datasets), 4):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Number of Trees")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# Plot 4: Combined summary (CRPS left, fit time right, all datasets overlaid)
# =============================================================================


def plot_combined_summary(
    results: dict,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 4.5),
) -> None:
    """Single figure with two panels: CRPS (left) and fit time (right), all datasets overlaid."""
    datasets = list(results["experiments"].keys())

    fig, (ax_crps, ax_time) = plt.subplots(1, 2, figsize=figsize, sharex=True)

    for dataset in datasets:
        color = DATASET_COLORS.get(dataset, "#333333")
        marker = DATASET_MARKERS.get(dataset, "o")
        label = DATASET_DISPLAY_NAMES.get(dataset, dataset)

        # CRPS panel
        n_trees, crps_means, crps_stds = extract_metric_series(results, dataset, "crps")
        crps_means = np.array(crps_means)
        crps_stds = np.array(crps_stds)

        ax_crps.plot(n_trees, crps_means, color=color, marker=marker, label=label, markersize=5, zorder=3)
        ax_crps.fill_between(n_trees, crps_means - crps_stds, crps_means + crps_stds, alpha=0.12, color=color)

        # Fit time panel
        n_trees, time_means, time_stds = extract_fit_time_series(results, dataset)
        time_means = np.array(time_means)
        time_stds = np.array(time_stds)

        ax_time.plot(n_trees, time_means, color=color, marker=marker, label=label, markersize=5, zorder=3)
        ax_time.fill_between(n_trees, time_means - time_stds, time_means + time_stds, alpha=0.12, color=color)

    ax_crps.set_xlabel("Number of Trees")
    ax_crps.set_ylabel("CRPS")
    ax_crps.set_xscale("log")

    ax_time.set_xlabel("Number of Trees")
    ax_time.set_ylabel("Fit Time (s)")
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")

    # Shared legend
    handles, labels = ax_crps.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(datasets), bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    plt.close(fig)


# =============================================================================
# LaTeX Summary Table
# =============================================================================


def find_plateau(means: list[float], threshold: float = 0.01) -> int | None:
    """Find first n_trees index where CRPS improvement < threshold (1%).

    Returns the index into the means list, or None if no plateau found.
    """
    for i in range(1, len(means)):
        if means[i - 1] == 0:
            continue
        rel_improvement = (means[i - 1] - means[i]) / abs(means[i - 1])
        if rel_improvement < threshold:
            return i
    return None


def generate_summary_table(
    results: dict,
    save_path: Path | None = None,
) -> str:
    """Generate LaTeX table: rows = n_trees, columns = datasets, cells = CRPS mean +/- std.

    Includes fit time row at bottom and plateau annotation.
    """
    datasets = list(results["experiments"].keys())
    n_trees_grid = results["config"]["n_trees_grid"]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{CRPS ($\\downarrow$) as a function of ensemble size.}")
    lines.append("\\label{tab:n-trees-crps}")

    # Column spec: n_trees label + one column per dataset
    col_spec = "l" + "c" * len(datasets)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    ds_headers = " & ".join(DATASET_DISPLAY_NAMES.get(d, d) for d in datasets)
    lines.append(f"$n_{{\\text{{trees}}}}$ & {ds_headers} \\\\")
    lines.append("\\midrule")

    # Find best CRPS per dataset (across all n_trees) and plateau indices
    best_per_dataset: dict[str, float] = {}
    plateau_per_dataset: dict[str, int | None] = {}

    for dataset in datasets:
        n_trees, means, _ = extract_metric_series(results, dataset, "crps")
        best_per_dataset[dataset] = min(means)
        plateau_idx = find_plateau(means)
        plateau_per_dataset[dataset] = n_trees[plateau_idx] if plateau_idx is not None else None

    # CRPS rows
    for nt in n_trees_grid:
        cells = [str(nt)]
        for dataset in datasets:
            ds = results["experiments"][dataset]
            key = str(nt) if str(nt) in ds["n_trees_results"] else nt
            agg = ds["n_trees_results"][key]["aggregated_metrics"]
            mean = agg["crps"]["mean"]
            std = agg["crps"]["std"]
            is_best = abs(mean - best_per_dataset[dataset]) < 1e-10
            cell = format_metric_value(mean, std, is_best=is_best, precision=3)
            # Annotate plateau
            if plateau_per_dataset[dataset] == nt:
                cell = cell + "$^{\\dagger}$"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    # Fit time row
    lines.append("\\midrule")
    cells = ["Fit time (s)"]
    for dataset in datasets:
        ds = results["experiments"][dataset]
        # Use last (largest) n_trees for representative fit time
        last_nt = n_trees_grid[-1]
        key = str(last_nt) if str(last_nt) in ds["n_trees_results"] else last_nt
        mean_time = ds["n_trees_results"][key]["mean_fit_time"]
        std_time = ds["n_trees_results"][key]["std_fit_time"]
        cells.append(format_metric_value(mean_time, std_time, precision=1))
    lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\par\\smallskip\\footnotesize{Bold indicates best CRPS per column. "
        "$^{\\dagger}$ marks the plateau point where CRPS improvement drops below 1\\%. "
        "Fit time shown for $n_{\\text{trees}}=" + str(n_trees_grid[-1]) + "$. "
        "Values are mean $\\pm$ std over 9 evaluation folds.}"
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
    """Generate all plots and tables from n_trees results."""
    logger.info("Loading n_trees experiment results...")
    results = load_results()

    if not results.get("experiments"):
        logger.error("No results found!")
        return

    datasets = list(results["experiments"].keys())
    logger.info(f"Loaded results for {len(datasets)} datasets: {datasets}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: CRPS vs n_trees
    plot_crps_vs_n_trees(results, save_path=PLOTS_DIR / "crps_vs_n_trees.pdf")

    # Plot 2: Fit time vs n_trees
    plot_fit_time_vs_n_trees(results, save_path=PLOTS_DIR / "fit_time_vs_n_trees.pdf")

    # Plot 3: Coverage vs n_trees
    plot_coverage_vs_n_trees(results, coverage_level=90, save_path=PLOTS_DIR / "coverage_vs_n_trees.pdf")

    # Plot 4: Combined summary
    plot_combined_summary(results, save_path=PLOTS_DIR / "combined_summary.pdf")

    # LaTeX table
    generate_summary_table(results, save_path=PLOTS_DIR / "crps_summary_table.tex")

    logger.success("All plots and tables generated.")


if __name__ == "__main__":
    main()
