"""
Plotting and analysis for the Distribution Misspecification Study.

Generates:
1. Misspecification heatmap (relative CRPS, 5x5)
2. PIT histogram panel (5x5 grid)
3. Full results LaTeX table
4. Statistical tests (Friedman + post-hoc per DGP row)

Usage:
    python -m benchmarks.plot_misspecification_results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.patches import Rectangle

from .utils.statistical_tests import (
    friedman_test,
    nemenyi_test,
    wilcoxon_signed_rank_test,
)

RESULTS_DIR = Path("benchmarks/results/dist_misspecification")
PLOTS_DIR = Path("benchmarks/plots/dist_misspecification")
TABLES_DIR = Path("benchmarks/tables/dist_misspecification")

DGP_ORDER = ["Gaussian", "Count", "Waiting-Time", "Heavy-Tailed", "Multimodal"]
DIST_ORDER = ["Normal", "Poisson", "Exponential", "Student-t", "KDE"]

# Home distribution for each DGP
HOME_MAP = {
    "Gaussian": "Normal",
    "Count": "Poisson",
    "Waiting-Time": "Exponential",
    "Heavy-Tailed": "Student-t",
    "Multimodal": "KDE",
}

# Known incompatible cells (domain mismatch)
INCOMPATIBLE = {
    ("Gaussian", "Poisson"),  # negative values
    ("Gaussian", "Exponential"),  # negative values
    ("Heavy-Tailed", "Poisson"),  # negative values
    ("Heavy-Tailed", "Exponential"),  # negative values
    ("Multimodal", "Poisson"),  # negative values
    ("Multimodal", "Exponential"),  # negative values
    ("Count", "Exponential"),  # Poisson produces zeros, Exponential requires y > 0
}


def load_results() -> pd.DataFrame:
    """Load results from parquet."""
    path = RESULTS_DIR / "results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run the study first: {path}")
    return pd.read_parquet(path)


# =============================================================================
# 1. Misspecification Heatmap
# =============================================================================


def plot_misspecification_heatmap(df: pd.DataFrame, metric: str = "crps", relative: bool = True):
    """Plot 5x5 heatmap of distributional misspecification impact.

    Args:
        df: Results DataFrame
        metric: Metric to plot (default: crps)
        relative: If True, normalize per row to HOME=1.0
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    compatible = df[df["compatible"]].copy()

    # Compute mean metric per (DGP, dist) cell
    cell_means = compatible.groupby(["dgp_short", "dist_short"])[metric].mean().reset_index()
    pivot = cell_means.pivot(index="dgp_short", columns="dist_short", values=metric)

    # Reorder
    pivot = pivot.reindex(index=[d for d in DGP_ORDER if d in pivot.index])
    pivot = pivot.reindex(columns=[d for d in DIST_ORDER if d in pivot.columns])

    # Compute std for annotations
    cell_stds = compatible.groupby(["dgp_short", "dist_short"])[metric].std().reset_index()
    pivot_std = cell_stds.pivot(index="dgp_short", columns="dist_short", values=metric)
    pivot_std = pivot_std.reindex(index=pivot.index, columns=pivot.columns)

    if relative:
        # Normalize each row by its HOME distribution
        for dgp in pivot.index:
            home_dist = HOME_MAP.get(dgp)
            if home_dist and home_dist in pivot.columns and not np.isnan(pivot.loc[dgp, home_dist]):
                home_val = pivot.loc[dgp, home_dist]
                pivot_std.loc[dgp] = pivot_std.loc[dgp] / home_val
                pivot.loc[dgp] = pivot.loc[dgp] / home_val

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap data with NaN for incompatible cells
    data = pivot.values.copy().astype(float)
    mask = np.zeros_like(data, dtype=bool)

    for i, dgp in enumerate(pivot.index):
        for j, dist in enumerate(pivot.columns):
            if (dgp, dist) in INCOMPATIBLE:
                mask[i, j] = True
                data[i, j] = np.nan

    # Color map: lower is better for CRPS
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    if relative:
        vmin = max(0.8, vmin)
        vmax = min(5.0, vmax)

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

    # Hatch incompatible cells
    for i, dgp in enumerate(pivot.index):
        for j, dist in enumerate(pivot.columns):
            if (dgp, dist) in INCOMPATIBLE:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="gray",
                    facecolor="lightgray",
                    hatch="///",
                    alpha=0.7,
                )
                ax.add_patch(rect)
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray", style="italic")
            elif not np.isnan(data[i, j]):
                # Annotate with mean +/- std
                val = data[i, j]
                std_val = pivot_std.iloc[i, j] if not np.isnan(pivot_std.iloc[i, j]) else 0
                is_home = HOME_MAP.get(dgp) == dist
                fontweight = "bold" if is_home else "normal"
                color = "white" if val > (vmin + vmax) / 2 else "black"
                if relative:
                    txt = f"{val:.2f}\n({std_val:.2f})" if std_val > 0 else f"{val:.2f}"
                else:
                    txt = f"{val:.3f}\n({std_val:.3f})" if std_val > 0 else f"{val:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, fontweight=fontweight, color=color)

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_xlabel("Assumed Distribution", fontsize=12)
    ax.set_ylabel("True DGP", fontsize=12)

    title = f"Relative {metric.upper()} (HOME = 1.0)" if relative else f"{metric.upper()} by (DGP, Distribution)"
    ax.set_title(title, fontsize=13, pad=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Relative CRPS" if relative else metric.upper(), fontsize=11)

    plt.tight_layout()

    suffix = "relative" if relative else "absolute"
    fig.savefig(PLOTS_DIR / f"misspecification_heatmap_{suffix}.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(PLOTS_DIR / f"misspecification_heatmap_{suffix}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to {PLOTS_DIR}")


# =============================================================================
# 2. PIT Histogram Panel
# =============================================================================


def plot_pit_panel(df: pd.DataFrame, n_bins: int = 10):
    """Plot 5x5 panel of PIT histograms.

    Each subplot shows the aggregated PIT histogram for one (DGP, distribution) cell.
    A perfectly calibrated model produces a uniform (flat) histogram.
    Characteristic deviations:
    - U-shape: underdispersed (too narrow intervals)
    - Inverse-U: overdispersed (too wide intervals)
    - Left-skewed: systematic overprediction
    - Right-skewed: systematic underprediction
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    compatible = df[df["compatible"]].copy()

    # Check if PIT bin columns exist
    pit_cols = [f"pit_bin_{b}" for b in range(n_bins)]
    has_pit_bins = all(col in compatible.columns for col in pit_cols)

    fig, axes = plt.subplots(
        len(DGP_ORDER),
        len(DIST_ORDER),
        figsize=(14, 12),
        sharex=True,
        sharey=True,
    )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins

    for i, dgp in enumerate(DGP_ORDER):
        for j, dist in enumerate(DIST_ORDER):
            ax = axes[i, j]

            # Always set row/column labels regardless of compatibility
            if j == 0:
                ax.set_ylabel(dgp, fontsize=10)
            if i == 0:
                ax.set_title(dist, fontsize=10)
            if i == len(DGP_ORDER) - 1:
                ax.set_xlabel("PIT", fontsize=8)

            if (dgp, dist) in INCOMPATIBLE:
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                    color="gray",
                    style="italic",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 2)
                continue

            cell_data = compatible[(compatible["dgp_short"] == dgp) & (compatible["dist_short"] == dist)]

            if cell_data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 2)
                continue

            is_home = HOME_MAP.get(dgp) == dist
            color = "forestgreen" if is_home else "steelblue"

            if has_pit_bins and not cell_data[pit_cols[0]].isna().all():
                # Sum PIT bin counts across all folds and seeds
                total_counts = cell_data[pit_cols].sum().values.astype(float)
                # Normalize to density (so uniform = 1.0 line)
                total_n = total_counts.sum()
                if total_n > 0:
                    density = total_counts / (total_n * bin_width)
                else:
                    density = np.ones(n_bins)

                ax.bar(
                    bin_centers,
                    density,
                    width=bin_width * 0.9,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.6)

                # Annotate with KS statistic
                if "pit_ks_statistic" in cell_data.columns:
                    ks = cell_data["pit_ks_statistic"].mean()
                    ax.text(
                        0.95,
                        0.95,
                        f"KS={ks:.3f}",
                        ha="right",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )
            elif "pit_ks_statistic" in cell_data.columns:
                # Fallback: just show KS value as text
                ks_mean = cell_data["pit_ks_statistic"].mean()
                ks_std = cell_data["pit_ks_statistic"].std()
                ax.text(
                    0.5,
                    0.5,
                    f"KS = {ks_mean:.3f}\n({ks_std:.3f})",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.6)
            else:
                ax.text(0.5, 0.5, "No PIT data", ha="center", va="center", transform=ax.transAxes, fontsize=9)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2.5)

            if is_home:
                for spine in ax.spines.values():
                    spine.set_edgecolor("forestgreen")
                    spine.set_linewidth(2.5)

    fig.suptitle("PIT Histograms (flat = well calibrated)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "pit_histogram_panel.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(PLOTS_DIR / "pit_histogram_panel.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved PIT histogram panel to {PLOTS_DIR}")


# =============================================================================
# 3. Coverage Comparison
# =============================================================================


def plot_coverage_comparison(df: pd.DataFrame):
    """Per-DGP coverage@90 bar chart comparing distributions."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    compatible = df[df["compatible"]].copy()

    fig, axes = plt.subplots(1, len(DGP_ORDER), figsize=(16, 4), sharey=True)

    for idx, dgp in enumerate(DGP_ORDER):
        ax = axes[idx]
        dgp_data = compatible[compatible["dgp_short"] == dgp]

        dists_present = [d for d in DIST_ORDER if (dgp, d) not in INCOMPATIBLE]
        means = []
        stds = []
        colors = []

        for dist in dists_present:
            cell = dgp_data[dgp_data["dist_short"] == dist]
            if cell.empty or "coverage_90" not in cell.columns:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(cell["coverage_90"].mean())
                stds.append(cell["coverage_90"].std())
            is_home = HOME_MAP.get(dgp) == dist
            colors.append("forestgreen" if is_home else "steelblue")

        x = np.arange(len(dists_present))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=3, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0.9, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Nominal 90%")
        ax.set_xticks(x)
        ax.set_xticklabels(dists_present, fontsize=8, rotation=45, ha="right")
        ax.set_title(dgp, fontsize=11)
        ax.set_ylim(0.5, 1.05)

        if idx == 0:
            ax.set_ylabel("Coverage@90%", fontsize=10)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "coverage_comparison.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(PLOTS_DIR / "coverage_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved coverage comparison to {PLOTS_DIR}")


# =============================================================================
# 4. Statistical Tests
# =============================================================================


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run Friedman + post-hoc tests per DGP row."""
    compatible = df[df["compatible"]].copy()
    results = {}

    for dgp in DGP_ORDER:
        dgp_data = compatible[compatible["dgp_short"] == dgp]
        if dgp_data.empty or "crps" not in dgp_data.columns:
            continue

        dists_present = sorted(dgp_data["dist_short"].unique())
        if len(dists_present) < 3:
            continue

        # Build matrix: rows = observations (seed x fold), columns = distributions
        # Each observation needs matched values across all distributions
        obs_data = dgp_data.groupby(["seed", "fold", "dist_short"])["crps"].mean().reset_index()
        pivot = obs_data.pivot(index=["seed", "fold"], columns="dist_short", values="crps").dropna()

        if pivot.shape[0] < 5:
            logger.warning(f"Not enough matched observations for {dgp}: {pivot.shape[0]}")
            continue

        # Friedman test
        try:
            algorithms = [col for col in DIST_ORDER if col in pivot.columns]
            matrix = pivot[algorithms].values
            friedman = friedman_test(matrix, algorithms)
            dgp_result = {
                "friedman_statistic": friedman.statistic,
                "friedman_p_value": friedman.p_value,
                "reject_null": friedman.reject_null,
                "avg_ranks": friedman.avg_ranks,
                "n_observations": int(pivot.shape[0]),
            }

            # Nemenyi post-hoc if significant
            if friedman.reject_null:
                nemenyi = nemenyi_test(matrix, algorithms)
                dgp_result["nemenyi_cd"] = nemenyi.critical_difference
                dgp_result["significant_pairs"] = nemenyi.significant_pairs
                dgp_result["non_significant_pairs"] = nemenyi.non_significant_pairs

            # Wilcoxon: HOME vs each other distribution
            home_dist = HOME_MAP.get(dgp)
            if home_dist and home_dist in pivot.columns:
                pairwise = {}
                for dist in algorithms:
                    if dist == home_dist:
                        continue
                    try:
                        wilcoxon = wilcoxon_signed_rank_test(pivot[home_dist].values, pivot[dist].values)
                        pairwise[f"{home_dist}_vs_{dist}"] = {
                            "p_value": wilcoxon.p_value,
                            "effect_size_r": wilcoxon.effect_size_r,
                            "reject_null": wilcoxon.reject_null,
                        }
                    except Exception as e:
                        logger.debug(f"Wilcoxon failed for {home_dist} vs {dist}: {e}")
                dgp_result["pairwise_vs_home"] = pairwise

            results[dgp] = dgp_result

        except Exception as e:
            logger.warning(f"Statistical tests failed for {dgp}: {e}")

    return results


# =============================================================================
# 5. LaTeX Table
# =============================================================================


def generate_latex_table(df: pd.DataFrame, metric: str = "crps") -> str:
    """Generate a publication-ready LaTeX table."""
    compatible = df[df["compatible"]].copy()

    cell_stats = (
        compatible.groupby(["dgp_short", "dist_short"])
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
        )
        .reset_index()
    )

    # Start building LaTeX
    n_dists = len(DIST_ORDER)
    col_spec = "l" + "c" * n_dists
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(["DGP"] + DIST_ORDER) + " \\\\",
        "\\midrule",
    ]

    for dgp in DGP_ORDER:
        row_vals = []
        best_val = float("inf")

        # Find best for this row
        for dist in DIST_ORDER:
            if (dgp, dist) in INCOMPATIBLE:
                continue
            match = cell_stats[(cell_stats["dgp_short"] == dgp) & (cell_stats["dist_short"] == dist)]
            if not match.empty:
                val = match["mean"].values[0]
                if not np.isnan(val) and val < best_val:
                    best_val = val

        for dist in DIST_ORDER:
            if (dgp, dist) in INCOMPATIBLE:
                row_vals.append("---")
                continue
            match = cell_stats[(cell_stats["dgp_short"] == dgp) & (cell_stats["dist_short"] == dist)]
            if match.empty:
                row_vals.append("---")
            else:
                mean = match["mean"].values[0]
                std = match["std"].values[0]
                is_best = abs(mean - best_val) < 1e-6
                is_home = HOME_MAP.get(dgp) == dist
                formatted = f"{mean:.3f} \\pm {std:.3f}"
                if is_best:
                    formatted = f"\\textbf{{{formatted}}}"
                if is_home:
                    formatted = f"{formatted}$^{{\\star}}$"
                row_vals.append(f"${formatted}$")

        lines.append(" & ".join([dgp] + row_vals) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    table = "\n".join(lines)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / f"misspecification_{metric}.tex", "w") as f:
        f.write(table)
    logger.info(f"Saved LaTeX table to {TABLES_DIR}")

    return table


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import json

    logger.info("Loading results...")
    df = load_results()

    logger.info(f"Loaded {len(df)} rows, {df['compatible'].sum()} compatible")

    # Generate all outputs
    plot_misspecification_heatmap(df, metric="crps", relative=True)
    plot_misspecification_heatmap(df, metric="crps", relative=False)
    plot_pit_panel(df)
    plot_coverage_comparison(df)

    latex = generate_latex_table(df, metric="crps")
    logger.info(f"\n{latex}")

    stat_results = run_statistical_tests(df)
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2, default=str)
    logger.info(f"Saved statistical tests to {RESULTS_DIR / 'statistical_tests.json'}")

    logger.info("\nAll plots and tables generated!")
