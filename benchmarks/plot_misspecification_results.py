"""
Plotting and analysis for the Distribution Misspecification Study.

Generates:

**Main paper figures:**
1. Misspecification heatmap (relative CRPS, NxM) — compact, publication-ready
2. Degradation summary dot plot — per-DGP worst-case vs median degradation
3. Critical difference diagrams — per-DGP Nemenyi CD diagrams

**Appendix figures:**
4. Full PIT histogram panel (NxM grid)
5. Coverage comparison bar chart
6. Absolute CRPS heatmap

**Tables:**
7. Full results LaTeX table (CRPS)
8. Statistical tests JSON

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

# Ordering for rows (DGPs) and columns (distributions) in all plots/tables
DGP_ORDER = ["Gaussian", "Skewed", "Count", "Waiting-Time", "Heavy-Tailed", "Multimodal"]
DIST_ORDER = ["Normal", "NIG", "Skew-Normal", "Poisson", "Exponential", "Student-t", "KDE"]

# Home distribution for each DGP (the "correct" one)
HOME_MAP = {
    "Gaussian": "Normal",
    "Skewed": "Skew-Normal",
    "Count": "Poisson",
    "Waiting-Time": "Exponential",
    "Heavy-Tailed": "Student-t",
    "Multimodal": "KDE",
}

# Known incompatible cells (domain mismatch)
INCOMPATIBLE = {
    # Poisson requires non-negative integers
    ("Gaussian", "Poisson"),
    ("Heavy-Tailed", "Poisson"),
    ("Multimodal", "Poisson"),
    ("Skewed", "Poisson"),
    # Exponential requires strictly positive (y > 0)
    ("Gaussian", "Exponential"),
    ("Heavy-Tailed", "Exponential"),
    ("Multimodal", "Exponential"),
    ("Count", "Exponential"),
    ("Skewed", "Exponential"),
}

# Colors
HOME_COLOR = "#2ca02c"  # green for home distribution
MISSPEC_COLOR = "#1f77b4"  # blue for misspecified
INCOMPAT_COLOR = "#d9d9d9"  # light gray for N/A cells


def load_results() -> pd.DataFrame:
    """Load results from parquet."""
    path = RESULTS_DIR / "results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run the study first: {path}")
    return pd.read_parquet(path)


def _build_pivot(df: pd.DataFrame, metric: str, relative: bool = False):
    """Build mean and std pivot tables, optionally relative to HOME."""
    compatible = df[df["compatible"]].copy()
    cell_means = compatible.groupby(["dgp_short", "dist_short"])[metric].mean().reset_index()
    pivot = cell_means.pivot(index="dgp_short", columns="dist_short", values=metric)

    cell_stds = compatible.groupby(["dgp_short", "dist_short"])[metric].std().reset_index()
    pivot_std = cell_stds.pivot(index="dgp_short", columns="dist_short", values=metric)

    # Reorder to canonical order (only include rows/cols that exist)
    pivot = pivot.reindex(index=[d for d in DGP_ORDER if d in pivot.index])
    pivot = pivot.reindex(columns=[d for d in DIST_ORDER if d in pivot.columns])
    pivot_std = pivot_std.reindex(index=pivot.index, columns=pivot.columns)

    if relative:
        for dgp in pivot.index:
            home_dist = HOME_MAP.get(dgp)
            if home_dist and home_dist in pivot.columns and not np.isnan(pivot.loc[dgp, home_dist]):
                home_val = pivot.loc[dgp, home_dist]
                pivot_std.loc[dgp] = pivot_std.loc[dgp] / home_val
                pivot.loc[dgp] = pivot.loc[dgp] / home_val

    return pivot, pivot_std


# =============================================================================
# MAIN PAPER FIGURE 1: Misspecification Heatmap (relative CRPS)
# =============================================================================


def plot_misspecification_heatmap(df: pd.DataFrame, metric: str = "crps", relative: bool = True):
    """Plot heatmap of distributional misspecification impact.

    Args:
        df: Results DataFrame
        metric: Metric to plot (default: crps)
        relative: If True, normalize per row to HOME=1.0
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pivot, pivot_std = _build_pivot(df, metric, relative=relative)

    n_rows, n_cols = pivot.shape

    # Adaptive figure size
    fig, ax = plt.subplots(figsize=(1.4 * n_cols + 1.5, 1.0 * n_rows + 1.0))

    data = pivot.values.copy().astype(float)

    # Mark incompatible cells as NaN
    for i, dgp in enumerate(pivot.index):
        for j, dist in enumerate(pivot.columns):
            if (dgp, dist) in INCOMPATIBLE:
                data[i, j] = np.nan

    # Color scale
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    if relative:
        vmin = max(0.85, vmin)
        vmax = min(3.0, vmax)

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=vmin, vmax=vmax)

    # Annotate cells
    for i, dgp in enumerate(pivot.index):
        for j, dist in enumerate(pivot.columns):
            if (dgp, dist) in INCOMPATIBLE:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    linewidth=0.5,
                    edgecolor="gray",
                    facecolor=INCOMPAT_COLOR,
                    hatch="///",
                    alpha=0.7,
                )
                ax.add_patch(rect)
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="gray", style="italic")
            elif not np.isnan(data[i, j]):
                val = data[i, j]
                std_val = pivot_std.iloc[i, j] if not np.isnan(pivot_std.iloc[i, j]) else 0
                is_home = HOME_MAP.get(dgp) == dist
                fontweight = "bold" if is_home else "normal"
                color = "white" if val > (vmin + vmax) / 2 else "black"
                if relative:
                    txt = f"{val:.2f}\n({std_val:.2f})" if std_val > 0 else f"{val:.2f}"
                else:
                    txt = f"{val:.3f}\n({std_val:.3f})" if std_val > 0 else f"{val:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, fontweight=fontweight, color=color)

    # Labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Assumed Distribution", fontsize=11)
    ax.set_ylabel("True DGP", fontsize=11)

    title = f"Relative {metric.upper()} (HOME = 1.0)" if relative else f"{metric.upper()} by (DGP, Distribution)"
    ax.set_title(title, fontsize=12, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Relative CRPS" if relative else metric.upper(), fontsize=10)

    plt.tight_layout()
    suffix = "relative" if relative else "absolute"
    fig.savefig(PLOTS_DIR / f"misspecification_heatmap_{suffix}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved heatmap ({suffix}) to {PLOTS_DIR}")


# =============================================================================
# MAIN PAPER FIGURE 2: Degradation Summary Dot Plot
# =============================================================================


def plot_degradation_summary(df: pd.DataFrame, metric: str = "crps"):
    """Plot per-DGP degradation dot plot showing how much misspecification costs.

    For each DGP row, shows relative CRPS of each non-home distribution as a dot.
    HOME distribution is marked at 1.0. Gives an at-a-glance answer to
    "how bad does misspecification get?"
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pivot, _ = _build_pivot(df, metric, relative=True)

    # Only DGPs that have data
    dgps = [d for d in DGP_ORDER if d in pivot.index]
    n_dgps = len(dgps)

    fig, ax = plt.subplots(figsize=(6, 0.7 * n_dgps + 1.2))

    y_positions = np.arange(n_dgps)

    for i, dgp in enumerate(dgps):
        home_dist = HOME_MAP.get(dgp)
        row = pivot.loc[dgp]

        for dist in row.index:
            val = row[dist]
            if np.isnan(val) or (dgp, dist) in INCOMPATIBLE:
                continue
            is_home = dist == home_dist
            marker = "D" if is_home else "o"
            color = HOME_COLOR if is_home else MISSPEC_COLOR
            size = 80 if is_home else 50
            zorder = 5 if is_home else 3

            ax.scatter(val, i, marker=marker, color=color, s=size, zorder=zorder, edgecolors="black", linewidths=0.5)
            # Label each dot
            offset_x = 0.01
            ax.annotate(
                dist,
                (val + offset_x, i),
                fontsize=6.5,
                va="center",
                ha="left",
                color="0.3",
            )

    # Reference line at 1.0 (no degradation)
    ax.axvline(x=1.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)

    # Shading for degradation zones
    ax.axvspan(1.0, 1.05, color="green", alpha=0.06)
    ax.axvspan(1.05, 1.15, color="orange", alpha=0.06)
    ax.axvspan(1.15, ax.get_xlim()[1] if ax.get_xlim()[1] > 1.15 else 1.5, color="red", alpha=0.06)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(dgps, fontsize=10)
    ax.set_xlabel("Relative CRPS (HOME = 1.0)", fontsize=11)
    ax.set_title("Distribution Misspecification Cost", fontsize=12, pad=8)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(left=0.9)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=HOME_COLOR,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Home distribution",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MISSPEC_COLOR,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Misspecified",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "degradation_summary.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved degradation summary to {PLOTS_DIR}")


# =============================================================================
# MAIN PAPER FIGURE 3: Critical Difference Diagrams
# =============================================================================


def plot_cd_diagrams(df: pd.DataFrame, metric: str = "crps"):
    """Plot one CD diagram per DGP (stacked vertically in a single figure).

    Shows average ranks of distributions and connects those that are
    not statistically significantly different (Nemenyi test).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    compatible = df[df["compatible"]].copy()

    # Collect CD diagram data per DGP
    cd_data = {}
    for dgp in DGP_ORDER:
        dgp_data = compatible[compatible["dgp_short"] == dgp]
        if dgp_data.empty or metric not in dgp_data.columns:
            continue

        obs = dgp_data.groupby(["seed", "fold", "dist_short"])[metric].mean().reset_index()
        pvt = obs.pivot(index=["seed", "fold"], columns="dist_short", values=metric).dropna()

        if pvt.shape[0] < 5:
            continue

        algorithms = [col for col in DIST_ORDER if col in pvt.columns]
        if len(algorithms) < 3:
            continue

        try:
            matrix = pvt[algorithms].values
            friedman = friedman_test(matrix, algorithms)
            nemenyi = nemenyi_test(matrix, algorithms)
            cd_data[dgp] = {
                "avg_ranks": friedman.avg_ranks,
                "cd": nemenyi.critical_difference,
                "n_obs": pvt.shape[0],
                "algorithms": algorithms,
                "non_sig_pairs": nemenyi.non_significant_pairs,
            }
        except Exception as e:
            logger.warning(f"CD diagram data failed for {dgp}: {e}")

    if not cd_data:
        logger.warning("No CD diagram data available")
        return

    n_panels = len(cd_data)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 1.8 * n_panels + 0.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (dgp, info) in zip(axes, cd_data.items()):
        avg_ranks = info["avg_ranks"]
        cd = info["cd"]
        algorithms = info["algorithms"]
        non_sig = info["non_sig_pairs"]

        # Sort by rank
        sorted_algs = sorted(algorithms, key=lambda a: avg_ranks[a])
        ranks = [avg_ranks[a] for a in sorted_algs]
        n_alg = len(sorted_algs)

        # Axis scale
        rank_min = 0.5
        rank_max = n_alg + 0.5
        ax.set_xlim(rank_min, rank_max)
        ax.set_ylim(-0.5, 1.5)

        # Main rank axis line
        ax.plot([rank_min, rank_max], [0, 0], "k-", linewidth=1.5)

        # CD bar at top
        cd_start = rank_min + 0.2
        ax.plot([cd_start, cd_start + cd], [1.2, 1.2], "k-", linewidth=2)
        ax.plot([cd_start, cd_start], [1.1, 1.3], "k-", linewidth=1.5)
        ax.plot([cd_start + cd, cd_start + cd], [1.1, 1.3], "k-", linewidth=1.5)
        ax.text(cd_start + cd / 2, 1.35, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)

        # Plot each algorithm
        home_dist = HOME_MAP.get(dgp)
        for j, (alg, rank) in enumerate(zip(sorted_algs, ranks)):
            is_home = alg == home_dist
            color = HOME_COLOR if is_home else "0.2"
            marker = "D" if is_home else "o"
            ms = 8 if is_home else 6

            ax.plot(
                rank,
                0,
                marker=marker,
                color=color,
                markersize=ms,
                zorder=5,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
            # Alternate label position above/below
            y_offset = 0.35 if j % 2 == 0 else -0.35
            va = "bottom" if j % 2 == 0 else "top"
            fontweight = "bold" if is_home else "normal"
            ax.annotate(
                f"{alg}\n({rank:.2f})",
                (rank, y_offset),
                ha="center",
                va=va,
                fontsize=7,
                fontweight=fontweight,
                color=color,
            )

        # Draw thick bars connecting non-significantly different pairs
        for pair in non_sig:
            r1 = avg_ranks[pair[0]]
            r2 = avg_ranks[pair[1]]
            y_bar = -0.15
            ax.plot([r1, r2], [y_bar, y_bar], "-", color="0.4", linewidth=3, alpha=0.5, solid_capstyle="round")

        ax.set_title(f"{dgp} DGP (n={info['n_obs']})", fontsize=10, loc="left", pad=2)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(f"Critical Difference Diagrams ({metric.upper()}, lower rank = better)", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "cd_diagrams.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"Saved CD diagrams to {PLOTS_DIR}")


# =============================================================================
# APPENDIX FIGURE: PIT Histogram Panel
# =============================================================================


def plot_pit_panel(df: pd.DataFrame, n_bins: int = 10):
    """Plot full NxM panel of PIT histograms (appendix).

    Each subplot shows the aggregated PIT histogram for one (DGP, distribution) cell.
    A perfectly calibrated model produces a uniform (flat) histogram.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    compatible = df[df["compatible"]].copy()

    pit_cols = [f"pit_bin_{b}" for b in range(n_bins)]
    has_pit_bins = all(col in compatible.columns for col in pit_cols)

    # Determine actual DGPs and dists present
    dgps_present = [
        d
        for d in DGP_ORDER
        if d in compatible["dgp_short"].unique() or any((d, dist) in INCOMPATIBLE for dist in DIST_ORDER)
    ]
    dists_present = [
        d
        for d in DIST_ORDER
        if d in compatible["dist_short"].unique() or any((dgp, d) in INCOMPATIBLE for dgp in DGP_ORDER)
    ]

    n_rows = len(dgps_present)
    n_cols = len(dists_present)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.2 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins

    for i, dgp in enumerate(dgps_present):
        for j, dist in enumerate(dists_present):
            ax = axes[i, j]

            if j == 0:
                ax.set_ylabel(dgp, fontsize=9)
            if i == 0:
                ax.set_title(dist, fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("PIT", fontsize=7)

            if (dgp, dist) in INCOMPATIBLE:
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="gray",
                    style="italic",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 2)
                continue

            cell_data = compatible[(compatible["dgp_short"] == dgp) & (compatible["dist_short"] == dist)]
            if cell_data.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 2)
                continue

            is_home = HOME_MAP.get(dgp) == dist
            color = HOME_COLOR if is_home else MISSPEC_COLOR

            if has_pit_bins and not cell_data[pit_cols[0]].isna().all():
                total_counts = cell_data[pit_cols].sum().values.astype(float)
                total_n = total_counts.sum()
                density = total_counts / (total_n * bin_width) if total_n > 0 else np.ones(n_bins)

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

                if "pit_ks_statistic" in cell_data.columns:
                    ks = cell_data["pit_ks_statistic"].mean()
                    ax.text(
                        0.95,
                        0.95,
                        f"KS={ks:.3f}",
                        ha="right",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )
            else:
                ax.text(0.5, 0.5, "No PIT", ha="center", va="center", transform=ax.transAxes, fontsize=8)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2.5)

            if is_home:
                for spine in ax.spines.values():
                    spine.set_edgecolor(HOME_COLOR)
                    spine.set_linewidth(2.5)

    fig.suptitle("PIT Histograms (flat = well calibrated)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "pit_histogram_panel.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved PIT histogram panel to {PLOTS_DIR}")


# =============================================================================
# APPENDIX FIGURE: Coverage Comparison
# =============================================================================


def plot_coverage_comparison(df: pd.DataFrame):
    """Per-DGP coverage@90 bar chart comparing distributions (appendix)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    compatible = df[df["compatible"]].copy()

    dgps_present = [d for d in DGP_ORDER if d in compatible["dgp_short"].unique()]
    n_dgps = len(dgps_present)

    fig, axes = plt.subplots(1, n_dgps, figsize=(2.8 * n_dgps, 4), sharey=True)
    if n_dgps == 1:
        axes = [axes]

    for idx, dgp in enumerate(dgps_present):
        ax = axes[idx]
        dgp_data = compatible[compatible["dgp_short"] == dgp]

        dists_here = [d for d in DIST_ORDER if (dgp, d) not in INCOMPATIBLE and d in dgp_data["dist_short"].unique()]
        means = []
        stds = []
        colors = []

        for dist in dists_here:
            cell = dgp_data[dgp_data["dist_short"] == dist]
            if cell.empty or "coverage_90" not in cell.columns:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(cell["coverage_90"].mean())
                stds.append(cell["coverage_90"].std())
            is_home = HOME_MAP.get(dgp) == dist
            colors.append(HOME_COLOR if is_home else MISSPEC_COLOR)

        x = np.arange(len(dists_here))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=3, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0.9, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(dists_here, fontsize=7, rotation=45, ha="right")
        ax.set_title(dgp, fontsize=10)
        ax.set_ylim(0.5, 1.05)

        if idx == 0:
            ax.set_ylabel("Coverage@90%", fontsize=10)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "coverage_comparison.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved coverage comparison to {PLOTS_DIR}")


# =============================================================================
# Statistical Tests
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

        obs_data = dgp_data.groupby(["seed", "fold", "dist_short"])["crps"].mean().reset_index()
        pivot = obs_data.pivot(index=["seed", "fold"], columns="dist_short", values="crps").dropna()

        if pivot.shape[0] < 5:
            logger.warning(f"Not enough matched observations for {dgp}: {pivot.shape[0]}")
            continue

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

            if friedman.reject_null:
                nemenyi = nemenyi_test(matrix, algorithms)
                dgp_result["nemenyi_cd"] = nemenyi.critical_difference
                dgp_result["significant_pairs"] = nemenyi.significant_pairs
                dgp_result["non_significant_pairs"] = nemenyi.non_significant_pairs

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
# LaTeX Table
# =============================================================================


def generate_latex_table(df: pd.DataFrame, metric: str = "crps") -> str:
    """Generate a publication-ready LaTeX table."""
    compatible = df[df["compatible"]].copy()

    cell_stats = (
        compatible.groupby(["dgp_short", "dist_short"]).agg(mean=(metric, "mean"), std=(metric, "std")).reset_index()
    )

    # Only include columns/rows that have data
    dists_in_data = [d for d in DIST_ORDER if d in compatible["dist_short"].unique()]
    dgps_in_data = [d for d in DGP_ORDER if d in compatible["dgp_short"].unique()]

    n_dists = len(dists_in_data)
    col_spec = "l" + "c" * n_dists
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(["DGP"] + dists_in_data) + " \\\\",
        "\\midrule",
    ]

    for dgp in dgps_in_data:
        row_vals = []
        best_val = float("inf")

        for dist in dists_in_data:
            if (dgp, dist) in INCOMPATIBLE:
                continue
            match = cell_stats[(cell_stats["dgp_short"] == dgp) & (cell_stats["dist_short"] == dist)]
            if not match.empty:
                val = match["mean"].values[0]
                if not np.isnan(val) and val < best_val:
                    best_val = val

        for dist in dists_in_data:
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

    lines.extend(["\\bottomrule", "\\end{tabular}"])

    table = "\n".join(lines)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / f"misspecification_{metric}.tex", "w") as f:
        f.write(table)
    logger.info(f"Saved LaTeX table to {TABLES_DIR}")

    return table


def generate_tree_complexity_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table of avg tree depth / nodes for each dist x DGP cell."""
    compatible = df[df["compatible"]].copy()

    # Check that the metrics exist
    if "avg_depth" not in compatible.columns or "avg_nodes" not in compatible.columns:
        logger.warning("avg_depth / avg_nodes not found in results — skipping tree complexity table")
        return ""

    cell_stats = (
        compatible.groupby(["dgp_short", "dist_short"])
        .agg(
            depth_mean=("avg_depth", "mean"),
            depth_std=("avg_depth", "std"),
            nodes_mean=("avg_nodes", "mean"),
            nodes_std=("avg_nodes", "std"),
        )
        .reset_index()
    )

    dists_in_data = [d for d in DIST_ORDER if d in compatible["dist_short"].unique()]
    dgps_in_data = [d for d in DGP_ORDER if d in compatible["dgp_short"].unique()]

    n_dists = len(dists_in_data)
    col_spec = "l" + "c" * n_dists
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(["DGP"] + dists_in_data) + " \\\\",
        "\\midrule",
    ]

    for dgp in dgps_in_data:
        row_vals = []
        for dist in dists_in_data:
            if (dgp, dist) in INCOMPATIBLE:
                row_vals.append("---")
                continue
            match = cell_stats[(cell_stats["dgp_short"] == dgp) & (cell_stats["dist_short"] == dist)]
            if match.empty:
                row_vals.append("---")
            else:
                depth = match["depth_mean"].values[0]
                nodes = match["nodes_mean"].values[0]
                row_vals.append(f"${depth:.1f}\\,/\\,{nodes:.0f}$")
        lines.append(" & ".join([dgp] + row_vals) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\multicolumn{"
            + str(n_dists + 1)
            + "}{l}{\\footnotesize Values: avg.\\ depth / avg.\\ nodes per tree.} \\\\",
            "\\end{tabular}",
        ]
    )

    table = "\n".join(lines)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "tree_complexity.tex"
    with open(out_path, "w") as f:
        f.write(table)
    logger.info(f"Saved tree complexity table to {out_path}")

    return table


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import json

    logger.info("Loading results...")
    df = load_results()

    logger.info(f"Loaded {len(df)} rows, {df['compatible'].sum()} compatible")

    # ---- Main paper figures ----
    logger.info("\n--- Main paper figures ---")
    plot_misspecification_heatmap(df, metric="crps", relative=True)
    plot_degradation_summary(df, metric="crps")
    plot_cd_diagrams(df, metric="crps")

    # ---- Appendix figures ----
    logger.info("\n--- Appendix figures ---")
    plot_misspecification_heatmap(df, metric="crps", relative=False)
    plot_pit_panel(df)
    plot_coverage_comparison(df)

    # ---- Tables and tests ----
    logger.info("\n--- Tables and statistical tests ---")
    latex = generate_latex_table(df, metric="crps")
    logger.info(f"\n{latex}")

    complexity_latex = generate_tree_complexity_table(df)
    if complexity_latex:
        logger.info(f"\n{complexity_latex}")

    stat_results = run_statistical_tests(df)
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2, default=str)
    logger.info(f"Saved statistical tests to {RESULTS_DIR / 'statistical_tests.json'}")

    logger.info("\nAll plots and tables generated!")
