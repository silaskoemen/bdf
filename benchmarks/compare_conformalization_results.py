"""Generate comparison plots from conformalization study results.

Workflow:
    1. pixi run -e default python -m benchmarks.conformalization_study
       → benchmarks/results/conformalization/*_conformalization_core.json   (BDF, ConfBDF)

    2. pixi run -e bench-models python -m benchmarks.conformalization_study
       → benchmarks/results/conformalization/*_conformalization.json        (CatBoost variants, ConfRF)
       (also includes BDF if importable in bench-models env)

    3. pixi run python -m benchmarks.compare_conformalization_results
       → merges both, generates plots in benchmarks/plots/conformalization/summary/

Plots generated:
    - Spread-skill diagram per DGP (all models on one figure)
    - Conditional calibration by uncertainty decile per DGP
    - Global coverage bar chart (marginal coverage @ 50/90/95%)
    - Interval width vs coverage scatter (efficiency-calibration trade-off)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from benchmarks.conformalization_study import (
    OUTPUT_DIR,
    load_and_merge_conformalization_results,
)
from benchmarks.utils.style import MODEL_COLORS, MODEL_DISPLAY_NAMES, apply_paper_style
from benchmarks.utils.synthetic_plotting import (
    order_models,
    plot_conditional_calibration_by_uncertainty,
    plot_spread_skill,
)

apply_paper_style()

PLOTS_DIR = Path("benchmarks/plots/conformalization/summary")

NOMINAL_LEVELS = {50: 0.50, 90: 0.90, 95: 0.95}


# =============================================================================
# Helpers
# =============================================================================


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#999999")


def _label(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


# =============================================================================
# Plot functions
# =============================================================================


def plot_global_coverage_summary(
    all_results: dict,
    save_path: Path | None = None,
) -> None:
    """Bar chart of marginal coverage at 50/90/95% across DGPs, one panel per level."""
    dgp_names = list(all_results.keys())
    all_model_names: set[str] = set()
    for dgp_data in all_results.values():
        all_model_names.update(dgp_data.get("models", {}).keys())
    models = order_models(list(all_model_names))

    n_levels = len(NOMINAL_LEVELS)
    n_dgps = len(dgp_names)
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 4), sharey=False)
    if n_levels == 1:
        axes = [axes]

    x = np.arange(len(models))
    bar_w = 0.7 / max(n_dgps, 1)

    for ax_idx, (pct, nominal) in enumerate(NOMINAL_LEVELS.items()):
        ax = axes[ax_idx]
        metric_key = f"global_coverage_{pct}"

        for dgp_idx, dgp_name in enumerate(dgp_names):
            models_data = all_results[dgp_name].get("models", {})
            means = []
            for model_name in models:
                agg = models_data.get(model_name, {}).get("aggregated_metrics", {})
                val = agg.get(metric_key, {}).get("mean", float("nan"))
                means.append(val)

            offset = (dgp_idx - (n_dgps - 1) / 2) * bar_w
            ax.bar(
                x + offset,
                means,
                width=bar_w * 0.9,
                label=dgp_name.replace("_", " ").title(),
                alpha=0.75,
            )

        ax.axhline(nominal, color="black", linestyle="--", linewidth=1.2, label=f"Nominal {pct}%", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in models], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Empirical coverage @ {pct}%")
        ax.set_title(f"{pct}% intervals", fontweight="bold")
        ax.grid(axis="y", alpha=0.15)
        if ax_idx == 0:
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle("Marginal Coverage: Conformal vs Raw Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved global coverage summary to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_width_vs_coverage(
    all_results: dict,
    level: int = 90,
    save_path: Path | None = None,
) -> None:
    """Scatter plot: mean interval width (x) vs empirical coverage (y) at given level.

    Points on the vertical dashed line (nominal coverage) with minimal width are ideal.
    """
    nominal = NOMINAL_LEVELS[level]
    cov_key = f"global_coverage_{level}"
    width_key = f"global_width_{level}"

    all_model_names: set[str] = set()
    for dgp_data in all_results.values():
        all_model_names.update(dgp_data.get("models", {}).keys())
    models = order_models(list(all_model_names))

    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "h"]
    dgp_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_idx, model_name in enumerate(models):
        widths, covs = [], []
        for dgp_name in dgp_names:
            agg = all_results[dgp_name].get("models", {}).get(model_name, {}).get("aggregated_metrics", {})
            cov = agg.get(cov_key, {}).get("mean", float("nan"))
            width = agg.get(width_key, {}).get("mean", float("nan"))
            if not (np.isnan(cov) or np.isnan(width)):
                widths.append(width)
                covs.append(cov)

        if widths:
            ax.scatter(
                widths,
                covs,
                color=_color(model_name),
                marker=markers[model_idx % len(markers)],
                label=_label(model_name),
                s=80,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.4,
                zorder=3,
            )
            # Connect per-DGP dots with a faint line
            if len(widths) > 1:
                sort_idx = np.argsort(widths)
                ax.plot(
                    [widths[i] for i in sort_idx],
                    [covs[i] for i in sort_idx],
                    color=_color(model_name),
                    alpha=0.2,
                    linewidth=0.8,
                    zorder=2,
                )

    ax.axhline(nominal, color="black", linestyle="--", linewidth=1.2, label=f"Nominal {level}%", alpha=0.7)
    ax.set_xlabel(f"Mean {level}% interval width", fontsize=11)
    ax.set_ylabel(f"Empirical {level}% coverage", fontsize=11)
    ax.set_title(f"Efficiency–Calibration Trade-off ({level}% intervals)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved width-vs-coverage plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_conditional_coverage_heatmap(
    all_results: dict,
    level: int = 90,
    save_path: Path | None = None,
) -> None:
    """Heatmap of conditional coverage by uncertainty decile: rows=models, cols=deciles.

    One panel per DGP. Colours show deviation from nominal: green=on-target,
    red=under-cover, blue=over-cover.
    """
    cov_key = f"cond_cov_{level}_by_bin"
    nominal = NOMINAL_LEVELS[level]

    all_model_names: set[str] = set()
    for dgp_data in all_results.values():
        all_model_names.update(dgp_data.get("models", {}).keys())
    models = order_models(list(all_model_names))

    dgp_names = list(all_results.keys())
    n_dgps = len(dgp_names)
    if n_dgps == 0 or not models:
        logger.warning("No data for conditional coverage heatmap — skipping")
        return

    fig, axes = plt.subplots(1, n_dgps, figsize=(4 * n_dgps, 0.6 * len(models) + 1.5), squeeze=False)

    from matplotlib.colors import TwoSlopeNorm

    cmap = plt.cm.RdYlGn  # red=under, yellow=on-target, green=over

    for col, dgp_name in enumerate(dgp_names):
        ax = axes[0, col]
        models_data = all_results[dgp_name].get("models", {})

        grid = np.full((len(models), 10), np.nan)
        for row, model_name in enumerate(models):
            agg = models_data.get(model_name, {}).get("aggregated_metrics", {})
            cov_raw = agg.get(cov_key, {}).get("mean")
            if cov_raw is not None:
                vals = np.array(cov_raw, dtype=float)
                grid[row, : len(vals)] = vals

        # Deviation from nominal
        deviation = grid - nominal
        norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
        im = ax.imshow(deviation, cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i + 1) for i in range(10)], fontsize=8)
        ax.set_xlabel("Uncertainty decile", fontsize=9)
        if col == 0:
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([_label(m) for m in models], fontsize=9)
        else:
            ax.set_yticks([])
        ax.set_title(dgp_name.replace("_", " ").title(), fontsize=10, fontweight="bold")

        plt.colorbar(im, ax=ax, label=f"Coverage − {nominal:.0%}", shrink=0.8)

    fig.suptitle(
        f"Conditional Coverage Deviation from Nominal ({level}% intervals)\n"
        "Green = on-target, Red = under-cover, Yellow = over-cover",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved conditional coverage heatmap to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Per-DGP summary plots (spread-skill + conditional calibration)
# =============================================================================


def generate_per_dgp_plots(all_results: dict, plots_dir: Path) -> None:
    """Generate spread-skill and conditional calibration plots per DGP."""
    from benchmarks.synthetic_dgp_benchmark import N_UNCERTAINTY_BINS

    for dgp_name, dgp_data in all_results.items():
        dgp_dir = plots_dir / dgp_name
        dgp_dir.mkdir(parents=True, exist_ok=True)

        models_data = dgp_data.get("models", {})

        spread_by_model: dict = {}
        cond_cal_by_model: dict = {}

        for model_name, model_data in models_data.items():
            agg = model_data.get("aggregated_metrics", {})

            if "spread_pred_std_by_bin" in agg and "spread_rmse_by_bin" in agg:
                spread_by_model[model_name] = {
                    "spread_skill_pred_std": agg["spread_pred_std_by_bin"],
                    "spread_skill_rmse": agg["spread_rmse_by_bin"],
                    "spread_skill_true_std": {"mean": [float("nan")] * N_UNCERTAINTY_BINS},
                }

            if "cond_cov_90_by_bin" in agg:
                cond_cal_by_model[model_name] = {
                    "cond_cal_cov_90": agg["cond_cov_90_by_bin"],
                }

        if spread_by_model:
            plot_spread_skill(
                results_by_model=spread_by_model,
                save_path=dgp_dir / "spread_skill_all_models.pdf",
            )

        if cond_cal_by_model:
            plot_conditional_calibration_by_uncertainty(
                results_by_model=cond_cal_by_model,
                save_path=dgp_dir / "cond_calibration_all_models.pdf",
            )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logger.info("\n" + "=" * 80)
    logger.info("Conformalization Study — Comparison & Plots")
    logger.info("=" * 80)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Merge core + full results
    logger.info(f"\nLoading results from {OUTPUT_DIR} ...")
    all_results = load_and_merge_conformalization_results(output_dir=OUTPUT_DIR)

    if not all_results:
        logger.error(
            "No conformalization results found.\n"
            "Run the study first:\n"
            "  pixi run python -m benchmarks.conformalization_study           (default env)\n"
            "  pixi run -e bench-models python -m benchmarks.conformalization_study"
        )
        return

    all_model_names: set[str] = set()
    for d in all_results.values():
        all_model_names.update(d.get("models", {}).keys())
    logger.info(f"Models found: {', '.join(sorted(all_model_names))}")
    logger.info(f"DGPs found:   {', '.join(all_results.keys())}")

    # Save merged JSON for later use
    merged_path = OUTPUT_DIR / "conformalization_merged.json"
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved merged results to {merged_path}")

    # --- Summary plots ---
    logger.info("\nGenerating summary plots ...")

    plot_global_coverage_summary(
        all_results,
        save_path=PLOTS_DIR / "global_coverage_summary.pdf",
    )

    plot_width_vs_coverage(
        all_results,
        level=90,
        save_path=PLOTS_DIR / "width_vs_coverage_90.pdf",
    )

    plot_conditional_coverage_heatmap(
        all_results,
        level=90,
        save_path=PLOTS_DIR / "cond_coverage_heatmap_90.pdf",
    )

    # --- Per-DGP plots ---
    logger.info("\nGenerating per-DGP plots ...")
    generate_per_dgp_plots(all_results, PLOTS_DIR)

    logger.info(f"\nAll plots saved to {PLOTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
