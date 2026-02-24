"""Plotting utilities for synthetic DGP benchmarks with ground truth visualization.

This module provides publication-quality plots specifically for synthetic datasets where
the true conditional distribution p(y|x) is known. Plots include:
- Ground truth overlays on predictions
- Conditional density comparisons
- Variance function estimation
- Error analysis by covariate regions
- Calibration diagnostics
"""

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.pipeline.synthetic_dgps import SyntheticDataset

# Publication-quality defaults
mpl.rcParams.update(
    {
        "font.size": 11,
        "figure.figsize": (10, 6),
        "font.family": "serif",
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    }
)

COLORS = {
    "ground_truth": "#2E86AB",  # Steel blue
    "bdf": "#A23B72",  # Plum
    "baseline": "#F18F01",  # Orange
    "data": "#4A4A4A",  # Dark gray
    "confidence": "#C73E1D",  # Burnt sienna
}

# Consistent model styling across all summary figures
MODEL_ORDER = ["BDFNormal", "BDFKDE", "ConformalRF", "NGBoost", "BART"]
MODEL_COLORS = {
    "BDFNormal": "#A23B72",  # Plum
    "BDFKDE": "#6A4C93",  # Purple
    "ConformalRF": "#F18F01",  # Orange
    "NGBoost": "#2E86AB",  # Steel blue
    "BART": "#6A994E",  # Green
}
MODEL_DISPLAY_NAMES = {
    "BDFNormal": "BDF (Normal)",
    "BDFKDE": "BDF (KDE)",
    "ConformalRF": "Conformal RF",
    "NGBoost": "NGBoost",
    "BART": "BART",
}


def _order_models(models: list[str]) -> list[str]:
    """Sort model names into canonical display order."""
    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: order_map.get(m, len(MODEL_ORDER)))


def _get_color(model_name: str) -> str:
    """Get consistent color for a model."""
    return MODEL_COLORS.get(model_name, "#999999")


def _get_display_name(model_name: str) -> str:
    """Get pretty display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def plot_predictions_with_ground_truth(
    dataset: SyntheticDataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_samples: Optional[np.ndarray] = None,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    n_plot_points: int = 200,
) -> None:
    """Plot predictions against ground truth mean and uncertainty.

    Args:
        dataset: SyntheticDataset with ground truth functions
        X_test: Test features
        y_test: Test targets
        y_pred: Predicted means
        y_samples: Predicted samples (n_test, n_samples) for uncertainty
        model_name: Name of model for title
        save_path: Optional path to save figure
        n_plot_points: Number of points for ground truth curves
    """
    # Only works for 1D features currently
    if X_test.shape[1] != 1:
        print(f"Skipping plot: requires 1D features, got {X_test.shape[1]}D")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    x_flat = X_test.flatten()
    gt = dataset.ground_truth

    # Create dense grid for ground truth
    x_grid = np.linspace(x_flat.min(), x_flat.max(), n_plot_points).reshape(-1, 1)
    y_gt_mean = gt.mean_fn(x_grid).flatten()

    # Top panel: Mean function
    ax_mean = axes[0]
    ax_mean.scatter(x_flat, y_test, alpha=0.3, s=10, c=COLORS["data"], label="Test data", zorder=1)
    ax_mean.plot(x_grid, y_gt_mean, c=COLORS["ground_truth"], label="True mean", linewidth=2.5, zorder=3)
    ax_mean.plot(x_flat, y_pred, ".", c=COLORS["bdf"], label=f"{model_name} pred", alpha=0.6, markersize=4, zorder=2)

    # Add prediction intervals if samples available
    if y_samples is not None:
        y_pred_sorted_idx = np.argsort(x_flat)
        x_sorted = x_flat[y_pred_sorted_idx]
        y_lower_90 = np.quantile(y_samples, 0.05, axis=-1)[y_pred_sorted_idx]
        y_upper_90 = np.quantile(y_samples, 0.95, axis=-1)[y_pred_sorted_idx]

        ax_mean.fill_between(
            x_sorted,
            y_lower_90,
            y_upper_90,
            alpha=0.2,
            color=COLORS["bdf"],
            label="90% pred. interval",
            zorder=0,
        )

    ax_mean.set_ylabel("y")
    ax_mean.legend(loc="best", framealpha=0.9)
    ax_mean.set_title(f"{model_name} on {dataset.name}", fontsize=13, fontweight="bold")

    # Bottom panel: Variance function (if available)
    ax_var = axes[1]
    if y_samples is not None and gt.variance_fn is not None:
        y_gt_var = gt.variance_fn(x_grid).flatten()
        y_pred_var = np.var(y_samples, axis=-1)

        ax_var.plot(x_grid, y_gt_var, c=COLORS["ground_truth"], label="True variance", linewidth=2.5)
        ax_var.plot(x_flat, y_pred_var, ".", c=COLORS["bdf"], label=f"{model_name} variance", alpha=0.6, markersize=4)

        ax_var.set_ylabel("Variance")
        ax_var.set_xlabel("x")
        ax_var.legend(loc="best", framealpha=0.9)
    else:
        # Show prediction errors instead
        errors = np.abs(y_test - y_pred)
        ax_var.scatter(x_flat, errors, alpha=0.5, s=15, c=COLORS["confidence"])
        ax_var.set_ylabel("Absolute error")
        ax_var.set_xlabel("x")
        ax_var.set_title("Prediction errors")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_conditional_densities(
    dataset: SyntheticDataset,
    y_samples: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    x_values: list[float],
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> None:
    """Plot conditional densities at specific x values comparing model to ground truth.

    Args:
        dataset: SyntheticDataset with ground truth functions
        y_samples: Predicted samples (n_test, n_samples)
        X_test: Test features
        y_test: Test targets
        x_values: List of x values to plot densities at
        model_name: Name of model
        save_path: Optional path to save figure
    """
    if X_test.shape[1] != 1:
        print(f"Skipping plot: requires 1D features, got {X_test.shape[1]}D")
        return

    x_flat = X_test.flatten()
    n_plots = len(x_values)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    gt = dataset.ground_truth

    for idx, x_val in enumerate(x_values):
        ax = axes[idx]

        # Find closest test point to x_val
        closest_idx = np.argmin(np.abs(x_flat - x_val))
        x_actual = x_flat[closest_idx]

        # Model predicted density (from samples)
        model_samples = y_samples[closest_idx]
        ax.hist(
            model_samples,
            bins=30,
            density=True,
            alpha=0.6,
            color=COLORS["bdf"],
            label=f"{model_name}",
            edgecolor="black",
            linewidth=0.5,
        )

        # Ground truth density
        if gt.density_fn is not None:
            y_grid = np.linspace(model_samples.min(), model_samples.max(), 200)
            # Evaluate ground truth density at this x value
            x_for_gt = np.array([[x_actual]])  # Shape (1, 1) to match expected input
            gt_density = np.array([gt.density_fn(y_val, x_for_gt)[0] for y_val in y_grid])
            ax.plot(y_grid, gt_density, c=COLORS["ground_truth"], linewidth=2.5, label="True density")

        # Mark actual test observation
        y_actual = y_test[closest_idx]
        ax.axvline(y_actual, color=COLORS["data"], linestyle="--", linewidth=1.5, label="Observed y")

        ax.set_title(f"x = {x_actual:.2f}", fontsize=11)
        ax.set_xlabel("y")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

    # Remove unused subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Conditional Densities: {model_name} on {dataset.name}", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_dgp_comparison_table(
    results_dict: dict,
    models: list[str],
    metrics: list[str] = ["rmse", "crps", "gt_mean_rmse", "gt_variance_rmse"],
    save_path: Optional[Path] = None,
) -> None:
    """Create a table comparing models across different DGPs.

    Args:
        results_dict: Dictionary from benchmark results (dgp_name -> models -> metrics)
        models: List of model names to include
        metrics: List of metrics to display
        save_path: Optional path to save figure
    """
    dgp_names = list(results_dict.keys())
    n_dgps = len(dgp_names)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, n_dgps, figsize=(3 * n_dgps, 2 * n_metrics))
    if n_dgps == 1 and n_metrics == 1:
        axes = np.array([[axes]])
    elif n_dgps == 1:
        axes = axes.reshape(-1, 1)
    elif n_metrics == 1:
        axes = axes.reshape(1, -1)

    for dgp_idx, dgp_name in enumerate(dgp_names):
        dgp_results = results_dict[dgp_name]

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx, dgp_idx]

            # Extract metric values for each model
            model_means = []
            model_stds = []
            model_labels = []

            for model_name in models:
                if model_name in dgp_results["models"]:
                    model_data = dgp_results["models"][model_name]
                    if "aggregated_metrics" in model_data:
                        agg = model_data["aggregated_metrics"]
                        if metric in agg:
                            model_means.append(agg[metric]["mean"])
                            model_stds.append(agg[metric]["std"])
                            model_labels.append(model_name)

            if model_means:
                # Bar plot with error bars
                x_pos = np.arange(len(model_labels))
                bars = ax.bar(x_pos, model_means, yerr=model_stds, capsize=5, alpha=0.7)

                # Color best model differently
                best_idx = np.argmin(model_means)
                bars[best_idx].set_color(COLORS["ground_truth"])

                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9)
                # Format metric name for display
                display_metric = metric.upper().replace("_", " ")
                ax.set_ylabel(display_metric, fontsize=10)
                ax.grid(alpha=0.15, axis="y")

                if metric_idx == 0:
                    ax.set_title(dgp_name.replace("_", " ").title(), fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved table to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_variance_estimation_quality(
    results_dict: dict,
    models: list[str],
    save_path: Optional[Path] = None,
) -> None:
    """Create scatter plot of variance estimation quality across DGPs.

    Args:
        results_dict: Dictionary from benchmark results
        models: List of model names to include
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors_list = [COLORS["bdf"], COLORS["baseline"], COLORS["confidence"], "#6A994E", "#BC4749"]

    for model_idx, model_name in enumerate(models):
        var_rmse_vals = []
        mean_rmse_vals = []
        dgp_labels = []

        for dgp_name, dgp_results in results_dict.items():
            if model_name in dgp_results["models"]:
                model_data = dgp_results["models"][model_name]
                if "aggregated_metrics" in model_data:
                    agg = model_data["aggregated_metrics"]
                    if "gt_variance_rmse" in agg and "gt_mean_rmse" in agg:
                        var_rmse_vals.append(agg["gt_variance_rmse"]["mean"])
                        mean_rmse_vals.append(agg["gt_mean_rmse"]["mean"])
                        dgp_labels.append(dgp_name)

        if var_rmse_vals:
            ax.scatter(
                mean_rmse_vals,
                var_rmse_vals,
                label=model_name,
                marker=markers[model_idx % len(markers)],
                color=colors_list[model_idx % len(colors_list)],
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

            # Annotate with DGP names
            for i, dgp_label in enumerate(dgp_labels):
                ax.annotate(
                    dgp_label.replace("_", " ")[:10],
                    (mean_rmse_vals[i], var_rmse_vals[i]),
                    fontsize=7,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    ax.set_xlabel("Ground Truth Mean RMSE", fontsize=12)
    ax.set_ylabel("Ground Truth Variance RMSE", fontsize=12)
    ax.set_title("Variance Estimation Quality vs Mean Estimation", fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)

    # Log scale if values span orders of magnitude
    if len(var_rmse_vals) > 0 and max(var_rmse_vals) / (min(var_rmse_vals) + 1e-10) > 100:
        ax.set_yscale("log")
        ax.set_xscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_coverage_by_region(
    dataset: SyntheticDataset,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_samples: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    n_bins: int = 10,
) -> None:
    """Plot calibration (coverage) by covariate regions.

    Tests whether uncertainty estimates are locally calibrated.

    Args:
        dataset: SyntheticDataset
        X_test: Test features
        y_test: Test targets
        y_samples: Predicted samples
        model_name: Model name
        save_path: Optional save path
        n_bins: Number of bins to divide X into
    """
    if X_test.shape[1] != 1:
        print(f"Skipping plot: requires 1D features, got {X_test.shape[1]}D")
        return

    x_flat = X_test.flatten()

    # Bin x values
    x_bins = np.percentile(x_flat, np.linspace(0, 100, n_bins + 1))
    x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2

    coverage_50 = []
    coverage_90 = []

    for i in range(n_bins):
        mask = (x_flat >= x_bins[i]) & (x_flat < x_bins[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (x_flat >= x_bins[i]) & (x_flat <= x_bins[i + 1])

        if mask.sum() == 0:
            continue

        y_test_bin = y_test[mask]
        y_samples_bin = y_samples[mask]

        # Compute coverage
        y_lower_50 = np.quantile(y_samples_bin, 0.25, axis=-1)
        y_upper_50 = np.quantile(y_samples_bin, 0.75, axis=-1)
        cov_50 = np.mean((y_test_bin >= y_lower_50) & (y_test_bin <= y_upper_50))

        y_lower_90 = np.quantile(y_samples_bin, 0.05, axis=-1)
        y_upper_90 = np.quantile(y_samples_bin, 0.95, axis=-1)
        cov_90 = np.mean((y_test_bin >= y_lower_90) & (y_test_bin <= y_upper_90))

        coverage_50.append(cov_50)
        coverage_90.append(cov_90)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_bin_centers, coverage_50, "o-", label="50% interval", color=COLORS["bdf"], linewidth=2, markersize=8)
    ax.plot(x_bin_centers, coverage_90, "s-", label="90% interval", color=COLORS["baseline"], linewidth=2, markersize=8)

    # Reference lines for nominal coverage
    ax.axhline(0.5, color=COLORS["ground_truth"], linestyle="--", linewidth=1.5, label="Nominal 50%")
    ax.axhline(0.9, color=COLORS["confidence"], linestyle="--", linewidth=1.5, label="Nominal 90%")

    ax.set_xlabel("x (binned)", fontsize=12)
    ax.set_ylabel("Empirical coverage", fontsize=12)
    ax.set_title(f"Local Calibration: {model_name} on {dataset.name}", fontsize=13, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_pit_histogram(
    y_test: np.ndarray,
    y_samples: np.ndarray,
    model_name: str = "Model",
    dgp_name: str = "",
    n_bins: int = 20,
    save_path: Optional[Path] = None,
) -> dict:
    """Plot PIT histogram for a single model on a single DGP.

    Well-calibrated models produce uniform PIT values. Deviations indicate:
    - U-shaped: underdispersed (intervals too narrow)
    - Inverse-U: overdispersed (intervals too wide)
    - Skewed left/right: systematic bias

    Args:
        y_test: True target values
        y_samples: Predicted samples (n_obs, n_samples)
        model_name: Model name for title
        dgp_name: DGP name for title
        n_bins: Number of histogram bins
        save_path: Optional path to save figure

    Returns:
        dict with bin_counts, bin_proportions, n_bins, pit_values for reuse
    """
    # Compute PIT values: fraction of samples <= true value
    pit_values = np.mean(y_samples <= y_test[:, None], axis=1)

    bin_counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))
    bin_proportions = bin_counts / bin_counts.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins
    uniform_height = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        bin_centers,
        bin_proportions,
        width=bin_width * 0.9,
        color=COLORS["bdf"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(uniform_height, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Uniform")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(bin_proportions.max() * 1.15, uniform_height * 1.5))
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Proportion")
    title = f"PIT Histogram: {model_name}"
    if dgp_name:
        title += f" — {dgp_name.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)

    return {
        "bin_counts": bin_counts.tolist(),
        "bin_proportions": bin_proportions.tolist(),
        "n_bins": n_bins,
        "n_samples": len(y_test),
    }


def plot_pit_summary_grid(
    results_dict: dict,
    models: list[str],
    n_bins: int = 20,
    save_path: Optional[Path] = None,
) -> None:
    """Plot PIT histogram grid: one column per model, one row per DGP.

    Extracts stored PIT histogram data from results and plots an overview grid
    suitable for an appendix figure.

    Args:
        results_dict: Benchmark results dict (dgp_name -> {models -> {aggregated_metrics -> ...}})
        models: Ordered list of model names to include
        n_bins: Expected number of bins (for uniform reference line)
        save_path: Optional path to save figure
    """
    dgp_names = list(results_dict.keys())

    # Filter to models that have PIT data in at least one DGP
    active_models = []
    for m in models:
        for dgp_name in dgp_names:
            dgp_res = results_dict[dgp_name]
            if m in dgp_res.get("models", {}):
                model_data = dgp_res["models"][m]
                agg = model_data.get("aggregated_metrics", {})
                if "pit_bin_proportions" in agg:
                    active_models.append(m)
                    break
    if not active_models:
        print("No PIT histogram data found in results — skipping PIT summary grid.")
        return

    n_rows = len(dgp_names)
    n_cols = len(active_models)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.4 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    bin_width = 1.0 / n_bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    uniform_height = 1.0 / n_bins

    # Model colours (cycle through a palette)
    palette = [COLORS["bdf"], COLORS["baseline"], COLORS["ground_truth"], COLORS["confidence"], "#6A994E"]

    for col_idx, model_name in enumerate(active_models):
        color = palette[col_idx % len(palette)]

        for row_idx, dgp_name in enumerate(dgp_names):
            ax = axes[row_idx, col_idx]

            dgp_res = results_dict[dgp_name]
            model_data = dgp_res.get("models", {}).get(model_name, {})
            agg = model_data.get("aggregated_metrics", {})
            pit_info = agg.get("pit_bin_proportions", None)

            if pit_info is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            else:
                # pit_info["mean"] is the average bin proportions across folds
                avg_props = np.array(pit_info["mean"])
                ax.bar(
                    bin_centers[: len(avg_props)],
                    avg_props,
                    width=bin_width * 0.88,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.4,
                )
                ax.axhline(uniform_height, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

                # Show KS statistic if available
                ks = agg.get("pit_ks_statistic", {})
                if isinstance(ks, dict) and "mean" in ks:
                    ax.text(
                        0.97,
                        0.93,
                        f"KS={ks['mean']:.3f}",
                        ha="right",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    )

            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.12, axis="y")

            # Row labels on left edge
            if col_idx == 0:
                ax.set_ylabel(dgp_name.replace("_", " ").title(), fontsize=10)

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(model_name, fontsize=11, fontweight="bold")

            # X labels on bottom row only
            if row_idx == n_rows - 1:
                ax.set_xlabel("PIT value", fontsize=9)

    fig.suptitle("PIT Histograms by DGP and Model", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved PIT summary grid to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_dgp_overview(
    dgps_to_run: list[dict],
    save_path: Optional[Path] = None,
) -> None:
    """Plot DGP overview: data scatter + true mean/variance for each DGP (Fig A.1).

    Args:
        dgps_to_run: List of DGP specification dicts (name + kwargs)
        save_path: Optional path to save figure
    """
    from benchmarks.pipeline.synthetic_dgps import DGP_REGISTRY

    n_dgps = len(dgps_to_run)
    fig, axes = plt.subplots(n_dgps, 2, figsize=(12, 3.0 * n_dgps))
    if n_dgps == 1:
        axes = axes.reshape(1, -1)

    for row_idx, dgp_spec in enumerate(dgps_to_run):
        dgp_name = dgp_spec["name"]
        dgp_kwargs = dgp_spec.get("kwargs", {})

        dataset = DGP_REGISTRY[dgp_name](**dgp_kwargs, seed=42)
        gt = dataset.ground_truth

        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        if dataset.X.shape[1] == 1:
            x_flat = dataset.X.flatten()
            sort_idx = np.argsort(x_flat)
            _x_sorted = x_flat[sort_idx]

            # Left: data scatter + true mean + ±2σ band
            ax_left.scatter(x_flat, dataset.y, alpha=0.15, s=4, c=COLORS["data"], rasterized=True)
            x_grid = np.linspace(x_flat.min(), x_flat.max(), 300).reshape(-1, 1)
            y_mean = gt.mean_fn(x_grid).flatten()
            ax_left.plot(x_grid.flatten(), y_mean, c=COLORS["ground_truth"], linewidth=2, label="True mean")

            if gt.variance_fn is not None:
                y_std = np.sqrt(gt.variance_fn(x_grid).flatten())
                ax_left.fill_between(
                    x_grid.flatten(),
                    y_mean - 2 * y_std,
                    y_mean + 2 * y_std,
                    alpha=0.2,
                    color=COLORS["ground_truth"],
                    label="±2σ",
                )

            ax_left.set_ylabel("y")
            ax_left.legend(fontsize=8, loc="best")
            ax_left.set_title(dgp_name.replace("_", " ").title(), fontsize=11, fontweight="bold")

            # Right: conditional densities at 2-3 x-values
            x_vals_map = {
                "heteroscedastic_sinusoidal": [1.0, 3.14, 5.5],
                "step_function": [0.5, 2.5, 5.5],
                "bimodal_mixture": [0.2, 0.5, 0.8],
                "sparse_sampling": [1.0, 3.0, 5.5],
            }
            x_vals = x_vals_map.get(
                dgp_name, [x_flat.min() + (x_flat.max() - x_flat.min()) * f for f in [0.25, 0.5, 0.75]]
            )

            if gt.density_fn is not None:
                y_range = np.linspace(dataset.y.min(), dataset.y.max(), 300)
                for x_val in x_vals:
                    x_pt = np.array([[x_val]])
                    densities = np.array([gt.density_fn(yv, x_pt)[0] for yv in y_range])
                    ax_right.plot(y_range, densities, linewidth=1.5, label=f"x={x_val:.1f}")
                ax_right.legend(fontsize=8, loc="best")
                ax_right.set_xlabel("y")
                ax_right.set_ylabel("p(y|x)")
                ax_right.set_title("Conditional densities", fontsize=10)
            else:
                ax_right.text(0.5, 0.5, "No density_fn", ha="center", va="center", transform=ax_right.transAxes)
        else:
            # 2D+ features (e.g. heavy_tailed): scatter X[:,0] vs y + residual hist
            ax_left.scatter(dataset.X[:, 0], dataset.y, alpha=0.15, s=4, c=COLORS["data"], rasterized=True)
            ax_left.set_xlabel("x₁")
            ax_left.set_ylabel("y")
            ax_left.set_title(dgp_name.replace("_", " ").title(), fontsize=11, fontweight="bold")

            # Right: residual histogram vs reference distribution
            y_mean_pred = gt.mean_fn(dataset.X).flatten()
            residuals = dataset.y - y_mean_pred
            ax_right.hist(
                residuals, bins=60, density=True, alpha=0.6, color=COLORS["data"], edgecolor="black", linewidth=0.3
            )
            # Overlay Student-t if heavy-tailed
            from scipy.stats import t as student_t

            df = dgp_kwargs.get("df", 3.0)
            scale = dgp_kwargs.get("scale", 0.5)
            r_grid = np.linspace(residuals.min(), residuals.max(), 300)
            ax_right.plot(
                r_grid,
                student_t.pdf(r_grid, df=df, scale=scale),
                c=COLORS["ground_truth"],
                linewidth=2,
                label=f"t(df={df})",
            )
            ax_right.legend(fontsize=8)
            ax_right.set_xlabel("Residual")
            ax_right.set_ylabel("Density")
            ax_right.set_title("Residual distribution", fontsize=10)

    # Only set x-label on bottom row
    for col in range(2):
        axes[-1, col].set_xlabel(axes[-1, col].get_xlabel() or "x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved DGP overview to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_calibration_curves(
    results_dict: dict,
    models: list[str],
    save_path: Optional[Path] = None,
) -> None:
    """Plot calibration curves: empirical vs nominal coverage (Fig A.4).

    2×3 grid (5 DGPs + 1 legend cell). Each panel shows diagonal reference
    + one line per model with ±1 std band across folds.

    Args:
        results_dict: Benchmark results dict (dgp_name -> {models -> ...})
        models: List of model names
        save_path: Optional path to save figure
    """
    dgp_names = list(results_dict.keys())
    ordered_models = _order_models(
        [m for m in models if any(m in results_dict[d].get("models", {}) for d in dgp_names)]
    )

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, dgp_name in enumerate(dgp_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        dgp_res = results_dict[dgp_name]

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Ideal")

        for model_name in ordered_models:
            model_data = dgp_res.get("models", {}).get(model_name, {})
            agg = model_data.get("aggregated_metrics", {})
            cov_info = agg.get("coverage_curve_empirical")
            levels_info = agg.get("coverage_curve_levels")

            if cov_info is None or levels_info is None:
                continue

            nominal = np.array(levels_info["mean"])
            empirical_mean = np.array(cov_info["mean"])
            empirical_std = np.array(cov_info["std"])

            color = _get_color(model_name)
            label = _get_display_name(model_name)
            ax.plot(nominal, empirical_mean, "o-", color=color, linewidth=1.5, markersize=3, label=label)
            ax.fill_between(
                nominal,
                empirical_mean - empirical_std,
                empirical_mean + empirical_std,
                alpha=0.15,
                color=color,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(dgp_name.replace("_", " ").title(), fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Empirical coverage")
        if row == n_rows - 1:
            ax.set_xlabel("Nominal coverage")
        ax.grid(True, alpha=0.15)

    # Last cell: legend only
    ax_legend = axes[n_rows - 1, n_cols - 1]
    if len(dgp_names) < n_rows * n_cols:
        ax_legend.axis("off")
        # Collect handles from a populated axis
        for idx in range(len(dgp_names)):
            r, c = divmod(idx, n_cols)
            handles, labels = axes[r, c].get_legend_handles_labels()
            if handles:
                ax_legend.legend(handles, labels, loc="center", fontsize=11, frameon=False)
                break

    fig.suptitle("Calibration Curves", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved calibration curves to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_predictions_grid(
    dgps_to_run: list[dict],
    plot_dir: Path,
    models: list[str],
    save_path: Optional[Path] = None,
) -> None:
    """Plot predictions grid: rows=1D DGPs, cols=models (Fig A.5).

    Each cell shows data scatter + true mean + model 90% PI band + model mean.
    Loads fold-1 .npz files from plot_dir/<dgp_name>/.

    Args:
        dgps_to_run: List of DGP specification dicts
        plot_dir: Base directory containing per-DGP subdirectories with .npz files
        models: List of model names
        save_path: Optional path to save figure
    """
    from benchmarks.pipeline.synthetic_dgps import DGP_REGISTRY

    # Filter to 1D DGPs only (skip heavy_tailed which is 2D)
    dgps_1d = []
    for dgp_spec in dgps_to_run:
        dgp_name = dgp_spec["name"]
        dgp_kwargs = dgp_spec.get("kwargs", {})
        dataset = DGP_REGISTRY[dgp_name](**dgp_kwargs, seed=42)
        if dataset.X.shape[1] == 1:
            dgps_1d.append((dgp_spec, dataset))

    if not dgps_1d:
        print("No 1D DGPs found — skipping predictions grid.")
        return

    ordered_models = _order_models(models)

    # Check which models have data
    active_models = []
    for m in ordered_models:
        for dgp_spec, _ in dgps_1d:
            npz_path = Path(plot_dir) / dgp_spec["name"] / f"{m}_fold1_data.npz"
            if npz_path.exists():
                active_models.append(m)
                break
    if not active_models:
        print("No fold-1 .npz files found — skipping predictions grid.")
        return

    n_rows = len(dgps_1d)
    n_cols = len(active_models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.0 * n_rows), squeeze=False)

    for row_idx, (dgp_spec, dataset) in enumerate(dgps_1d):
        dgp_name = dgp_spec["name"]
        gt = dataset.ground_truth

        for col_idx, model_name in enumerate(active_models):
            ax = axes[row_idx, col_idx]
            npz_path = Path(plot_dir) / dgp_name / f"{model_name}_fold1_data.npz"

            if not npz_path.exists():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            data = np.load(npz_path)
            X_test = data["X_test"]
            y_test = data["y_test"]
            y_pred = data["y_pred"]
            y_samples = data["y_samples"]

            x_flat = X_test.flatten()
            sort_idx = np.argsort(x_flat)
            x_sorted = x_flat[sort_idx]

            # Data scatter
            ax.scatter(x_flat, y_test, alpha=0.12, s=3, c=COLORS["data"], rasterized=True)

            # True mean
            x_grid = np.linspace(x_flat.min(), x_flat.max(), 200).reshape(-1, 1)
            y_mean_true = gt.mean_fn(x_grid).flatten()
            ax.plot(x_grid.flatten(), y_mean_true, c=COLORS["ground_truth"], linewidth=1.5, label="True mean")

            # Model mean prediction
            color = _get_color(model_name)
            ax.plot(x_sorted, y_pred[sort_idx], c=color, linewidth=1.2, alpha=0.8, label=_get_display_name(model_name))

            # Model 90% PI band
            y_lower = np.quantile(y_samples, 0.05, axis=-1)[sort_idx]
            y_upper = np.quantile(y_samples, 0.95, axis=-1)[sort_idx]
            ax.fill_between(x_sorted, y_lower, y_upper, alpha=0.18, color=color)

            # Labels
            if col_idx == 0:
                ax.set_ylabel(dgp_name.replace("_", " ").title(), fontsize=10)
            if row_idx == 0:
                ax.set_title(_get_display_name(model_name), fontsize=11, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("x")

            ax.grid(True, alpha=0.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved predictions grid to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_main_body_predictions(
    dgp_name: str,
    dgp_kwargs: dict,
    plot_dir: Path,
    models: list[str],
    save_path: Optional[Path] = None,
) -> None:
    """Main-body figure: single DGP, all models side-by-side with ground truth.

    Shows data scatter + true mean + true ±2σ band + model mean + model 90% PI
    in a 1×N strip (one column per model). Designed for the main paper body.

    Args:
        dgp_name: Name of the DGP to feature
        dgp_kwargs: kwargs for the DGP generator
        plot_dir: Base directory containing per-DGP subdirs with .npz files
        models: List of model names
        save_path: Optional path to save figure
    """
    from benchmarks.pipeline.synthetic_dgps import DGP_REGISTRY

    dataset = DGP_REGISTRY[dgp_name](**dgp_kwargs, seed=42)
    gt = dataset.ground_truth

    if dataset.X.shape[1] != 1:
        print(f"Skipping main-body predictions: {dgp_name} is not 1D.")
        return

    ordered_models = _order_models(models)
    active_models = [m for m in ordered_models if (Path(plot_dir) / dgp_name / f"{m}_fold1_data.npz").exists()]
    if not active_models:
        print(f"No fold-1 .npz files found for {dgp_name} — skipping main-body predictions.")
        return

    n_cols = len(active_models)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.5), squeeze=False, sharey=True)

    for col_idx, model_name in enumerate(active_models):
        ax = axes[0, col_idx]
        npz_path = Path(plot_dir) / dgp_name / f"{model_name}_fold1_data.npz"
        data = np.load(npz_path)
        X_test, y_test = data["X_test"], data["y_test"]
        y_pred, y_samples = data["y_pred"], data["y_samples"]

        x_flat = X_test.flatten()
        sort_idx = np.argsort(x_flat)
        x_sorted = x_flat[sort_idx]

        # Data scatter
        ax.scatter(x_flat, y_test, alpha=0.10, s=3, c=COLORS["data"], rasterized=True, zorder=0)

        # True mean + ±2σ
        x_grid = np.linspace(x_flat.min(), x_flat.max(), 250).reshape(-1, 1)
        y_mean_true = gt.mean_fn(x_grid).flatten()
        ax.plot(x_grid.flatten(), y_mean_true, c=COLORS["ground_truth"], linewidth=1.8, label="True mean", zorder=3)

        if gt.variance_fn is not None:
            y_std_true = np.sqrt(gt.variance_fn(x_grid).flatten())
            ax.fill_between(
                x_grid.flatten(),
                y_mean_true - 2 * y_std_true,
                y_mean_true + 2 * y_std_true,
                alpha=0.12,
                color=COLORS["ground_truth"],
                label="True ±2σ",
                zorder=1,
            )

        # Model mean
        color = _get_color(model_name)
        ax.plot(x_sorted, y_pred[sort_idx], c=color, linewidth=1.3, alpha=0.85, label="Predicted mean", zorder=4)

        # Model 90% PI
        y_lower = np.quantile(y_samples, 0.05, axis=-1)[sort_idx]
        y_upper = np.quantile(y_samples, 0.95, axis=-1)[sort_idx]
        ax.fill_between(x_sorted, y_lower, y_upper, alpha=0.20, color=color, label="90% PI", zorder=2)

        ax.set_title(_get_display_name(model_name), fontsize=12, fontweight="bold")
        ax.set_xlabel("x")
        if col_idx == 0:
            ax.set_ylabel("y")
        ax.grid(True, alpha=0.1)

    # Single shared legend from first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(handles), fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02)
    )

    dgp_display = dgp_name.replace("_", " ").title()
    fig.suptitle(f"Predictive Distributions — {dgp_display}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved main-body predictions to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_main_body_metric_table(
    results_dict: dict,
    models: list[str],
    metrics: list[str] = ["crps", "coverage_90", "gt_mean_rmse", "gt_variance_rmse"],
    save_path: Optional[Path] = None,
) -> None:
    """Main-body figure: compact metric table across all DGPs.

    Renders a publication-quality table with models as columns and (DGP × metric)
    as rows. Best value per row is bold. Includes ±1 std.

    Args:
        results_dict: Benchmark results dict (dgp_name -> {models -> ...})
        models: List of model names
        metrics: Metrics to display (must exist in aggregated_metrics)
        save_path: Optional path to save figure
    """
    ordered_models = _order_models(
        [m for m in models if any(m in results_dict[d].get("models", {}) for d in results_dict)]
    )
    dgp_names = list(results_dict.keys())

    metric_display = {
        "crps": "CRPS",
        "rmse": "RMSE",
        "coverage_90": "Cov. 90%",
        "coverage_50": "Cov. 50%",
        "gt_mean_rmse": "Mean RMSE",
        "gt_variance_rmse": "Var. RMSE",
        "gt_variance_rel_error": "Var. Rel. Err.",
        "pica": "PICA",
        "sharpness": "Sharpness",
        "pit_ks_statistic": "PIT KS",
    }

    # Coverage targets: higher-is-better for coverage (closer to nominal), lower-is-better for everything else
    higher_is_better = {"coverage_90", "coverage_50", "r2"}

    # Build table data: rows = (dgp, metric), cols = models
    row_labels = []
    cell_text = []
    cell_colors = []

    for dgp_name in dgp_names:
        dgp_res = results_dict[dgp_name]
        dgp_display = dgp_name.replace("_", " ").title()

        for metric in metrics:
            row_labels.append(f"{dgp_display}\n{metric_display.get(metric, metric)}")

            row_vals = []  # (formatted_str, raw_value) per model
            for model_name in ordered_models:
                model_data = dgp_res.get("models", {}).get(model_name, {})
                agg = model_data.get("aggregated_metrics", {})
                m_info = agg.get(metric)

                if m_info is None or not isinstance(m_info, dict) or "mean" not in m_info:
                    row_vals.append(("—", None))
                else:
                    mean_val = m_info["mean"]
                    std_val = m_info.get("std", 0)
                    # For coverage, compute absolute deviation from nominal
                    if metric == "coverage_90":
                        raw = abs(mean_val - 0.90)  # lower deviation = better
                    elif metric == "coverage_50":
                        raw = abs(mean_val - 0.50)
                    else:
                        raw = mean_val
                    row_vals.append((f"{mean_val:.3f}±{std_val:.3f}", raw))

            # Find best (bold) — lower is better unless in higher_is_better
            raw_values = [v[1] for v in row_vals if v[1] is not None]
            if raw_values:
                if metric in higher_is_better:
                    best_raw = max(raw_values)
                else:
                    best_raw = min(raw_values)
            else:
                best_raw = None

            row_text = []
            row_color = []
            for text, raw in row_vals:
                if raw is not None and best_raw is not None and abs(raw - best_raw) < 1e-10:
                    row_text.append(text)
                    row_color.append("#d4edda")  # Light green
                else:
                    row_text.append(text)
                    row_color.append("white")

            cell_text.append(row_text)
            cell_colors.append(row_color)

    if not cell_text:
        print("No metric data found — skipping main-body metric table.")
        return

    # Create table figure
    n_rows_table = len(cell_text)
    n_cols_table = len(ordered_models)
    fig_height = max(3.0, 0.45 * n_rows_table + 1.0)
    fig_width = max(6.0, 1.8 * n_cols_table + 2.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    col_labels = [_get_display_name(m) for m in ordered_models]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(n_cols_table):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#f0f0f0")

    # Style row labels
    for i in range(n_rows_table):
        cell = table[i + 1, -1]
        cell.set_text_props(fontsize=8)

    # Bold the best cells
    for i, row_color in enumerate(cell_colors):
        for j, color in enumerate(row_color):
            if color == "#d4edda":
                table[i + 1, j].set_text_props(fontweight="bold")

    fig.suptitle("Synthetic DGP Benchmark Results", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved main-body metric table to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def create_synthetic_benchmark_summary(
    results_dict: dict,
    models: list[str],
    output_dir: Path,
) -> None:
    """Create comprehensive summary plots for synthetic benchmark.

    Args:
        results_dict: Dictionary from benchmark results
        models: List of model names
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating synthetic benchmark summary plots...")

    # 1. Comparison table across DGPs
    print("  Creating comparison table...")
    plot_dgp_comparison_table(
        results_dict,
        models,
        metrics=["rmse", "crps", "gt_mean_rmse", "gt_variance_rmse"],
        save_path=output_dir / "dgp_comparison_table.png",
    )

    # 2. Variance estimation quality scatter
    print("  Creating variance estimation plot...")
    plot_variance_estimation_quality(
        results_dict,
        models,
        save_path=output_dir / "variance_estimation_quality.png",
    )

    # 3. PIT histogram summary grid
    print("  Creating PIT histogram summary grid...")
    plot_pit_summary_grid(
        results_dict,
        models,
        save_path=output_dir / "pit_histograms.png",
    )

    # 4. DGP overview (generated from DGP registry, no results needed)
    print("  Creating DGP overview...")
    from benchmarks.synthetic_dgp_benchmark import DGPS_TO_RUN

    plot_dgp_overview(
        DGPS_TO_RUN,
        save_path=output_dir / "dgp_overview.png",
    )

    # 5. Calibration curves
    print("  Creating calibration curves...")
    plot_calibration_curves(
        results_dict,
        models,
        save_path=output_dir / "calibration_curves.png",
    )

    # 6. Predictions grid (from fold-1 saved data)
    print("  Creating predictions grid...")
    plot_dir = Path("benchmarks/plots/synthetic_dgp")
    plot_predictions_grid(
        DGPS_TO_RUN,
        plot_dir,
        models,
        save_path=output_dir / "predictions_grid.png",
    )

    # 7. Main-body predictions panel (heteroscedastic sinusoidal — most visually compelling)
    print("  Creating main-body predictions panel...")
    plot_dir = Path("benchmarks/plots/synthetic_dgp")
    featured_dgp = DGPS_TO_RUN[0]  # heteroscedastic_sinusoidal
    plot_main_body_predictions(
        dgp_name=featured_dgp["name"],
        dgp_kwargs=featured_dgp.get("kwargs", {}),
        plot_dir=plot_dir,
        models=models,
        save_path=output_dir / "main_predictions.png",
    )

    # 8. Main-body compact metric table
    print("  Creating main-body metric table...")
    plot_main_body_metric_table(
        results_dict,
        models,
        save_path=output_dir / "main_metric_table.png",
    )

    print(f"\nSummary plots saved to {output_dir}")
