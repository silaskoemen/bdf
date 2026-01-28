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

    print(f"\nSummary plots saved to {output_dir}")
