"""
Experiment: Effect of BDF Hyperparameters (Ablation Study)

Systematic one-at-a-time (OAT) ablation study of BDF regularization parameters
and scoring methods. This script investigates the sensitivity of BDF performance
to its key hyperparameters.

Parameters ablated:
- reg_lambda: Prior split probability penalty (0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0)
- reg_gamma: Multiplicity correction (0, 1e-3, 0.01, 0.1, 0.5, 1.0)
- reg_nu: Depth penalty (0, 1e-3, 0.01, 0.1, 0.5, 1.0)
- min_samples_leaf: Minimum samples per leaf (5, 10, 25, 50)
- score_method: nle (Bayesian), nll+bic, nll (plug-in only)

Uses Friedman 1-3 and make_regression DGPs with multiple seeds.
Generates publication-quality plots and summary tables.

Extensible for classification (where NLE makes larger difference due to Bernoulli).
"""

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3, make_regression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFRegressor
from benchmarks.metrics.regression import (
    REG_POINT_METRICS,
    REG_PROB_METRICS,
    coverage_90,
    crps_wrapper,
    precompute_percentiles,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_SEEDS = 10  # Number of random seeds for confidence intervals
N_SAMPLES = 1000  # Dataset size
N_FEATURES = 10  # For make_regression and friedman1
TEST_SIZE = 0.2
N_SAMPLES_PRED = 500  # Number of samples for probabilistic predictions

# Output directories
RESULTS_DIR = Path("benchmarks/results/effect_bdf_params")
PLOTS_DIR = Path("benchmarks/plots/effect_bdf_params")

# =============================================================================
# Parameter Grids for Ablation
# =============================================================================

# Default values (used when varying other parameters)
DEFAULT_PARAMS = {
    "reg_lambda": 0.01,
    "reg_gamma": 0.1,
    "reg_nu": 0.01,
    "min_samples_leaf": 10,
    "n_trees": 50,
    "max_depth": 50,
    "subsample": 0.9,
    "colsample": 0.9,
    "eta": 0.01,
}

# Distribution params defaults
DEFAULT_DIST_PARAMS = {
    "mu_mu": "auto",
    "sigma_mu": "auto",
    "sigma_mu_auto_scale": 1.0,
    "score_method": "nle",
    "score_correction": "bic",
}

# Parameter grids for ablation
ABLATION_GRIDS = {
    "reg_lambda": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0],
    "reg_gamma": [0.0, 1e-3, 0.01, 0.1, 0.5, 1.0],
    "reg_nu": [0.0, 1e-3, 0.01, 0.1, 0.5, 1.0],
    "min_samples_leaf": [5, 10, 25, 50],
}

# Scoring method configurations
SCORING_CONFIGS = {
    "nle": {"score_method": "nle", "score_correction": "bic"},  # correction ignored for nle
    "nll_bic": {"score_method": "nll", "score_correction": "bic"},
    "nll": {"score_method": "nll", "score_correction": None},
}

# =============================================================================
# Data Generating Processes (DGPs)
# =============================================================================

DGPS: dict[str, Callable] = {
    "friedman1": lambda n, seed: make_friedman1(n_samples=n, n_features=N_FEATURES, noise=1.0, random_state=seed),
    "friedman2": lambda n, seed: make_friedman2(n_samples=n, noise=1.0, random_state=seed),
    "friedman3": lambda n, seed: make_friedman3(n_samples=n, noise=0.1, random_state=seed),
    "make_regression": lambda n, seed: make_regression(
        n_samples=n, n_features=N_FEATURES, n_informative=5, noise=10.0, random_state=seed
    ),
}

# =============================================================================
# Metrics
# =============================================================================

# Select subset of metrics for ablation (focus on key ones)
POINT_METRICS = ["rmse", "mae"]
PROB_METRICS = ["crps", "coverage_90", "interval_score_90"]

# Primary metric for summary tables (lower is better)
PRIMARY_METRIC = "crps"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExperimentResult:
    """Single experiment result."""

    dgp: str
    seed: int
    param_name: str
    param_value: Any
    scoring_config: str
    metrics: dict[str, float]
    fit_time: float


@dataclass
class AblationResults:
    """Results for a complete ablation study."""

    results: list[ExperimentResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        records = []
        for r in self.results:
            record = {
                "dgp": r.dgp,
                "seed": r.seed,
                "param_name": r.param_name,
                "param_value": r.param_value,
                "scoring_config": r.scoring_config,
                "fit_time": r.fit_time,
            }
            record.update(r.metrics)
            records.append(record)
        return pd.DataFrame(records)

    def save(self, filepath: Path):
        """Save results to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": self.config,
            "timestamp": self.timestamp,
            "results": [
                {
                    "dgp": r.dgp,
                    "seed": r.seed,
                    "param_name": r.param_name,
                    "param_value": r.param_value,
                    "scoring_config": r.scoring_config,
                    "metrics": r.metrics,
                    "fit_time": r.fit_time,
                }
                for r in self.results
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "AblationResults":
        """Load results from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        results = [
            ExperimentResult(
                dgp=r["dgp"],
                seed=r["seed"],
                param_name=r["param_name"],
                param_value=r["param_value"],
                scoring_config=r["scoring_config"],
                metrics=r["metrics"],
                fit_time=r["fit_time"],
            )
            for r in data["results"]
        ]
        return cls(results=results, config=data["config"], timestamp=data["timestamp"])


# =============================================================================
# Core Experiment Functions
# =============================================================================


def generate_data(dgp_name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate train/test data for a DGP."""
    X, y = DGPS[dgp_name](N_SAMPLES, seed)
    X = np.asarray(X)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    return X_train, X_test, y_train, y_test


def compute_metrics(
    model: BDFRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics for a fitted model."""
    metrics = {}

    # Point predictions
    y_pred_point = model.predict(X_test)

    # Point metrics
    for metric_name in POINT_METRICS:
        if metric_name in REG_POINT_METRICS:
            metrics[metric_name] = float(REG_POINT_METRICS[metric_name](y_test, y_pred_point))

    # Probabilistic predictions (samples)
    try:
        y_pred_samples = model.predict_samples(X_test, n_samples=N_SAMPLES_PRED)

        # Pre-compute percentiles for efficiency
        precomputed = precompute_percentiles(y_pred_samples)

        # Probabilistic metrics
        for metric_name in PROB_METRICS:
            if metric_name in REG_PROB_METRICS:
                metric_spec = REG_PROB_METRICS[metric_name]
                try:
                    metrics[metric_name] = float(
                        metric_spec.func(y_test, y_pred_samples, quantile_levels=None, precomputed=precomputed)
                    )
                except TypeError:
                    # Some metrics don't take precomputed
                    metrics[metric_name] = float(metric_spec.func(y_test, y_pred_samples))
    except Exception as e:
        logger.warning(f"Failed to compute probabilistic metrics: {e}")
        for metric_name in PROB_METRICS:
            metrics[metric_name] = float("nan")

    return metrics


def run_single_experiment(
    dgp_name: str,
    seed: int,
    param_name: str,
    param_value: Any,
    scoring_config: str,
) -> ExperimentResult:
    """Run a single ablation experiment."""
    # Generate data
    X_train, X_test, y_train, y_test = generate_data(dgp_name, seed)

    # Build model kwargs
    model_kwargs = DEFAULT_PARAMS.copy()
    model_kwargs["random_state"] = seed

    # Override the ablated parameter
    if param_name in model_kwargs:
        model_kwargs[param_name] = param_value

    # Build distribution params
    dist_params = DEFAULT_DIST_PARAMS.copy()
    dist_params.update(SCORING_CONFIGS[scoring_config])

    # Create and fit model
    model = BDFRegressor(dist="NormalMuNormal", params=dist_params, **model_kwargs)

    start_time = time()
    model.fit(X_train, y_train)
    fit_time = time() - start_time

    # Compute metrics
    metrics = compute_metrics(model, X_test, y_test)

    return ExperimentResult(
        dgp=dgp_name,
        seed=seed,
        param_name=param_name,
        param_value=param_value,
        scoring_config=scoring_config,
        metrics=metrics,
        fit_time=fit_time,
    )


def run_regularization_ablation(
    param_name: str,
    param_grid: list,
    n_seeds: int = N_SEEDS,
) -> AblationResults:
    """Run ablation study for a single regularization parameter."""
    logger.info(f"Running ablation for {param_name}: {param_grid}")

    results = AblationResults(
        config={
            "param_name": param_name,
            "param_grid": param_grid,
            "n_seeds": n_seeds,
            "default_params": DEFAULT_PARAMS,
            "dgps": list(DGPS.keys()),
        }
    )

    total_experiments = len(DGPS) * len(param_grid) * n_seeds
    pbar = tqdm(total=total_experiments, desc=f"Ablation: {param_name}")

    for dgp_name in DGPS.keys():
        for param_value in param_grid:
            for seed in range(SEED, SEED + n_seeds):
                try:
                    # Use default scoring (nle) for regularization ablation
                    result = run_single_experiment(
                        dgp_name=dgp_name,
                        seed=seed,
                        param_name=param_name,
                        param_value=param_value,
                        scoring_config="nle",
                    )
                    results.results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {dgp_name}, {param_name}={param_value}, seed={seed}: {e}")
                pbar.update(1)

    pbar.close()
    return results


def run_scoring_ablation(n_seeds: int = N_SEEDS) -> AblationResults:
    """Run ablation study for scoring method configurations."""
    logger.info(f"Running scoring method ablation: {list(SCORING_CONFIGS.keys())}")

    results = AblationResults(
        config={
            "param_name": "scoring",
            "scoring_configs": SCORING_CONFIGS,
            "n_seeds": n_seeds,
            "default_params": DEFAULT_PARAMS,
            "dgps": list(DGPS.keys()),
        }
    )

    total_experiments = len(DGPS) * len(SCORING_CONFIGS) * n_seeds
    pbar = tqdm(total=total_experiments, desc="Ablation: scoring")

    for dgp_name in DGPS.keys():
        for scoring_name in SCORING_CONFIGS.keys():
            for seed in range(SEED, SEED + n_seeds):
                try:
                    result = run_single_experiment(
                        dgp_name=dgp_name,
                        seed=seed,
                        param_name="scoring",
                        param_value=scoring_name,
                        scoring_config=scoring_name,
                    )
                    results.results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {dgp_name}, scoring={scoring_name}, seed={seed}: {e}")
                pbar.update(1)

    pbar.close()
    return results


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
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_sensitivity_curve(
    df: pd.DataFrame,
    param_name: str,
    metric: str = PRIMARY_METRIC,
    log_scale: bool = True,
    save_path: Path | None = None,
):
    """
    Plot sensitivity curve for a single parameter across all DGPs.

    Shows mean ± 95% CI band across seeds.
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    dgps = df["dgp"].unique()
    colors = sns.color_palette("husl", len(dgps))

    for ax, (dgp, color) in zip(axes, zip(dgps, colors)):
        dgp_df = df[df["dgp"] == dgp]

        # Group by parameter value
        grouped = dgp_df.groupby("param_value")[metric]
        means = grouped.mean()
        stds = grouped.std()
        n = grouped.count()

        # 95% confidence interval
        ci = 1.96 * stds / np.sqrt(n)

        x = means.index.values
        y = means.values

        # Handle log scale for x-axis
        if log_scale and param_name != "min_samples_leaf":
            # Replace 0 with small value for log scale
            x_plot = np.where(x == 0, 1e-5, x)
            ax.set_xscale("log")
        else:
            x_plot = x

        # Sort for proper line plotting
        sort_idx = np.argsort(x_plot)
        x_plot = x_plot[sort_idx]
        y = y[sort_idx]
        ci_sorted = ci.values[sort_idx]

        # Plot
        ax.plot(x_plot, y, "o-", color=color, linewidth=2, markersize=6, label=dgp)
        ax.fill_between(x_plot, y - ci_sorted, y + ci_sorted, alpha=0.2, color=color)

        # Highlight default value
        default_val = DEFAULT_PARAMS.get(param_name, None)
        if default_val is not None:
            ax.axvline(default_val, color="gray", linestyle="--", alpha=0.5, label=f"Default ({default_val})")

        ax.set_xlabel(param_name)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{dgp}")
        ax.legend(loc="best")

    plt.suptitle(f"Sensitivity Analysis: {param_name} (metric: {metric})", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_scoring_comparison(
    df: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    save_path: Path | None = None,
):
    """
    Plot comparison of scoring methods across DGPs.

    Bar plot with error bars showing mean ± 95% CI.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    dgps = df["dgp"].unique()
    scoring_configs = df["param_value"].unique()
    x = np.arange(len(dgps))
    width = 0.25

    colors = {"nle": "#2ecc71", "nll_bic": "#3498db", "nll": "#e74c3c"}

    for i, scoring in enumerate(scoring_configs):
        means = []
        cis = []
        for dgp in dgps:
            subset = df[(df["dgp"] == dgp) & (df["param_value"] == scoring)]
            mean = subset[metric].mean()
            std = subset[metric].std()
            n = len(subset)
            ci = 1.96 * std / np.sqrt(n)
            means.append(mean)
            cis.append(ci)

        ax.bar(
            x + i * width,
            means,
            width,
            label=scoring,
            color=colors.get(scoring, f"C{i}"),
            yerr=cis,
            capsize=3,
        )

    ax.set_xlabel("DGP")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Scoring Method Comparison (metric: {metric})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(dgps)
    ax.legend(title="Scoring Method")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_combined_sensitivity(
    all_results: dict[str, AblationResults],
    metric: str = PRIMARY_METRIC,
    save_path: Path | None = None,
):
    """
    Plot combined sensitivity curves for all regularization parameters.

    One subplot per parameter, lines for each DGP, aggregated across seeds.
    """
    setup_plot_style()

    params = [p for p in all_results.keys() if p != "scoring"]
    n_params = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    colors = sns.color_palette("husl", len(DGPS))
    dgp_colors = dict(zip(DGPS.keys(), colors))

    for ax, param_name in zip(axes, params):
        df = all_results[param_name].to_dataframe()

        for dgp in DGPS.keys():
            dgp_df = df[df["dgp"] == dgp]
            grouped = dgp_df.groupby("param_value")[metric]
            means = grouped.mean()
            stds = grouped.std()
            n = grouped.count()
            ci = 1.96 * stds / np.sqrt(n)

            x = means.index.values
            y = means.values

            # Handle log scale
            if param_name != "min_samples_leaf":
                x_plot = np.where(x == 0, 1e-5, x)
                ax.set_xscale("log")
            else:
                x_plot = x

            sort_idx = np.argsort(x_plot)
            x_plot = x_plot[sort_idx]
            y = y[sort_idx]
            ci_sorted = ci.values[sort_idx]

            ax.plot(x_plot, y, "o-", color=dgp_colors[dgp], linewidth=1.5, markersize=4, label=dgp)
            ax.fill_between(x_plot, y - ci_sorted, y + ci_sorted, alpha=0.15, color=dgp_colors[dgp])

        # Default value
        default_val = DEFAULT_PARAMS.get(param_name, None)
        if default_val is not None and default_val > 0:
            ax.axvline(default_val, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel(param_name)
        ax.set_ylabel(metric.upper() if ax == axes[0] else "")
        ax.set_title(param_name)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(DGPS), bbox_to_anchor=(0.5, 1.08))

    plt.suptitle(f"BDF Parameter Sensitivity (metric: {metric})", fontsize=14, y=1.12)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_heatmap_summary(
    all_results: dict[str, AblationResults],
    metric: str = PRIMARY_METRIC,
    save_path: Path | None = None,
):
    """
    Create a heatmap showing relative performance across parameters and DGPs.

    Normalized to show % change from default.
    """
    setup_plot_style()

    params = [p for p in all_results.keys() if p != "scoring"]

    # Build summary data
    summary_data = []
    for param_name in params:
        df = all_results[param_name].to_dataframe()
        default_val = DEFAULT_PARAMS.get(param_name)

        for dgp in DGPS.keys():
            dgp_df = df[df["dgp"] == dgp]

            # Get default performance
            default_perf = dgp_df[dgp_df["param_value"] == default_val][metric].mean()

            # Get best and worst
            grouped = dgp_df.groupby("param_value")[metric].mean()
            best_val = grouped.idxmin()
            best_perf = grouped.min()
            worst_perf = grouped.max()

            # Relative change (%)
            rel_change = 100 * (best_perf - default_perf) / default_perf if default_perf != 0 else 0

            summary_data.append(
                {
                    "param": param_name,
                    "dgp": dgp,
                    "default_perf": default_perf,
                    "best_perf": best_perf,
                    "best_val": best_val,
                    "worst_perf": worst_perf,
                    "rel_change_pct": rel_change,
                    "range_pct": 100 * (worst_perf - best_perf) / best_perf if best_perf != 0 else 0,
                }
            )

    summary_df = pd.DataFrame(summary_data)

    # Create heatmap of relative change
    pivot = summary_df.pivot(index="dgp", columns="param", values="rel_change_pct")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=0,
        ax=ax,
        cbar_kws={"label": f"% Change in {metric.upper()} from Default"},
    )
    ax.set_title(f"Parameter Sensitivity Summary\n(negative = improvement over default)")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("DGP")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")

    plt.close()

    return summary_df


# =============================================================================
# Summary Table Generation
# =============================================================================


def generate_summary_tables(
    all_results: dict[str, AblationResults],
    save_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Generate summary tables for the ablation study."""
    tables = {}

    # Table 1: Best values per parameter per DGP
    best_values = []
    for param_name, results in all_results.items():
        if param_name == "scoring":
            continue

        df = results.to_dataframe()
        for dgp in DGPS.keys():
            dgp_df = df[df["dgp"] == dgp]
            grouped = dgp_df.groupby("param_value")[PRIMARY_METRIC].mean()
            best_val = grouped.idxmin()
            best_metric = grouped.min()
            default_val = DEFAULT_PARAMS.get(param_name)
            default_metric = grouped.get(default_val, float("nan"))

            best_values.append(
                {
                    "Parameter": param_name,
                    "DGP": dgp,
                    "Best Value": best_val,
                    f"Best {PRIMARY_METRIC.upper()}": f"{best_metric:.4f}",
                    f"Default {PRIMARY_METRIC.upper()}": f"{default_metric:.4f}",
                    "Improvement (%)": f"{100*(default_metric-best_metric)/default_metric:.1f}",
                }
            )

    tables["best_values"] = pd.DataFrame(best_values)

    # Table 2: Scoring method comparison
    if "scoring" in all_results:
        df = all_results["scoring"].to_dataframe()
        scoring_summary = []

        for dgp in DGPS.keys():
            dgp_df = df[df["dgp"] == dgp]
            row = {"DGP": dgp}

            for scoring in SCORING_CONFIGS.keys():
                subset = dgp_df[dgp_df["param_value"] == scoring]
                mean = subset[PRIMARY_METRIC].mean()
                std = subset[PRIMARY_METRIC].std()
                row[f"{scoring} (mean±std)"] = f"{mean:.4f}±{std:.4f}"

            # Statistical test: nle vs nll
            nle_vals = dgp_df[dgp_df["param_value"] == "nle"][PRIMARY_METRIC].values
            nll_vals = dgp_df[dgp_df["param_value"] == "nll"][PRIMARY_METRIC].values
            if len(nle_vals) > 1 and len(nll_vals) > 1:
                stat, pval = stats.wilcoxon(nle_vals, nll_vals, alternative="less")
                row["NLE < NLL (p-value)"] = f"{pval:.4f}"
            else:
                row["NLE < NLL (p-value)"] = "N/A"

            scoring_summary.append(row)

        tables["scoring_comparison"] = pd.DataFrame(scoring_summary)

    # Table 3: Overall sensitivity ranking
    sensitivity_ranking = []
    for param_name, results in all_results.items():
        if param_name == "scoring":
            continue

        df = results.to_dataframe()
        grouped = df.groupby(["dgp", "param_value"])[PRIMARY_METRIC].mean().reset_index()

        # Compute range (max - min) per DGP, then average
        ranges = []
        for dgp in DGPS.keys():
            dgp_vals = grouped[grouped["dgp"] == dgp][PRIMARY_METRIC]
            ranges.append(dgp_vals.max() - dgp_vals.min())

        avg_range = np.mean(ranges)
        avg_metric = df[PRIMARY_METRIC].mean()

        sensitivity_ranking.append(
            {
                "Parameter": param_name,
                f"Avg {PRIMARY_METRIC.upper()} Range": f"{avg_range:.4f}",
                "Relative Sensitivity (%)": f"{100*avg_range/avg_metric:.1f}",
            }
        )

    tables["sensitivity_ranking"] = pd.DataFrame(sensitivity_ranking).sort_values(
        "Relative Sensitivity (%)", ascending=False
    )

    # Save tables
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        csv_path = save_dir / f"{name}.csv"
        table.to_csv(csv_path, index=False)
        logger.info(f"Saved table to {csv_path}")

        # Also save as formatted markdown
        md_path = save_dir / f"{name}.md"
        with open(md_path, "w") as f:
            f.write(f"# {name.replace('_', ' ').title()}\n\n")
            f.write(table.to_markdown(index=False))
        logger.info(f"Saved table to {md_path}")

    return tables


# =============================================================================
# Main Execution
# =============================================================================


def run_full_ablation_study(n_seeds: int = N_SEEDS) -> dict[str, AblationResults]:
    """Run the complete ablation study."""
    logger.info("Starting BDF parameter ablation study")
    logger.info(f"DGPs: {list(DGPS.keys())}")
    logger.info(f"Seeds: {n_seeds}")
    logger.info(f"Parameters: {list(ABLATION_GRIDS.keys())} + scoring")

    all_results = {}

    # Run regularization parameter ablations
    for param_name, param_grid in ABLATION_GRIDS.items():
        results = run_regularization_ablation(param_name, param_grid, n_seeds)
        results.save(RESULTS_DIR / f"ablation_{param_name}.json")
        all_results[param_name] = results

    # Run scoring method ablation
    scoring_results = run_scoring_ablation(n_seeds)
    scoring_results.save(RESULTS_DIR / f"ablation_scoring.json")
    all_results["scoring"] = scoring_results

    return all_results


def generate_all_plots(all_results: dict[str, AblationResults]):
    """Generate all plots from results."""
    logger.info("Generating plots...")

    # Individual sensitivity curves
    for param_name, results in all_results.items():
        if param_name == "scoring":
            continue
        df = results.to_dataframe()
        plot_sensitivity_curve(
            df,
            param_name,
            metric=PRIMARY_METRIC,
            log_scale=(param_name != "min_samples_leaf"),
            save_path=PLOTS_DIR / f"sensitivity_{param_name}.png",
        )

    # Scoring comparison
    if "scoring" in all_results:
        df = all_results["scoring"].to_dataframe()
        plot_scoring_comparison(
            df,
            metric=PRIMARY_METRIC,
            save_path=PLOTS_DIR / "scoring_comparison.png",
        )

    # Combined sensitivity plot
    plot_combined_sensitivity(
        all_results,
        metric=PRIMARY_METRIC,
        save_path=PLOTS_DIR / "combined_sensitivity.png",
    )

    # Heatmap summary
    plot_heatmap_summary(
        all_results,
        metric=PRIMARY_METRIC,
        save_path=PLOTS_DIR / "sensitivity_heatmap.png",
    )

    # Additional plots for other metrics
    for metric in ["rmse", "coverage_90"]:
        plot_combined_sensitivity(
            all_results,
            metric=metric,
            save_path=PLOTS_DIR / f"combined_sensitivity_{metric}.png",
        )

    logger.info(f"All plots saved to {PLOTS_DIR}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("BDF Parameter Ablation Study")
    logger.info("=" * 60)

    # Check if results already exist (allow resuming)
    existing_results = {}
    for param_name in list(ABLATION_GRIDS.keys()) + ["scoring"]:
        result_path = RESULTS_DIR / f"ablation_{param_name}.json"
        if result_path.exists():
            logger.info(f"Loading existing results for {param_name}")
            existing_results[param_name] = AblationResults.load(result_path)

    if len(existing_results) == len(ABLATION_GRIDS) + 1:
        logger.info("All results already exist, generating plots and tables only")
        all_results = existing_results
    else:
        # Run the full study
        all_results = run_full_ablation_study()

    # Generate plots
    generate_all_plots(all_results)

    # Generate summary tables
    tables = generate_summary_tables(all_results, RESULTS_DIR / "tables")

    # Print summary to console
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if "sensitivity_ranking" in tables:
        logger.info("\nParameter Sensitivity Ranking:")
        print(tables["sensitivity_ranking"].to_string(index=False))

    if "scoring_comparison" in tables:
        logger.info("\nScoring Method Comparison:")
        print(tables["scoring_comparison"].to_string(index=False))

    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Plots saved to: {PLOTS_DIR}")
    logger.success("Ablation study complete!")


if __name__ == "__main__":
    main()
