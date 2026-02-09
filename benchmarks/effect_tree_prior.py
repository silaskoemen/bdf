"""
Experiment: Effect of Tree Structure Prior Formulations

Compares three tree structure prior formulations:
1. LINEAR (current): penalty = alpha + delta * d
2. DEFER (CART prior): penalty = log((1-p_d)/p_d) where p_d = alpha * delta^d
3. BERNOULLI (full branching process): penalty = log((1-p_d)/p_d) + 2*log(1-p_{d+1})

Mathematical Background:
- CART Prior (Chipman et al. 1998): p_d = alpha * delta^d where alpha, delta in (0,1)
- p_d = probability of splitting at depth d
- DEFER: log-odds against splitting at current node only
- BERNOULLI: includes probability that both new children will stop (terminal leaves)

**IMPORTANT**: This benchmark uses Optuna to tune alpha/delta for each mode separately,
ensuring a fair comparison. Without tuning, different defaults would make comparison unfair.

Generates performance comparison and tree structure statistics.
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.datasets import (
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_regression,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFRegressor
from benchmarks.metrics.regression import (
    REG_POINT_METRICS,
    REG_PROB_METRICS,
    coverage_at_level,
    crps_wrapper,
    precompute_percentiles,
)

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_SEEDS = 5  # Number of random seeds for confidence intervals
N_SAMPLES = 5000  # Dataset size
N_FEATURES = 10  # For make_regression and friedman1
TEST_SIZE = 0.2
N_SAMPLES_PRED = 500  # Number of samples for probabilistic predictions

# Optuna tuning settings
N_OPTUNA_TRIALS = 30  # Number of trials per mode per DGP
TUNING_SAMPLE_SIZE = 200  # Samples for CRPS during tuning (faster)

# Output directories
RESULTS_DIR = Path("benchmarks/results/effect_tree_prior")
PLOTS_DIR = Path("benchmarks/plots/effect_tree_prior")

# =============================================================================
# Tree Prior Search Spaces (for Optuna tuning)
# =============================================================================

# Each mode has different valid parameter ranges and semantics
TREE_PRIOR_SEARCH_SPACES = {
    "linear": {
        # LINEAR mode: alpha is base penalty, delta is depth increment
        # Both can be any non-negative values (typically small)
        "alpha": {"type": "float", "low": 1e-5, "high": 1.0, "log": True},
        "delta": {"type": "float", "low": 1e-5, "high": 1.0, "log": True},
    },
    "defer": {
        # DEFER mode: alpha and delta define p_d = alpha * delta^d
        # Must be in (0, 1) for valid probabilities
        "alpha": {"type": "float", "low": 0.5, "high": 0.999, "log": False},
        "delta": {"type": "float", "low": 0.1, "high": 0.99, "log": False},
    },
    "bernoulli": {
        # BERNOULLI mode: same constraints as DEFER
        "alpha": {"type": "float", "low": 0.5, "high": 0.999, "log": False},
        "delta": {"type": "float", "low": 0.1, "high": 0.99, "log": False},
    },
}

# Default parameters (used when not tuning, e.g., for quick tests)
TREE_PRIOR_DEFAULTS = {
    "linear": {"alpha": 0.0001, "delta": 0.001},
    "defer": {"alpha": 0.95, "delta": 0.5},
    "bernoulli": {"alpha": 0.95, "delta": 0.5},
}

# Common model parameters (excluding alpha, delta which are mode-specific)
COMMON_MODEL_PARAMS = {
    "n_trees": 50,
    "max_depth": 50,
    "min_samples_leaf": 5,
    "min_samples_split": 20,
    "min_child_weight": 10,
    "subsample": 0.9,
    "colsample": 0.9,
    "gamma": 0.1,
    "eta": 0.01,
}

# Distribution params (NormalMuNormal)
DIST_PARAMS = {
    "mu_mu": "auto",
    "sigma_mu": "auto",
    "sigma_mu_auto_scale": 3.0,
    "score_method": "nle",
    "score_correction": "bic",
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

POINT_METRICS = ["rmse", "mae"]
PROB_METRICS = ["crps", "interval_score_90"]
COVERAGE_LEVELS = [0.50, 0.90, 0.95]
PRIMARY_METRIC = "crps"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExperimentResult:
    """Single experiment result."""

    dgp: str
    seed: int
    tree_prior_mode: str
    alpha: float
    delta: float
    metrics: dict[str, float]
    tree_stats: dict[str, float]  # avg_depth, avg_nodes, etc.
    fit_time: float


@dataclass
class TreePriorResults:
    """Results for tree prior comparison study."""

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
                "tree_prior_mode": r.tree_prior_mode,
                "alpha": r.alpha,
                "delta": r.delta,
                "fit_time": r.fit_time,
            }
            record.update(r.metrics)
            record.update(r.tree_stats)
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
                    "tree_prior_mode": r.tree_prior_mode,
                    "alpha": r.alpha,
                    "delta": r.delta,
                    "metrics": r.metrics,
                    "tree_stats": r.tree_stats,
                    "fit_time": r.fit_time,
                }
                for r in self.results
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "TreePriorResults":
        """Load results from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        results = [
            ExperimentResult(
                dgp=r["dgp"],
                seed=r["seed"],
                tree_prior_mode=r["tree_prior_mode"],
                alpha=r["alpha"],
                delta=r["delta"],
                metrics=r["metrics"],
                tree_stats=r["tree_stats"],
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


def compute_tree_stats(model: BDFRegressor) -> dict[str, float]:
    """Compute tree structure statistics."""
    depths = [tree.get_max_depth() for tree in model.trees]
    nodes = [tree.count_nodes() for tree in model.trees]

    return {
        "avg_depth": float(np.mean(depths)),
        "std_depth": float(np.std(depths)),
        "max_depth": float(np.max(depths)),
        "min_depth": float(np.min(depths)),
        "avg_nodes": float(np.mean(nodes)),
        "std_nodes": float(np.std(nodes)),
        "max_nodes": float(np.max(nodes)),
        "min_nodes": float(np.min(nodes)),
    }


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
                    metrics[metric_name] = float(metric_spec.func(y_test, y_pred_samples))

        # Coverage metrics
        for level in COVERAGE_LEVELS:
            level_pct = int(level * 100)
            metrics[f"coverage_{level_pct}"] = float(
                coverage_at_level(y_test, y_pred_samples, level=level, precomputed=precomputed)
            )
    except Exception as e:
        logger.warning(f"Failed to compute probabilistic metrics: {e}")
        for metric_name in PROB_METRICS:
            metrics[metric_name] = float("nan")
        for level in COVERAGE_LEVELS:
            level_pct = int(level * 100)
            metrics[f"coverage_{level_pct}"] = float("nan")

    return metrics


def tune_tree_prior_mode(
    mode: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
    seed: int = SEED,
) -> dict[str, float]:
    """Tune alpha and delta for a specific tree prior mode using Optuna.

    Uses CRPS as the optimization objective for fair probabilistic comparison.

    Args:
        mode: Tree prior mode ("linear", "defer", or "bernoulli")
        X_train, y_train: Training data
        X_val, y_val: Validation data for evaluating trials
        n_trials: Number of Optuna trials
        seed: Random seed for reproducibility

    Returns:
        Best alpha and delta parameters for this mode
    """
    search_space = TREE_PRIOR_SEARCH_SPACES[mode]

    def objective(trial: optuna.Trial) -> float:
        np.random.seed(seed + trial.number)

        # Sample parameters from mode-specific search space
        alpha = trial.suggest_float(
            "alpha",
            search_space["alpha"]["low"],
            search_space["alpha"]["high"],
            log=search_space["alpha"]["log"],
        )
        delta = trial.suggest_float(
            "delta",
            search_space["delta"]["low"],
            search_space["delta"]["high"],
            log=search_space["delta"]["log"],
        )

        # Build model
        model_kwargs = COMMON_MODEL_PARAMS.copy()
        model_kwargs["random_state"] = seed
        model_kwargs["tree_prior_mode"] = mode
        model_kwargs["alpha"] = alpha
        model_kwargs["delta"] = delta
        # Use fewer trees for faster tuning
        model_kwargs["n_trees"] = 20

        try:
            model = BDFRegressor(dist="NormalMuNormal", params=DIST_PARAMS, **model_kwargs)
            model.fit(X_train, y_train)

            # Compute CRPS on validation set (use fewer samples for speed)
            y_pred_samples = model.predict_samples(X_val, n_samples=TUNING_SAMPLE_SIZE)
            crps = crps_wrapper(y_val, y_pred_samples, quantile_levels=None)
            return float(crps)
        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return float("inf")

    # Create study with reproducible sampler
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "alpha": study.best_params["alpha"],
        "delta": study.best_params["delta"],
        "best_crps": study.best_value,
    }


def run_single_experiment(
    dgp_name: str,
    seed: int,
    prior_config: dict[str, Any],
) -> ExperimentResult:
    """Run a single tree prior experiment with pre-tuned parameters."""
    # Generate data
    X_train, X_test, y_train, y_test = generate_data(dgp_name, seed)

    # Build model kwargs
    model_kwargs = COMMON_MODEL_PARAMS.copy()
    model_kwargs["random_state"] = seed
    model_kwargs.update(prior_config)

    # Create and fit model
    model = BDFRegressor(dist="NormalMuNormal", params=DIST_PARAMS, **model_kwargs)

    start_time = time()
    model.fit(X_train, y_train)
    fit_time = time() - start_time

    # Compute metrics and tree stats
    metrics = compute_metrics(model, X_test, y_test)
    tree_stats = compute_tree_stats(model)

    return ExperimentResult(
        dgp=dgp_name,
        seed=seed,
        tree_prior_mode=prior_config["tree_prior_mode"],
        alpha=prior_config["alpha"],
        delta=prior_config["delta"],
        metrics=metrics,
        tree_stats=tree_stats,
        fit_time=fit_time,
    )


def run_tree_prior_comparison(
    n_seeds: int = N_SEEDS,
    tune: bool = True,
    n_trials: int = N_OPTUNA_TRIALS,
) -> TreePriorResults:
    """Run the complete tree prior comparison study.

    Args:
        n_seeds: Number of evaluation seeds (for confidence intervals)
        tune: If True, tune alpha/delta for each mode using Optuna. If False, use defaults.
        n_trials: Number of Optuna trials per mode per DGP (if tune=True)
    """
    modes = list(TREE_PRIOR_SEARCH_SPACES.keys())

    logger.info("Starting tree prior comparison study")
    logger.info(f"DGPs: {list(DGPS.keys())}")
    logger.info(f"Seeds: {n_seeds}")
    logger.info(f"Prior modes: {modes}")
    logger.info(f"Tuning: {tune} (n_trials={n_trials if tune else 'N/A'})")

    results = TreePriorResults(
        config={
            "n_seeds": n_seeds,
            "n_samples": N_SAMPLES,
            "n_optuna_trials": n_trials if tune else 0,
            "tuned": tune,
            "search_spaces": TREE_PRIOR_SEARCH_SPACES if tune else None,
            "common_params": COMMON_MODEL_PARAMS,
            "dgps": list(DGPS.keys()),
        }
    )

    # Store tuned parameters for each (dgp, mode) combination
    tuned_params: dict[tuple[str, str], dict[str, float]] = {}

    # Phase 1: Tune parameters for each DGP and mode
    if tune:
        logger.info("Phase 1: Tuning parameters with Optuna...")
        total_tuning = len(DGPS) * len(modes)
        pbar = tqdm(total=total_tuning, desc="Tuning")

        for dgp_name in DGPS.keys():
            # Generate tuning data (use seed 0 for tuning, different seeds for eval)
            X, y = DGPS[dgp_name](N_SAMPLES, SEED)
            X = np.asarray(X)
            y = np.asarray(y)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

            for mode in modes:
                pbar.set_description(f"Tuning {dgp_name}/{mode}")
                try:
                    best_params = tune_tree_prior_mode(
                        mode=mode,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        n_trials=n_trials,
                        seed=SEED,
                    )
                    tuned_params[(dgp_name, mode)] = best_params
                    logger.debug(
                        f"Tuned {dgp_name}/{mode}: alpha={best_params['alpha']:.4f}, "
                        f"delta={best_params['delta']:.4f}, CRPS={best_params['best_crps']:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Tuning failed for {dgp_name}/{mode}: {e}")
                    # Fall back to defaults
                    tuned_params[(dgp_name, mode)] = TREE_PRIOR_DEFAULTS[mode].copy()
                pbar.update(1)

        pbar.close()

        # Store tuned params in results
        results.config["tuned_params"] = {f"{dgp}_{mode}": params for (dgp, mode), params in tuned_params.items()}
    else:
        # Use defaults
        for dgp_name in DGPS.keys():
            for mode in modes:
                tuned_params[(dgp_name, mode)] = TREE_PRIOR_DEFAULTS[mode].copy()

    # Phase 2: Evaluate with tuned/default parameters
    logger.info("Phase 2: Evaluating with tuned parameters...")
    total_experiments = len(DGPS) * len(modes) * n_seeds
    pbar = tqdm(total=total_experiments, desc="Evaluating")

    for dgp_name in DGPS.keys():
        for mode in modes:
            params = tuned_params[(dgp_name, mode)]
            prior_config = {
                "tree_prior_mode": mode,
                "alpha": params["alpha"],
                "delta": params["delta"],
            }

            for seed in range(SEED, SEED + n_seeds):
                pbar.set_description(f"Eval {dgp_name}/{mode}/seed{seed}")
                try:
                    result = run_single_experiment(
                        dgp_name=dgp_name,
                        seed=seed,
                        prior_config=prior_config,
                    )
                    results.results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {dgp_name}, {mode}, seed={seed}: {e}")
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


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    save_path: Path | None = None,
):
    """
    Plot comparison of tree prior modes across DGPs.

    Bar plot with error bars showing mean +/- 95% CI.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    dgps = df["dgp"].unique()
    modes = df["tree_prior_mode"].unique()
    x = np.arange(len(dgps))
    width = 0.25

    colors = {"linear": "#2ecc71", "defer": "#3498db", "bernoulli": "#e74c3c"}

    for i, mode in enumerate(modes):
        means = []
        cis = []
        for dgp in dgps:
            subset = df[(df["dgp"] == dgp) & (df["tree_prior_mode"] == mode)]
            mean = subset[metric].mean()
            std = subset[metric].std()
            n = len(subset)
            ci = 1.96 * std / np.sqrt(n) if n > 1 else 0
            means.append(mean)
            cis.append(ci)

        ax.bar(
            x + i * width,
            means,
            width,
            label=mode.upper(),
            color=colors.get(mode, f"C{i}"),
            yerr=cis,
            capsize=3,
        )

    ax.set_xlabel("DGP")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Tree Prior Comparison (metric: {metric})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(dgps)
    ax.legend(title="Prior Mode")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_tree_structure_comparison(
    df: pd.DataFrame,
    save_path: Path | None = None,
):
    """Plot tree structure statistics comparison."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"linear": "#2ecc71", "defer": "#3498db", "bernoulli": "#e74c3c"}
    dgps = df["dgp"].unique()
    modes = df["tree_prior_mode"].unique()

    # Plot 1: Average depth
    ax = axes[0]
    x = np.arange(len(dgps))
    width = 0.25

    for i, mode in enumerate(modes):
        means = []
        stds = []
        for dgp in dgps:
            subset = df[(df["dgp"] == dgp) & (df["tree_prior_mode"] == mode)]
            means.append(subset["avg_depth"].mean())
            stds.append(subset["avg_depth"].std())

        ax.bar(
            x + i * width,
            means,
            width,
            label=mode.upper(),
            color=colors.get(mode, f"C{i}"),
            yerr=stds,
            capsize=3,
        )

    ax.set_xlabel("DGP")
    ax.set_ylabel("Average Tree Depth")
    ax.set_title("Tree Depth by Prior Mode")
    ax.set_xticks(x + width)
    ax.set_xticklabels(dgps)
    ax.legend(title="Prior Mode")

    # Plot 2: Average nodes
    ax = axes[1]
    for i, mode in enumerate(modes):
        means = []
        stds = []
        for dgp in dgps:
            subset = df[(df["dgp"] == dgp) & (df["tree_prior_mode"] == mode)]
            means.append(subset["avg_nodes"].mean())
            stds.append(subset["avg_nodes"].std())

        ax.bar(
            x + i * width,
            means,
            width,
            label=mode.upper(),
            color=colors.get(mode, f"C{i}"),
            yerr=stds,
            capsize=3,
        )

    ax.set_xlabel("DGP")
    ax.set_ylabel("Average Tree Nodes")
    ax.set_title("Tree Size by Prior Mode")
    ax.set_xticks(x + width)
    ax.set_xticklabels(dgps)
    ax.legend(title="Prior Mode")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_multi_metric_comparison(
    df: pd.DataFrame,
    metrics: list[str],
    save_path: Path | None = None,
):
    """Plot comparison across multiple metrics."""
    setup_plot_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = {"LINEAR": "#2ecc71", "DEFER": "#3498db", "BERNOULLI": "#e74c3c"}
    modes = df["tree_prior_mode"].unique()

    for ax, metric in zip(axes, metrics):
        # Aggregate across DGPs
        data_for_plot = []
        for mode in modes:
            values = df[df["tree_prior_mode"] == mode][metric].values
            for v in values:
                data_for_plot.append({"mode": mode.upper(), "value": v})

        plot_df = pd.DataFrame(data_for_plot)

        sns.boxplot(
            data=plot_df,
            x="mode",
            y="value",
            hue="mode",
            palette=colors,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("Prior Mode")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()}")

    plt.suptitle("Multi-Metric Comparison (Aggregated Across DGPs)", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.close()


# =============================================================================
# Summary Table Generation
# =============================================================================


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without external dependencies."""
    lines = []
    # Header
    lines.append("| " + " | ".join(str(c) for c in df.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in df.columns) + " |")
    # Rows
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def generate_summary_tables(
    results: TreePriorResults,
    save_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Generate summary tables."""
    tables = {}
    df = results.to_dataframe()

    modes = df["tree_prior_mode"].unique()

    # Table 1: Overall performance by prior mode
    overall_summary = []
    for mode in modes:
        mode_df = df[df["tree_prior_mode"] == mode]
        # Get average alpha/delta across DGPs (they may differ if tuned per-DGP)
        avg_alpha = mode_df["alpha"].mean()
        avg_delta = mode_df["delta"].mean()

        row = {
            "Prior Mode": mode.upper(),
            "Avg Alpha": f"{avg_alpha:.4f}",
            "Avg Delta": f"{avg_delta:.4f}",
        }

        for metric in POINT_METRICS + PROB_METRICS:
            if metric in mode_df.columns:
                mean = mode_df[metric].mean()
                std = mode_df[metric].std()
                row[f"{metric.upper()} (mean+/-std)"] = f"{mean:.4f}+/-{std:.4f}"

        row["Avg Depth"] = f"{mode_df['avg_depth'].mean():.1f}"
        row["Avg Nodes"] = f"{mode_df['avg_nodes'].mean():.1f}"
        row["Fit Time (s)"] = f"{mode_df['fit_time'].mean():.2f}"

        overall_summary.append(row)

    tables["overall_summary"] = pd.DataFrame(overall_summary)

    # Table 2: Performance by DGP (with tuned params)
    dgp_summary = []
    for dgp in DGPS.keys():
        for mode in modes:
            subset = df[(df["dgp"] == dgp) & (df["tree_prior_mode"] == mode)]
            if subset.empty:
                continue
            row = {
                "DGP": dgp,
                "Prior Mode": mode.upper(),
                "Alpha": f"{subset['alpha'].iloc[0]:.4f}",
                "Delta": f"{subset['delta'].iloc[0]:.4f}",
                "CRPS": f"{subset['crps'].mean():.4f}",
                "RMSE": f"{subset['rmse'].mean():.4f}",
                "Avg Depth": f"{subset['avg_depth'].mean():.1f}",
                "Avg Nodes": f"{subset['avg_nodes'].mean():.1f}",
            }
            dgp_summary.append(row)

    tables["dgp_summary"] = pd.DataFrame(dgp_summary)

    # Table 3: Coverage summary
    coverage_summary = []
    for mode in modes:
        mode_df = df[df["tree_prior_mode"] == mode]
        row = {"Prior Mode": mode.upper()}
        for level in COVERAGE_LEVELS:
            level_pct = int(level * 100)
            col = f"coverage_{level_pct}"
            if col in mode_df.columns:
                mean = mode_df[col].mean()
                row[f"Coverage {level_pct}%"] = f"{mean:.3f}"
        coverage_summary.append(row)

    tables["coverage_summary"] = pd.DataFrame(coverage_summary)

    # Save tables
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        csv_path = save_dir / f"{name}.csv"
        table.to_csv(csv_path, index=False)
        logger.info(f"Saved table to {csv_path}")

        # Save markdown version (simple format, no tabulate dependency)
        md_path = save_dir / f"{name}.md"
        with open(md_path, "w") as f:
            f.write(f"# {name.replace('_', ' ').title()}\n\n")
            f.write(_dataframe_to_markdown(table))
        logger.info(f"Saved table to {md_path}")

    return tables


# =============================================================================
# Main Execution
# =============================================================================


def generate_all_plots(results: TreePriorResults):
    """Generate all plots from results."""
    logger.info("Generating plots...")
    df = results.to_dataframe()

    # Primary metric comparison
    plot_metric_comparison(
        df,
        metric=PRIMARY_METRIC,
        save_path=PLOTS_DIR / "crps_comparison.png",
    )

    # RMSE comparison
    plot_metric_comparison(
        df,
        metric="rmse",
        save_path=PLOTS_DIR / "rmse_comparison.png",
    )

    # Tree structure comparison
    plot_tree_structure_comparison(
        df,
        save_path=PLOTS_DIR / "tree_structure.png",
    )

    # Multi-metric comparison
    plot_multi_metric_comparison(
        df,
        metrics=["crps", "rmse", "coverage_90"],
        save_path=PLOTS_DIR / "multi_metric.png",
    )

    logger.info(f"All plots saved to {PLOTS_DIR}")


def main(
    n_seeds: int = N_SEEDS,
    force_rerun: bool = False,
    tune: bool = True,
    n_trials: int = N_OPTUNA_TRIALS,
):
    """Main entry point.

    Args:
        n_seeds: Number of evaluation seeds for confidence intervals
        force_rerun: If True, rerun even if results exist
        tune: If True, tune alpha/delta with Optuna for fair comparison.
              If False, use default parameters (faster but potentially unfair).
        n_trials: Number of Optuna trials per mode per DGP
    """
    logger.info("=" * 60)
    logger.info("Tree Structure Prior Comparison Study")
    logger.info("=" * 60)

    suffix = "_tuned" if tune else "_defaults"
    result_path = RESULTS_DIR / f"tree_prior_comparison{suffix}.json"

    if result_path.exists() and not force_rerun:
        logger.info("Loading existing results...")
        results = TreePriorResults.load(result_path)
    else:
        results = run_tree_prior_comparison(n_seeds, tune=tune, n_trials=n_trials)
        results.save(result_path)

    generate_all_plots(results)
    tables = generate_summary_tables(results, RESULTS_DIR / "tables")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if "overall_summary" in tables:
        logger.info("\nOverall Performance Summary:")
        print(tables["overall_summary"].to_string(index=False))

    if "coverage_summary" in tables:
        logger.info("\nCoverage Summary:")
        print(tables["coverage_summary"].to_string(index=False))

    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Plots saved to: {PLOTS_DIR}")
    logger.success("Tree prior comparison complete!")


if __name__ == "__main__":
    # Default behavior: tune with 30 trials, 5 seeds, always rerun for fresh results
    # This ensures fair comparison by tuning alpha/delta for each mode
    main(n_seeds=N_SEEDS, force_rerun=True, tune=True, n_trials=N_OPTUNA_TRIALS)
