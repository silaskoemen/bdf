"""
Computational Complexity Analysis: BDF vs Random Forest

Empirical analysis of BDF's computational complexity compared to Random Forest:
1. Sample size scaling (n): Fix d, T, vary n
2. Feature scaling (d): Fix n, T, vary d
3. Tree scaling (T): Fix n, d, vary T

Outputs:
- JSON results with all measurements
- Log-log plots with fitted scaling exponents and O(x^1) reference lines
- Main-text 3-panel comparison figure (BDF vs RF)
- Overhead ratio table (BDF/RF constant factor)
- Markdown report
"""

import gc
import json
import os
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFRegressor
from benchmarks.utils.style import MODEL_COLORS, MODEL_DISPLAY_NAMES, MODEL_MARKERS

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_REPEATS = 5  # Number of repetitions for timing stability

# Scaling grids
N_SAMPLES_GRID = [100, 250, 500, 1000, 2500, 5000, 10000]
N_FEATURES_GRID = [5, 10, 20, 50, 100, 200]
N_TREES_GRID = [10, 20, 50, 100, 200]

# Fixed values when varying other dimensions
FIXED_N_SAMPLES = 2000
FIXED_N_FEATURES = 10
FIXED_N_TREES = 50

# BDF configuration (fixed hyperparameters for complexity analysis - no tuning)
BDF_CONFIG = {
    "dist": "NormalMuNormal",
    "n_trees": FIXED_N_TREES,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "alpha": 0.001,
    "gamma": 0.1,
    "delta": 0.01,
    "subsample": 0.9,
    "colsample": 0.9,
    "eta": 0.01,
    "random_state": SEED,
}

BDF_PARAMS = {
    "mu_mu": "auto",
    "sigma_mu": "auto",
    "sigma_mu_auto_scale": 1.0,
    "score_method": "nll",
    "score_correction": "bic",
}

# Random Forest configuration (matched to BDF where possible)
RF_CONFIG = {
    "n_estimators": FIXED_N_TREES,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "max_features": 0.9,
    "random_state": SEED,
}

RESULTS_DIR = "benchmarks/results/complexity_analysis"
PLOTS_DIR = "benchmarks/plots/complexity_analysis"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TimingResult:
    """Single timing measurement."""

    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    times: list[float] = field(default_factory=list)


@dataclass
class MemoryResult:
    """Memory usage measurement."""

    peak_memory_mb: float
    current_memory_mb: float


@dataclass
class ScalingResult:
    """Result for a single scaling experiment."""

    parameter_name: str
    parameter_values: list[int]
    timing_results: dict[int, TimingResult]
    memory_results: dict[int, MemoryResult]
    fixed_params: dict[str, int]
    estimated_slope: float | None = None
    r_squared: float | None = None


# =============================================================================
# Utility Functions
# =============================================================================


def generate_data(n_samples: int, n_features: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, max(5, n_features // 2)),
        noise=1.0,
        random_state=seed,
    )
    return X.astype(np.float64), y.astype(np.float64)


def warmup_models() -> None:
    """Run a dummy fit for each model to eliminate JIT/import overhead from timing."""
    logger.info("Warming up models (dummy fit to eliminate startup overhead)...")
    X, y = generate_data(200, 5, seed=0)

    # Warmup BDF (triggers Rust library loading, numba JIT, etc.)
    bdf = BDFRegressor(**{**BDF_CONFIG, "n_trees": 5}, params=BDF_PARAMS)
    bdf.fit(X, y)
    del bdf

    # Warmup RF
    rf = RandomForestRegressor(**{**RF_CONFIG, "n_estimators": 5})
    rf.fit(X, y)
    del rf

    gc.collect()
    logger.info("Warmup complete")


def measure_fit_time(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = N_REPEATS,
) -> TimingResult:
    """Measure fit time with multiple repeats.

    A throwaway warmup call is run first (not timed) to absorb any
    per-configuration overhead (memory allocation, code-path specialization).
    """
    # Warmup: untimed fit to absorb first-call overhead for this config
    warmup = model_factory()
    warmup.fit(X, y)
    del warmup
    gc.collect()

    times = []

    for _ in range(n_repeats):
        gc.collect()
        model = model_factory()

        start = time()
        model.fit(X, y)
        elapsed = time() - start
        times.append(elapsed)

        del model

    return TimingResult(
        mean_time=float(np.mean(times)),
        std_time=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        times=times,
    )


def measure_memory(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
) -> MemoryResult:
    """Measure peak memory usage during fitting."""
    gc.collect()

    tracemalloc.start()
    model = model_factory()
    model.fit(X, y)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    del model
    gc.collect()

    return MemoryResult(
        peak_memory_mb=peak / (1024 * 1024),
        current_memory_mb=current / (1024 * 1024),
    )


def fit_log_log_slope(x_values: list[int], y_values: list[float]) -> tuple[float, float]:
    """Fit slope in log-log space to determine scaling exponent.

    Returns:
        (slope, r_squared) where slope is the scaling exponent
        e.g., slope=1.0 means O(n), slope=2.0 means O(n^2)
    """
    log_x = np.log(x_values)
    log_y = np.log(y_values)

    slope, _, r_value, _, _ = stats.linregress(log_x, log_y)

    return float(slope), float(r_value**2)


def compute_overhead_ratios(all_results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compute BDF/RF timing ratio across all grid points for each axis."""
    ratios = {}

    for axis in ["n_scaling", "d_scaling", "tree_scaling"]:
        bdf_timing = all_results["models"]["BDF"][axis]["timing"]
        rf_timing = all_results["models"]["RandomForest"][axis]["timing"]

        axis_ratios = []
        for key in bdf_timing:
            if key in rf_timing:
                ratio = bdf_timing[key]["mean_time"] / rf_timing[key]["mean_time"]
                axis_ratios.append(ratio)

        ratios[axis] = {
            "mean": float(np.mean(axis_ratios)),
            "std": float(np.std(axis_ratios)),
            "min": float(np.min(axis_ratios)),
            "max": float(np.max(axis_ratios)),
            "per_point": axis_ratios,
        }

    return ratios


# =============================================================================
# Model Factories
# =============================================================================


def create_bdf_factory(n_trees: int | None = None):
    """Create BDF model factory with optional n_trees override."""

    def factory():
        config = dict(BDF_CONFIG)
        if n_trees is not None:
            config["n_trees"] = n_trees
        return BDFRegressor(**config, params=BDF_PARAMS)

    return factory


def create_rf_factory(n_estimators: int | None = None):
    """Create Random Forest model factory."""

    def factory():
        config = dict(RF_CONFIG)
        if n_estimators is not None:
            config["n_estimators"] = n_estimators
        return RandomForestRegressor(**config)

    return factory


# =============================================================================
# Scaling Experiments
# =============================================================================


def run_sample_size_scaling(
    model_name: str,
    model_factory_creator,
    n_samples_grid: list[int] | None = None,
    n_features: int = FIXED_N_FEATURES,
    n_trees: int = FIXED_N_TREES,
) -> ScalingResult:
    """Measure scaling with sample size n."""
    if n_samples_grid is None:
        n_samples_grid = list(N_SAMPLES_GRID)

    logger.info(f"Running sample size scaling for {model_name}")

    timing_results = {}
    memory_results = {}

    for n_samples in tqdm(n_samples_grid, desc=f"{model_name} n-scaling"):
        X, y = generate_data(n_samples, n_features)
        factory = model_factory_creator(n_trees)

        timing_results[n_samples] = measure_fit_time(factory, X, y)
        memory_results[n_samples] = measure_memory(factory, X, y)

        logger.debug(
            f"  n={n_samples}: time={timing_results[n_samples].mean_time:.3f}s, "
            f"memory={memory_results[n_samples].peak_memory_mb:.1f}MB"
        )

    mean_times = [timing_results[n].mean_time for n in n_samples_grid]
    slope, r_squared = fit_log_log_slope(n_samples_grid, mean_times)

    return ScalingResult(
        parameter_name="n_samples",
        parameter_values=n_samples_grid,
        timing_results=timing_results,
        memory_results=memory_results,
        fixed_params={"n_features": n_features, "n_trees": n_trees},
        estimated_slope=slope,
        r_squared=r_squared,
    )


def run_feature_scaling(
    model_name: str,
    model_factory_creator,
    n_features_grid: list[int] | None = None,
    n_samples: int = FIXED_N_SAMPLES,
    n_trees: int = FIXED_N_TREES,
) -> ScalingResult:
    """Measure scaling with number of features d."""
    if n_features_grid is None:
        n_features_grid = list(N_FEATURES_GRID)

    logger.info(f"Running feature scaling for {model_name}")

    timing_results = {}
    memory_results = {}

    for n_features in tqdm(n_features_grid, desc=f"{model_name} d-scaling"):
        X, y = generate_data(n_samples, n_features)
        factory = model_factory_creator(n_trees)

        timing_results[n_features] = measure_fit_time(factory, X, y)
        memory_results[n_features] = measure_memory(factory, X, y)

        logger.debug(
            f"  d={n_features}: time={timing_results[n_features].mean_time:.3f}s, "
            f"memory={memory_results[n_features].peak_memory_mb:.1f}MB"
        )

    mean_times = [timing_results[d].mean_time for d in n_features_grid]
    slope, r_squared = fit_log_log_slope(n_features_grid, mean_times)

    return ScalingResult(
        parameter_name="n_features",
        parameter_values=n_features_grid,
        timing_results=timing_results,
        memory_results=memory_results,
        fixed_params={"n_samples": n_samples, "n_trees": n_trees},
        estimated_slope=slope,
        r_squared=r_squared,
    )


def run_tree_scaling(
    model_name: str,
    model_factory_creator,
    n_trees_grid: list[int] | None = None,
    n_samples: int = FIXED_N_SAMPLES,
    n_features: int = FIXED_N_FEATURES,
) -> ScalingResult:
    """Measure scaling with number of trees."""
    if n_trees_grid is None:
        n_trees_grid = list(N_TREES_GRID)

    logger.info(f"Running tree scaling for {model_name}")

    timing_results = {}
    memory_results = {}

    for n_trees in tqdm(n_trees_grid, desc=f"{model_name} tree-scaling"):
        X, y = generate_data(n_samples, n_features)
        factory = model_factory_creator(n_trees)

        timing_results[n_trees] = measure_fit_time(factory, X, y)
        memory_results[n_trees] = measure_memory(factory, X, y)

        logger.debug(
            f"  n_trees={n_trees}: time={timing_results[n_trees].mean_time:.3f}s, "
            f"memory={memory_results[n_trees].peak_memory_mb:.1f}MB"
        )

    mean_times = [timing_results[t].mean_time for t in n_trees_grid]
    slope, r_squared = fit_log_log_slope(n_trees_grid, mean_times)

    return ScalingResult(
        parameter_name="n_trees",
        parameter_values=n_trees_grid,
        timing_results=timing_results,
        memory_results=memory_results,
        fixed_params={"n_samples": n_samples, "n_features": n_features},
        estimated_slope=slope,
        r_squared=r_squared,
    )


# =============================================================================
# Results Serialization
# =============================================================================


def scaling_result_to_dict(result: ScalingResult) -> dict[str, Any]:
    """Convert ScalingResult to JSON-serializable dict."""
    return {
        "parameter_name": result.parameter_name,
        "parameter_values": result.parameter_values,
        "fixed_params": result.fixed_params,
        "estimated_slope": result.estimated_slope,
        "r_squared": result.r_squared,
        "timing": {
            str(k): {
                "mean_time": v.mean_time,
                "std_time": v.std_time,
                "min_time": v.min_time,
                "max_time": v.max_time,
            }
            for k, v in result.timing_results.items()
        },
        "memory": {
            str(k): {
                "peak_memory_mb": v.peak_memory_mb,
                "current_memory_mb": v.current_memory_mb,
            }
            for k, v in result.memory_results.items()
        },
    }


def save_results(results: dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filepath}")


# =============================================================================
# Plotting
# =============================================================================


def setup_plot_style():
    """Set up publication-quality plot style."""
    from benchmarks.utils.style import apply_paper_style

    apply_paper_style()


def plot_main_text_figure(
    all_results: dict[str, Any],
    save_path: str,
):
    """Create the main-text 3-panel figure: BDF vs RF scaling on n, d, T.

    Each panel shows measured data with error bars, fitted slopes in legend,
    and a dashed O(x^1) reference line.
    """
    setup_plot_style()

    _, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        ("n_scaling", "Sample size $n$", "$n$"),
        ("d_scaling", "Features $d$", "$d$"),
        ("tree_scaling", "Trees $T$", "$T$"),
    ]

    for ax, (axis_key, xlabel, var) in zip(axes, panels):
        for model_name in ["BDF", "RandomForest"]:
            data = all_results["models"][model_name][axis_key]
            x_vals = data["parameter_values"]
            y_means = [data["timing"][str(x)]["mean_time"] for x in x_vals]
            y_stds = [data["timing"][str(x)]["std_time"] for x in x_vals]
            slope = data["estimated_slope"]
            r2 = data["r_squared"]

            color = MODEL_COLORS[model_name]
            marker = MODEL_MARKERS[model_name]
            label = MODEL_DISPLAY_NAMES.get(model_name, model_name)

            ax.errorbar(
                x_vals,
                y_means,
                yerr=y_stds,
                fmt=f"{marker}-",
                color=color,
                linewidth=1.5,
                markersize=6,
                capsize=3,
                label=rf"{label} $\mathcal{{O}}({var}^{{{slope:.2f}}})$" + f", $R^2$={r2:.3f}",
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fit time (s)")

        # Add O(x^1) reference line anchored to midpoint of BDF data
        bdf_data = all_results["models"]["BDF"][axis_key]
        bdf_x = bdf_data["parameter_values"]
        bdf_y = [bdf_data["timing"][str(x)]["mean_time"] for x in bdf_x]
        x_arr = np.array(bdf_x, dtype=float)
        y_arr = np.array(bdf_y, dtype=float)
        # Anchor at geometric midpoint
        log_x_mid = np.mean(np.log(x_arr))
        log_y_mid = np.mean(np.log(y_arr))
        c = np.exp(log_y_mid - log_x_mid)  # slope=1 in log-log
        y_ref = c * x_arr
        ax.plot(
            x_arr,
            y_ref,
            "--",
            color="0.5",
            linewidth=1.0,
            alpha=0.6,
            label=rf"$\mathcal{{O}}({var}^{{1}})$ ref.",
            zorder=1,
        )

        ax.legend(fontsize=7.5, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved main-text figure to {save_path}")
    plt.close()


def plot_memory_comparison(
    all_results: dict[str, Any],
    save_path: str,
):
    """Create 3-panel memory comparison figure (appendix material)."""
    setup_plot_style()

    _, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        ("n_scaling", "Sample size $n$"),
        ("d_scaling", "Features $d$"),
        ("tree_scaling", "Trees $T$"),
    ]

    for ax, (axis_key, xlabel) in zip(axes, panels):
        for model_name in ["BDF", "RandomForest"]:
            data = all_results["models"][model_name][axis_key]
            x_vals = data["parameter_values"]
            y_vals = [data["memory"][str(x)]["peak_memory_mb"] for x in x_vals]

            color = MODEL_COLORS[model_name]
            marker = MODEL_MARKERS[model_name]
            label = MODEL_DISPLAY_NAMES.get(model_name, model_name)

            ax.plot(
                x_vals,
                y_vals,
                f"{marker}-",
                color=color,
                linewidth=1.5,
                markersize=6,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Peak memory (MB)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved memory figure to {save_path}")
    plt.close()


def plot_overhead_ratio(
    all_results: dict[str, Any],
    save_path: str,
):
    """Plot BDF/RF overhead ratio across each scaling axis."""
    setup_plot_style()

    _, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        ("n_scaling", "Sample size $n$"),
        ("d_scaling", "Features $d$"),
        ("tree_scaling", "Trees $T$"),
    ]

    for ax, (axis_key, xlabel) in zip(axes, panels):
        bdf_data = all_results["models"]["BDF"][axis_key]
        rf_data = all_results["models"]["RandomForest"][axis_key]

        x_vals = bdf_data["parameter_values"]
        ratios = []
        for x in x_vals:
            bdf_t = bdf_data["timing"][str(x)]["mean_time"]
            rf_t = rf_data["timing"][str(x)]["mean_time"]
            ratios.append(bdf_t / rf_t)

        ax.plot(x_vals, ratios, "o-", color="#2E86AB", linewidth=1.5, markersize=6)
        ax.axhline(y=1.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("BDF / RF time ratio")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logger.info(f"Saved overhead ratio figure to {save_path}")
    plt.close()


# =============================================================================
# Markdown Report Generation
# =============================================================================


def generate_markdown_report(results: dict[str, Any], filepath: str) -> None:
    """Generate markdown report summarizing complexity analysis."""

    overhead = results.get("overhead_ratios", {})

    lines = [
        "# BDF Computational Complexity Analysis",
        "",
        f"Generated: {results['metadata']['timestamp']}",
        "",
        "## Summary",
        "",
        "Empirical analysis of BDF vs Random Forest computational complexity across",
        "three dimensions: sample size (n), feature count (d), and ensemble size (T).",
        "",
        "## Theoretical Complexity",
        "",
        "| Model | Training Complexity | Notes |",
        "|-------|---------------------|-------|",
        "| BDF | O(n x d x T) | Bayesian tree ensemble (Rust split finding) |",
        "| Random Forest | O(n x d x T) | Standard tree ensemble (sklearn) |",
        "",
        "Both scale identically in theory; BDF has a constant overhead from Bayesian",
        "posterior updates and distributional scoring at each split.",
        "",
        "## Empirical Scaling Exponents",
        "",
        "| Model | n Scaling | d Scaling | T Scaling |",
        "|-------|-----------|-----------|-----------|",
    ]

    for model_name in ["BDF", "RandomForest"]:
        data = results["models"][model_name]
        n_s = data["n_scaling"]["estimated_slope"]
        n_r = data["n_scaling"]["r_squared"]
        d_s = data["d_scaling"]["estimated_slope"]
        d_r = data["d_scaling"]["r_squared"]
        t_s = data["tree_scaling"]["estimated_slope"]
        t_r = data["tree_scaling"]["r_squared"]

        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        lines.append(
            f"| {label} | O(n^{n_s:.2f}) R²={n_r:.3f} | "
            f"O(d^{d_s:.2f}) R²={d_r:.3f} | "
            f"O(T^{t_s:.2f}) R²={t_r:.3f} |"
        )

    # Overhead ratio
    if overhead:
        lines.extend(
            [
                "",
                "## Overhead Ratio (BDF / Random Forest)",
                "",
                "| Axis | Mean | Std | Min | Max |",
                "|------|------|-----|-----|-----|",
            ]
        )
        for axis, label in [("n_scaling", "n"), ("d_scaling", "d"), ("tree_scaling", "T")]:
            r = overhead[axis]
            lines.append(f"| {label} | {r['mean']:.2f}x | {r['std']:.2f} | {r['min']:.2f}x | {r['max']:.2f}x |")

    # Timing tables
    for axis_key, axis_label, grid in [
        ("n_scaling", "Sample Size (n)", N_SAMPLES_GRID),
        ("d_scaling", "Feature Count (d)", N_FEATURES_GRID),
        ("tree_scaling", "Trees (T)", N_TREES_GRID),
    ]:
        fixed = results["models"]["BDF"][axis_key]["fixed_params"]
        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())

        lines.extend(
            [
                "",
                f"### {axis_label} Scaling",
                "",
                f"Fixed: {fixed_str}",
                "",
                f"| {axis_label.split('(')[1].rstrip(')')} | BDF Time (s) | RF Time (s) | Ratio |",
                "|---|---|---|---|",
            ]
        )

        for val in grid:
            bdf_t = results["models"]["BDF"][axis_key]["timing"].get(str(val))
            rf_t = results["models"]["RandomForest"][axis_key]["timing"].get(str(val))
            if bdf_t and rf_t:
                ratio = bdf_t["mean_time"] / rf_t["mean_time"]
                lines.append(
                    f"| {val} | {bdf_t['mean_time']:.3f} +/- {bdf_t['std_time']:.3f} | "
                    f"{rf_t['mean_time']:.3f} +/- {rf_t['std_time']:.3f} | {ratio:.2f}x |"
                )

    # Memory
    lines.extend(
        [
            "",
            "### Memory Usage",
            "",
            f"| Model | Peak Memory at n={max(N_SAMPLES_GRID)} (MB) | Peak Memory at d={max(N_FEATURES_GRID)} (MB) |",
            "|-------|---|---|",
        ]
    )

    for model_name in ["BDF", "RandomForest"]:
        data = results["models"][model_name]
        n_mem = data["n_scaling"]["memory"].get(str(max(N_SAMPLES_GRID)), {}).get("peak_memory_mb", "-")
        d_mem = data["d_scaling"]["memory"].get(str(max(N_FEATURES_GRID)), {}).get("peak_memory_mb", "-")

        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        n_str = f"{n_mem:.1f}" if isinstance(n_mem, float) else str(n_mem)
        d_str = f"{d_mem:.1f}" if isinstance(d_mem, float) else str(d_mem)
        lines.append(f"| {label} | {n_str} | {d_str} |")

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            "```python",
            f"SEED = {SEED}",
            f"N_REPEATS = {N_REPEATS}",
            f"N_SAMPLES_GRID = {N_SAMPLES_GRID}",
            f"N_FEATURES_GRID = {N_FEATURES_GRID}",
            f"N_TREES_GRID = {N_TREES_GRID}",
            "```",
            "",
        ]
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved markdown report to {filepath}")


def generate_latex_table(results: dict[str, Any], filepath: str) -> None:
    """Generate LaTeX table with scaling exponents and overhead ratios."""
    overhead = results.get("overhead_ratios", {})

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical scaling exponents and overhead ratios (BDF vs.\ Random Forest).}",
        r"\label{tab:complexity}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & $n$ scaling & $d$ scaling & $T$ scaling & Overhead \\",
        r"\midrule",
    ]

    for model_name in ["BDF", "RandomForest"]:
        data = results["models"][model_name]
        n_s = data["n_scaling"]["estimated_slope"]
        d_s = data["d_scaling"]["estimated_slope"]
        t_s = data["tree_scaling"]["estimated_slope"]
        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        if model_name == "BDF" and overhead:
            # Average overhead across all axes
            all_means = [overhead[a]["mean"] for a in overhead]
            avg_overhead = np.mean(all_means)
            overhead_str = rf"${avg_overhead:.1f}\times$"
        else:
            overhead_str = r"$1\times$ (baseline)"

        lines.append(
            rf"{label} & $\mathcal{{O}}(n^{{{n_s:.2f}}})$ & "
            rf"$\mathcal{{O}}(d^{{{d_s:.2f}}})$ & "
            rf"$\mathcal{{O}}(T^{{{t_s:.2f}}})$ & "
            rf"{overhead_str} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved LaTeX table to {filepath}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run full complexity analysis."""
    logger.info("Starting BDF Computational Complexity Analysis")
    logger.info(f"Configuration: seed={SEED}, repeats={N_REPEATS}")
    logger.info(f"n_samples_grid: {N_SAMPLES_GRID}")
    logger.info(f"n_features_grid: {N_FEATURES_GRID}")
    logger.info(f"n_trees_grid: {N_TREES_GRID}")

    # Warmup to eliminate JIT/import overhead
    warmup_models()

    all_results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "n_repeats": N_REPEATS,
        },
        "config": {
            "n_samples_grid": N_SAMPLES_GRID,
            "n_features_grid": N_FEATURES_GRID,
            "n_trees_grid": N_TREES_GRID,
            "fixed_n_samples": FIXED_N_SAMPLES,
            "fixed_n_features": FIXED_N_FEATURES,
            "fixed_n_trees": FIXED_N_TREES,
            "bdf_config": BDF_CONFIG,
            "bdf_params": BDF_PARAMS,
            "rf_config": RF_CONFIG,
        },
        "models": {},
    }

    # Run experiments for BDF and Random Forest
    models = {
        "BDF": create_bdf_factory,
        "RandomForest": create_rf_factory,
    }

    for model_name, factory_creator in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {model_name}")
        logger.info(f"{'='*50}")

        n_scaling = run_sample_size_scaling(model_name, factory_creator)
        logger.success(f"{model_name} n-scaling: O(n^{n_scaling.estimated_slope:.2f}), R²={n_scaling.r_squared:.3f}")

        d_scaling = run_feature_scaling(model_name, factory_creator)
        logger.success(f"{model_name} d-scaling: O(d^{d_scaling.estimated_slope:.2f}), R²={d_scaling.r_squared:.3f}")

        tree_scaling = run_tree_scaling(model_name, factory_creator)
        logger.success(
            f"{model_name} tree-scaling: O(T^{tree_scaling.estimated_slope:.2f}), R²={tree_scaling.r_squared:.3f}"
        )

        all_results["models"][model_name] = {
            "n_scaling": scaling_result_to_dict(n_scaling),
            "d_scaling": scaling_result_to_dict(d_scaling),
            "tree_scaling": scaling_result_to_dict(tree_scaling),
        }

        save_results(all_results, f"{RESULTS_DIR}/complexity_analysis.json")

    # Compute overhead ratios
    overhead = compute_overhead_ratios(all_results)
    all_results["overhead_ratios"] = overhead
    logger.info(
        f"Overhead BDF/RF — n: {overhead['n_scaling']['mean']:.2f}x, "
        f"d: {overhead['d_scaling']['mean']:.2f}x, "
        f"T: {overhead['tree_scaling']['mean']:.2f}x"
    )

    # Generate plots
    logger.info("\nGenerating plots...")
    plot_main_text_figure(all_results, f"{PLOTS_DIR}/scaling_comparison.pdf")
    plot_memory_comparison(all_results, f"{PLOTS_DIR}/memory_comparison.pdf")
    plot_overhead_ratio(all_results, f"{PLOTS_DIR}/overhead_ratio.pdf")

    # Generate reports
    generate_markdown_report(all_results, f"{RESULTS_DIR}/COMPLEXITY_ANALYSIS.md")
    generate_latex_table(all_results, f"{RESULTS_DIR}/complexity_table.tex")

    # Final save
    save_results(all_results, f"{RESULTS_DIR}/complexity_analysis.json")

    logger.success("\nComplexity analysis complete!")
    logger.info(f"Results: {RESULTS_DIR}/complexity_analysis.json")
    logger.info(f"Report: {RESULTS_DIR}/COMPLEXITY_ANALYSIS.md")
    logger.info(f"LaTeX: {RESULTS_DIR}/complexity_table.tex")
    logger.info(f"Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
