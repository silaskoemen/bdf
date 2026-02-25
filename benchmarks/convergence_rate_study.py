"""
Convergence Rate Study: BDF vs Random Forest

Empirical comparison of learning curves (MSE vs training set size) for BDF
and Random Forest across synthetic DGPs stratified by target function smoothness.

Design:
- DGPs stratified by smoothness class: smooth (trig, polynomial), non-smooth (step),
  and mixed (smooth + jump), at dimensionalities d=1, d=5, d=10
- Sample sizes: n in [50, 100, 200, 500, 1000, 2000, 5000, 10000]
- Models: BDF (NormalMuNormal), Random Forest
- Metric: MSE against true mean function (no observation noise)
- Analysis: log-log plots with fitted convergence rate exponents
- Multiple seeds for error bars

Outputs:
- JSON results with per-seed MSE at each (DGP, model, n) triple
- Log-log convergence plots with fitted slopes and CI bands
- Summary table of convergence exponents
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFRegressor
from benchmarks.utils.style import MODEL_COLORS, MODEL_MARKERS

# =============================================================================
# Configuration
# =============================================================================

SEEDS = [42, 123, 456, 789, 1024]
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
DIMENSIONALITIES = [1, 5, 10]
TEST_N = 2000  # Fixed test set size for stable MSE estimation

RESULTS_DIR = "benchmarks/results/convergence_rate"
PLOTS_DIR = "benchmarks/plots/convergence_rate"

# =============================================================================
# DGP Definitions
# =============================================================================


@dataclass
class DGP:
    """Data generating process with known mean function."""

    name: str
    smoothness_class: str  # "smooth", "non_smooth", "mixed"
    mean_fn: Callable[[np.ndarray], np.ndarray]  # f: (n, d) -> (n,)
    noise_std: float
    description: str
    x_range: tuple[float, float] = (0.0, 1.0)


def _make_smooth_sinusoidal(d: int) -> DGP:
    """sin(2*pi*x1) + 0.5*sin(4*pi*x2) + ... (additive trig)."""

    def mean_fn(X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        result = np.zeros(n)
        for j in range(min(p, 3)):  # First 3 features are informative
            result += np.sin(2 * np.pi * (j + 1) * X[:, j]) / (j + 1)
        return result

    return DGP(
        name=f"sinusoidal_d{d}",
        smoothness_class="smooth",
        mean_fn=mean_fn,
        noise_std=0.1,
        description=f"Additive sinusoidal (d={d})",
    )


def _make_smooth_polynomial(d: int) -> DGP:
    """Degree-3 polynomial in first 3 features."""

    def mean_fn(X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        result = np.zeros(n)
        # Centered polynomial to avoid numerical issues
        for j in range(min(p, 3)):
            xc = X[:, j] - 0.5
            result += (4 * xc) ** 3 - 3 * (4 * xc)  # Chebyshev-like
        result /= min(p, 3)
        return result

    return DGP(
        name=f"polynomial_d{d}",
        smoothness_class="smooth",
        mean_fn=mean_fn,
        noise_std=0.1,
        description=f"Cubic polynomial (d={d})",
    )


def _make_smooth_friedman(d: int) -> DGP:
    """Friedman #1 function (requires d >= 5)."""

    def mean_fn(X: np.ndarray) -> np.ndarray:
        # Standard Friedman #1: 10*sin(pi*x1*x2) + 20*(x3 - 0.5)^2 + 10*x4 + 5*x5
        n, p = X.shape
        result = 10.0 * np.sin(np.pi * X[:, 0] * X[:, min(1, p - 1)])
        if p >= 3:
            result += 20.0 * (X[:, 2] - 0.5) ** 2
        if p >= 4:
            result += 10.0 * X[:, 3]
        if p >= 5:
            result += 5.0 * X[:, 4]
        return result

    return DGP(
        name=f"friedman1_d{d}",
        smoothness_class="smooth",
        mean_fn=mean_fn,
        noise_std=1.0,
        description=f"Friedman #1 (d={d})",
    )


def _make_step_function(d: int) -> DGP:
    """Multi-dimensional step function with 5 levels per informative feature."""

    def mean_fn(X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        step_heights = np.array([0.0, 1.5, -0.5, 2.0, 0.5])
        n_steps = len(step_heights)
        boundaries = np.linspace(0, 1, n_steps + 1)

        result = np.zeros(n)
        for j in range(min(p, 3)):  # First 3 features are informative
            for i in range(n_steps):
                mask = (X[:, j] >= boundaries[i]) & (X[:, j] < boundaries[i + 1])
                result[mask] += step_heights[i] / min(p, 3)
            result[X[:, j] >= boundaries[-1]] += step_heights[-1] / min(p, 3)
        return result

    return DGP(
        name=f"step_d{d}",
        smoothness_class="non_smooth",
        mean_fn=mean_fn,
        noise_std=0.1,
        description=f"Multi-step function (d={d})",
    )


def _make_indicator_function(d: int) -> DGP:
    """XOR-like indicator function: I(x1 > 0.5) XOR I(x2 > 0.5)."""

    def mean_fn(X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        if p >= 2:
            a = (X[:, 0] > 0.5).astype(float)
            b = (X[:, 1] > 0.5).astype(float)
            # XOR: high if exactly one is true
            result = 2.0 * (a + b - 2 * a * b)  # Scale for reasonable signal
        else:
            result = 2.0 * (X[:, 0] > 0.5).astype(float)
        return result

    return DGP(
        name=f"indicator_d{d}",
        smoothness_class="non_smooth",
        mean_fn=mean_fn,
        noise_std=0.1,
        description=f"XOR indicator (d={d})",
    )


def _make_mixed_function(d: int) -> DGP:
    """Smooth function + discontinuous jump: sin(2*pi*x) + 2*I(x1 > 0.5)."""

    def mean_fn(X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        smooth_part = np.sin(2 * np.pi * X[:, 0])
        jump_part = 2.0 * (X[:, min(1, p - 1)] > 0.5).astype(float)
        return smooth_part + jump_part

    return DGP(
        name=f"mixed_d{d}",
        smoothness_class="mixed",
        mean_fn=mean_fn,
        noise_std=0.1,
        description=f"Sinusoidal + jump (d={d})",
    )


def build_dgp_suite() -> list[DGP]:
    """Build the full DGP suite across smoothness classes and dimensionalities."""
    dgps = []
    for d in DIMENSIONALITIES:
        # Smooth
        dgps.append(_make_smooth_sinusoidal(d))
        dgps.append(_make_smooth_polynomial(d))
        if d >= 5:
            dgps.append(_make_smooth_friedman(d))
        # Non-smooth
        dgps.append(_make_step_function(d))
        dgps.append(_make_indicator_function(d))
        # Mixed
        dgps.append(_make_mixed_function(d))
    return dgps


# =============================================================================
# Model Factories
# =============================================================================


def create_bdf(n_train: int) -> BDFRegressor:
    """Create BDF model with reasonable defaults scaled to training size."""
    return BDFRegressor(
        dist="NormalMuNormal",
        n_trees=100,
        max_depth=50,
        min_samples_leaf=max(5, n_train // 100),
        alpha=0.001,
        gamma=0.1,
        delta=0.01,
        subsample=0.9,
        colsample=0.9,
        eta=0.01,
        random_state=42,
        params={
            "mu_mu": "auto",
            "sigma_mu": "auto",
            "sigma_mu_auto_scale": 1.0,
            "score_method": "nle",
            "score_correction": "bic",
        },
    )


def create_rf(n_train: int) -> RandomForestRegressor:
    """Create Random Forest with reasonable defaults."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        min_samples_leaf=max(5, n_train // 100),
        max_features="sqrt",
        random_state=42,
    )


MODEL_FACTORIES: dict[str, Callable] = {
    "BDF": lambda n: create_bdf(n),
    "RandomForest": lambda n: create_rf(n),
}


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_single(
    dgp: DGP,
    model_name: str,
    n_train: int,
    seed: int,
) -> dict[str, float]:
    """Run a single (DGP, model, n, seed) experiment.

    Returns dict with mse (against true mean), fit_time.
    """
    rng = np.random.RandomState(seed)

    d = int(dgp.name.split("_d")[-1])

    # Generate train data
    X_train = rng.uniform(dgp.x_range[0], dgp.x_range[1], size=(n_train, d))
    y_true_train = dgp.mean_fn(X_train)
    y_train = y_true_train + rng.randn(n_train) * dgp.noise_std

    # Generate fixed test data (same seed offset for all n)
    rng_test = np.random.RandomState(seed + 10000)
    X_test = rng_test.uniform(dgp.x_range[0], dgp.x_range[1], size=(TEST_N, d))
    y_true_test = dgp.mean_fn(X_test)

    # Fit model
    factory = MODEL_FACTORIES[model_name]
    model = factory(n_train)

    start = time()
    model.fit(X_train, y_train)
    fit_time = time() - start

    # Predict and evaluate against TRUE mean (not noisy y)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_true_test, y_pred))

    return {"mse": mse, "fit_time": fit_time, "skipped": False}


# =============================================================================
# Convergence Rate Fitting
# =============================================================================


def fit_convergence_rate(
    n_values: list[int],
    mse_values: list[float],
) -> dict[str, float]:
    """Fit log(MSE) = a + b*log(n) via OLS. Returns slope, intercept, R^2, and SE."""
    # Filter out NaN
    valid = [(n, m) for n, m in zip(n_values, mse_values) if np.isfinite(m) and m > 0]
    if len(valid) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan, "slope_se": np.nan}

    ns, ms = zip(*valid)
    log_n = np.log(np.array(ns))
    log_mse = np.log(np.array(ms))

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_mse)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "slope_se": float(std_err),
    }


# =============================================================================
# Plotting
# =============================================================================


def setup_plot_style():
    from benchmarks.utils.style import apply_paper_style

    apply_paper_style()


COLORS = {
    "BDF": MODEL_COLORS.get("BDF", "#C03028"),
    "RandomForest": MODEL_COLORS.get("RandomForest", "#2D7D46"),
}
MARKERS = {
    "BDF": MODEL_MARKERS.get("BDF", "o"),
    "RandomForest": MODEL_MARKERS.get("RandomForest", "s"),
}


def plot_convergence_single_dgp(
    dgp_name: str,
    dgp_description: str,
    results_by_model: dict[str, dict],
    save_path: str,
):
    """Plot log MSE vs log n for a single DGP, all models overlaid."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name, data in results_by_model.items():
        n_vals = data["n_values"]
        mean_mse = data["mean_mse"]
        std_mse = data["std_mse"]
        rate = data["rate"]

        # Filter valid points
        valid = [i for i, m in enumerate(mean_mse) if np.isfinite(m) and m > 0]
        if not valid:
            continue

        ns = [n_vals[i] for i in valid]
        ms = [mean_mse[i] for i in valid]
        ss = [std_mse[i] for i in valid]

        color = COLORS.get(model_name, "gray")
        marker = MARKERS.get(model_name, "o")
        slope = rate["slope"]
        r2 = rate["r_squared"]

        label = f"{model_name} (rate={slope:.2f}, R\u00b2={r2:.2f})"

        ax.errorbar(
            ns,
            ms,
            yerr=ss,
            fmt=f"{marker}-",
            color=color,
            linewidth=2,
            markersize=7,
            capsize=3,
            label=label,
        )

        # Overlay fitted line
        if np.isfinite(slope):
            log_n_fit = np.linspace(np.log(min(ns)), np.log(max(ns)), 100)
            log_mse_fit = rate["intercept"] + slope * log_n_fit
            ax.plot(
                np.exp(log_n_fit),
                np.exp(log_mse_fit),
                "--",
                color=color,
                alpha=0.5,
                linewidth=1.5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training samples (n)")
    ax.set_ylabel("MSE vs true mean")
    ax.set_title(dgp_description)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_convergence_grid(
    all_results: dict[str, Any],
    dimensionality: int,
    save_path: str,
):
    """Plot convergence grid: rows = smoothness class, columns = DGPs, for one d value."""
    setup_plot_style()

    # Group DGPs by smoothness class for this dimensionality
    dgp_groups: dict[str, list[str]] = {"smooth": [], "non_smooth": [], "mixed": []}
    for dgp_name, dgp_data in all_results["dgps"].items():
        if dgp_data["dimensionality"] != dimensionality:
            continue
        sc = dgp_data["smoothness_class"]
        dgp_groups[sc].append(dgp_name)

    # Determine grid dimensions
    n_rows = len([g for g in dgp_groups.values() if g])
    n_cols = max(len(g) for g in dgp_groups.values() if g)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    row_labels = {"smooth": "Smooth", "non_smooth": "Non-smooth", "mixed": "Mixed"}
    row_idx = 0
    for sc_key in ["smooth", "non_smooth", "mixed"]:
        dgp_names = dgp_groups.get(sc_key, [])
        if not dgp_names:
            continue

        for col_idx, dgp_name in enumerate(dgp_names):
            ax = axes[row_idx, col_idx]
            dgp_data = all_results["dgps"][dgp_name]

            for model_name, model_data in dgp_data["models"].items():
                n_vals = model_data["n_values"]
                mean_mse = model_data["mean_mse"]
                rate = model_data["rate"]

                valid = [i for i, m in enumerate(mean_mse) if np.isfinite(m) and m > 0]
                if not valid:
                    continue

                ns = [n_vals[i] for i in valid]
                ms = [mean_mse[i] for i in valid]

                color = COLORS.get(model_name, "gray")
                marker = MARKERS.get(model_name, "o")
                slope = rate["slope"]

                ax.plot(
                    ns,
                    ms,
                    f"{marker}-",
                    color=color,
                    linewidth=1.5,
                    markersize=5,
                    label=f"{model_name} ({slope:.2f})",
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(dgp_data["description"], fontsize=10)
            ax.grid(True, alpha=0.3, which="both")
            if col_idx == 0:
                ax.set_ylabel(f"{row_labels[sc_key]}\nMSE vs true mean")
            if row_idx == n_rows - 1:
                ax.set_xlabel("n")
            ax.legend(fontsize=7, loc="best")

        # Hide unused axes
        for col_idx in range(len(dgp_names), n_cols):
            axes[row_idx, col_idx].set_visible(False)

        row_idx += 1

    fig.suptitle(f"Convergence Rates (d={dimensionality})", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_rate_comparison_bar(all_results: dict[str, Any], save_path: str):
    """Bar chart comparing convergence rates across smoothness classes."""
    setup_plot_style()

    # Aggregate rates by (smoothness_class, model)
    from collections import defaultdict

    rates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for dgp_name, dgp_data in all_results["dgps"].items():
        sc = dgp_data["smoothness_class"]
        for model_name, model_data in dgp_data["models"].items():
            slope = model_data["rate"]["slope"]
            if np.isfinite(slope):
                rates[sc][model_name].append(slope)

    # Plot
    sc_order = ["smooth", "non_smooth", "mixed"]
    sc_labels = {"smooth": "Smooth", "non_smooth": "Non-smooth", "mixed": "Mixed"}
    model_names = list(MODEL_FACTORIES.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sc_order))
    width = 0.3

    for i, model_name in enumerate(model_names):
        means = []
        stds = []
        for sc in sc_order:
            vals = rates[sc].get(model_name, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        color = COLORS.get(model_name, "gray")
        ax.bar(
            x + i * width,
            [-m for m in means],  # Negate so more negative = taller bar = faster rate
            width,
            yerr=stds,
            label=model_name,
            color=color,
            alpha=0.85,
            capsize=3,
        )

    ax.set_xlabel("Function smoothness class")
    ax.set_ylabel("Convergence rate (-slope in log-log)")
    ax.set_title("Convergence Rates by Smoothness Class")
    ax.set_xticks(x + width)
    ax.set_xticklabels([sc_labels[sc] for sc in sc_order])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(all_results: dict[str, Any], filepath: str):
    """Generate markdown summary of convergence rates."""
    lines = [
        "# Convergence Rate Study: BDF vs RF",
        "",
        f"Generated: {all_results['metadata']['timestamp']}",
        "",
        "## Summary of Convergence Rates",
        "",
        "Rate = slope in log(MSE) vs log(n). More negative = faster convergence.",
        "",
        "| DGP | Class | d | BDF rate | RF rate |",
        "|-----|-------|---|----------|---------|",
    ]

    for dgp_name, dgp_data in sorted(all_results["dgps"].items()):
        sc = dgp_data["smoothness_class"]
        d = dgp_data["dimensionality"]
        rates = {}
        for model_name in ["BDF", "RandomForest"]:
            if model_name in dgp_data["models"]:
                r = dgp_data["models"][model_name]["rate"]
                slope = r["slope"]
                se = r["slope_se"]
                if np.isfinite(slope):
                    rates[model_name] = f"{slope:.3f} ({se:.3f})"
                else:
                    rates[model_name] = "N/A"
            else:
                rates[model_name] = "N/A"

        lines.append(
            f"| {dgp_name} | {sc} | {d} | " f"{rates.get('BDF', 'N/A')} | " f"{rates.get('RandomForest', 'N/A')} |"
        )

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Seeds: {SEEDS}",
            f"- Sample sizes: {SAMPLE_SIZES}",
            f"- Dimensionalities: {DIMENSIONALITIES}",
            f"- Test set size: {TEST_N}",
            "",
        ]
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved to {filepath}")


# =============================================================================
# Main
# =============================================================================


def main():
    logger.info("Starting Convergence Rate Study")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Sample sizes: {SAMPLE_SIZES}")
    logger.info(f"Dimensionalities: {DIMENSIONALITIES}")
    logger.info(f"Models: {list(MODEL_FACTORIES.keys())}")

    dgps = build_dgp_suite()
    logger.info(f"DGPs: {[d.name for d in dgps]}")

    all_results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seeds": SEEDS,
            "sample_sizes": SAMPLE_SIZES,
            "dimensionalities": DIMENSIONALITIES,
            "test_n": TEST_N,
        },
        "dgps": {},
    }

    total_experiments = len(dgps) * len(MODEL_FACTORIES) * len(SAMPLE_SIZES) * len(SEEDS)
    logger.info(f"Total experiments: {total_experiments}")

    pbar = tqdm(total=total_experiments, desc="Convergence study")

    for dgp in dgps:
        d = int(dgp.name.split("_d")[-1])
        dgp_result = {
            "description": dgp.description,
            "smoothness_class": dgp.smoothness_class,
            "dimensionality": d,
            "noise_std": dgp.noise_std,
            "models": {},
        }

        for model_name in MODEL_FACTORIES:
            # Collect MSE values: shape (len(SAMPLE_SIZES), len(SEEDS))
            mse_matrix = np.full((len(SAMPLE_SIZES), len(SEEDS)), np.nan)

            for i, n_train in enumerate(SAMPLE_SIZES):
                for j, seed in enumerate(SEEDS):
                    try:
                        result = evaluate_single(dgp, model_name, n_train, seed)
                        mse_matrix[i, j] = result["mse"]
                    except Exception as e:
                        logger.warning(f"Failed: {dgp.name}/{model_name}/n={n_train}/seed={seed}: {e}")

                    pbar.update(1)

            # Aggregate across seeds
            mean_mse = [float(np.nanmean(mse_matrix[i, :])) for i in range(len(SAMPLE_SIZES))]
            std_mse = [float(np.nanstd(mse_matrix[i, :])) for i in range(len(SAMPLE_SIZES))]

            # Fit convergence rate on mean MSE
            rate = fit_convergence_rate(SAMPLE_SIZES, mean_mse)

            dgp_result["models"][model_name] = {  # ty:ignore[invalid-assignment]
                "n_values": SAMPLE_SIZES,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "all_mse": mse_matrix.tolist(),
                "rate": rate,
            }

            slope_str = f"{rate['slope']:.3f}" if np.isfinite(rate["slope"]) else "N/A"
            logger.info(f"  {dgp.name} / {model_name}: rate={slope_str}")

        all_results["dgps"][dgp.name] = dgp_result

        # Save intermediate results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/convergence_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    pbar.close()

    # Generate plots
    logger.info("Generating plots...")

    # Individual DGP plots
    for dgp_name, dgp_data in all_results["dgps"].items():
        plot_convergence_single_dgp(
            dgp_name,
            dgp_data["description"],
            dgp_data["models"],
            f"{PLOTS_DIR}/individual/{dgp_name}.png",
        )

    # Grid plots per dimensionality
    for d in DIMENSIONALITIES:
        plot_convergence_grid(all_results, d, f"{PLOTS_DIR}/grid_d{d}.png")

    # Bar chart comparison
    plot_rate_comparison_bar(all_results, f"{PLOTS_DIR}/rate_comparison.png")

    # Generate report
    generate_report(all_results, f"{RESULTS_DIR}/CONVERGENCE_RATES.md")

    # Final save
    with open(f"{RESULTS_DIR}/convergence_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.success("Convergence rate study complete!")
    logger.info(f"Results: {RESULTS_DIR}/convergence_results.json")
    logger.info(f"Report: {RESULTS_DIR}/CONVERGENCE_RATES.md")
    logger.info(f"Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
