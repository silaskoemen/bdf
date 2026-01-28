"""
Experiment: Effect of Number of Trees on BDF Performance

Investigates how the number of trees (n_trees) affects BDF performance metrics,
calibration, and computational cost. This helps determine the optimal ensemble size
and whether performance plateaus at a certain number of trees.

For each n_trees value:
1. Tune other hyperparameters on fold 0
2. Evaluate with those hyperparameters on folds 1-9
3. Track both predictive metrics and fitting time

Uses sklearn synthetic datasets (make_friedman1, make_friedman2, make_friedman3, make_regression).
"""

import json
import os
from time import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3, make_regression
from sklearn.model_selection import KFold
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFRegressor
from benchmarks.metrics.regression import REG_POINT_METRICS, REG_PROB_METRICS, precompute_percentiles

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
SEED = 42
N_FOLDS = 10  # 1 for tuning (fold 0), 9 for evaluation (folds 1-9)
N_TRIALS = 50  # Optuna trials for hyperparameter tuning
N_FEATURES = 10
SAMPLE_SIZE = 2000  # Fixed sample size
N_SAMPLES_PRED = 500  # Number of samples for probabilistic predictions

# Range of n_trees to evaluate
N_TREES_GRID = [5, 10, 25, 50, 100, 200]

# Dataset generators
DATASET_GENERATORS: dict[str, Callable] = {
    "friedman1": lambda n, seed: make_friedman1(n_samples=n, n_features=N_FEATURES, noise=1.0, random_state=seed),
    "friedman2": lambda n, seed: make_friedman2(n_samples=n, noise=1.0, random_state=seed),
    "friedman3": lambda n, seed: make_friedman3(n_samples=n, noise=0.1, random_state=seed),
    "make_regression": lambda n, seed: make_regression(
        n_samples=n, n_features=N_FEATURES, n_informative=5, noise=10.0, random_state=seed
    ),
}

# Metrics to track
POINT_METRICS = ["rmse", "mae", "r2"]
PROB_METRICS = ["crps", "coverage_90", "interval_score_90", "pica"]

# BDF configuration (hyperparameters to tune, excluding n_trees)
TUNABLE_INIT_KWARGS = {
    "reg_lambda": {"type": "float", "low": 0.00001, "high": 0.1, "log": True},
    "reg_gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
    "reg_nu": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
    "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample": {"type": "float", "low": 0.5, "high": 1.0},
    "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
}

TUNABLE_PARAMS = {
    "sigma_mu_auto_scale": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
    "score_method": {"type": "categorical", "categories": ["nll", "nle"]},
}

FIXED_INIT_KWARGS = {
    "dist": "NormalMuNormal",
    "random_state": SEED,
}

FIXED_PARAMS = {
    "mu_mu": "auto",
    "sigma_mu": "auto",
    "score_correction": "bic",
}


# =============================================================================
# Hyperparameter Tuning
# =============================================================================


def suggest_hyperparameters(trial: optuna.Trial, n_trees: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Suggest hyperparameters from optuna trial."""
    init_kwargs = {"n_trees": n_trees}

    # Suggest tunable init kwargs
    for name, args in TUNABLE_INIT_KWARGS.items():
        if args["type"] == "int":
            init_kwargs[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            init_kwargs[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            init_kwargs[name] = trial.suggest_categorical(name, args["categories"])

    # Build params dict
    params = dict(FIXED_PARAMS)
    for name, args in TUNABLE_PARAMS.items():
        if args["type"] == "int":
            params[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            params[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, args["categories"])

    return init_kwargs, params


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trees: int,
    dataset_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune hyperparameters for a given n_trees using Optuna."""

    def objective(trial: optuna.Trial) -> float:
        np.random.seed(SEED + trial.number)
        init_kwargs, params = suggest_hyperparameters(trial, n_trees)

        try:
            model = BDFRegressor(**FIXED_INIT_KWARGS, **init_kwargs, params=params)
            model.fit(X_train, y_train)
            y_pred_samples = model.predict_samples(X_val, n_samples=N_SAMPLES_PRED)

            # Use CRPS as optimization metric
            from benchmarks.metrics.regression import crps_wrapper

            crps = crps_wrapper(y_val, y_pred_samples, quantile_levels=None)
            return crps
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")

    # Create study
    study_name = f"effect_n_trees-{dataset_name}-ntrees{n_trees}"
    os.makedirs("benchmarks/results/optuna/", exist_ok=True)
    storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"

    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except KeyError:
        pass
    except Exception as e:
        logger.warning(f"Could not delete existing study: {e}")

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        sampler=sampler,
        load_if_exists=False,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Extract best params
    best_params = study.best_params
    best_init_kwargs = {"n_trees": n_trees}
    best_dist_params = dict(FIXED_PARAMS)

    for param, value in best_params.items():
        if param in TUNABLE_INIT_KWARGS:
            best_init_kwargs[param] = value
        elif param in TUNABLE_PARAMS:
            best_dist_params[param] = value

    logger.info(f"  Best CRPS: {study.best_value:.4f}")
    return best_init_kwargs, best_dist_params


# =============================================================================
# Metrics Computation
# =============================================================================


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
    except Exception as e:
        logger.warning(f"Failed to compute probabilistic metrics: {e}")
        for metric_name in PROB_METRICS:
            metrics[metric_name] = float("nan")

    return metrics


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment_for_dataset(dataset_name: str, results: dict[str, Any]) -> dict[str, Any]:
    """Run full experiment for a single dataset."""
    logger.info(f"🔬 Processing {dataset_name}")

    # Generate dataset
    X, y = DATASET_GENERATORS[dataset_name](SAMPLE_SIZE, SEED)
    X = np.asarray(X)
    y = np.asarray(y)
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Create K-fold splits
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kfold.split(X))

    # Fold 0 for tuning, folds 1-9 for evaluation
    tune_train_idx, tune_test_idx = folds[0]
    X_tune = X[tune_train_idx]
    y_tune = y[tune_train_idx]

    # Further split fold 0 into train/val for Optuna
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X_tune, y_tune, test_size=0.25, random_state=SEED)

    logger.info(f"Tuning split: train={len(X_train)}, val={len(X_val)}")

    # Initialize result structure
    results["experiments"][dataset_name] = {
        "n_trees_grid": N_TREES_GRID,
        "n_trees_results": {},
    }

    # For each n_trees value
    for n_trees in N_TREES_GRID:
        logger.info(f"  🌲 n_trees = {n_trees}")

        # Tune hyperparameters on fold 0
        logger.info("    Tuning hyperparameters...")
        best_init_kwargs, best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trees, dataset_name)

        # Evaluate on folds 1-9
        fold_results = {
            "fold_metrics": [],
            "fold_fit_times": [],
            "best_init_kwargs": best_init_kwargs,
            "best_params": best_params,
        }

        for fold_idx in range(1, N_FOLDS):
            train_idx, test_idx = folds[fold_idx]
            X_fold_train = X[train_idx]
            y_fold_train = y[train_idx]
            X_fold_test = X[test_idx]
            y_fold_test = y[test_idx]

            # Fit model with best hyperparameters
            np.random.seed(SEED + fold_idx)
            model = BDFRegressor(**FIXED_INIT_KWARGS, **best_init_kwargs, params=best_params)

            start_time = time()
            model.fit(X_fold_train, y_fold_train)
            fit_time = time() - start_time

            # Compute metrics
            metrics = compute_metrics(model, X_fold_test, y_fold_test)

            fold_results["fold_metrics"].append(metrics)
            fold_results["fold_fit_times"].append(fit_time)

        # Aggregate metrics across folds 1-9
        aggregated = {}
        for metric_name in fold_results["fold_metrics"][0].keys():
            values = [fold[metric_name] for fold in fold_results["fold_metrics"]]
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        fold_results["aggregated_metrics"] = aggregated
        fold_results["mean_fit_time"] = float(np.mean(fold_results["fold_fit_times"]))
        fold_results["std_fit_time"] = float(np.std(fold_results["fold_fit_times"]))

        results["experiments"][dataset_name]["n_trees_results"][n_trees] = fold_results

        logger.success(
            f"    ✅ n_trees={n_trees}: CRPS={aggregated['crps']['mean']:.4f}±{aggregated['crps']['std']:.4f}, "
            f"fit_time={fold_results['mean_fit_time']:.2f}±{fold_results['std_fit_time']:.2f}s"
        )

    logger.success(f"✅ Finished {dataset_name}")
    return results


def save_results(results: dict[str, Any], filepath: str = "benchmarks/results/effect_n_trees/effect_n_trees.json"):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to {filepath}")


# =============================================================================
# Plotting
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
        }
    )


def plot_metric_vs_n_trees(
    results: dict[str, Any],
    metric: str,
    save_dir: str = "benchmarks/plots/effect_n_trees",
):
    """Create 2x2 plot showing metric vs n_trees for all datasets."""
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    datasets = list(results["experiments"].keys())
    colors = sns.color_palette("husl", len(datasets))

    for ax, (dataset, color) in zip(axes, zip(datasets, colors)):
        dataset_results = results["experiments"][dataset]
        n_trees_grid = dataset_results["n_trees_grid"]

        means = []
        stds = []

        for n_trees in n_trees_grid:
            agg_metrics = dataset_results["n_trees_results"][n_trees]["aggregated_metrics"]
            means.append(agg_metrics[metric]["mean"])
            stds.append(agg_metrics[metric]["std"])

        means = np.array(means)
        stds = np.array(stds)

        # Plot with error bars
        ax.errorbar(
            n_trees_grid,
            means,
            yerr=stds,
            fmt="o-",
            color=color,
            linewidth=2,
            markersize=6,
            capsize=4,
            label=dataset,
        )

        # Fill between for visual clarity
        ax.fill_between(n_trees_grid, means - stds, means + stds, alpha=0.2, color=color)

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{dataset}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(f"Effect of Number of Trees on {metric.upper()}", fontsize=14, y=1.00)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{metric}_vs_n_trees.png")
    plt.savefig(save_path)
    logger.info(f"Saved plot to {save_path}")
    plt.close()


def plot_fit_time_vs_n_trees(
    results: dict[str, Any],
    save_dir: str = "benchmarks/plots/effect_n_trees",
):
    """Create 2x2 plot showing fit time vs n_trees for all datasets."""
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    datasets = list(results["experiments"].keys())
    colors = sns.color_palette("husl", len(datasets))

    for ax, (dataset, color) in zip(axes, zip(datasets, colors)):
        dataset_results = results["experiments"][dataset]
        n_trees_grid = dataset_results["n_trees_grid"]

        mean_times = []
        std_times = []

        for n_trees in n_trees_grid:
            n_trees_result = dataset_results["n_trees_results"][n_trees]
            mean_times.append(n_trees_result["mean_fit_time"])
            std_times.append(n_trees_result["std_fit_time"])

        mean_times = np.array(mean_times)
        std_times = np.array(std_times)

        # Plot with error bars
        ax.errorbar(
            n_trees_grid,
            mean_times,
            yerr=std_times,
            fmt="s-",
            color=color,
            linewidth=2,
            markersize=6,
            capsize=4,
            label=dataset,
        )

        ax.fill_between(n_trees_grid, mean_times - std_times, mean_times + std_times, alpha=0.2, color=color)

        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Fit Time (seconds)")
        ax.set_title(f"{dataset}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Computational Cost: Fit Time vs Number of Trees", fontsize=14, y=1.00)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fit_time_vs_n_trees.png")
    plt.savefig(save_path)
    logger.info(f"Saved plot to {save_path}")
    plt.close()


def generate_all_plots(results: dict[str, Any]):
    """Generate all plots."""
    logger.info("📊 Generating plots...")

    # Plot for each metric
    all_metrics = POINT_METRICS + PROB_METRICS
    for metric in all_metrics:
        plot_metric_vs_n_trees(results, metric)

    # Fit time plot
    plot_fit_time_vs_n_trees(results)

    logger.success("✅ All plots generated")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point for the experiment."""
    logger.info("🚀 Starting Effect of Number of Trees Experiment")
    logger.info(f"Datasets: {list(DATASET_GENERATORS.keys())}")
    logger.info(f"n_trees grid: {N_TREES_GRID}")
    logger.info(f"K-fold splits: {N_FOLDS} (fold 0 for tuning, folds 1-{N_FOLDS-1} for evaluation)")

    # Initialize results
    results: dict[str, Any] = {
        "config": {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "n_trials": N_TRIALS,
            "n_features": N_FEATURES,
            "sample_size": SAMPLE_SIZE,
            "n_trees_grid": N_TREES_GRID,
            "datasets": list(DATASET_GENERATORS.keys()),
        },
        "experiments": {},
    }

    # Run experiments
    total_experiments = len(DATASET_GENERATORS)
    pbar = tqdm(total=total_experiments, desc="Datasets")

    for dataset_name in DATASET_GENERATORS.keys():
        try:
            results = run_experiment_for_dataset(dataset_name, results)
        except Exception as e:
            logger.error(f"❌ Failed for {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            results["experiments"][dataset_name] = {"error": str(e)}

        # Save intermediate results
        save_results(results)
        pbar.update(1)

    pbar.close()

    # Generate plots
    generate_all_plots(results)

    # Final save
    save_results(results)
    logger.success("🎉 Experiment complete!")

    return results


if __name__ == "__main__":
    main()
