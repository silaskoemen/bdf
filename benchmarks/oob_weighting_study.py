"""
OOB Weighting Study

Compares BDF NormalMuNormal with and without OOB tree weighting on
sklearn regression datasets (friedman1, friedman2, friedman3, make_regression).

Design:
    4 datasets x 2 modes (uniform vs OOB) x 5 seeds = 40 cells
    Each cell: Optuna tuning on fold 0 (50 trials, CRPS), evaluate on folds 1-9

Usage:
    python -m benchmarks.oob_weighting_study
    python -m benchmarks.oob_weighting_study --quick
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
from loguru import logger
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3, make_regression
from sklearn.model_selection import KFold

from bdf.tree_classes.bdf_regressor import BDFRegressor

from .metrics.regression import crps_wrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEEDS = [42, 123, 456, 789, 1024]
N_SPLITS = 10
N_TRIALS = 50
N_POSTERIOR_SAMPLES = 1000
N_SAMPLES = 2000

RESULTS_DIR = Path("benchmarks/results/oob_weighting")
PLOTS_DIR = Path("benchmarks/plots/oob_weighting")

DATASETS = {
    "friedman1": lambda seed: make_friedman1(n_samples=N_SAMPLES, noise=1.0, random_state=seed),
    "friedman2": lambda seed: make_friedman2(n_samples=N_SAMPLES, noise=1.0, random_state=seed),
    "friedman3": lambda seed: make_friedman3(n_samples=N_SAMPLES, noise=0.1, random_state=seed),
    "make_regression": lambda seed: make_regression(
        n_samples=N_SAMPLES, n_features=10, n_informative=5, noise=10.0, random_state=seed
    ),
}

# Shared tunable init_kwargs (tree construction)
TUNABLE_INIT_KWARGS = {
    "n_trees": {"type": "int", "low": 15, "high": 100},
    "alpha": {"type": "float", "low": 0.00001, "high": 0.1, "log": True},
    "gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
    "delta": {"type": "float", "low": 0.0001, "high": 0.5, "log": True},
    "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
    "colsample": {"type": "float", "low": 0.7, "high": 1.0},
    "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
}

# Shared tunable distribution params
TUNABLE_PARAMS = {
    "sigma_mu": {"type": "float", "low": 0.1, "high": 50.0, "log": True},
    "score_method": {"type": "categorical", "categories": ["nll", "nle"]},
}

FIXED_PARAMS = {
    "mu_mu": "auto",
    "score_correction": "bic",
}

# Mode-specific configs
MODES = {
    "uniform": {
        "fixed_init_kwargs": {
            "dist": "NormalMuNormal",
            "oob_weights": False,
            "bootstrap": True,
        },
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    },
    "oob": {
        "fixed_init_kwargs": {
            "dist": "NormalMuNormal",
            "oob_weights": True,
            "bootstrap": False,
        },
        "subsample": {"type": "float", "low": 0.5, "high": 0.9},
        "extra_tunable": {
            "oob_temperature": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
        },
    },
}


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class CellResult:
    dataset_name: str
    mode: str
    seed: int
    best_params: dict = field(default_factory=dict)
    tuning_time: float = 0.0
    fold_crps: list[float] = field(default_factory=list)
    fold_fit_times: list[float] = field(default_factory=list)


# =============================================================================
# Tuning & evaluation
# =============================================================================


def suggest_hyperparameters(trial: optuna.Trial, mode: str) -> tuple[dict, dict]:
    """Suggest hyperparameters for a given mode."""
    mode_config = MODES[mode]
    init_kwargs: dict[str, Any] = {}

    # Tree construction params
    for name, args in TUNABLE_INIT_KWARGS.items():
        if args["type"] == "int":
            init_kwargs[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            init_kwargs[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))

    # Subsample (mode-specific range)
    sub_args = mode_config["subsample"]
    init_kwargs["subsample"] = trial.suggest_float("subsample", sub_args["low"], sub_args["high"])

    # OOB-specific params
    if "extra_tunable" in mode_config:
        for name, args in mode_config["extra_tunable"].items():
            if args["type"] == "float":
                init_kwargs[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))

    # Distribution params
    params: dict[str, Any] = dict(FIXED_PARAMS)
    for name, args in TUNABLE_PARAMS.items():
        if args["type"] == "float":
            params[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, args["categories"])

    return init_kwargs, params


def tune_cell(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mode: str,
    study_name: str,
    n_trials: int = N_TRIALS,
    seed: int = 42,
) -> tuple[dict, dict, float]:
    """Tune BDF with Optuna, return (best_init_kwargs, best_params, time)."""
    mode_config = MODES[mode]
    fixed_init_kwargs = mode_config["fixed_init_kwargs"]

    def objective(trial: optuna.Trial) -> float:
        np.random.seed(seed + trial.number)
        init_kwargs, params = suggest_hyperparameters(trial, mode)
        try:
            model = BDFRegressor(
                **fixed_init_kwargs,
                **init_kwargs,
                random_state=seed,
                params=params,
            )
            model.fit(X_train, y_train)
            y_samples = model.predict_samples(X_val, n_samples=N_POSTERIOR_SAMPLES)
            return float(crps_wrapper(y_val, y_samples))
        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return float("inf")

    os.makedirs("benchmarks/results/optuna/", exist_ok=True)
    storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"
    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except (KeyError, Exception):
        pass

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        sampler=sampler,
        load_if_exists=False,
    )

    start = time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tuning_time = time() - start

    if study.best_value == float("inf"):
        raise RuntimeError(f"All {n_trials} tuning trials failed for mode={mode}")

    logger.info(f"    Best CRPS: {study.best_value:.4f} ({tuning_time:.1f}s)")

    # Split best params back into init_kwargs and params
    best = study.best_params
    init_kwargs: dict[str, Any] = {}
    params = dict(FIXED_PARAMS)

    for name in TUNABLE_INIT_KWARGS:
        if name in best:
            init_kwargs[name] = best[name]
    if "subsample" in best:
        init_kwargs["subsample"] = best["subsample"]
    if "extra_tunable" in mode_config:
        for name in mode_config["extra_tunable"]:
            if name in best:
                init_kwargs[name] = best[name]
    for name in TUNABLE_PARAMS:
        if name in best:
            params[name] = best[name]

    return init_kwargs, params, tuning_time


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    init_kwargs: dict,
    params: dict,
    mode: str,
    seed: int,
) -> tuple[float, float]:
    """Train and evaluate on a single fold. Returns (crps, fit_time)."""
    fixed_init_kwargs = MODES[mode]["fixed_init_kwargs"]
    np.random.seed(seed)

    model = BDFRegressor(
        **fixed_init_kwargs,
        **init_kwargs,
        random_state=seed,
        params=params,
    )

    start = time()
    model.fit(X_train, y_train)
    fit_time = time() - start

    y_samples = model.predict_samples(X_test, n_samples=N_POSTERIOR_SAMPLES)
    crps = float(crps_wrapper(y_test, y_samples))

    return crps, fit_time


# =============================================================================
# Main study
# =============================================================================


def run_study(quick: bool = False) -> list[CellResult]:
    """Run the full OOB weighting study."""
    seeds = SEEDS[:1] if quick else SEEDS
    n_trials = 5 if quick else N_TRIALS
    dataset_names = list(DATASETS.keys())[:2] if quick else list(DATASETS.keys())
    mode_names = list(MODES.keys())

    all_results: list[CellResult] = []
    total_cells = len(seeds) * len(dataset_names) * len(mode_names)
    completed = 0

    for seed in seeds:
        for ds_name in dataset_names:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Dataset: {ds_name} | Seed: {seed}")

            X, y = DATASETS[ds_name](seed)

            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            folds = list(enumerate(kf.split(X)))

            for mode in mode_names:
                completed += 1
                logger.info(f"\n  [{completed}/{total_cells}] {ds_name} / {mode}")

                cell = CellResult(dataset_name=ds_name, mode=mode, seed=seed)

                # Tune on fold 0
                _, (train_idx_0, test_idx_0) = folds[0]
                X_train_0, X_val_0 = X[train_idx_0], X[test_idx_0]
                y_train_0, y_val_0 = y[train_idx_0], y[test_idx_0]

                study_name = f"oob_{ds_name}_{mode}_s{seed}"
                try:
                    best_init_kwargs, best_params, tuning_time = tune_cell(
                        X_train_0,
                        y_train_0,
                        X_val_0,
                        y_val_0,
                        mode,
                        study_name,
                        n_trials=n_trials,
                        seed=seed,
                    )
                    cell.best_params = {"init_kwargs": best_init_kwargs, "params": best_params}
                    cell.tuning_time = tuning_time
                except Exception as e:
                    logger.error(f"Tuning failed: {e}")
                    all_results.append(cell)
                    continue

                # Evaluate on folds 1-9
                for fold_idx, (train_idx, test_idx) in folds[1:]:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    try:
                        crps, fit_time = evaluate_fold(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            best_init_kwargs,
                            best_params,
                            mode,
                            seed,
                        )
                        cell.fold_crps.append(crps)
                        cell.fold_fit_times.append(fit_time)
                    except Exception as e:
                        logger.warning(f"Fold {fold_idx} failed: {e}")

                if cell.fold_crps:
                    mean_crps = np.mean(cell.fold_crps)
                    logger.info(f"    Mean CRPS: {mean_crps:.4f} ({len(cell.fold_crps)} folds)")

                all_results.append(cell)

    return all_results


# =============================================================================
# Plotting
# =============================================================================


def plot_results(results: list[CellResult]) -> None:
    """Side-by-side CRPS comparison: uniform vs OOB for each dataset."""
    from benchmarks.utils.style import apply_paper_style

    apply_paper_style()

    # Collect per-(dataset, mode) CRPS values across seeds and folds
    data: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if not r.fold_crps:
            continue
        data.setdefault(r.dataset_name, {}).setdefault(r.mode, []).extend(r.fold_crps)

    dataset_names = [ds for ds in DATASETS if ds in data]
    n_datasets = len(dataset_names)
    if n_datasets == 0:
        logger.warning("No results to plot")
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(3.5 * n_datasets, 4), sharey=False)
    if n_datasets == 1:
        axes = [axes]

    colors = {"uniform": "#4A90D9", "oob": "#E07B54"}

    for ax, ds_name in zip(axes, dataset_names):
        ds_data = data[ds_name]
        positions = []
        box_data = []
        box_colors = []

        for i, mode in enumerate(["uniform", "oob"]):
            if mode in ds_data:
                positions.append(i)
                box_data.append(ds_data[mode])
                box_colors.append(colors[mode])

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 3, "alpha": 0.5},
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        # Annotate with mean ± std
        for pos, vals in zip(positions, box_data):
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            ax.text(
                pos,
                ax.get_ylim()[1] * 0.98,
                f"{mean_val:.3f}\n±{std_val:.3f}",
                ha="center",
                va="top",
                fontsize=8,
                style="italic",
            )

        ax.set_xticks(range(len(["uniform", "oob"])))
        ax.set_xticklabels(["Uniform", "OOB"], fontsize=10)
        ax.set_title(ds_name.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel("CRPS" if ax == axes[0] else "", fontsize=10)

    fig.suptitle("OOB Tree Weighting: CRPS Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "oob_crps_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    logger.info(f"Plot saved to {out_path}")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOB Weighting Study")
    parser.add_argument("--quick", action="store_true", help="Smoke test (2 datasets, 1 seed, 5 trials)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("OOB Weighting Study")
    logger.info(f"Datasets: {list(DATASETS.keys())}")
    logger.info(f"Modes: {list(MODES.keys())}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Trials: {N_TRIALS}")
    logger.info("=" * 70)

    results = run_study(quick=args.quick)

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = [asdict(r) for r in results]
    with open(RESULTS_DIR / "raw_results.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)
    logger.info(f"Results saved to {RESULTS_DIR / 'raw_results.json'}")

    # Print summary
    logger.info(f"\n{'=' * 70}")
    logger.info("Summary (mean CRPS ± std):")
    for ds_name in DATASETS:
        logger.info(f"\n  {ds_name}:")
        for mode in MODES:
            cells = [r for r in results if r.dataset_name == ds_name and r.mode == mode and r.fold_crps]
            if cells:
                all_crps = [c for r in cells for c in r.fold_crps]
                logger.info(f"    {mode:8s}: {np.mean(all_crps):.4f} ± {np.std(all_crps):.4f}")

    # Plot
    plot_results(results)

    logger.info("\nStudy complete!")
