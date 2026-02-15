"""
Distribution Misspecification Study

Evaluates the impact of distributional misspecification on BDF performance.
Crosses 5 DGPs (each the "home" of one distribution) with 5 BDF distribution variants,
measuring degradation when the assumed distribution is wrong.

Design:
    5 DGPs x 5 Distributions = 25 cells (6 incompatible, 19 runnable)
    Each cell: Optuna tuning on fold 0 (50 trials, CRPS), evaluate on folds 1-9
    5 seeds for confidence intervals -> 45 eval observations per cell

Usage:
    python -m benchmarks.dist_misspecification_study
    python -m benchmarks.dist_misspecification_study --quick   # smoke test
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from bdf.tree_classes.bdf_regressor import BDFRegressor

from .metrics.regression import (
    crps_wrapper,
    interval_score_samples,
    pica,
    pit_ks_statistic,
)
from .pipeline.synthetic_dgps import DGP_REGISTRY, SyntheticDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEEDS = [42, 123, 456, 789, 1024]
N_SPLITS = 10  # Fold 0 for tuning, folds 1-9 for evaluation
N_TRIALS = 50
N_POSTERIOR_SAMPLES = 1000

RESULTS_DIR = Path("benchmarks/results/dist_misspecification")
PLOTS_DIR = Path("benchmarks/plots/dist_misspecification")

# DGPs: each is the "home" of one distribution
DGPS = [
    {"name": "gaussian_heteroscedastic", "kwargs": {"n_samples": 5000}, "home_dist": "NormalMuNormal"},
    {"name": "poisson_count", "kwargs": {"n_samples": 5000}, "home_dist": "GammaMVLambdaPoisson"},
    {"name": "exponential_waiting_time", "kwargs": {"n_samples": 5000}, "home_dist": "GammaMVLambdaExponential"},
    {
        "name": "heavy_tailed",
        "kwargs": {"n_samples": 5000, "df": 3.0, "scale": 0.5},
        "home_dist": "FrequentistStudentT",
    },
    {"name": "multimodal_mixture_fixed", "kwargs": {"n_samples": 5000}, "home_dist": "KDE"},
]

# Distribution model configs: hyperparameter search spaces
DIST_CONFIGS: dict[str, dict[str, Any]] = {
    "NormalMuNormal": {
        "fixed_init_kwargs": {"dist": "NormalMuNormal"},
        "tunable_init_kwargs": {
            "n_trees": {"type": "int", "low": 15, "high": 100},
            "alpha": {"type": "float", "low": 0.00001, "high": 0.1, "log": True},
            "gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "delta": {"type": "float", "low": 0.0001, "high": 0.5, "log": True},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0},
            "colsample": {"type": "float", "low": 0.7, "high": 1.0},
            "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        },
        "tunable_params": {
            "sigma_mu": {"type": "float", "low": 0.1, "high": 50.0, "log": True},
            "score_method": {"type": "categorical", "categories": ["nll", "nle"]},
        },
        "fixed_params": {
            "mu_mu": "auto",
            "score_correction": "bic",
        },
    },
    "GammaMVLambdaPoisson": {
        "fixed_init_kwargs": {"dist": "GammaMVLambdaPoisson"},
        "tunable_init_kwargs": {
            "n_trees": {"type": "int", "low": 5, "high": 200},
            "alpha": {"type": "float", "low": 0.00001, "high": 1.0, "log": True},
            "gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "delta": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample": {"type": "float", "low": 0.5, "high": 1.0},
            "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        },
        "tunable_params": {
            "var_lambda_auto_scale": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        },
        "fixed_params": {
            "mean_lambda": "auto",
            "var_lambda": "auto",
            "raise_on_non_integer": False,
        },
    },
    "GammaMVLambdaExponential": {
        "fixed_init_kwargs": {"dist": "GammaMVLambdaExponential"},
        "tunable_init_kwargs": {
            "n_trees": {"type": "int", "low": 5, "high": 200},
            "alpha": {"type": "float", "low": 0.0001, "high": 10.0, "log": True},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            "min_samples_split": {"type": "int", "low": 10, "high": 100},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample": {"type": "float", "low": 0.5, "high": 1.0},
            "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        },
        "tunable_params": {
            "var_lambda": {"type": "float", "low": 0.1, "high": 20.0, "log": True},
        },
        "fixed_params": {
            "mean_lambda": "auto",
        },
    },
    "FrequentistStudentT": {
        "fixed_init_kwargs": {"dist": "FrequentistStudentT"},
        "tunable_init_kwargs": {
            "n_trees": {"type": "int", "low": 5, "high": 200},
            "alpha": {"type": "float", "low": 0.00001, "high": 1.0, "log": True},
            "gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "delta": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample": {"type": "float", "low": 0.5, "high": 1.0},
            "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        },
        "tunable_params": {},
        "fixed_params": {
            "df": None,  # estimate from data
            "tree_prior_mode": "linear",
            "score_correction": "bic",
        },
    },
    "KDE": {
        "fixed_init_kwargs": {"dist": "KDE"},
        "tunable_init_kwargs": {
            "n_trees": {"type": "int", "low": 15, "high": 100},
            "alpha": {"type": "float", "low": 0.00001, "high": 0.1, "log": True},
            "gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "delta": {"type": "float", "low": 0.0001, "high": 0.1, "log": True},
            "min_samples_leaf": {"type": "int", "low": 10, "high": 100},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0},
            "colsample": {"type": "float", "low": 0.7, "high": 1.0},
            "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        },
        "tunable_params": {
            "bandwidth": {"type": "categorical", "categories": ["scott", "silverman"]},
            "score_correction": {"type": "categorical", "categories": ["loo_cv", "bic", None]},
        },
        "fixed_params": {
            "kernel": "gaussian",
            "parent_bw_refine_top_k": 3,
            "score_cv_folds": 3,
            "kde_backend": "switch",
            "bandwidth_policy": "parent",
            "use_compact_support": False,
        },
    },
}

# Known domain incompatibilities: (dgp_name, dist_name) pairs that cannot work.
# These are skipped before attempting a fit.
KNOWN_INCOMPATIBLE = {
    # Poisson requires non-negative integers; these DGPs produce negative values
    ("gaussian_heteroscedastic", "GammaMVLambdaPoisson"),
    ("heavy_tailed", "GammaMVLambdaPoisson"),
    ("multimodal_mixture_fixed", "GammaMVLambdaPoisson"),
    # Exponential requires strictly positive (y > 0); these produce negative or zero values
    ("gaussian_heteroscedastic", "GammaMVLambdaExponential"),
    ("heavy_tailed", "GammaMVLambdaExponential"),
    ("multimodal_mixture_fixed", "GammaMVLambdaExponential"),
    ("poisson_count", "GammaMVLambdaExponential"),  # Poisson produces zeros
}

DISTRIBUTION_NAMES = list(DIST_CONFIGS.keys())
SHORT_NAMES = {
    "NormalMuNormal": "Normal",
    "GammaMVLambdaPoisson": "Poisson",
    "GammaMVLambdaExponential": "Exponential",
    "FrequentistStudentT": "Student-t",
    "KDE": "KDE",
}
DGP_SHORT_NAMES = {
    "gaussian_heteroscedastic": "Gaussian",
    "poisson_count": "Count",
    "exponential_waiting_time": "Waiting-Time",
    "heavy_tailed": "Heavy-Tailed",
    "multimodal_mixture_fixed": "Multimodal",
}


# =============================================================================
# Result storage
# =============================================================================


@dataclass
class CellResult:
    """Result for one (DGP, distribution, seed) cell."""

    dgp_name: str
    dist_name: str
    seed: int
    compatible: bool = True
    incompatibility_reason: str = ""
    best_params: dict = field(default_factory=dict)
    tuning_time: float = 0.0
    fold_metrics: list[dict] = field(default_factory=list)
    fold_fit_times: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dgp_name": self.dgp_name,
            "dist_name": self.dist_name,
            "seed": self.seed,
            "compatible": self.compatible,
            "incompatibility_reason": self.incompatibility_reason,
            "best_params": self.best_params,
            "tuning_time": self.tuning_time,
            "fold_metrics": self.fold_metrics,
            "fold_fit_times": self.fold_fit_times,
        }


# =============================================================================
# Helper functions
# =============================================================================


def suggest_hyperparameters(trial: optuna.Trial, dist_name: str) -> tuple[dict, dict]:
    """Suggest hyperparameters from an Optuna trial."""
    config = DIST_CONFIGS[dist_name]
    init_kwargs = {}
    for name, args in config["tunable_init_kwargs"].items():
        if args["type"] == "int":
            init_kwargs[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            init_kwargs[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            init_kwargs[name] = trial.suggest_categorical(name, args["categories"])

    params = dict(config.get("fixed_params", {}))
    for name, args in config.get("tunable_params", {}).items():
        if args["type"] == "int":
            params[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            params[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, args["categories"])

    return init_kwargs, params


def split_best_params(best_params: dict, dist_name: str) -> tuple[dict, dict]:
    """Split Optuna best_params into init_kwargs and params dict."""
    config = DIST_CONFIGS[dist_name]
    init_kwargs = {}
    tuned_params = {}
    for param, value in best_params.items():
        if param in config["tunable_init_kwargs"]:
            init_kwargs[param] = value
        elif param in config.get("tunable_params", {}):
            tuned_params[param] = value
    combined_params = dict(config.get("fixed_params", {}))
    combined_params.update(tuned_params)
    return init_kwargs, combined_params


def _get_default_params(dist_name: str) -> dict:
    """Get params with sensible defaults for tunable params (used for compatibility check)."""
    config = DIST_CONFIGS[dist_name]
    params = dict(config.get("fixed_params", {}))
    # Fill in tunable params with midpoint defaults so the model can instantiate
    for name, args in config.get("tunable_params", {}).items():
        if args["type"] == "float":
            if args.get("log"):
                params[name] = (args["low"] * args["high"]) ** 0.5  # geometric mean
            else:
                params[name] = (args["low"] + args["high"]) / 2
        elif args["type"] == "int":
            params[name] = (args["low"] + args["high"]) // 2
        elif args["type"] == "categorical":
            params[name] = args["categories"][0]
    return params


def check_compatibility(dist_name: str, y_train: np.ndarray) -> tuple[bool, str]:
    """Check if a distribution is compatible with the data domain.

    Attempts a quick fit with minimal trees to catch domain errors early.
    """
    config = DIST_CONFIGS[dist_name]
    try:
        n = min(50, len(y_train))
        X_tiny = np.random.RandomState(0).randn(n, 2)
        y_tiny = y_train[:n]
        params = _get_default_params(dist_name)
        model = BDFRegressor(
            **config["fixed_init_kwargs"],
            random_state=0,
            n_trees=2,
            min_samples_leaf=5,
            params=params,
        )
        model.fit(X_tiny, y_tiny)
        return True, ""
    except (ValueError, AssertionError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected: {e}"


def tune_cell(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dist_name: str,
    study_name: str,
    n_trials: int = N_TRIALS,
    seed: int = 42,
) -> tuple[dict, dict, float]:
    """Tune a BDF model with a specific distribution on train/val split."""
    config = DIST_CONFIGS[dist_name]
    fixed_init_kwargs = config["fixed_init_kwargs"]

    def objective(trial: optuna.Trial) -> float:
        np.random.seed(seed + trial.number)
        init_kwargs, params = suggest_hyperparameters(trial, dist_name)
        try:
            model = BDFRegressor(
                **fixed_init_kwargs,
                **init_kwargs,
                random_state=seed,
                params=params,
            )
            model.fit(X_train, y_train)
            # Tune on CRPS (distributional quality)
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

    sampler = TPESampler(seed=seed)
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
        raise RuntimeError(f"All {n_trials} tuning trials failed for {dist_name}")

    logger.info(f"    Best CRPS: {study.best_value:.4f} ({tuning_time:.1f}s)")
    best_init_kwargs, best_params = split_best_params(study.best_params, dist_name)
    return best_init_kwargs, best_params, tuning_time


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset: SyntheticDataset,
    init_kwargs: dict,
    params: dict,
    dist_name: str,
    seed: int,
) -> tuple[dict, float]:
    """Train and evaluate on a single fold."""
    config = DIST_CONFIGS[dist_name]
    fixed_init_kwargs = config["fixed_init_kwargs"]
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

    # Point predictions
    y_pred = model.predict(X_test)
    metrics: dict[str, float] = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    # Probabilistic metrics
    try:
        y_samples = model.predict_samples(X_test, n_samples=N_POSTERIOR_SAMPLES)
        y_pred_std = np.std(y_samples, axis=-1)
        metrics["sharpness"] = float(np.mean(y_pred_std))
        metrics["crps"] = float(crps_wrapper(y_test, y_samples))

        for width in [0.5, 0.9, 0.95]:
            alpha = 1 - width
            y_lower = np.quantile(y_samples, alpha / 2, axis=-1)
            y_upper = np.quantile(y_samples, 1 - alpha / 2, axis=-1)
            coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
            ci_width = np.mean(y_upper - y_lower)
            int_score = interval_score_samples(y_test, y_samples, alpha=alpha)
            metrics[f"coverage_{int(width * 100)}"] = float(coverage)
            metrics[f"ci_width_{int(width * 100)}"] = float(ci_width)
            metrics[f"interval_score_{int(width * 100)}"] = float(np.mean(int_score))

        try:
            metrics["pica"] = float(pica(y_test, y_samples))
            metrics["pit_ks_statistic"] = float(pit_ks_statistic(y_test, y_samples))
            # Store PIT values for histogram plotting
            pit_values = np.mean(y_samples <= y_test[:, None], axis=1)
            n_bins = 10
            pit_hist, _ = np.histogram(pit_values, bins=n_bins, range=(0, 1))
            for b in range(n_bins):
                metrics[f"pit_bin_{b}"] = int(pit_hist[b])
        except Exception as e:
            logger.debug(f"Calibration metric failed: {e}")

        # Ground truth metrics
        gt = dataset.ground_truth
        y_true_mean = gt.mean_fn(X_test).flatten()
        metrics["gt_mean_rmse"] = float(np.sqrt(mean_squared_error(y_true_mean, y_pred)))

        if gt.variance_fn is not None:
            y_true_var = gt.variance_fn(X_test).flatten()
            y_pred_var = y_pred_std**2
            metrics["gt_variance_rmse"] = float(np.sqrt(mean_squared_error(y_true_var, y_pred_var)))
            nonzero = y_true_var > 1e-8
            if nonzero.any():
                metrics["gt_variance_rel_error"] = float(
                    np.mean(np.abs(y_pred_var[nonzero] - y_true_var[nonzero]) / y_true_var[nonzero])
                )

    except Exception as e:
        logger.warning(f"Probabilistic evaluation failed: {e}")

    return metrics, fit_time


# =============================================================================
# Main study
# =============================================================================


def run_study(quick: bool = False) -> list[CellResult]:
    """Run the full misspecification study."""
    seeds = SEEDS[:1] if quick else SEEDS
    n_trials = 5 if quick else N_TRIALS
    dgps = DGPS[:2] if quick else DGPS
    dists = DISTRIBUTION_NAMES[:2] if quick else DISTRIBUTION_NAMES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[CellResult] = []
    total_cells = len(seeds) * len(dgps) * len(dists)
    completed = 0

    for seed in seeds:
        for dgp_spec in dgps:
            dgp_name = dgp_spec["name"]
            dgp_kwargs = {**dgp_spec["kwargs"], "seed": seed}

            logger.info(f"\n{'=' * 70}")
            logger.info(f"DGP: {dgp_name} | Seed: {seed}")
            logger.info(f"{'=' * 70}")

            dataset = DGP_REGISTRY[dgp_name](**dgp_kwargs)
            X, y = dataset.X, dataset.y

            # Set up CV folds (deterministic per seed)
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            folds = list(enumerate(kf.split(X)))
            train_idx_0, test_idx_0 = folds[0][1]
            X_train_0, X_val_0 = X[train_idx_0], X[test_idx_0]
            y_train_0, y_val_0 = y[train_idx_0], y[test_idx_0]

            for dist_name in dists:
                completed += 1
                logger.info(f"\n  [{completed}/{total_cells}] {dist_name} on {dgp_name} (seed={seed})")

                cell = CellResult(dgp_name=dgp_name, dist_name=dist_name, seed=seed)

                # Check known incompatibilities first (skip without attempting fit)
                if (dgp_name, dist_name) in KNOWN_INCOMPATIBLE:
                    cell.compatible = False
                    cell.incompatibility_reason = "Known domain incompatibility"
                    logger.info("    INCOMPATIBLE (known domain mismatch)")
                    all_results.append(cell)
                    continue

                # Check compatibility via trial fit
                compatible, reason = check_compatibility(dist_name, y_train_0)
                if not compatible:
                    cell.compatible = False
                    cell.incompatibility_reason = reason
                    logger.info(f"    INCOMPATIBLE: {reason[:80]}")
                    all_results.append(cell)
                    continue

                # Tune on fold 0
                study_name = f"misspec_{dgp_name}_{dist_name}_s{seed}"
                try:
                    best_init_kwargs, best_params, tuning_time = tune_cell(
                        X_train_0,
                        y_train_0,
                        X_val_0,
                        y_val_0,
                        dist_name,
                        study_name,
                        n_trials=n_trials,
                        seed=seed,
                    )
                    cell.best_params = {"init_kwargs": best_init_kwargs, "params": best_params}
                    cell.tuning_time = tuning_time
                except Exception as e:
                    logger.error(f"    Tuning failed: {e}")
                    cell.compatible = False
                    cell.incompatibility_reason = f"Tuning failed: {e}"
                    all_results.append(cell)
                    continue

                # Evaluate on folds 1-9
                for fold_idx, (train_idx, test_idx) in folds[1:]:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    try:
                        metrics, fit_time = evaluate_fold(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            dataset,
                            best_init_kwargs,
                            best_params,
                            dist_name,
                            seed,
                        )
                        cell.fold_metrics.append(metrics)
                        cell.fold_fit_times.append(fit_time)
                    except Exception as e:
                        logger.warning(f"    Fold {fold_idx} failed: {e}")

                if cell.fold_metrics:
                    crps_vals = [m.get("crps", float("nan")) for m in cell.fold_metrics]
                    mean_crps = np.nanmean(crps_vals)
                    logger.info(f"    CRPS: {mean_crps:.4f} ({len(cell.fold_metrics)} folds)")

                all_results.append(cell)

                # Checkpoint after each cell
                _save_checkpoint(all_results)

    return all_results


def _save_checkpoint(results: list[CellResult]):
    """Save intermediate results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = [r.to_dict() for r in results]
    with open(RESULTS_DIR / "checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


# =============================================================================
# Aggregation and analysis
# =============================================================================


def results_to_dataframe(results: list[CellResult]) -> pd.DataFrame:
    """Convert results to a tidy DataFrame for analysis."""
    rows = []
    for cell in results:
        if not cell.compatible:
            rows.append(
                {
                    "dgp": cell.dgp_name,
                    "dgp_short": DGP_SHORT_NAMES.get(cell.dgp_name, cell.dgp_name),
                    "dist": cell.dist_name,
                    "dist_short": SHORT_NAMES.get(cell.dist_name, cell.dist_name),
                    "seed": cell.seed,
                    "compatible": False,
                    "fold": -1,
                }
            )
            continue

        for fold_idx, metrics in enumerate(cell.fold_metrics):
            row = {
                "dgp": cell.dgp_name,
                "dgp_short": DGP_SHORT_NAMES.get(cell.dgp_name, cell.dgp_name),
                "dist": cell.dist_name,
                "dist_short": SHORT_NAMES.get(cell.dist_name, cell.dist_name),
                "seed": cell.seed,
                "compatible": True,
                "fold": fold_idx + 1,
                "tuning_time": cell.tuning_time,
                "fit_time": cell.fold_fit_times[fold_idx] if fold_idx < len(cell.fold_fit_times) else None,
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std CRPS per (DGP, distribution) cell."""
    compatible = df[df["compatible"]].copy()
    if compatible.empty:
        return pd.DataFrame()

    summary = (
        compatible.groupby(["dgp_short", "dist_short"])
        .agg(
            crps_mean=("crps", "mean"),
            crps_std=("crps", "std"),
            pit_ks_mean=("pit_ks_statistic", "mean"),
            pit_ks_std=("pit_ks_statistic", "std"),
            pica_mean=("pica", "mean"),
            pica_std=("pica", "std"),
            coverage_90_mean=("coverage_90", "mean"),
            coverage_90_std=("coverage_90", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            n_obs=("crps", "count"),
        )
        .reset_index()
    )

    return summary


def save_results(results: list[CellResult], df: pd.DataFrame):
    """Save final results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Raw results
    raw = [r.to_dict() for r in results]
    with open(RESULTS_DIR / "raw_results.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)

    # Tidy DataFrame
    df.to_parquet(RESULTS_DIR / "results.parquet", index=False)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)

    # Summary table
    summary = compute_summary_table(df)
    if not summary.empty:
        summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    logger.info(f"Results saved to {RESULTS_DIR}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution Misspecification Study")
    parser.add_argument("--quick", action="store_true", help="Smoke test with reduced configuration")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Distribution Misspecification Study")
    logger.info("=" * 70)

    if args.quick:
        logger.info("QUICK MODE: reduced seeds, DGPs, distributions, trials")

    results = run_study(quick=args.quick)

    df = results_to_dataframe(results)
    save_results(results, df)

    # Print summary
    summary = compute_summary_table(df)
    if not summary.empty:
        logger.info("\n" + "=" * 70)
        logger.info("Summary (CRPS, lower is better):")
        logger.info("=" * 70)

        # Pivot for display
        pivot = summary.pivot(index="dgp_short", columns="dist_short", values="crps_mean")
        dgp_order = ["Gaussian", "Count", "Waiting-Time", "Heavy-Tailed", "Multimodal"]
        dist_order = ["Normal", "Poisson", "Exponential", "Student-t", "KDE"]
        pivot = pivot.reindex(index=[d for d in dgp_order if d in pivot.index])
        pivot = pivot.reindex(columns=[d for d in dist_order if d in pivot.columns])
        logger.info(f"\n{pivot.to_string(float_format='%.4f')}")

    logger.info("\nStudy complete!")
