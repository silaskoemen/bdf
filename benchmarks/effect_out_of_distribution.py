"""
Experiment: Out-of-Distribution (OOD) / Covariate Shift Analysis

Investigates how distributional regression models behave under covariate shift,
specifically whether uncertainty estimates increase appropriately in OOD regions.

Uses Friedman 1-3 as base functions (known ground truth, more complex than toy DGPs):
- Training on restricted feature range (e.g., X ∈ [0, 0.7])
- Testing on shifted range (e.g., X ∈ [0.3, 1.0])
- Evaluates separately on overlap region vs OOD region

Shift Configurations:
- none: train [0,1], test [0,1] (baseline, no shift)
- mild: train [0, 0.85], test [0.15, 1.0] (15% OOD)
- moderate: train [0, 0.7], test [0.3, 1.0] (30% OOD)
- strong: train [0, 0.5], test [0.5, 1.0] (50% extrapolation)

Key Questions:
1. Does uncertainty increase in OOD regions? (desirable for UQ)
2. Does coverage degrade gracefully or catastrophically?
3. Which models are most robust to distribution shift?

Following benchmark pattern: tune on fold 0, evaluate on folds 1-9.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from benchmarks.metrics.regression import (
    coverage_at_level,
    crps_wrapper,
    interval_score_samples,
    pica,
    precompute_percentiles,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_FOLDS = 10  # Fold 0 for tuning, folds 1-9 for evaluation
N_TRIALS = 50
N_TRAIN_SAMPLES = 2000
N_TEST_SAMPLES = 1000
N_POSTERIOR_SAMPLES = 500
GENERATE_INDIVIDUAL_PLOTS = True

# Output directories
RESULTS_DIR = Path("benchmarks/results/effect_ood")
PLOTS_DIR = Path("benchmarks/plots/effect_ood")

# =============================================================================
# Shift Configurations
# =============================================================================

ShiftLevel = Literal["none", "mild", "moderate", "strong"]

SHIFT_CONFIGS: dict[ShiftLevel, dict[str, tuple[float, float]]] = {
    "none": {"train": (0.0, 1.0), "test": (0.0, 1.0)},
    "mild": {"train": (0.0, 0.85), "test": (0.15, 1.0)},
    "moderate": {"train": (0.0, 0.7), "test": (0.3, 1.0)},
    "strong": {"train": (0.0, 0.5), "test": (0.5, 1.0)},
}


def get_region_masks(
    X: np.ndarray,
    train_range: tuple[float, float],
    test_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Get boolean masks for overlap and OOD regions.

    Args:
        X: Feature array (n_samples, n_features), values assumed in [0, 1]
        train_range: (low, high) of training feature range
        test_range: (low, high) of test feature range

    Returns:
        overlap_mask: True where X is in overlap region
        ood_mask: True where X is in OOD region (test but not train)
    """
    # For multi-dimensional X, use the first (most important) feature
    x_primary = X[:, 0] if X.ndim > 1 else X

    overlap_low = max(train_range[0], test_range[0])
    overlap_high = min(train_range[1], test_range[1])

    overlap_mask = (x_primary >= overlap_low) & (x_primary <= overlap_high)
    ood_mask = (x_primary > train_range[1]) | (x_primary < train_range[0])

    return overlap_mask, ood_mask


# =============================================================================
# Friedman DGPs with Controlled Feature Ranges
# =============================================================================


@dataclass
class FriedmanDGP:
    """Friedman function with controlled feature range."""

    name: str
    mean_fn: Callable[[np.ndarray], np.ndarray]
    n_features: int
    n_informative: int
    noise_std: float


def friedman1_fn(X: np.ndarray) -> np.ndarray:
    """Friedman #1: 10*sin(π*x1*x2) + 20*(x3-0.5)² + 10*x4 + 5*x5"""
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]


def friedman2_fn(X: np.ndarray) -> np.ndarray:
    """Friedman #2: sqrt(x1² + (x2*x3 - 1/(x2*x4))²)

    Note: We rescale features to avoid numerical issues.
    Original ranges: x1∈[0,100], x2∈[40π,560π], x3∈[0,1], x4∈[1,11]
    We use [0,1] and rescale internally.
    """
    # Rescale from [0,1] to original ranges
    x1 = X[:, 0] * 100  # [0, 100]
    x2 = 40 * np.pi + X[:, 1] * (560 - 40) * np.pi  # [40π, 560π]
    x3 = X[:, 2]  # [0, 1]
    x4 = 1 + X[:, 3] * 10  # [1, 11]

    return np.sqrt(x1**2 + (x2 * x3 - 1 / (x2 * x4 + 1e-8)) ** 2)


def friedman3_fn(X: np.ndarray) -> np.ndarray:
    """Friedman #3: arctan((x2*x3 - 1/(x2*x4)) / x1)

    Similar rescaling as Friedman #2.
    """
    x1 = X[:, 0] * 100 + 1e-8  # [0, 100], avoid division by zero
    x2 = 40 * np.pi + X[:, 1] * (560 - 40) * np.pi
    x3 = X[:, 2]
    x4 = 1 + X[:, 3] * 10

    return np.arctan((x2 * x3 - 1 / (x2 * x4 + 1e-8)) / x1)


DGPS = {
    "friedman1": FriedmanDGP(
        name="friedman1",
        mean_fn=friedman1_fn,
        n_features=10,  # 5 informative + 5 noise
        n_informative=5,
        noise_std=1.0,
    ),
    "friedman2": FriedmanDGP(
        name="friedman2",
        mean_fn=friedman2_fn,
        n_features=4,
        n_informative=4,
        noise_std=125.0,  # Scaled to match original
    ),
    "friedman3": FriedmanDGP(
        name="friedman3",
        mean_fn=friedman3_fn,
        n_features=4,
        n_informative=4,
        noise_std=0.1,
    ),
}


def generate_data(
    dgp: FriedmanDGP,
    n_samples: int,
    feature_range: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data from Friedman DGP with controlled feature range.

    Args:
        dgp: FriedmanDGP specification
        n_samples: Number of samples
        feature_range: (low, high) for uniform feature distribution
        seed: Random seed

    Returns:
        X: Features (n_samples, n_features)
        y: Targets (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # Generate features in specified range
    low, high = feature_range
    X = rng.uniform(low, high, size=(n_samples, dgp.n_features))

    # Generate targets
    y_mean = dgp.mean_fn(X)
    y = y_mean + rng.randn(n_samples) * dgp.noise_std

    return X, y


# =============================================================================
# Model Configurations (Dynamic Loading)
# =============================================================================


def get_available_models() -> dict[str, dict[str, Any]]:
    """Detect available models in current environment."""
    available = {}

    # BDF Models
    try:
        from bdf.tree_classes.bdf_regressor import BDFRegressor

        available["BDFNormal"] = {
            "class": BDFRegressor,
            "fixed_init_kwargs": {"dist": "NormalMuNormal", "random_state": SEED},
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
            "fixed_params": {"mu_mu": "auto", "score_correction": "bic"},
            "has_params_dict": True,
            "probabilistic": True,
        }

        available["BDFKDE"] = {
            "class": BDFRegressor,
            "fixed_init_kwargs": {"dist": "KDE", "random_state": SEED},
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
            "has_params_dict": True,
            "probabilistic": True,
        }

        # RandomForest baseline
        available["RandomForest"] = {
            "class": RandomForestRegressor,
            "fixed_init_kwargs": {"random_state": SEED},
            "tunable_init_kwargs": {
                "criterion": {"type": "categorical", "categories": ["squared_error", "absolute_error"]},
                "max_depth": {"type": "int", "low": 5, "high": 30},
                "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
                "min_samples_split": {"type": "int", "low": 10, "high": 100},
                "max_features": {"type": "float", "low": 0.1, "high": 1.0},
                "n_estimators": {"type": "int", "low": 25, "high": 200},
            },
            "tunable_params": {},
            "fixed_params": {},
            "has_params_dict": False,
            "probabilistic": False,
        }
        logger.info("✓ BDF and RandomForest available")
    except ImportError:
        logger.info("✗ BDF not available")

    # NGBoost
    try:
        from benchmarks.models.wrappers import NGBRegressorWrapper

        available["NGBoost"] = {
            "class": NGBRegressorWrapper,
            "fixed_init_kwargs": {"random_state": SEED, "verbose": False, "dist_name": "Normal"},
            "tunable_init_kwargs": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                "minibatch_frac": {"type": "float", "low": 0.5, "high": 1.0},
            },
            "tunable_params": {},
            "fixed_params": {},
            "has_params_dict": False,
            "probabilistic": True,
        }
        logger.info("✓ NGBoost available")
    except ImportError:
        pass

    # Conformalized LightGBM
    try:
        from benchmarks.models.wrappers import ConformalizedLGBMWrapper

        available["ConformalLGBM"] = {
            "class": ConformalizedLGBMWrapper,
            "fixed_init_kwargs": {"random_state": SEED, "verbose": -1},
            "tunable_init_kwargs": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "num_leaves": {"type": "int", "low": 15, "high": 100},
                "min_child_samples": {"type": "int", "low": 10, "high": 100},
            },
            "tunable_params": {},
            "fixed_params": {},
            "has_params_dict": False,
            "probabilistic": True,
        }
        logger.info("✓ ConformalLGBM available")
    except ImportError:
        pass

    return available


MODEL_CONFIGS = get_available_models()


# =============================================================================
# Hyperparameter Tuning
# =============================================================================


def suggest_hyperparameters(trial: optuna.Trial, model_name: str) -> tuple[dict, dict]:
    """Suggest hyperparameters for Optuna trial."""
    config = MODEL_CONFIGS[model_name]
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


def split_best_params(best_params: dict, model_name: str) -> tuple[dict, dict]:
    """Split Optuna best params into init_kwargs and params."""
    config = MODEL_CONFIGS[model_name]
    init_kwargs = {}
    params = dict(config.get("fixed_params", {}))

    for param, value in best_params.items():
        if param in config["tunable_init_kwargs"]:
            init_kwargs[param] = value
        elif param in config.get("tunable_params", {}):
            params[param] = value

    return init_kwargs, params


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    study_name: str,
    model_name: str,
) -> tuple[dict, dict, float]:
    """Tune model hyperparameters using Optuna."""
    config = MODEL_CONFIGS[model_name]
    model_cls = config["class"]
    fixed_init_kwargs = config["fixed_init_kwargs"]
    has_params_dict = config.get("has_params_dict", False)

    def objective(trial: optuna.Trial) -> float:
        np.random.seed(SEED + trial.number)
        init_kwargs, params = suggest_hyperparameters(trial, model_name)

        try:
            if has_params_dict and params:
                model = model_cls(**fixed_init_kwargs, **init_kwargs, params=params)
            else:
                model = model_cls(**fixed_init_kwargs, **init_kwargs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, y_pred))
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")

    os.makedirs("benchmarks/results/optuna/", exist_ok=True)
    storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"

    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except (KeyError, Exception):
        pass

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        sampler=sampler,
        load_if_exists=False,
    )

    start_time = time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    tuning_time = time() - start_time

    best_init_kwargs, best_params = split_best_params(study.best_params, model_name)
    return best_init_kwargs, best_params, tuning_time


# =============================================================================
# Metrics Computation (Region-Specific)
# =============================================================================


def compute_metrics_by_region(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_samples: np.ndarray | None,
    overlap_mask: np.ndarray,
    ood_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute metrics separately for overlap and OOD regions.

    Returns dict with keys: overall, overlap, ood
    """
    results = {}

    for region_name, mask in [
        ("overall", np.ones(len(y_true), dtype=bool)),
        ("overlap", overlap_mask),
        ("ood", ood_mask),
    ]:
        if mask.sum() == 0:
            results[region_name] = {"n_samples": 0}
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        region_metrics = {
            "n_samples": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "mae": float(mean_absolute_error(y_t, y_p)),
        }

        if y_samples is not None:
            y_s = y_samples[mask]

            # Uncertainty metrics
            y_std = np.std(y_s, axis=1)
            region_metrics["mean_uncertainty"] = float(np.mean(y_std))
            region_metrics["median_uncertainty"] = float(np.median(y_std))

            # Probabilistic metrics
            try:
                precomputed = precompute_percentiles(y_s)
                region_metrics["crps"] = float(crps_wrapper(y_t, y_s))

                # Coverage at 90%
                region_metrics["coverage_90"] = float(
                    coverage_at_level(y_t, y_s, level=0.90, quantile_levels=None, precomputed=precomputed)
                )
                lower = np.percentile(y_s, 5, axis=1)
                upper = np.percentile(y_s, 95, axis=1)

                # Interval score
                int_score = interval_score_samples(y_t, y_s, alpha=0.1, precomputed=precomputed)
                region_metrics["interval_score_90"] = float(np.mean(int_score))

                # Interval width
                region_metrics["ci_width_90"] = float(np.mean(upper - lower))

                # PICA (Prediction Interval Coverage Accuracy)
                try:
                    region_metrics["pica"] = float(pica(y_t, y_s))
                except Exception:
                    pass

            except Exception as e:
                logger.warning(f"Probabilistic metric computation failed for {region_name}: {e}")

        results[region_name] = region_metrics

    # Compute uncertainty ratio (OOD / overlap) - key metric for UQ quality
    if results.get("ood", {}).get("mean_uncertainty") and results.get("overlap", {}).get("mean_uncertainty"):
        results["uncertainty_ratio_ood_vs_overlap"] = (
            results["ood"]["mean_uncertainty"] / results["overlap"]["mean_uncertainty"]
        )

    return results


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_range: tuple[float, float],
    test_range: tuple[float, float],
    init_kwargs: dict,
    params: dict,
    model_name: str,
) -> tuple[dict, float]:
    """Evaluate model on a single fold with region-specific metrics."""
    config = MODEL_CONFIGS[model_name]
    model_cls = config["class"]
    fixed_init_kwargs = config["fixed_init_kwargs"]
    has_params_dict = config.get("has_params_dict", False)
    is_probabilistic = config.get("probabilistic", False)

    np.random.seed(SEED)

    if has_params_dict and params:
        model = model_cls(**fixed_init_kwargs, **init_kwargs, params=params)
    else:
        model = model_cls(**fixed_init_kwargs, **init_kwargs)

    start_time = time()
    model.fit(X_train, y_train)
    fit_time = time() - start_time

    y_pred = model.predict(X_test)

    # Get samples for probabilistic models
    y_samples = None
    if is_probabilistic:
        try:
            if hasattr(model, "predict_samples"):
                y_samples = model.predict_samples(X_test, n_samples=N_POSTERIOR_SAMPLES)
            elif hasattr(model, "pred_dist"):
                dist = model.pred_dist(X_test)
                y_samples = dist.sample(N_POSTERIOR_SAMPLES).T
        except Exception as e:
            logger.warning(f"Failed to get samples: {e}")

    # Get region masks
    overlap_mask, ood_mask = get_region_masks(X_test, train_range, test_range)

    # Compute metrics by region
    metrics = compute_metrics_by_region(y_test, y_pred, y_samples, overlap_mask, ood_mask)

    return metrics, fit_time


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
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_uncertainty_by_shift(
    results: dict,
    save_dir: Path,
):
    """Plot uncertainty ratio (OOD/overlap) vs shift level."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dgp_names = list(DGPS.keys())
    shift_levels = ["none", "mild", "moderate", "strong"]
    colors = sns.color_palette("husl", len(MODEL_CONFIGS))
    model_colors = dict(zip(MODEL_CONFIGS.keys(), colors))

    for ax, dgp_name in zip(axes, dgp_names):
        for model_name in MODEL_CONFIGS.keys():
            ratios = []
            for shift in shift_levels:
                key = f"{dgp_name}_{shift}"
                if key in results and model_name in results[key].get("models", {}):
                    agg = results[key]["models"][model_name].get("aggregated_metrics", {})
                    ratio = agg.get("uncertainty_ratio_ood_vs_overlap", {}).get("mean", np.nan)
                    ratios.append(ratio)
                else:
                    ratios.append(np.nan)

            ax.plot(
                range(len(shift_levels)),
                ratios,
                "o-",
                label=model_name,
                color=model_colors[model_name],
                linewidth=2,
                markersize=6,
            )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No change")
        ax.set_xticks(range(len(shift_levels)))
        ax.set_xticklabels(shift_levels)
        ax.set_xlabel("Shift Level")
        ax.set_ylabel("Uncertainty Ratio (OOD / Overlap)")
        ax.set_title(f"{dgp_name}")
        ax.legend(loc="best")

    plt.suptitle("Uncertainty Increase in OOD Regions\n(Higher = better UQ)", fontsize=14)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "uncertainty_ratio_by_shift.png")
    logger.info(f"Saved plot to {save_dir / 'uncertainty_ratio_by_shift.png'}")
    plt.close()


def plot_coverage_degradation(
    results: dict,
    save_dir: Path,
):
    """Plot coverage in OOD region vs shift level."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dgp_names = list(DGPS.keys())
    shift_levels = ["none", "mild", "moderate", "strong"]
    colors = sns.color_palette("husl", len(MODEL_CONFIGS))
    model_colors = dict(zip(MODEL_CONFIGS.keys(), colors))

    for ax, dgp_name in zip(axes, dgp_names):
        for model_name in MODEL_CONFIGS.keys():
            if not MODEL_CONFIGS[model_name].get("probabilistic", False):
                continue

            coverages_ood = []
            coverages_overlap = []
            for shift in shift_levels:
                key = f"{dgp_name}_{shift}"
                if key in results and model_name in results[key].get("models", {}):
                    agg = results[key]["models"][model_name].get("aggregated_metrics", {})

                    ood_cov = agg.get("ood", {}).get("coverage_90", {}).get("mean", np.nan)
                    overlap_cov = agg.get("overlap", {}).get("coverage_90", {}).get("mean", np.nan)

                    coverages_ood.append(ood_cov)
                    coverages_overlap.append(overlap_cov)
                else:
                    coverages_ood.append(np.nan)
                    coverages_overlap.append(np.nan)

            ax.plot(
                range(len(shift_levels)),
                coverages_ood,
                "o-",
                label=f"{model_name} (OOD)",
                color=model_colors[model_name],
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                range(len(shift_levels)),
                coverages_overlap,
                "s--",
                label=f"{model_name} (Overlap)",
                color=model_colors[model_name],
                linewidth=1.5,
                markersize=5,
                alpha=0.6,
            )

        ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.7, label="Nominal 90%")
        ax.set_xticks(range(len(shift_levels)))
        ax.set_xticklabels(shift_levels)
        ax.set_xlabel("Shift Level")
        ax.set_ylabel("Coverage @ 90%")
        ax.set_title(f"{dgp_name}")
        ax.set_ylim(0.5, 1.0)
        ax.legend(loc="best", fontsize=8)

    plt.suptitle("Coverage Degradation in OOD Regions", fontsize=14)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "coverage_by_shift.png")
    logger.info(f"Saved plot to {save_dir / 'coverage_by_shift.png'}")
    plt.close()


def plot_rmse_by_region(
    results: dict,
    save_dir: Path,
):
    """Plot RMSE comparison between overlap and OOD regions."""
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dgp_names = list(DGPS.keys())
    shift_levels = ["none", "mild", "moderate", "strong"]
    colors = sns.color_palette("husl", len(MODEL_CONFIGS))
    model_colors = dict(zip(MODEL_CONFIGS.keys(), colors))

    for ax, dgp_name in zip(axes, dgp_names):
        for model_name in MODEL_CONFIGS.keys():
            rmse_ood = []
            rmse_overlap = []
            for shift in shift_levels:
                key = f"{dgp_name}_{shift}"
                if key in results and model_name in results[key].get("models", {}):
                    agg = results[key]["models"][model_name].get("aggregated_metrics", {})

                    ood_rmse = agg.get("ood", {}).get("rmse", {}).get("mean", np.nan)
                    overlap_rmse = agg.get("overlap", {}).get("rmse", {}).get("mean", np.nan)

                    rmse_ood.append(ood_rmse)
                    rmse_overlap.append(overlap_rmse)
                else:
                    rmse_ood.append(np.nan)
                    rmse_overlap.append(np.nan)

            ax.plot(
                range(len(shift_levels)),
                rmse_ood,
                "o-",
                label=f"{model_name} (OOD)",
                color=model_colors[model_name],
                linewidth=2,
                markersize=6,
            )

        ax.set_xticks(range(len(shift_levels)))
        ax.set_xticklabels(shift_levels)
        ax.set_xlabel("Shift Level")
        ax.set_ylabel("RMSE (OOD Region)")
        ax.set_title(f"{dgp_name}")
        ax.legend(loc="best")

    plt.suptitle("Prediction Error in OOD Regions", fontsize=14)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "rmse_ood_by_shift.png")
    logger.info(f"Saved plot to {save_dir / 'rmse_ood_by_shift.png'}")
    plt.close()


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark():
    """Run the OOD/covariate shift benchmark."""
    logger.info("=" * 80)
    logger.info("Out-of-Distribution Analysis Benchmark")
    logger.info("=" * 80)

    available_models = list(MODEL_CONFIGS.keys())
    logger.info(f"Available models: {', '.join(available_models)}")
    logger.info(f"DGPs: {list(DGPS.keys())}")
    logger.info(f"Shift levels: {list(SHIFT_CONFIGS.keys())}")

    # Detect environment
    core_models = {"BDFNormal", "BDFKDE", "RandomForest"}
    has_extended = any(m not in core_models for m in available_models)
    env_suffix = "" if has_extended else "_core"

    all_results = {}

    for dgp_name, dgp in DGPS.items():
        for shift_level, shift_config in SHIFT_CONFIGS.items():
            experiment_key = f"{dgp_name}_{shift_level}"
            train_range = shift_config["train"]
            test_range = shift_config["test"]

            logger.info(f"\n{'='*60}")
            logger.info(f"DGP: {dgp_name}, Shift: {shift_level}")
            logger.info(f"Train range: {train_range}, Test range: {test_range}")
            logger.info(f"{'='*60}")

            experiment_results = {
                "dgp_name": dgp_name,
                "shift_level": shift_level,
                "train_range": train_range,
                "test_range": test_range,
                "models": {},
            }

            # Generate training data
            X_train_full, y_train_full = generate_data(dgp, N_TRAIN_SAMPLES, train_range, SEED)

            # K-fold on training data
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            folds = list(enumerate(kf.split(X_train_full)))

            for model_name in available_models:
                logger.info(f"\n  Model: {model_name}")

                model_results = {
                    "fold_metrics": [],
                    "fold_fit_times": [],
                    "best_params": None,
                    "tuning_time": None,
                }

                # Tune on fold 0
                train_idx_0, val_idx_0 = folds[0][1]
                X_tune_train = X_train_full[train_idx_0]
                y_tune_train = y_train_full[train_idx_0]
                X_tune_val = X_train_full[val_idx_0]
                y_tune_val = y_train_full[val_idx_0]

                study_name = f"ood_{dgp_name}_{shift_level}_{model_name}"
                try:
                    logger.info("    Tuning hyperparameters...")
                    best_init_kwargs, best_params, tuning_time = tune_model(
                        X_tune_train, y_tune_train, X_tune_val, y_tune_val, study_name, model_name
                    )
                    model_results["best_params"] = {
                        "init_kwargs": best_init_kwargs,
                        "params": best_params,
                    }
                    model_results["tuning_time"] = tuning_time
                except Exception as e:
                    logger.error(f"    Tuning failed: {e}")
                    continue

                # Evaluate on folds 1-9 with OOD test data
                logger.info("    Evaluating on folds 1-9...")
                for fold_idx, (train_idx, _) in tqdm(folds[1:], desc="Folds"):
                    X_fold_train = X_train_full[train_idx]
                    y_fold_train = y_train_full[train_idx]

                    # Generate fresh OOD test data for each fold
                    X_test, y_test = generate_data(dgp, N_TEST_SAMPLES, test_range, SEED + fold_idx)

                    try:
                        metrics, fit_time = evaluate_fold(
                            X_fold_train,
                            y_fold_train,
                            X_test,
                            y_test,
                            train_range,
                            test_range,
                            best_init_kwargs,
                            best_params,
                            model_name,
                        )
                        model_results["fold_metrics"].append(metrics)
                        model_results["fold_fit_times"].append(fit_time)
                    except Exception as e:
                        logger.error(f"    Fold {fold_idx} failed: {e}")

                # Aggregate results
                if model_results["fold_metrics"]:
                    aggregated = {}

                    # Aggregate each region's metrics
                    for region in ["overall", "overlap", "ood"]:
                        region_agg = {}
                        for metric_key in model_results["fold_metrics"][0].get(region, {}).keys():
                            if metric_key == "n_samples":
                                continue
                            values = [
                                m[region][metric_key]
                                for m in model_results["fold_metrics"]
                                if region in m and metric_key in m[region] and not np.isnan(m[region][metric_key])
                            ]
                            if values:
                                region_agg[metric_key] = {
                                    "mean": float(np.mean(values)),
                                    "std": float(np.std(values)),
                                }
                        aggregated[region] = region_agg

                    # Aggregate uncertainty ratio
                    ratios = [
                        m.get("uncertainty_ratio_ood_vs_overlap")
                        for m in model_results["fold_metrics"]
                        if m.get("uncertainty_ratio_ood_vs_overlap") is not None
                    ]
                    if ratios:
                        aggregated["uncertainty_ratio_ood_vs_overlap"] = {
                            "mean": float(np.mean(ratios)),
                            "std": float(np.std(ratios)),
                        }

                    model_results["aggregated_metrics"] = aggregated
                    model_results["mean_fit_time"] = float(np.mean(model_results["fold_fit_times"]))

                    # Log summary
                    logger.info(
                        f"    Overall RMSE: {aggregated.get('overall', {}).get('rmse', {}).get('mean', 'N/A'):.4f}"
                    )
                    if "uncertainty_ratio_ood_vs_overlap" in aggregated:
                        logger.info(
                            f"    Uncertainty ratio (OOD/Overlap): {aggregated['uncertainty_ratio_ood_vs_overlap']['mean']:.2f}"
                        )

                experiment_results["models"][model_name] = model_results

            all_results[experiment_key] = experiment_results

            # Save incrementally
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            output_file = RESULTS_DIR / f"{experiment_key}_results{env_suffix}.json"
            with open(output_file, "w") as f:
                json.dump(experiment_results, f, indent=2)
            logger.info(f"  Saved to {output_file}")

    # Save combined results
    combined_file = RESULTS_DIR / f"all_results{env_suffix}.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved combined results to {combined_file}")

    # Generate plots
    logger.info("\nGenerating plots...")
    plot_uncertainty_by_shift(all_results, PLOTS_DIR)
    plot_coverage_degradation(all_results, PLOTS_DIR)
    plot_rmse_by_region(all_results, PLOTS_DIR)

    logger.success("\n" + "=" * 80)
    logger.success("OOD Benchmark Complete!")
    logger.success("=" * 80)

    return all_results


if __name__ == "__main__":
    run_benchmark()
