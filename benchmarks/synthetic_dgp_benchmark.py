"""Rigorous benchmarking on synthetic DGPs with known ground truth.

This script evaluates distributional regression methods on carefully designed synthetic
datasets where the true conditional distribution p(y|x) is known. This enables computing:
- Mean squared error against true mean function
- Variance estimation error against true variance
- Quantile coverage and sharpness
- Calibration metrics (PIT, PICA)
- Distributional distance metrics (when tractable)

Experimental design follows JMLR standards:
- Proper cross-validation: tune on fold 0, evaluate on folds 1-9
- Multiple baseline comparisons
- Statistical significance testing
- Publication-quality visualizations with ground truth overlays
"""

import json
import os
from pathlib import Path
from time import time
from typing import Any, Optional

import numpy as np
import optuna
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from .metrics.regression import (
    crps_wrapper,
    interval_score_samples,
    pica,
    pit_histogram,
    pit_ks_statistic,
    quantile_loss,
)
from .pipeline.synthetic_dgps import DGP_REGISTRY, SyntheticDataset
from .utils.synthetic_plotting import (
    plot_conditional_densities,
    plot_coverage_by_region,
    plot_pit_histogram,
    plot_predictions_with_ground_truth,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_SPLITS = 10  # 10-fold CV: fold 0 for tuning, folds 1-9 for evaluation
N_TRIALS = 50  # More trials for synthetic (cleaner signal)
N_POSTERIOR_SAMPLES = 1000  # For computing predictive distributions
GENERATE_INDIVIDUAL_PLOTS = True  # Generate diagnostic plots during benchmark

# DGPs to benchmark
DGPS_TO_RUN = [
    {"name": "heteroscedastic_sinusoidal", "kwargs": {"n_samples": 5000, "noise_scale": 0.1}},
    {"name": "step_function", "kwargs": {"n_samples": 5000, "noise_std": 0.15, "n_steps": 5}},
    {"name": "bimodal_mixture", "kwargs": {"n_samples": 5000, "noise_std": 0.1, "separation": 2.0}},
    {"name": "heavy_tailed", "kwargs": {"n_samples": 5000, "df": 3.0, "scale": 0.5}},
    {"name": "sparse_sampling", "kwargs": {"n_samples": 1000, "noise_std": 0.1, "sparsity_type": "linear_decay"}},
]

# Quantiles for evaluation
EVAL_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# Coverage levels for calibration curves (19 levels: 0.05, 0.10, ..., 0.95)
COVERAGE_LEVELS = [round(0.05 * i, 2) for i in range(1, 20)]

# =============================================================================
# Dynamic Model Loading based on Available Dependencies
# =============================================================================


def get_available_models() -> dict[str, dict[str, Any]]:
    """Detect which models are available in current environment.

    This allows the benchmark to run in both:
    - Default environment: BDF models + sklearn baseline
    - Bench-models environment: Full comparison with NGBoost, LightGBM, etc.

    Returns:
        Dictionary mapping model names to their configurations
    """
    available = {}

    # =============================================================================
    # BDF Models - Available from default environment
    # =============================================================================
    try:
        from bdf.tree_classes.bdf_regressor import BDFRegressor

        available["BDFNormal"] = {
            "class": BDFRegressor,
            "fixed_init_kwargs": {
                "dist": "NormalMuNormal",
                "random_state": SEED,
            },
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
            "has_params_dict": True,
            "probabilistic": True,
        }

        available["BDFKDE"] = {
            "class": BDFRegressor,
            "fixed_init_kwargs": {
                "dist": "KDE",
                "random_state": SEED,
            },
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

    except (ImportError, ModuleNotFoundError):
        logger.info("✗ BDF not available (run in default environment)")

    # =============================================================================
    # Optional Models (bench-models environment)
    # =============================================================================

    # Conformal RF
    try:
        from .models.wrappers import ConformalizedRFWrapper

        available["ConformalRF"] = {
            "class": ConformalizedRFWrapper,
            "fixed_init_kwargs": {
                "random_state": SEED,
            },
            "tunable_init_kwargs": {
                "n_estimators": {"type": "int", "low": 25, "high": 200},
                "max_depth": {"type": "int", "low": 5, "high": 30},
                "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
                "max_features": {"type": "float", "low": 0.1, "high": 1.0},
            },
            "tunable_params": {},
            "fixed_params": {},
            "has_params_dict": False,
            "probabilistic": True,
        }
        logger.info("✓ ConformalRF available")
    except ImportError:
        logger.info("✗ ConformalRF not available (run in benchmark environment)")

    # NGBoost
    try:
        from .models.wrappers import NGBRegressorWrapper

        available["NGBoost"] = {
            "class": NGBRegressorWrapper,
            "fixed_init_kwargs": {
                "random_state": SEED,
                "verbose": False,
                "dist_name": "Normal",
                "col_sample": 1.0,  # Fixed to 1.0 for low-dimensional synthetic DGPs
            },
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
        logger.info("✗ NGBoost not available (install ngboost or run in bench-models environment)")

    # BART (Bayesian Additive Regression Trees)
    try:
        from .models.wrappers import BARTPyRegressorWrapper

        available["BART"] = {
            "class": BARTPyRegressorWrapper,
            "fixed_init_kwargs": {},
            "tunable_init_kwargs": {
                "n_trees": {"type": "int", "low": 20, "high": 200},
                "n_burn": {"type": "int", "low": 50, "high": 250},
                "n_samples": {"type": "int", "low": 100, "high": 500},
                "alpha": {"type": "float", "low": 0.5, "high": 0.99},
                "beta": {"type": "float", "low": 0.5, "high": 3.0},
            },
            "tunable_params": {},
            "fixed_params": {},
            "has_params_dict": False,
            "probabilistic": True,
        }
        logger.info("✓ BART available")
    except ImportError:
        logger.info("✗ BART not available (install bartpy or run in bench-models environment)")

    return available


# Initialize MODEL_CONFIGS with available models
MODEL_CONFIGS = get_available_models()


# =============================================================================
# Helper Functions
# =============================================================================


def suggest_hyperparameters(trial: optuna.Trial, model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Suggest hyperparameters from optuna trial for a specific model."""
    config = MODEL_CONFIGS[model_name]
    init_kwargs = {}

    # Suggest tunable init kwargs
    for name, args in config["tunable_init_kwargs"].items():
        if args["type"] == "int":
            init_kwargs[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            init_kwargs[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            init_kwargs[name] = trial.suggest_categorical(name, args["categories"])

    # Build params dict
    params = dict(config.get("fixed_params", {}))
    tunable_params = config.get("tunable_params", {})

    for name, args in tunable_params.items():
        if args["type"] == "int":
            params[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            params[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, args["categories"])

    return init_kwargs, params


def split_best_params(best_params: dict[str, Any], model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split optuna best params into init_kwargs and params dict."""
    config = MODEL_CONFIGS[model_name]
    tuned_init_kwargs = {}
    tuned_params = {}

    for param, value in best_params.items():
        if param in config["tunable_init_kwargs"]:
            tuned_init_kwargs[param] = value
        elif param in config.get("tunable_params", {}):
            tuned_params[param] = value

    # Combine fixed_params with tuned_params
    combined_params = dict(config.get("fixed_params", {}))
    if tuned_params:
        combined_params.update(tuned_params)

    return tuned_init_kwargs, combined_params


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    study_name: str,
    model_name: str,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Tune model using optuna on train/val split from fold 0.

    Returns:
        tuple of (best_init_kwargs, best_params, tuning_time)
    """
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
            return np.sqrt(mean_squared_error(y_val, y_pred))  # RMSE for tuning
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")

    # Create optuna storage
    os.makedirs("benchmarks/results/optuna/", exist_ok=True)
    storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"

    # Clean up existing study
    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except (KeyError, Exception):
        pass

    # Create study
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        sampler=sampler,
        load_if_exists=False,
    )

    start_time = time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    tuning_time = time() - start_time

    logger.info(f"  Best CV RMSE: {study.best_value:.4f}")

    best_init_kwargs, best_params = split_best_params(study.best_params, model_name)
    return best_init_kwargs, best_params, tuning_time


def compute_ground_truth_metrics(
    dataset: SyntheticDataset,
    X_test: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute metrics against ground truth functions.

    Args:
        dataset: SyntheticDataset with ground truth functions
        X_test: Test features
        y_pred_mean: Predicted means
        y_pred_std: Predicted standard deviations (if available)

    Returns:
        Dictionary of ground truth metrics
    """
    gt = dataset.ground_truth
    metrics = {}

    # Mean function error
    y_true_mean = gt.mean_fn(X_test).flatten()
    metrics["gt_mean_mse"] = float(mean_squared_error(y_true_mean, y_pred_mean))
    metrics["gt_mean_rmse"] = float(np.sqrt(mean_squared_error(y_true_mean, y_pred_mean)))
    metrics["gt_mean_mae"] = float(mean_absolute_error(y_true_mean, y_pred_mean))

    # Variance function error (if model provides std and ground truth has variance)
    if y_pred_std is not None and gt.variance_fn is not None:
        y_true_var = gt.variance_fn(X_test).flatten()
        y_pred_var = y_pred_std**2
        metrics["gt_variance_mse"] = float(mean_squared_error(y_true_var, y_pred_var))
        metrics["gt_variance_rmse"] = float(np.sqrt(mean_squared_error(y_true_var, y_pred_var)))
        metrics["gt_variance_mae"] = float(mean_absolute_error(y_true_var, y_pred_var))
        # Relative error
        metrics["gt_variance_rel_error"] = float(np.mean(np.abs(y_pred_var - y_true_var) / (y_true_var + 1e-8)))

    return metrics


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset: SyntheticDataset,
    init_kwargs: dict[str, Any],
    params: dict[str, Any],
    model_name: str,
    fold_idx: int = -1,
    dgp_name: str = "",
) -> tuple[dict[str, float], float, Any, Optional[np.ndarray]]:
    """Train and evaluate model on a single fold with ground truth metrics.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        dataset: SyntheticDataset with ground truth
        init_kwargs: Model initialization kwargs
        params: Model params dict
        model_name: Name of the model
        fold_idx: Fold index (for plotting on fold 1)
        dgp_name: DGP name (for plot directory)

    Returns:
        tuple of (metrics_dict, fit_time, model, y_samples)
    """
    config = MODEL_CONFIGS[model_name]
    model_cls = config["class"]
    fixed_init_kwargs = config["fixed_init_kwargs"]
    has_params_dict = config.get("has_params_dict", False)
    is_probabilistic = config.get("probabilistic", False)

    np.random.seed(SEED)

    # Train model
    if has_params_dict and params:
        model = model_cls(**fixed_init_kwargs, **init_kwargs, params=params)
    else:
        model = model_cls(**fixed_init_kwargs, **init_kwargs)

    start_time = time()
    model.fit(X_train, y_train)
    fit_time = time() - start_time

    # Point predictions
    y_pred = model.predict(X_test)

    # Basic metrics
    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    # Store samples for potential plotting
    y_samples = None

    # Probabilistic metrics
    if is_probabilistic:
        try:
            # Get predictive samples
            if hasattr(model, "predict_samples"):
                y_samples = model.predict_samples(X_test, n_samples=N_POSTERIOR_SAMPLES)
            elif hasattr(model, "pred_dist"):
                # NGBoost interface
                dist = model.pred_dist(X_test)
                y_samples = dist.sample(N_POSTERIOR_SAMPLES).T
            else:
                y_samples = None

            if y_samples is not None:
                # Store for plotting
                # y_samples already assigned above

                # Compute probabilistic metrics
                y_pred_std = np.std(y_samples, axis=-1)
                metrics["sharpness"] = float(np.mean(y_pred_std))

                # CRPS
                metrics["crps"] = float(crps_wrapper(y_test, y_samples))

                # Interval scores and coverage
                for width in [0.5, 0.9, 0.95]:
                    alpha = 1 - width
                    lower_q = alpha / 2
                    upper_q = 1 - alpha / 2
                    y_lower = np.quantile(y_samples, lower_q, axis=-1)
                    y_upper = np.quantile(y_samples, upper_q, axis=-1)

                    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
                    interval_width = np.mean(y_upper - y_lower)

                    metrics[f"coverage_{int(width*100)}"] = float(coverage)
                    metrics[f"ci_width_{int(width*100)}"] = float(interval_width)

                    # Interval score
                    int_score = interval_score_samples(y_test, y_samples, alpha=alpha)
                    metrics[f"interval_score_{int(width*100)}"] = float(np.mean(int_score))

                # Quantile losses
                for tau in EVAL_QUANTILES:
                    q_loss = quantile_loss(y_test, y_samples, quantile=tau)
                    metrics[f"quantile_loss_{int(tau*100)}"] = float(q_loss)

                # Calibration (PIT-based)
                try:
                    metrics["pica"] = float(pica(y_test, y_samples))
                    metrics["pit_ks_statistic"] = float(pit_ks_statistic(y_test, y_samples))
                    pit_data = pit_histogram(y_test, y_samples, n_bins=20)
                    bin_proportions = (np.array(pit_data["bin_counts"]) / sum(pit_data["bin_counts"])).tolist()
                    metrics["pit_bin_proportions"] = bin_proportions
                except Exception as e:
                    logger.warning(f"Calibration metric calculation failed: {e}")

                # Coverage curve for calibration plots (19 levels)
                coverage_curve = []
                for level in COVERAGE_LEVELS:
                    alpha = 1 - level
                    y_lower = np.quantile(y_samples, alpha / 2, axis=-1)
                    y_upper = np.quantile(y_samples, 1 - alpha / 2, axis=-1)
                    coverage_curve.append(float(np.mean((y_test >= y_lower) & (y_test <= y_upper))))
                metrics["coverage_curve_levels"] = COVERAGE_LEVELS
                metrics["coverage_curve_empirical"] = coverage_curve

                # Ground truth metrics
                gt_metrics = compute_ground_truth_metrics(dataset, X_test, y_pred, y_pred_std)
                metrics.update(gt_metrics)

        except Exception as e:
            logger.warning(f"Probabilistic evaluation failed: {e}")
    else:
        # For non-probabilistic models, just compute ground truth mean error
        gt_metrics = compute_ground_truth_metrics(dataset, X_test, y_pred, None)
        metrics.update(gt_metrics)

    # Generate diagnostic plots if this is fold 1 and plotting is enabled
    if GENERATE_INDIVIDUAL_PLOTS and fold_idx == 1 and y_samples is not None:
        try:
            dgp_plot_dir = Path(f"benchmarks/plots/synthetic_dgp/{dgp_name}")
            dgp_plot_dir.mkdir(parents=True, exist_ok=True)

            # X values for conditional density plots (DGP-specific)
            conditional_x_values = {
                "heteroscedastic_sinusoidal": [1.0, 3.0, 3.14, 3.3, 5.0],
                "step_function": [0.5, 1.5, 2.5, 4.0, 5.5],
                "bimodal_mixture": [0.1, 0.3, 0.5, 0.7, 0.9],
                "heavy_tailed": None,  # 2D features, skip
                "sparse_sampling": [0.5, 1.5, 3.0, 4.5, 6.0],
            }

            # 1. Predictions with ground truth overlay
            plot_predictions_with_ground_truth(
                dataset=dataset,
                X_test=X_test,
                y_test=y_test,
                y_pred=y_pred,
                y_samples=y_samples,
                model_name=model_name,
                save_path=dgp_plot_dir / f"{model_name}_predictions.pdf",
            )

            # 2. Conditional densities (if 1D)
            if X_test.shape[1] == 1:
                x_values = conditional_x_values.get(dgp_name)
                if x_values:
                    plot_conditional_densities(
                        dataset=dataset,
                        y_samples=y_samples,
                        X_test=X_test,
                        y_test=y_test,
                        x_values=x_values,
                        model_name=model_name,
                        save_path=dgp_plot_dir / f"{model_name}_conditional_densities.pdf",
                    )

                # 3. Local calibration by region
                plot_coverage_by_region(
                    dataset=dataset,
                    X_test=X_test,
                    y_test=y_test,
                    y_samples=y_samples,
                    model_name=model_name,
                    save_path=dgp_plot_dir / f"{model_name}_local_calibration.pdf",
                )

            # 4. PIT histogram (works for any dimensionality)
            plot_pit_histogram(
                y_test=y_test,
                y_samples=y_samples,
                model_name=model_name,
                dgp_name=dgp_name,
                save_path=dgp_plot_dir / f"{model_name}_pit_histogram.pdf",
            )

            # Save fold-1 data for predictions grid figure
            np.savez_compressed(
                dgp_plot_dir / f"{model_name}_fold1_data.npz",
                X_test=X_test,
                y_test=y_test,
                y_pred=y_pred,
                y_samples=y_samples,
            )

            logger.info(f"    Generated diagnostic plots in {dgp_plot_dir}")

        except Exception as e:
            logger.warning(f"    Failed to generate diagnostic plots: {e}")

    return metrics, fit_time, model, y_samples


# =============================================================================
# Main Benchmark Loop
# =============================================================================


def run_benchmark(dgps: list[dict], models: list[str], output_dir: str = "benchmarks/results/synthetic_dgp"):
    """Run synthetic DGP benchmark.

    Args:
        dgps: List of DGP specifications (name + kwargs)
        models: List of model names to evaluate
        output_dir: Directory for saving results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detect environment: core models only vs full comparison
    core_models = {"BDFNormal", "BDFKDE"}
    has_extended_models = any(m not in core_models for m in models)
    env_suffix = "" if has_extended_models else "_core"

    env_type = "full comparison" if has_extended_models else "core models only"
    logger.info(f"Running benchmark with {env_type}: {', '.join(models)}")
    if not has_extended_models:
        logger.info("Tip: Run in bench-models environment for comparison with NGBoost, LightGBM, etc.")

    all_results = {}

    for dgp_spec in dgps:
        dgp_name = dgp_spec["name"]
        dgp_kwargs = dgp_spec.get("kwargs", {})

        logger.info(f"\n{'='*80}")
        logger.info(f"DGP: {dgp_name}")
        logger.info(f"{'='*80}")

        # Generate dataset
        dataset = DGP_REGISTRY[dgp_name](**dgp_kwargs, seed=SEED)
        X, y = dataset.X, dataset.y

        logger.info(f"Generated {len(y)} samples, {X.shape[1]} features")

        dgp_results = {
            "dgp_name": dgp_name,
            "dgp_type": dataset.dgp_type.value,
            "dgp_metadata": dataset.metadata,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "seed": SEED,
            "models": {},
        }

        for model_name in models:
            logger.info(f"\n{'-'*80}")
            logger.info(f"Model: {model_name}")
            logger.info(f"{'-'*80}")

            model_results = {
                "fold_metrics": [],
                "fold_fit_times": [],
                "best_params": None,
                "tuning_time": None,
            }

            # Cross-validation: fold 0 for tuning, folds 1-9 for evaluation
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            folds = list(enumerate(kf.split(X)))

            # Tune on fold 0: use train split for training, test split for validation
            train_idx_0, test_idx_0 = folds[0][1]
            X_train_0, X_val_0 = X[train_idx_0], X[test_idx_0]
            y_train_0, y_val_0 = y[train_idx_0], y[test_idx_0]

            study_name = f"synthetic_{dgp_name}_{model_name}"
            try:
                logger.info("Tuning hyperparameters on fold 0...")
                best_init_kwargs, best_params, tuning_time = tune_model(
                    X_train_0, y_train_0, X_val_0, y_val_0, study_name, model_name, n_trials=N_TRIALS
                )
                model_results["best_params"] = {"init_kwargs": best_init_kwargs, "params": best_params}
                model_results["tuning_time"] = tuning_time
                logger.info(f"Tuning completed in {tuning_time:.1f}s")
            except Exception as e:
                logger.error(f"Tuning failed for {model_name}: {e}")
                continue

            # Evaluate on folds 1-9
            logger.info("Evaluating on folds 1-9...")
            for fold_idx, (train_idx, test_idx) in tqdm(folds[1:], desc="Eval folds"):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                try:
                    metrics, fit_time, _, _ = evaluate_fold(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        dataset,
                        best_init_kwargs,
                        best_params,
                        model_name,
                        fold_idx=fold_idx,
                        dgp_name=dgp_name,
                    )
                    model_results["fold_metrics"].append(metrics)
                    model_results["fold_fit_times"].append(fit_time)
                except Exception as e:
                    logger.error(f"Evaluation failed for {model_name} on fold {fold_idx}: {e}")
                    continue

            # Aggregate results across evaluation folds
            if model_results["fold_metrics"]:
                aggregated_metrics = {}
                metric_keys = model_results["fold_metrics"][0].keys()
                for key in metric_keys:
                    raw = [m[key] for m in model_results["fold_metrics"] if key in m]
                    if not raw:
                        continue

                    # List-valued metrics (e.g. pit_bin_proportions): average element-wise
                    if isinstance(raw[0], list):
                        arr = np.array(raw)
                        aggregated_metrics[key] = {
                            "mean": np.mean(arr, axis=0).tolist(),
                            "std": np.std(arr, axis=0).tolist(),
                        }
                        continue

                    values = [v for v in raw if not np.isnan(v)]
                    if values:
                        aggregated_metrics[key] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "values": [float(v) for v in values],
                        }

                model_results["aggregated_metrics"] = aggregated_metrics
                model_results["mean_fit_time"] = float(np.mean(model_results["fold_fit_times"]))

                # Log key metrics
                logger.info(
                    f"  RMSE: {aggregated_metrics['rmse']['mean']:.4f} ± {aggregated_metrics['rmse']['std']:.4f}"
                )
                if "gt_mean_rmse" in aggregated_metrics:
                    logger.info(
                        f"  GT Mean RMSE: {aggregated_metrics['gt_mean_rmse']['mean']:.4f} ± {aggregated_metrics['gt_mean_rmse']['std']:.4f}"
                    )
                if "crps" in aggregated_metrics:
                    logger.info(
                        f"  CRPS: {aggregated_metrics['crps']['mean']:.4f} ± {aggregated_metrics['crps']['std']:.4f}"
                    )
                if "gt_variance_rmse" in aggregated_metrics:
                    logger.info(
                        f"  GT Variance RMSE: {aggregated_metrics['gt_variance_rmse']['mean']:.4f} ± {aggregated_metrics['gt_variance_rmse']['std']:.4f}"
                    )

            dgp_results["models"][model_name] = model_results

        all_results[dgp_name] = dgp_results

        # Save results incrementally with environment suffix
        output_file = Path(output_dir) / f"{dgp_name}_results{env_suffix}.json"
        with open(output_file, "w") as f:
            json.dump(dgp_results, f, indent=2)
        logger.info(f"\nSaved results to {output_file}")

    # Save combined results with environment suffix
    combined_output = Path(output_dir) / f"all_results{env_suffix}.json"
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved combined results to {combined_output}")

    return all_results


if __name__ == "__main__":
    # Detect available models from environment
    available_models = list(MODEL_CONFIGS.keys())

    logger.info("\n" + "=" * 80)
    logger.info("Synthetic DGP Benchmark")
    logger.info("=" * 80)
    logger.info(f"Available models: {', '.join(available_models)}")
    logger.info(f"DGPs to evaluate: {len(DGPS_TO_RUN)}")

    results = run_benchmark(
        dgps=DGPS_TO_RUN,
        models=available_models,
        output_dir="benchmarks/results/synthetic_dgp",
    )

    logger.info("\n" + "=" * 80)
    logger.info("Benchmark complete!")
    logger.info("=" * 80)
