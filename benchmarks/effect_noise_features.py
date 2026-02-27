"""
Experiment: Effect of Noise Features on Probabilistic Prediction Quality

Investigates how adding pure noise features degrades the CRPS of BDF and baseline
probabilistic models. Uses sklearn synthetic datasets (friedman1–3, make_regression)
with fixed sample size and progressively more noise features: 0, 10, 50, 100, 500.

Also tracks feature selection behavior for BDF variants — proportion of times each
noise feature is selected for splits.

Models: BDFNormal, BDFKDE (always), ConformalRF, NGBoost, BART (bench-models env).
"""

import json
import os
from time import time
from typing import Any, Callable

import numpy as np
import optuna
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3, make_regression
from sklearn.model_selection import KFold
from tqdm import tqdm

from .metrics.regression import crps_wrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_SPLITS = 10  # 10-fold CV: fold 0 for tuning, folds 1-9 for evaluation
N_TRIALS = 50
N_FEATURES = 5  # Base number of informative features
SAMPLE_SIZE = 2000
N_POSTERIOR_SAMPLES = 1000

NOISE_FEATURE_COUNTS = [0, 10, 50, 100, 500]

# Dataset generators (return base features only)
DATASET_GENERATORS: dict[str, Callable] = {
    "friedman1": lambda n, seed: make_friedman1(n_samples=n, n_features=N_FEATURES, noise=1.0, random_state=seed),
    "friedman2": lambda n, seed: make_friedman2(n_samples=n, noise=1.0, random_state=seed),
    "friedman3": lambda n, seed: make_friedman3(n_samples=n, noise=0.1, random_state=seed),
    "make_regression": lambda n, seed: make_regression(
        n_samples=n, n_features=N_FEATURES, n_informative=5, noise=10.0, random_state=seed
    ),
}

# Number of truly informative features per dataset
INFORMATIVE_FEATURES = {
    "friedman1": 5,  # Uses features 0-4
    "friedman2": 4,  # Uses features 0-3
    "friedman3": 4,  # Uses features 0-3
    "make_regression": 5,  # n_informative=5
}


# =============================================================================
# Dynamic Model Loading
# =============================================================================


def get_available_models() -> dict[str, dict[str, Any]]:
    """Detect which models are available in current environment.

    Returns:
        Dictionary mapping model names to their configurations.
    """
    available = {}

    # =========================================================================
    # BDF Models — always available in default environment
    # =========================================================================
    try:
        from bdf.tree_classes.bdf_regressor import BDFRegressor

        available["BDFNormal"] = {
            "class": BDFRegressor,
            "fixed_init_kwargs": {
                "dist": "NormalMuNormal",
                "random_state": SEED,
            },
            "tunable_init_kwargs": {
                "n_trees": {"type": "int", "low": 15, "high": 200},
                "max_depth": {"type": "int", "low": 2, "high": 30},
                "alpha": {"type": "float", "low": 0.00001, "high": 0.1, "log": True},
                "gamma": {"type": "float", "low": 0.1, "high": 1.0},
                "delta": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
                "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample": {"type": "float", "low": 0.5, "high": 1.0},
                "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            },
            "tunable_params": {
                "sigma_mu_auto_scale": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
                "score_method": {"type": "categorical", "categories": ["nll", "nle"]},
            },
            "fixed_params": {
                "mu_mu": "auto",
                "sigma_mu": "auto",
                "score_correction": "bic",
            },
            "has_params_dict": True,
            "probabilistic": True,
            "has_feature_selection": True,
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
                "score_correction": {"type": "categorical", "categories": ["loo_cv", "bic"]},
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
            "has_feature_selection": True,
        }

        logger.info("✓ BDF models available (BDFNormal, BDFKDE)")
    except (ImportError, ModuleNotFoundError):
        logger.info("✗ BDF not available")

    # =========================================================================
    # Optional Models — bench-models environment
    # =========================================================================

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
            "has_feature_selection": True,
        }
        logger.info("✓ ConformalRF available")
    except ImportError:
        logger.info("✗ ConformalRF not available (run in bench-models environment)")

    try:
        from .models.wrappers import NGBRegressorWrapper

        available["NGBoost"] = {
            "class": NGBRegressorWrapper,
            "fixed_init_kwargs": {
                "random_state": SEED,
                "verbose": False,
                "dist_name": "Normal",
            },
            "tunable_init_kwargs": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                "minibatch_frac": {"type": "float", "low": 0.5, "high": 1.0},
                "col_sample": {"type": "float", "low": 0.5, "high": 1.0},
            },
            "tunable_params": {},
            "fixed_params": {},
            "has_params_dict": False,
            "probabilistic": True,
            "has_feature_selection": True,
        }
        logger.info("✓ NGBoost available")
    except ImportError:
        logger.info("✗ NGBoost not available (run in bench-models environment)")

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
            "has_feature_selection": True,
        }
        logger.info("✓ BART available")
    except ImportError:
        logger.info("✗ BART not available (run in bench-models environment)")

    return available


MODEL_CONFIGS = get_available_models()


# =============================================================================
# Feature Selection Analysis
# =============================================================================


def count_bdf_feature_selections(model, n_features: int) -> np.ndarray:
    """Count how many times each feature was selected for splits across all BDF trees."""
    counts = np.zeros(n_features, dtype=int)

    def traverse_node(node):
        if node is None or node._is_leaf():
            return
        if hasattr(node, "best_feature") and node.best_feature is not None:
            counts[node.best_feature] += 1
        traverse_node(node.left_node)
        traverse_node(node.right_node)

    for tree in model.trees:
        traverse_node(tree.root)

    return counts


def count_sklearn_tree_feature_selections(estimators, n_features: int) -> np.ndarray:
    """Count feature splits across a list of sklearn DecisionTree estimators."""
    counts = np.zeros(n_features, dtype=int)
    for tree in estimators:
        for feat_idx in tree.tree_.feature:
            if feat_idx >= 0:  # -2 indicates leaf node
                counts[feat_idx] += 1
    return counts


def count_confrf_feature_selections(model, n_features: int) -> np.ndarray:
    """Count feature splits from ConformalizedRFWrapper's internal RandomForest."""
    return count_sklearn_tree_feature_selections(model.estimator_.estimators_, n_features)


def count_ngboost_feature_selections(model, n_features: int) -> np.ndarray:
    """Count feature splits across all NGBoost base learners (multiple per boosting iteration)."""
    counts = np.zeros(n_features, dtype=int)
    for iteration_trees in model.base_models:
        for tree in iteration_trees:
            if hasattr(tree, "tree_"):
                for feat_idx in tree.tree_.feature:
                    if feat_idx >= 0:
                        counts[feat_idx] += 1
    return counts


def count_bart_feature_selections(model, n_features: int) -> np.ndarray:
    """Count feature splits across all BART MCMC posterior samples and trees."""
    counts = np.zeros(n_features, dtype=int)
    if not hasattr(model.model_, "_model_samples"):
        return counts
    for model_sample in model.model_._model_samples:
        for tree in model_sample.trees:
            for decision_node in tree.decision_nodes:
                if hasattr(decision_node, "split") and hasattr(decision_node.split, "splitting_variable"):
                    feat_idx = decision_node.split.splitting_variable
                    if feat_idx is not None and 0 <= feat_idx < n_features:
                        counts[feat_idx] += 1
    return counts


def count_feature_selections(model, model_name: str, n_features: int) -> np.ndarray:
    """Dispatch feature counting to the right implementation."""
    config = MODEL_CONFIGS[model_name]
    if not config.get("has_feature_selection", False):
        return np.zeros(n_features, dtype=int)

    try:
        if model_name in ("BDFNormal", "BDFKDE"):
            return count_bdf_feature_selections(model, n_features)
        elif model_name == "ConformalRF":
            return count_confrf_feature_selections(model, n_features)
        elif model_name == "NGBoost":
            return count_ngboost_feature_selections(model, n_features)
        elif model_name == "BART":
            return count_bart_feature_selections(model, n_features)
    except Exception as e:
        logger.warning(f"Feature selection counting failed for {model_name}: {e}")

    return np.zeros(n_features, dtype=int)


def compute_feature_selection_stats(
    feature_counts: np.ndarray,
    n_informative: int,
    n_base_features: int,
) -> dict[str, Any]:
    """Compute feature selection statistics from raw counts."""
    total_selections = feature_counts.sum()
    if total_selections == 0:
        return {
            "informative_selection_rate": 0.0,
            "base_selection_rate": 0.0,
            "noise_selection_rate": 0.0,
            "total_selections": 0,
            "informative_selections": 0,
            "base_selections": 0,
            "noise_selections": 0,
        }

    n_noise = len(feature_counts) - n_base_features

    informative_selections = feature_counts[:n_informative].sum()
    base_selections = feature_counts[:n_base_features].sum()
    noise_selections = feature_counts[n_base_features:].sum() if n_noise > 0 else 0

    return {
        "informative_selection_rate": float(informative_selections / total_selections),
        "base_selection_rate": float(base_selections / total_selections),
        "noise_selection_rate": float(noise_selections / total_selections) if n_noise > 0 else 0.0,
        "total_selections": int(total_selections),
        "informative_selections": int(informative_selections),
        "base_selections": int(base_selections),
        "noise_selections": int(noise_selections),
        "per_feature_counts": feature_counts.tolist(),
    }


# =============================================================================
# Core Functions
# =============================================================================


def suggest_hyperparameters(trial: optuna.Trial, model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Suggest hyperparameters from optuna trial for a specific model."""
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

    combined_params = dict(config.get("fixed_params", {}))
    if tuned_params:
        combined_params.update(tuned_params)

    return tuned_init_kwargs, combined_params


def generate_dataset_with_noise(
    generator_name: str, n_samples: int, n_noise_features: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate synthetic dataset with specified number of noise features appended."""
    generator = DATASET_GENERATORS[generator_name]
    X_base, y = generator(n_samples, seed)
    X_base = np.asarray(X_base)
    y = np.asarray(y)

    n_base_features = X_base.shape[1]

    if n_noise_features > 0:
        rng = np.random.default_rng(seed)
        X_noise = rng.standard_normal((n_samples, n_noise_features))
        X = np.hstack([X_base, X_noise])
    else:
        X = X_base

    return X, y, n_base_features


def _fit_model(model_cls, fixed_init_kwargs, init_kwargs, params, has_params_dict):
    """Instantiate and return an unfitted model."""
    if has_params_dict:
        return model_cls(**fixed_init_kwargs, **init_kwargs, params=params)
    else:
        return model_cls(**fixed_init_kwargs, **init_kwargs)


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    study_name: str,
    model_name: str,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Tune model hyperparameters using Optuna, minimizing CRPS on validation set."""
    config = MODEL_CONFIGS[model_name]
    model_cls = config["class"]
    fixed_init_kwargs = config["fixed_init_kwargs"]
    has_params_dict = config.get("has_params_dict", False)

    def objective(trial: optuna.Trial) -> float:
        np.random.seed(SEED + trial.number)
        init_kwargs, params = suggest_hyperparameters(trial, model_name)

        try:
            model = _fit_model(model_cls, fixed_init_kwargs, init_kwargs, params, has_params_dict)
            model.fit(X_train, y_train)
            y_samples = model.predict_samples(X_val, n_samples=N_POSTERIOR_SAMPLES)
            return float(crps_wrapper(y_val, y_samples))
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")

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

    start_time = time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    tuning_time = time() - start_time

    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best CRPS: {study.best_value:.4f}")

    best_init_kwargs, best_params = split_best_params(study.best_params, model_name)
    return best_init_kwargs, best_params, tuning_time


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    init_kwargs: dict[str, Any],
    params: dict[str, Any],
    model_name: str,
    n_informative: int,
    n_base_features: int,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Train and evaluate model on a single fold.

    Returns:
        tuple of (metrics_dict, fit_time, feature_selection_stats)
    """
    config = MODEL_CONFIGS[model_name]
    model_cls = config["class"]
    fixed_init_kwargs = config["fixed_init_kwargs"]
    has_params_dict = config.get("has_params_dict", False)

    np.random.seed(SEED)

    model = _fit_model(model_cls, fixed_init_kwargs, init_kwargs, params, has_params_dict)

    start_time = time()
    model.fit(X_train, y_train)
    fit_time = time() - start_time

    # Point predictions
    y_pred = model.predict(X_test)

    metrics: dict[str, float] = {
        "mse": float(np.mean((y_test - y_pred) ** 2)),
        "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
    }

    # Probabilistic metrics
    y_samples = model.predict_samples(X_test, n_samples=N_POSTERIOR_SAMPLES)
    metrics["crps"] = float(crps_wrapper(y_test, y_samples))

    # Coverage and interval widths
    for width in [0.5, 0.9, 0.95]:
        alpha = 1 - width
        y_lower = np.quantile(y_samples, alpha / 2, axis=-1)
        y_upper = np.quantile(y_samples, 1 - alpha / 2, axis=-1)
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
        ci_width = np.mean(y_upper - y_lower)
        metrics[f"coverage_{int(width * 100)}"] = float(coverage)
        metrics[f"ci_width_{int(width * 100)}"] = float(ci_width)

    # Feature selection statistics
    n_features = X_train.shape[1]
    feature_counts = count_feature_selections(model, model_name, n_features)
    feature_stats = compute_feature_selection_stats(feature_counts, n_informative, n_base_features)

    return metrics, fit_time, feature_stats


def run_experiment_for_dataset_and_noise(
    dataset_name: str,
    n_noise_features: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Run full experiment for a single dataset and noise feature count combination.

    Uses 10-fold CV: tune on fold 0, evaluate on folds 1-9.
    """
    logger.info(f"Processing {dataset_name} with n_noise_features={n_noise_features}")

    X, y, n_base_features = generate_dataset_with_noise(dataset_name, SAMPLE_SIZE, n_noise_features, SEED)
    n_informative = INFORMATIVE_FEATURES[dataset_name]

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Base features: {n_base_features}, Informative: {n_informative}, Noise: {n_noise_features}")

    key = f"{dataset_name}_noise{n_noise_features}"
    results["experiments"][key] = {
        "dataset": dataset_name,
        "n_samples": SAMPLE_SIZE,
        "n_base_features": n_base_features,
        "n_informative_features": n_informative,
        "n_noise_features": n_noise_features,
        "n_total_features": X.shape[1],
        "models": {},
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(enumerate(kf.split(X)))

    for model_name in MODEL_CONFIGS.keys():
        logger.info(f"  Model: {model_name}")

        model_results: dict[str, Any] = {
            "fold_metrics": [],
            "fold_fit_times": [],
            "fold_feature_stats": [],
            "best_params": None,
            "tuning_time": None,
        }

        # Tune on fold 0: train split for training, test split for validation
        train_idx_0, val_idx_0 = folds[0][1]
        X_train_0, X_val_0 = X[train_idx_0], X[val_idx_0]
        y_train_0, y_val_0 = y[train_idx_0], y[val_idx_0]

        study_name = f"noise-{dataset_name}-noise{n_noise_features}-{model_name}"
        try:
            logger.info("    Tuning on fold 0...")
            best_init_kwargs, best_params, tuning_time = tune_model(
                X_train_0, y_train_0, X_val_0, y_val_0, study_name, model_name
            )
            model_results["best_params"] = {"init_kwargs": best_init_kwargs, "params": best_params}
            model_results["tuning_time"] = tuning_time
            logger.info(f"    Tuning completed in {tuning_time:.1f}s")
        except Exception as e:
            logger.error(f"    Tuning failed for {model_name}: {e}")
            continue

        # Evaluate on folds 1-9
        logger.info("    Evaluating on folds 1-9...")
        for fold_idx, (train_idx, test_idx) in tqdm(folds[1:], desc=f"    {model_name} eval"):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                metrics, fit_time, feature_stats = evaluate_fold(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    best_init_kwargs,
                    best_params,
                    model_name,
                    n_informative,
                    n_base_features,
                )
                model_results["fold_metrics"].append(metrics)
                model_results["fold_fit_times"].append(fit_time)
                model_results["fold_feature_stats"].append(feature_stats)
            except Exception as e:
                logger.error(f"    Eval failed on fold {fold_idx}: {e}")
                continue

        if not model_results["fold_metrics"]:
            logger.warning(f"    No successful folds for {model_name}, skipping")
            continue

        # Aggregate metrics across evaluation folds
        aggregated_metrics: dict[str, Any] = {}
        for metric_name in model_results["fold_metrics"][0].keys():
            values = [fold[metric_name] for fold in model_results["fold_metrics"] if metric_name in fold]
            if values:
                aggregated_metrics[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        # Aggregate feature selection stats across folds
        aggregated_feature_stats: dict[str, Any] = {}
        for stat_name in ["informative_selection_rate", "base_selection_rate", "noise_selection_rate"]:
            values = [fold[stat_name] for fold in model_results["fold_feature_stats"]]
            aggregated_feature_stats[stat_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        total_selections = sum(fold["total_selections"] for fold in model_results["fold_feature_stats"])
        total_noise_selections = sum(fold["noise_selections"] for fold in model_results["fold_feature_stats"])
        aggregated_feature_stats["total_selections"] = total_selections
        aggregated_feature_stats["total_noise_selections"] = total_noise_selections

        model_results["aggregated_metrics"] = aggregated_metrics
        model_results["aggregated_feature_stats"] = aggregated_feature_stats
        model_results["mean_fit_time"] = float(np.mean(model_results["fold_fit_times"]))

        results["experiments"][key]["models"][model_name] = model_results

        logger.success(
            f"    {model_name}: CRPS={aggregated_metrics['crps']['mean']:.4f}"
            f"±{aggregated_metrics['crps']['std']:.4f}, "
            f"Noise sel. rate={aggregated_feature_stats['noise_selection_rate']['mean']:.3f}"
            f"±{aggregated_feature_stats['noise_selection_rate']['std']:.3f}"
        )

    logger.success(f"Finished {key}")
    return results


def save_results(
    results: dict[str, Any], filepath: str = "benchmarks/results/effect_noise_features/effect_noise_features.json"
):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filepath}")


def main():
    """Main entry point."""
    logger.info("Starting Effect of Noise Features Experiment")
    logger.info(f"Datasets: {list(DATASET_GENERATORS.keys())}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")
    logger.info(f"Noise feature counts: {NOISE_FEATURE_COUNTS}")
    logger.info(f"Models: {list(MODEL_CONFIGS.keys())}")
    logger.info(f"CV: {N_SPLITS}-fold (tune on fold 0, eval on folds 1-{N_SPLITS - 1})")
    logger.info(f"Posterior samples: {N_POSTERIOR_SAMPLES}")

    results: dict[str, Any] = {
        "config": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "n_trials": N_TRIALS,
            "n_base_features": N_FEATURES,
            "sample_size": SAMPLE_SIZE,
            "n_posterior_samples": N_POSTERIOR_SAMPLES,
            "noise_feature_counts": NOISE_FEATURE_COUNTS,
            "datasets": list(DATASET_GENERATORS.keys()),
            "informative_features": INFORMATIVE_FEATURES,
            "models": list(MODEL_CONFIGS.keys()),
        },
        "experiments": {},
    }

    total_experiments = len(DATASET_GENERATORS) * len(NOISE_FEATURE_COUNTS)
    pbar = tqdm(total=total_experiments, desc="Experiments")

    for dataset_name in DATASET_GENERATORS.keys():
        for n_noise in NOISE_FEATURE_COUNTS:
            try:
                results = run_experiment_for_dataset_and_noise(dataset_name, n_noise, results)
            except Exception as e:
                logger.error(f"Failed for {dataset_name} noise={n_noise}: {e}")
                import traceback

                traceback.print_exc()
                results["experiments"][f"{dataset_name}_noise{n_noise}"] = {"error": str(e)}

            save_results(results)
            pbar.update(1)

    pbar.close()
    save_results(results)
    logger.success("Experiment complete!")

    return results


if __name__ == "__main__":
    main()
