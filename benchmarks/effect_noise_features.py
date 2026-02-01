"""
Experiment: Effect of Noise Features on BDF Performance

Investigates the effect of adding pure noise features on the relative performance of BDF
compared to baseline models. Uses sklearn synthetic datasets (make_friedman1, make_friedman2,
make_friedman3, make_regression) with fixed sample size and progressively more noise features
added: 0, 10, 50, 100, 500.

Also tracks feature selection behavior - proportion of times each noise feature is selected
for splits in BDF and RandomForest.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
SEED = 42
N_SPLITS = 5
VAL_SIZE = 0.25  # Of the remaining data after test split
N_TRIALS = 50
N_FEATURES = 5  # Base number of informative features
SAMPLE_SIZE = 2000  # Fixed sample size for this experiment

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
# Model Configurations
# =============================================================================

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "BDFNormal": {
        "class": BDFRegressor,
        "fixed_init_kwargs": {
            "dist": "NormalMuNormal",
            "random_state": SEED,
            # "gamma": 1.0,
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
            "sigma_mu": {"type": "float", "low": 0.1, "high": 150.0},
            "score_method": {"type": "categorical", "categories": ["nll", "nle"]},
        },
        "fixed_params": {
            "mu_mu": "auto",
            "score_correction": "bic",
        },
        "has_params_dict": True,
    },
    "RandomForest": {
        "class": RandomForestRegressor,
        "fixed_init_kwargs": {
            "random_state": SEED,
        },
        "tunable_init_kwargs": {
            "criterion": {"type": "categorical", "categories": ["squared_error", "absolute_error"]},
            "max_leaf_nodes": {"type": "int", "low": 5, "high": 250},
            "max_depth": {"type": "int", "low": 2, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            "min_samples_split": {"type": "int", "low": 10, "high": 100},
            "max_features": {"type": "float", "low": 0.1, "high": 1.0},
            "n_estimators": {"type": "int", "low": 10, "high": 200},
        },
        "tunable_params": {},
        "fixed_params": {},
        "has_params_dict": False,
    },
}


# =============================================================================
# Feature Selection Analysis
# =============================================================================


def count_bdf_feature_selections(model: BDFRegressor, n_features: int) -> np.ndarray:
    """Count how many times each feature was selected for splits across all trees in BDF.

    Args:
        model: Fitted BDFRegressor
        n_features: Total number of features

    Returns:
        Array of shape (n_features,) with selection counts per feature
    """
    counts = np.zeros(n_features, dtype=int)

    def traverse_node(node):
        """Recursively traverse node and count feature selections."""
        if node is None or node._is_leaf():
            return
        if hasattr(node, "best_feature") and node.best_feature is not None:
            counts[node.best_feature] += 1
        traverse_node(node.left_node)
        traverse_node(node.right_node)

    for tree in model.trees:
        traverse_node(tree.root)

    return counts


def count_rf_feature_selections(model: RandomForestRegressor, n_features: int) -> np.ndarray:
    """Count how many times each feature was selected for splits across all trees in RF.

    Uses the tree structure to count actual split usage (not importance weights).

    Args:
        model: Fitted RandomForestRegressor
        n_features: Total number of features

    Returns:
        Array of shape (n_features,) with selection counts per feature
    """
    counts = np.zeros(n_features, dtype=int)

    for tree in model.estimators_:
        tree_struct = tree.tree_
        # feature array contains feature index for each node (-2 for leaves)
        features = tree_struct.feature
        for feat_idx in features:
            if feat_idx >= 0:  # Not a leaf node
                counts[feat_idx] += 1

    return counts


def compute_feature_selection_stats(
    feature_counts: np.ndarray,
    n_informative: int,
    n_base_features: int,
) -> dict[str, float]:
    """Compute feature selection statistics.

    Args:
        feature_counts: Array of selection counts per feature
        n_informative: Number of truly informative features
        n_base_features: Number of original (non-noise) features

    Returns:
        Dictionary with selection statistics
    """
    total_selections = feature_counts.sum()
    if total_selections == 0:
        return {
            "informative_selection_rate": 0.0,
            "base_selection_rate": 0.0,
            "noise_selection_rate": 0.0,
            "total_selections": 0,
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
# Core Functions (adapted from effect_sample_size.py)
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
    if not tunable_params:
        return init_kwargs, params

    for name, args in tunable_params.items():
        if args["type"] == "int":
            params[name] = trial.suggest_int(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "float":
            params[name] = trial.suggest_float(name, args["low"], args["high"], log=args.get("log", False))
        elif args["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, args["categories"])

    return init_kwargs, params


def split_best_params(best_params: dict[str, Any], model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split optuna best params into init_kwargs and params dict for a specific model."""
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
    """Generate synthetic dataset with specified number of noise features added.

    Returns:
        X: Feature matrix with noise features appended
        y: Target values
        n_base_features: Number of original (non-noise) features
    """
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


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    study_name: str,
    model_name: str,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Tune model using optuna on train/val split."""
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
            return mean_squared_error(y_val, y_pred)
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")

    os.makedirs("benchmarks/results/optuna/", exist_ok=True)
    storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"

    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except KeyError:
        # Study does not exist yet; safe to ignore and proceed with creation.
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
    logger.info(f"Best MSE: {study.best_value:.4f}")

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
    """Train and evaluate model on a single fold, also extracting feature selection stats.

    Returns:
        tuple of (metrics_dict, fit_time, feature_selection_stats)
    """
    config = MODEL_CONFIGS[model_name]
    model_cls = config["class"]
    fixed_init_kwargs = config["fixed_init_kwargs"]
    has_params_dict = config.get("has_params_dict", False)

    np.random.seed(SEED)

    if has_params_dict and params:
        model = model_cls(**fixed_init_kwargs, **init_kwargs, params=params)
    else:
        model = model_cls(**fixed_init_kwargs, **init_kwargs)

    start_time = time()
    model.fit(X_train, y_train)
    fit_time = time() - start_time

    y_pred = model.predict(X_test)

    metrics = {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(np.mean(np.abs(y_test - y_pred))),
    }

    # Extract feature selection statistics
    n_features = X_train.shape[1]
    if model_name == "BDFNormal":
        feature_counts = count_bdf_feature_selections(model, n_features)
    elif model_name == "RandomForest":
        feature_counts = count_rf_feature_selections(model, n_features)
    else:
        feature_counts = np.zeros(n_features, dtype=int)

    feature_stats = compute_feature_selection_stats(feature_counts, n_informative, n_base_features)

    return metrics, fit_time, feature_stats


def run_experiment_for_dataset_and_noise(
    dataset_name: str,
    n_noise_features: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Run full experiment for a single dataset and noise feature count combination."""
    logger.info(f"🔬 Processing {dataset_name} with n_noise_features={n_noise_features}")

    # Generate dataset with noise features
    X, y, n_base_features = generate_dataset_with_noise(dataset_name, SAMPLE_SIZE, n_noise_features, SEED)
    n_informative = INFORMATIVE_FEATURES[dataset_name]

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Base features: {n_base_features}, Informative: {n_informative}, Noise: {n_noise_features}")

    # Initialize result structure
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

    # K-fold CV: each fold has its own train/val/test split
    cv_splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Run for each model
    for model_name in MODEL_CONFIGS.keys():
        logger.info(f"  📊 Model: {model_name}")

        model_results = {
            "fold_metrics": [],
            "fold_fit_times": [],
            "fold_tuning_times": [],
            "fold_best_params": [],
            "fold_feature_stats": [],
        }

        for fold_idx, (trainval_idx, test_idx) in enumerate(cv_splitter.split(X)):
            logger.info(f"    Fold {fold_idx + 1}/{N_SPLITS}")

            X_trainval = X[trainval_idx]
            y_trainval = y[trainval_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=VAL_SIZE, random_state=SEED + fold_idx
            )

            # Tune on train/val
            study_name = f"effect_noise-{dataset_name}-noise{n_noise_features}-{model_name}-fold{fold_idx}"
            best_init_kwargs, best_params, tuning_time = tune_model(
                X_train, y_train, X_val, y_val, study_name, model_name
            )

            model_results["fold_tuning_times"].append(tuning_time)
            model_results["fold_best_params"].append(
                {
                    "init_kwargs": best_init_kwargs,
                    "params": best_params,
                }
            )

            # Refit on full trainval set with best params and evaluate on test
            metrics, fit_time, feature_stats = evaluate_fold(
                X_trainval,
                y_trainval,
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

            logger.info(
                f"      MSE: {metrics['mse']:.4f}, " f"Noise sel. rate: {feature_stats['noise_selection_rate']:.3f}"
            )

        # Aggregate metrics across folds
        aggregated_metrics = {}
        for metric_name in model_results["fold_metrics"][0].keys():
            values = [fold[metric_name] for fold in model_results["fold_metrics"]]
            aggregated_metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        # Aggregate feature selection stats across folds
        aggregated_feature_stats = {}
        for stat_name in ["informative_selection_rate", "base_selection_rate", "noise_selection_rate"]:
            values = [fold[stat_name] for fold in model_results["fold_feature_stats"]]
            aggregated_feature_stats[stat_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        # Also aggregate total counts
        total_selections = sum(fold["total_selections"] for fold in model_results["fold_feature_stats"])
        total_noise_selections = sum(fold["noise_selections"] for fold in model_results["fold_feature_stats"])
        aggregated_feature_stats["total_selections"] = total_selections
        aggregated_feature_stats["total_noise_selections"] = total_noise_selections

        model_results["aggregated_metrics"] = aggregated_metrics
        model_results["aggregated_feature_stats"] = aggregated_feature_stats
        model_results["mean_fit_time"] = float(np.mean(model_results["fold_fit_times"]))
        model_results["mean_tuning_time"] = float(np.mean(model_results["fold_tuning_times"]))

        results["experiments"][key]["models"][model_name] = model_results

        logger.success(
            f"    ✅ {model_name}: MSE={aggregated_metrics['mse']['mean']:.4f}±{aggregated_metrics['mse']['std']:.4f}, "
            f"Noise sel. rate={aggregated_feature_stats['noise_selection_rate']['mean']:.3f}"
            f"±{aggregated_feature_stats['noise_selection_rate']['std']:.3f}"
        )

    logger.success(f"✅ Finished {key}")

    return results


def save_results(
    results: dict[str, Any], filepath: str = "benchmarks/results/effect_noise_features/effect_noise_features.json"
):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to {filepath}")


def main():
    """Main entry point for the experiment."""
    logger.info("🚀 Starting Effect of Noise Features Experiment")
    logger.info(f"Datasets: {list(DATASET_GENERATORS.keys())}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")
    logger.info(f"Noise feature counts: {NOISE_FEATURE_COUNTS}")
    logger.info(f"Models: {list(MODEL_CONFIGS.keys())}")
    logger.info(f"K-fold splits: {N_SPLITS}, Val size (of trainval): {VAL_SIZE}")

    # Initialize results
    results: dict[str, Any] = {
        "config": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "val_size": VAL_SIZE,
            "n_trials": N_TRIALS,
            "n_base_features": N_FEATURES,
            "sample_size": SAMPLE_SIZE,
            "noise_feature_counts": NOISE_FEATURE_COUNTS,
            "datasets": list(DATASET_GENERATORS.keys()),
            "informative_features": INFORMATIVE_FEATURES,
            "models": list(MODEL_CONFIGS.keys()),
        },
        "experiments": {},
    }

    # Run experiments
    total_experiments = len(DATASET_GENERATORS) * len(NOISE_FEATURE_COUNTS)
    pbar = tqdm(total=total_experiments, desc="Experiments")

    for dataset_name in DATASET_GENERATORS.keys():
        for n_noise in NOISE_FEATURE_COUNTS:
            try:
                results = run_experiment_for_dataset_and_noise(dataset_name, n_noise, results)
            except Exception as e:
                logger.error(f"❌ Failed for {dataset_name} noise={n_noise}: {e}")
                import traceback

                traceback.print_exc()
                results["experiments"][f"{dataset_name}_noise{n_noise}"] = {"error": str(e)}

            # Save intermediate results
            save_results(results)
            pbar.update(1)

    pbar.close()

    # Final save
    save_results(results)
    logger.success("🎉 Experiment complete!")

    return results


if __name__ == "__main__":
    main()
