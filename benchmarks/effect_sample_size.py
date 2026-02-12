"""
Experiment: Effect of Sample Size on BDF Performance

Investigates the effect of sample size on the relative performance of BDF compared to baseline models.
Uses sklearn synthetic datasets (make_friedman1, make_friedman2, make_friedman3, make_regression)
with controlled sample sizes: 100, 200, 500, 1000, 2000, 5000.
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

# from benchmarks.models.wrappers import NGBRegressorWrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
SEED = 42
N_SPLITS = 10
TEST_SIZE = 0.25
VAL_SIZE = 0.25  # Of the remaining data after test split
N_TRIALS = 50
N_FEATURES = 10

SAMPLE_SIZES = [100, 200, 500, 1000, 2000, 5000]

# Dataset generators
DATASET_GENERATORS: dict[str, Callable] = {
    "friedman1": lambda n, seed: make_friedman1(n_samples=n, n_features=N_FEATURES, noise=1.0, random_state=seed),
    "friedman2": lambda n, seed: make_friedman2(n_samples=n, noise=1.0, random_state=seed),
    "friedman3": lambda n, seed: make_friedman3(n_samples=n, noise=0.1, random_state=seed),
    "make_regression": lambda n, seed: make_regression(
        n_samples=n, n_features=N_FEATURES, n_informative=5, noise=10.0, random_state=seed
    ),
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
        },
        "tunable_init_kwargs": {
            "n_trees": {"type": "int", "low": 15, "high": 200},
            "alpha": {"type": "float", "low": 0.00001, "high": 0.1, "log": True},
            "gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "delta": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
            # "max_depth": {"type": "int", "low": 3, "high": 50},
            # "min_samples_split": {"type": "int", "low": 10, "high": 100},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample": {"type": "float", "low": 0.5, "high": 1.0},
            "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        },
        "tunable_params": {
            "sigma_mu": {
                "type": "float",
                "low": 0.1,
                "high": 150.0,
            },  # TODO: implement `sigma_mu_mult` for `sigma_mu='auto'` s.t. it scales with data std
            "score_method": {"type": "categorical", "categories": ["nll", "nle"]},
            # "score_correction": {"type": "categorical", "categories": ["bic", None]},  # Could remove again, fix 'bic'
        },
        "fixed_params": {
            "mu_mu": "auto",
            "score_correction": "bic",
        },
        "has_params_dict": True,  # BDF uses a separate params dict
    },
    # TODO: merge KDE branch to main and include as comparison
    # "BDFKDE": {
    #     "class": BDFRegressor,
    #     "fixed_init_kwargs": {
    #         "dist": "KDE",
    #         "random_state": SEED,
    #     },
    #     "tunable_init_kwargs": {
    #         "n_trees": {"type": "int", "low": 15, "high": 100},
    #         "alpha": {"type": "float", "low": 0.00001, "high": 0.1, "log": True},
    #         "gamma": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
    #         "delta": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
    #         "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
    #         # "max_depth": {"type": "int", "low": 3, "high": 50},
    #         # "min_samples_split": {"type": "int", "low": 10, "high": 100},
    #         "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    #         "colsample": {"type": "float", "low": 0.5, "high": 1.0},
    #         "eta": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    #     },
    #     "tunable_params": {
    #         "bandwidth": {
    #             "type": "float",
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
    # "NGBoost": {
    #     "class": NGBRegressorWrapper,
    #     "fixed_init_kwargs": {
    #         "random_state": SEED,
    #         "verbose": False,
    #         "dist_name": "Normal",  # Only Normal distribution allowed
    #     },
    #     "tunable_init_kwargs": {
    #         "n_estimators": {"type": "int", "low": 25, "high": 250},
    #         "learning_rate": {"type": "float", "low": 0.01, "high": 0.5},
    #         "minibatch_frac": {"type": "float", "low": 0.1, "high": 1.0},
    #         "col_sample": {"type": "float", "low": 0.5, "high": 1.0},
    #     },
    #     "tunable_params": {},
    #     "fixed_params": {},
    #     "has_params_dict": False,
    # },
}


def suggest_hyperparameters(trial: optuna.Trial, model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Suggest hyperparameters from optuna trial for a specific model.

    Returns:
        tuple of (init_kwargs, params_dict)
    """
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

    # Build params dict: start with fixed_params, then add tunable_params
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

    # Combine fixed_params with tuned_params
    combined_params = dict(config.get("fixed_params", {}))
    if tuned_params:
        combined_params.update(tuned_params)

    return tuned_init_kwargs, combined_params


def generate_dataset(generator_name: str, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset with specified sample size."""
    generator = DATASET_GENERATORS[generator_name]
    X, y = generator(n_samples, seed)
    return np.asarray(X), np.asarray(y)


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    study_name: str,
    model_name: str,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Tune model using optuna on train/val split.

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
            return mean_squared_error(y_val, y_pred)
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")

    # Create optuna storage directory
    os.makedirs("benchmarks/results/optuna/", exist_ok=True)
    storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"

    # Clean up existing study
    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except KeyError:
        logger.debug(f"No existing study named '{study_name}' found in storage '{storage_name}'; nothing to delete.")
    except Exception as e:
        logger.warning(f"Could not delete existing study: {e}")

    # Create study with seeded sampler
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
) -> tuple[dict[str, float], float]:
    """Train and evaluate model on a single fold.

    Returns:
        tuple of (metrics_dict, fit_time)
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

    return metrics, fit_time


def run_experiment_for_dataset_and_size(
    dataset_name: str,
    n_samples: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Run full experiment for a single dataset and sample size combination."""
    logger.info(f"🔬 Processing {dataset_name} with n_samples={n_samples}")

    # Generate dataset
    X, y = generate_dataset(dataset_name, n_samples, SEED)
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Split: 20% test, then 20% val from remaining 80%
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    X_train_tune, X_val, y_train_tune, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=SEED
    )

    logger.info(f"Split sizes: train={len(X_train_tune)}, val={len(X_val)}, test={len(X_test)}")

    # Initialize result structure
    key = f"{dataset_name}_n{n_samples}"
    results["experiments"][key] = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_features": X.shape[1],
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
        }

        for fold_idx, (trainval_idx, test_idx) in enumerate(cv_splitter.split(X)):
            logger.info(f"    Fold {fold_idx + 1}/{N_SPLITS}")

            # Split into trainval and test for this fold
            X_trainval = X[trainval_idx]
            y_trainval = y[trainval_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Further split trainval into train and val for tuning
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=VAL_SIZE, random_state=SEED + fold_idx
            )

            # Tune on train/val
            study_name = f"effect_sample_size-{dataset_name}-n{n_samples}-{model_name}-fold{fold_idx}"
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
            metrics, fit_time = evaluate_fold(
                X_trainval, y_trainval, X_test, y_test, best_init_kwargs, best_params, model_name
            )

            model_results["fold_metrics"].append(metrics)
            model_results["fold_fit_times"].append(fit_time)

            logger.info(f"      MSE: {metrics['mse']:.4f}")

        # Aggregate metrics across folds
        aggregated = {}
        for metric_name in model_results["fold_metrics"][0].keys():
            values = [fold[metric_name] for fold in model_results["fold_metrics"]]
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        model_results["aggregated_metrics"] = aggregated
        model_results["mean_fit_time"] = float(np.mean(model_results["fold_fit_times"]))
        model_results["mean_tuning_time"] = float(np.mean(model_results["fold_tuning_times"]))

        results["experiments"][key]["models"][model_name] = model_results

        logger.success(f"    ✅ {model_name}: MSE={aggregated['mse']['mean']:.4f}±{aggregated['mse']['std']:.4f}")

    logger.success(f"✅ Finished {key}")

    return results


def save_results(
    results: dict[str, Any], filepath: str = "benchmarks/results/effect_sample_size/effect_sample_size.json"
):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to {filepath}")


def main():
    """Main entry point for the experiment."""
    logger.info("🚀 Starting Effect of Sample Size Experiment")
    logger.info(f"Datasets: {list(DATASET_GENERATORS.keys())}")
    logger.info(f"Sample sizes: {SAMPLE_SIZES}")
    logger.info(f"Models: {list(MODEL_CONFIGS.keys())}")
    logger.info(f"K-fold splits: {N_SPLITS}, Val size (of trainval): {VAL_SIZE}")

    # Initialize results
    results: dict[str, Any] = {
        "config": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "val_size": VAL_SIZE,
            "n_trials": N_TRIALS,
            "n_features": N_FEATURES,
            "sample_sizes": SAMPLE_SIZES,
            "datasets": list(DATASET_GENERATORS.keys()),
            "models": list(MODEL_CONFIGS.keys()),
        },
        "experiments": {},
    }

    # Run experiments
    total_experiments = len(DATASET_GENERATORS) * len(SAMPLE_SIZES)
    pbar = tqdm(total=total_experiments, desc="Experiments")

    for dataset_name in DATASET_GENERATORS.keys():
        for n_samples in SAMPLE_SIZES:
            try:
                results = run_experiment_for_dataset_and_size(dataset_name, n_samples, results)
            except Exception as e:
                logger.error(f"❌ Failed for {dataset_name} n={n_samples}: {e}")
                import traceback

                traceback.print_exc()
                results["experiments"][f"{dataset_name}_n{n_samples}"] = {"error": str(e)}

            # Save intermediate results
            save_results(results)
            pbar.update(1)

    pbar.close()

    # Final save
    save_results(results)
    logger.success("🎉 Experiment complete!")

    return results


if __name__ == "__main__":
    # TODO: Add second experiment for probabilistic models, tuning via CRPS, metrics incl. WIS, IS, coverage, width, pit ks
    main()
