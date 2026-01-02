import json
import os
from time import time

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import train_test_split as TTS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..metrics.classification import CLAS_POINT_METRICS, CLAS_PROB_METRICS
from ..metrics.regression import REG_POINT_METRICS, REG_PROB_METRICS
from ..pipeline.data import DatasetMetadata, available_classification_datasets, available_regression_datasets
from ..utils.benchmark_utils import LogTransformTransformer


class BaseOrchestrator:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.model_cfg = cfg.model  # type: ignore[attr-defined]
        self.target_type = self.model_cfg.target_type
        self.standardize_target = self.model_cfg.get("standardize_target", "no")
        self.log_transform_target = self.model_cfg.get("log_transform_target", False)

    def _is_compatible(self, dataset_metadata: DatasetMetadata) -> bool:
        """Check if model can handle this dataset's target domain."""
        compatible_domains = self.model_cfg["compatible_target_domains"]
        return dataset_metadata.target_domain.value in compatible_domains

    def _maybe_apply_target_standardization(self, y_train) -> tuple[np.ndarray | pd.Series, Pipeline | None]:
        """Apply target standardization based on config."""
        steps = []
        if self.target_type != "regression":
            return y_train, None
        else:
            if self.target_type in ["positive_real", "positive_integer"]:
                if self.log_transform_target:
                    steps.append(("log_transform", LogTransformTransformer()))

        if self.standardize_target == "no":
            if not steps:
                return y_train, None
            pipeline = Pipeline(steps)
            return pipeline.fit_transform(y_train.reshape(-1, 1)).ravel(), pipeline

        scaler = StandardScaler()
        steps.append(("scaler", scaler))
        pipeline = Pipeline(steps)
        y_train_scaled = pipeline.fit_transform(y_train.reshape(-1, 1)).ravel()

        return y_train_scaled, pipeline

    def _get_score_metric(self, metadata: DatasetMetadata):
        if self.target_type == "regression":
            return make_scorer(mean_squared_error, greater_is_better=False)
        else:
            # Discern binary from multi-class
            if metadata.target_domain.value == "binary":
                return make_scorer(log_loss, greater_is_better=False)
            else:
                return make_scorer(log_loss, greater_is_better=False)

    def calc_metrics(self, X, y, model, pipeline=None) -> dict:
        """Calculate metrics with optional inverse standardization."""
        do_reg = self.target_type == "regression"
        point_metrics = REG_POINT_METRICS if do_reg else CLAS_POINT_METRICS
        proba_metrics = REG_PROB_METRICS if do_reg else CLAS_PROB_METRICS

        metric_dict = {}
        try:
            y_pred = model.predict(X) if do_reg else model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            y_pred = model.predict(X)

        # Inverse transform if we standardized test targets
        if pipeline is not None and self.standardize_target in ["only", "both"]:
            y_pred = pipeline.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        # TODO: Change to tqdm over eval folds, NOT metrics, could pass pbar to set description
        pbar = tqdm(point_metrics.items(), desc="Point metrics", leave=False)
        for m, m_func in pbar:
            pbar.set_description(f"📊 {m}")
            try:
                metric_dict[m] = float(m_func(y, y_pred))
            except Exception as e:
                logger.error(f"❌ Point metric {m} failed: {e}")
                metric_dict[m] = float("nan")

        # Probabilistic metrics
        # TODO: Adapt for quantile predictions
        # TODO: Check whether seed needs to be reset here
        if self.model_cfg.probabilistic and proba_metrics:
            try:
                y_pred_samples = model.predict_samples(X, n_samples=self.cfg.sample_size)  # type: ignore[attr-defined]

                # Inverse transform samples if needed
                if pipeline is not None and self.standardize_target in ["only", "both"]:
                    y_pred_samples = pipeline.inverse_transform(y_pred_samples)

                pbar = tqdm(proba_metrics.items(), desc="Prob metrics", leave=False)
                for m, m_func in pbar:
                    pbar.set_description(f"🎲 {m}")
                    try:
                        metric_dict[m] = float(m_func(y, y_pred_samples))
                    except Exception as e:
                        logger.error(f"❌ Prob metric {m} failed: {e}")
                        metric_dict[m] = float("nan")
            except Exception as e:
                logger.error(f"❌ predict_samples failed: {e}")
                import traceback

                traceback.print_exc()

        return metric_dict


class CustomOrchestrator(BaseOrchestrator):

    def run(self):
        # Set global random seed for reproducibility
        np.random.seed(self.cfg.seed)  # type: ignore[attr-defined]

        if self.model_cfg.class_name.startswith("BDF"):
            from ..models.bdf_factory import ModelFactory
        else:
            from ..models.model_factory import ModelFactory
        model_cls = ModelFactory.get(self.model_cfg)
        logger.success(f"⚙ Loaded model class {model_cls.__name__}")
        results = {"model_config": OmegaConf.to_container(self.model_cfg, resolve=True), "datasets": {}}

        dataset_iterator = (
            available_regression_datasets if self.target_type == "regression" else available_classification_datasets
        )

        # Create optuna storage directory
        os.makedirs("benchmarks/results/optuna/", exist_ok=True)

        for metadata, X, y in dataset_iterator():
            if not self._is_compatible(metadata):
                logger.warning(f"⏭️ Skipping {metadata.name}: incompatible target domain {metadata.target_domain.value}")
                continue

            logger.info(f"🔬 Processing {metadata.name} with target domain {metadata.target_domain.value}")
            logger.info(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")
            results["datasets"][metadata.name] = {"metadata": metadata.to_dict(), "metrics": {}, "best_params": {}}

            # Create CV splitter with explicit seed for reproducible folds
            if self.target_type == "regression":
                cv_splitter = KFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=self.cfg.seed)  # type: ignore[arg-type]
            else:
                cv_splitter = StratifiedKFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=self.cfg.seed)  # type: ignore[arg-type]

            # Use fold 0 for tuning
            self.tuned_init_kwargs = None
            fit_times = []
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
                if fold_idx == 0:
                    results, self.tuned_init_kwargs = self._tune_on_fold(
                        X.iloc[train_idx],
                        y.iloc[train_idx],
                        X.iloc[test_idx],
                        y.iloc[test_idx],
                        model_cls,
                        results,
                        metadata,
                    )
                else:
                    # Evaluate on test set with seed reset
                    np.random.seed(self.cfg.seed)  # type: ignore[attr-defined]
                    best_model = model_cls(**self.model_cfg.fixed_init_kwargs, **self.tuned_init_kwargs)
                    start_time = time()
                    best_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                    end_time = time()
                    fit_times.append(end_time - start_time)

                    # Calculate metrics (handle inverse transform if needed)
                    metrics_dict = self.calc_metrics(X.iloc[test_idx], y.iloc[test_idx], best_model, None)
                    fold_metrics.append(metrics_dict)

            # Aggregate metrics across folds (mean & std)
            if fold_metrics:
                aggregated = {}
                for m in fold_metrics[0].keys():
                    values = [fold[m] for fold in fold_metrics if m in fold and not np.isnan(fold[m])]
                    aggregated[m] = {
                        "mean": float(np.mean(values)) if values else float("nan"),
                        "std": float(np.std(values)) if values else float("nan"),
                    }
                results["datasets"][metadata.name]["metrics"] = aggregated
                logger.info(f"✅ Finished {metadata.name} with aggregated metrics: {aggregated}")
            else:
                logger.warning(f"No fold metrics computed for task {metadata.name}")
            if fit_times:
                results["datasets"][metadata.name]["fitting_time"] = float(np.mean(fit_times))
            # Save intermediate results
            self._save_results(results)

        return results

    def _tune_on_fold(self, X_train, y_train, X_test, y_test, model_cls, results, metadata):
        """Tune model on fold 0 and evaluate on its test set."""

        # Apply target standard
        # Tune model
        def objective(trial):
            # Reset NumPy seed at each trial for reproducibility within CVS
            np.random.seed(self.cfg.seed + trial.number)  # type: ignore[attr-defined]

            # 1. Build init_kwargs from tunable init parameters
            iter_init_kwargs = {}
            for name, args in self.model_cfg.tunable_init_kwargs.items():
                if args["type"] == "int":
                    iter_init_kwargs[name] = trial.suggest_int(
                        name, args["low"], args["high"], log=args.get("log", False)
                    )
                elif args["type"] == "float":
                    iter_init_kwargs[name] = trial.suggest_float(
                        name, args["low"], args["high"], log=args.get("log", False)
                    )
                elif args["type"] == "categorical":
                    iter_init_kwargs[name] = trial.suggest_categorical(name, args["categories"])
                else:
                    raise ValueError(f"Parameter type '{args['type']}' unknown!")

            # 2. Build params dict: always include fixed_params (if present), then add tunable_params
            iter_params = {}

            # Always add fixed_params first (if model has them)
            if "fixed_params" in self.model_cfg:
                iter_params.update(self.model_cfg.fixed_params)

            # Add tunable_params suggestions (if model has them)
            if "tunable_params" in self.model_cfg:
                for name, args in self.model_cfg.tunable_params.items():
                    if args["type"] == "int":
                        iter_params[name] = trial.suggest_int(
                            name, args["low"], args["high"], log=args.get("log", False)
                        )
                    elif args["type"] == "float":
                        iter_params[name] = trial.suggest_float(
                            name, args["low"], args["high"], log=args.get("log", False)
                        )
                    elif args["type"] == "categorical":
                        iter_params[name] = trial.suggest_categorical(name, args["categories"])
                    else:
                        raise ValueError(f"Parameter type '{args['type']}' unknown!")

            # 3. Only pass params dict if it has content
            if iter_params:
                iter_init_kwargs["params"] = iter_params

            # Create model with fixed init kwargs + trial-suggested init kwargs (+ params if present)
            model = model_cls(**self.model_cfg.fixed_init_kwargs, **iter_init_kwargs)
            score_metric = mean_squared_error if self.target_type == "regression" else log_loss

            # Use explicit CV splitter for reproducible fold splits
            try:
                model.fit(X_train, y_train)
                y_pred = (
                    model.predict(X_test) if self.target_type == "regression" else model.predict_proba(X_test)[:, 1]
                )
            except Exception as e:
                logger.warning(f"⚠️ Warning: Fitting or prediction failed: {e}")
                return float("inf")
            return score_metric(y_test, y_pred)

        study_name = f"{self.model_cfg.name}-{metadata.name}"
        storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"

        # Clean up existing study - only if storage exists
        try:
            optuna.delete_study(study_name=study_name, storage=storage_name)
        except KeyError:
            pass  # Study doesn't exist yet, that's fine
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not delete existing study: {e}")

        # Create study with seeded sampler for reproducible trial suggestions
        sampler = TPESampler(seed=self.cfg.seed)  # type: ignore[arg-type]
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            sampler=sampler,
            load_if_exists=False,
        )
        start_time = time()
        study.optimize(objective, n_trials=self.cfg.n_trials)  # type: ignore[arg-type]
        end_time = time()
        logger.info(f"⏱ Tuning completed in {end_time - start_time:.2f} seconds")
        results["datasets"][metadata.name]["tuning_time_seconds"] = end_time - start_time

        optuna_best_params = study.best_params
        tuned_init_kwargs = {}
        tuned_params = {}

        # Split best params into init kwargs and distribution params
        for param, value in optuna_best_params.items():
            if param in self.model_cfg.tunable_init_kwargs.keys():
                tuned_init_kwargs[param] = value
            elif "tunable_params" in self.model_cfg and param in self.model_cfg.tunable_params.keys():
                tuned_params[param] = value

        # Reconstruct params dict: fixed_params + tuned_params
        # Only if the model has fixed_params or tunable_params sections
        if "fixed_params" in self.model_cfg or "tunable_params" in self.model_cfg:
            combined_params = {}

            # Always start with fixed_params (if present)
            if "fixed_params" in self.model_cfg:
                combined_params.update(self.model_cfg.fixed_params)

            # Update with tuned params
            if tuned_params:
                combined_params.update(tuned_params)

            # Only pass params if non-empty
            if combined_params:
                tuned_init_kwargs["params"] = combined_params

        logger.info(f"🏆 Best params for {metadata.name}: tuned_init_kwargs={tuned_init_kwargs}")
        results["datasets"][metadata.name]["best_params"] = study.best_params
        return results, tuned_init_kwargs

    def _save_results(self, results):
        """Save intermediate results to avoid losing progress."""
        os.makedirs("benchmarks/results/custom/", exist_ok=True)
        with open(f"benchmarks/results/custom/{self.model_cfg.name}.json", "w") as f:
            json.dump(results, f, indent=2)


class RegressionSuiteOrchestrator(BaseOrchestrator):
    """
    Run models over an OpenML suite (OpenML-CTR23 id 353).
    - Uses fold 0 for tuning (optuna) using internal CV on the fold-0 training set.
    - Evaluates the chosen hyperparameters on folds 1..9 (fit on each fold's train set,
      evaluate on the fold's test set), then reports mean/std across folds.
    """

    def run(self):
        import openml

        # Load suite 353 (OpenML-CTR23)
        suite = openml.study.get_suite(353)
        results = {"model_config": OmegaConf.to_container(self.model_cfg, resolve=True), "datasets": {}}

        if self.model_cfg.class_name.startswith("BDF"):
            from ..models.bdf_factory import ModelFactory
        else:
            from ..models.model_factory import ModelFactory

        model_cls = ModelFactory.get(self.model_cfg)
        logger.success(f"⚙ Loaded model class {model_cls.__name__}")

        # Iterate tasks in suite
        total_tasks = len(suite.tasks)
        logger.info(f"🚀 Starting regression suite run over {total_tasks} tasks")
        for i, task_id in enumerate(suite.tasks):
            try:
                task = openml.tasks.get_task(task_id)
            except Exception as e:
                logger.warning(f"Skipping task {task_id}: could not load task ({e})")
                continue

            try:
                dataset = task.get_dataset()
                X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            except Exception as e:
                logger.warning(f"Skipping task {task_id}: could not load dataset ({e})")
                continue

            dataset_name = getattr(dataset, "name", f"task_{task_id}")
            logger.info(f"🔬 Processing OpenML dataset {dataset_name} (task {task_id}) | {i+1}/{total_tasks}")
            logger.info(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")
            results["datasets"][dataset_name] = {
                "metadata": {
                    "task_id": task_id,
                    "name": dataset_name,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                },
                "metrics": {},
                "best_params": {},
            }

            # Use provided fold 0 for tuning
            try:
                train_idx_0, test_idx_0 = task.get_train_test_split_indices(fold=0)
            except Exception:
                # Some task versions return a tuple-of-lists; try the generic method
                splits = task.get_train_test_split_indices()
                if len(splits) > 0:
                    train_idx_0, test_idx_0 = splits[0]
                else:
                    logger.warning(f"No splits available for task {task_id}, skipping.")
                    continue

            X_fold0_train = X.iloc[train_idx_0]
            y_fold0_train = y.iloc[train_idx_0]
            X_fold0_test = X.iloc[test_idx_0]
            y_fold0_test = y.iloc[test_idx_0]

            # Tuning: copy of objective used elsewhere but restricted to X_fold0_train
            def objective(trial):
                np.random.seed(self.cfg.seed + trial.number)  # reproducibility

                iter_init_kwargs = {}
                for name, args in self.model_cfg.tunable_init_kwargs.items():
                    if args["type"] == "int":
                        iter_init_kwargs[name] = trial.suggest_int(
                            name, args["low"], args["high"], log=args.get("log", False)
                        )
                    elif args["type"] == "float":
                        iter_init_kwargs[name] = trial.suggest_float(
                            name, args["low"], args["high"], log=args.get("log", False)
                        )
                    elif args["type"] == "categorical":
                        iter_init_kwargs[name] = trial.suggest_categorical(name, args["categories"])
                    else:
                        raise ValueError(f"Parameter type '{args['type']}' unknown!")

                iter_params = {}
                if "fixed_params" in self.model_cfg:
                    iter_params.update(self.model_cfg.fixed_params)
                if "tunable_params" in self.model_cfg:
                    for name, args in self.model_cfg.tunable_params.items():
                        if args["type"] == "int":
                            iter_params[name] = trial.suggest_int(
                                name, args["low"], args["high"], log=args.get("log", False)
                            )
                        elif args["type"] == "float":
                            iter_params[name] = trial.suggest_float(
                                name, args["low"], args["high"], log=args.get("log", False)
                            )
                        elif args["type"] == "categorical":
                            iter_params[name] = trial.suggest_categorical(name, args["categories"])
                        else:
                            raise ValueError(f"Parameter type '{args['type']}' unknown!")

                if iter_params:
                    iter_init_kwargs["params"] = iter_params

                model = model_cls(**self.model_cfg.fixed_init_kwargs, **iter_init_kwargs)
                score_metric = mean_squared_error if self.target_type == "regression" else log_loss

                # Don't use CVS, just train/test on fold0 itself
                try:
                    model.fit(X_fold0_train, y_fold0_train)
                    y_pred = (
                        model.predict(X_fold0_test)
                        if self.target_type == "regression"
                        else model.predict_proba(X_fold0_test)[:, 1]
                    )
                    return score_metric(y_fold0_test, y_pred)
                except Exception as e:
                    logger.warning(f"Error during tuning (model fitting or prediction): {e}")
                    return float("nan")

            study_name = f"{self.model_cfg.name}-reg-suite-{dataset_name}"
            storage_name = f"sqlite:///benchmarks/results/optuna/{study_name}.db"

            # Clean up existing study
            try:
                optuna.delete_study(study_name=study_name, storage=storage_name)
            except KeyError:
                pass
            except Exception as e:
                logger.warning(f"Could not delete existing study {study_name}: {e}")

            sampler = TPESampler(seed=self.cfg.seed)  # type: ignore[arg-type]
            study = optuna.create_study(
                study_name=study_name, storage=storage_name, direction="minimize", sampler=sampler, load_if_exists=False
            )

            start_tune_time = time()
            study.optimize(objective, n_trials=self.cfg.n_trials)  # type: ignore[arg-type]
            tuning_time = time() - start_tune_time
            logger.info(f"🏆 Best params for {dataset_name}: {study.best_params}")
            results["datasets"][dataset_name]["best_params"] = study.best_params
            results["datasets"][dataset_name]["tuning_time"] = tuning_time

            # Convert best params into init kwargs + params dict as before
            optuna_best_params = study.best_params
            tuned_init_kwargs = {}
            tuned_params = {}
            for param, value in optuna_best_params.items():
                if param in self.model_cfg.tunable_init_kwargs.keys():
                    tuned_init_kwargs[param] = value
                elif "tunable_params" in self.model_cfg and param in self.model_cfg.tunable_params.keys():
                    tuned_params[param] = value

            if "fixed_params" in self.model_cfg or "tunable_params" in self.model_cfg:
                combined_params = {}
                if "fixed_params" in self.model_cfg:
                    combined_params.update(self.model_cfg.fixed_params)
                if tuned_params:
                    combined_params.update(tuned_params)
                if combined_params:
                    tuned_init_kwargs["params"] = combined_params

            # Evaluate on folds 1..9
            fold_metrics = []
            fit_times = []
            for fold_num in tqdm(range(1, 10), "Evaluating folds", leave=False):
                try:
                    train_idx, test_idx = task.get_train_test_split_indices(fold=fold_num)
                except Exception:
                    splits = task.get_train_test_split_indices()
                    if len(splits) > fold_num:
                        train_idx, test_idx = splits[fold_num]
                    else:
                        logger.warning(f"Missing fold {fold_num} for task {task_id}; skipping fold.")
                        continue

                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_test_fold = X.iloc[test_idx]
                y_test_fold = y.iloc[test_idx]

                best_model = model_cls(**self.model_cfg.fixed_init_kwargs, **tuned_init_kwargs)
                try:
                    start_fit_time = time()
                    best_model.fit(X_train_fold, y_train_fold)
                    fit_times.append(time() - start_fit_time)
                except Exception as e:
                    logger.error(f"Model fit failed on fold {fold_num} for task {task_id}: {e}")
                    continue

                metrics_dict = self.calc_metrics(X_test_fold, y_test_fold, best_model, None)
                fold_metrics.append(metrics_dict)

            # Aggregate metrics across folds (mean & std)
            if fold_metrics:
                aggregated = {}
                for m in fold_metrics[0].keys():
                    values = [fold[m] for fold in fold_metrics if m in fold and not np.isnan(fold[m])]
                    aggregated[m] = {
                        "mean": float(np.mean(values)) if values else float("nan"),
                        "std": float(np.std(values)) if values else float("nan"),
                    }
                results["datasets"][dataset_name]["metrics"] = aggregated
                logger.info(f"✅ Finished {dataset_name} with aggregated metrics: {aggregated}")
            else:
                logger.warning(f"No fold metrics computed for task {task_id} / {dataset_name}")
            if fit_times:
                results["datasets"][dataset_name]["fitting_time"] = float(np.mean(fit_times))
            # Save intermediate results
            self._save_results(results)

        return results

    def _save_results(self, results):
        """Save intermediate results to avoid losing progress."""
        os.makedirs("benchmarks/results/reg_suite/", exist_ok=True)
        with open(f"benchmarks/results/reg_suite/{self.model_cfg.name}.json", "w") as f:
            json.dump(results, f, indent=2)
