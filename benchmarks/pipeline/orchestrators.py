import json
import os
from time import time
from typing import Callable, Literal

import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger
from omegaconf import OmegaConf
from optuna.samplers import TPESampler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import train_test_split as TTS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..metrics.classification import CLAS_POINT_METRICS, CLAS_PROB_METRICS, compute_calibration_curve
from ..metrics.regression import REG_POINT_METRICS, REG_PROB_METRICS
from ..pipeline.data import DatasetMetadata, available_classification_datasets, available_regression_datasets
from ..utils.benchmark_utils import LogTransformTransformer


def _aggregate_calibration_curves(curves: list[dict]) -> dict:
    """Aggregate calibration curves across folds using weighted averaging.

    Args:
        curves: List of calibration curve dicts with keys 'prob_true', 'prob_pred', 'bin_counts'

    Returns:
        Aggregated calibration curve dict with weighted averages
    """
    if not curves:
        return {"prob_true": [], "prob_pred": [], "bin_counts": []}

    n_bins = len(curves[0]["prob_true"])
    total_counts = np.zeros(n_bins)
    weighted_prob_true = np.zeros(n_bins)
    weighted_prob_pred = np.zeros(n_bins)

    for curve in curves:
        counts = np.array(curve["bin_counts"])
        prob_true = np.array(curve["prob_true"])
        prob_pred = np.array(curve["prob_pred"])

        # Only include bins with samples (non-nan values)
        valid = ~np.isnan(prob_true) & (counts > 0)
        total_counts += np.where(valid, counts, 0)
        weighted_prob_true += np.where(valid, prob_true * counts, 0)
        weighted_prob_pred += np.where(valid, prob_pred * counts, 0)

    # Compute weighted averages (nan for bins with no samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_prob_true = np.where(total_counts > 0, weighted_prob_true / total_counts, np.nan)
        avg_prob_pred = np.where(total_counts > 0, weighted_prob_pred / total_counts, np.nan)

    return {
        "prob_true": [float(x) if not np.isnan(x) else None for x in avg_prob_true],
        "prob_pred": [float(x) if not np.isnan(x) else None for x in avg_prob_pred],
        "bin_counts": [int(x) for x in total_counts],
    }


# Type alias for prediction type
PredictionType = Literal["samples", "quantiles"]


def get_model_prediction_type(model) -> PredictionType:
    """Detect the native prediction type of a model."""
    if hasattr(model, "PREDICTION_TYPE"):
        return model.PREDICTION_TYPE
    # Fallback: duck typing - check samples first (more common)
    if hasattr(model, "predict_samples"):
        return "samples"
    if hasattr(model, "predict_quantiles") and hasattr(model, "quantiles"):
        return "quantiles"
    # Default to samples for standard sklearn models
    return "samples"


def get_model_quantiles(model) -> np.ndarray | None:
    """Get quantile levels from a quantile-native model."""
    if hasattr(model, "quantiles"):
        return np.array(sorted(model.quantiles))
    return None


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

    def _get_nan_metrics_dict(self) -> dict:
        """Create a metrics dict with all expected metrics set to NaN.

        This maintains consistent structure when model fitting fails.
        """
        do_reg = self.target_type == "regression"
        point_metrics = REG_POINT_METRICS if do_reg else CLAS_POINT_METRICS
        proba_metrics = REG_PROB_METRICS if do_reg else CLAS_PROB_METRICS

        metric_dict = {m: float("nan") for m in point_metrics.keys()}

        # Add calibration curve for classification
        if not do_reg:
            metric_dict["calibration_curve"] = None

        # Add probabilistic metrics if model is probabilistic
        if self.model_cfg.probabilistic and proba_metrics:
            for m in proba_metrics.keys():
                metric_dict[m] = float("nan")

        return metric_dict

    def calc_metrics(self, X, y, model, pipeline=None, status_callback: Callable[[str], None] | None = None) -> dict:
        """Calculate metrics with optional inverse standardization.

        Args:
            X: Features for prediction
            y: True target values
            model: Fitted model
            pipeline: Optional pipeline for inverse transformation
            status_callback: Optional callback to update status (e.g., "Fold 3: CRPS")
        """
        do_reg = self.target_type == "regression"
        point_metrics = REG_POINT_METRICS if do_reg else CLAS_POINT_METRICS
        proba_metrics = REG_PROB_METRICS if do_reg else CLAS_PROB_METRICS

        metric_dict = {}
        if status_callback:
            status_callback("Predicting")
        try:
            y_pred = model.predict(X) if do_reg else model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            y_pred = model.predict(X)

        # Inverse transform if we standardized test targets
        if pipeline is not None and self.standardize_target in ["only", "both"]:
            y_pred = pipeline.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        for m, m_func in point_metrics.items():
            if status_callback:
                status_callback(f"📊 {m}")
            try:
                metric_dict[m] = float(m_func(y, y_pred))
            except Exception as e:
                logger.error(f"❌ Point metric {m} failed: {e}")
                metric_dict[m] = float("nan")

        # Calibration curve for classification (stores dict, not scalar)
        if not do_reg:
            if status_callback:
                status_callback("📈 calibration_curve")
            try:
                y_np = y.values if hasattr(y, "values") else y
                metric_dict["calibration_curve"] = compute_calibration_curve(y_np, y_pred, n_bins=10)
            except Exception as e:
                logger.error(f"❌ Calibration curve failed: {e}")
                metric_dict["calibration_curve"] = None

        # Probabilistic metrics
        if self.model_cfg.probabilistic and proba_metrics:
            try:
                prediction_type = get_model_prediction_type(model)
                quantile_levels = get_model_quantiles(model)

                if status_callback:
                    status_callback(f"🎲 Predicting ({prediction_type})")

                # Get predictions in native format
                if prediction_type == "quantiles":
                    y_pred_native = model.predict_quantiles(X)  # type: ignore[attr-defined]
                    # Also get samples for metrics that need them
                    y_pred_samples = model.predict_samples(X, n_samples=self.cfg.sample_size)  # type: ignore[attr-defined]
                else:
                    y_pred_native = model.predict_samples(X, n_samples=self.cfg.sample_size)  # type: ignore[attr-defined]
                    y_pred_samples = y_pred_native

                # Inverse transform if needed
                if pipeline is not None and self.standardize_target in ["only", "both"]:
                    if prediction_type == "quantiles":
                        y_pred_native = pipeline.inverse_transform(y_pred_native)
                    y_pred_samples = pipeline.inverse_transform(y_pred_samples)

                # Route each metric to appropriate prediction format
                for m, spec in proba_metrics.items():
                    if status_callback:
                        status_callback(f"🎲 {m}")
                    try:
                        if prediction_type in spec.accepts:
                            # Native format works - pass quantile_levels to all metrics
                            metric_dict[m] = float(spec(y, y_pred_native, quantile_levels=quantile_levels))
                        else:
                            # Need samples, use converted version (quantile_levels=None for samples)
                            metric_dict[m] = float(spec(y, y_pred_samples, quantile_levels=None))
                    except Exception as e:
                        logger.error(f"❌ Prob metric {m} failed: {e}")
                        metric_dict[m] = float("nan")

            except Exception as e:
                logger.error(f"❌ Probabilistic prediction failed: {e}")
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

            # Use fold 0 for tuning, folds 1-9 for evaluation
            self.tuned_init_kwargs = None
            fit_times = []
            fold_metrics = []
            all_splits = list(cv_splitter.split(X, y))

            # Fold 0: tuning only
            train_idx_0, test_idx_0 = all_splits[0]
            results, self.tuned_init_kwargs = self._tune_on_fold(
                X.iloc[train_idx_0],
                y.iloc[train_idx_0],
                X.iloc[test_idx_0],
                y.iloc[test_idx_0],
                model_cls,
                results,
                metadata,
            )

            # Folds 1-9: evaluation with rich progress
            eval_splits = all_splits[1:]  # Folds 1-9
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(f"Evaluating {metadata.name}", total=len(eval_splits))

                for fold_idx, (train_idx, test_idx) in enumerate(eval_splits, start=1):
                    # Evaluate on test set with seed reset
                    np.random.seed(self.cfg.seed)  # type: ignore[attr-defined]
                    progress.update(task, description=f"Fold {fold_idx}: Fitting")
                    best_model = model_cls(**self.model_cfg.fixed_init_kwargs, **self.tuned_init_kwargs)

                    try:
                        start_time = time()
                        best_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                        end_time = time()
                        fit_times.append(end_time - start_time)

                        # Calculate metrics with status callback for progress updates
                        def update_status(status: str) -> None:
                            progress.update(task, description=f"Fold {fold_idx}: {status}")

                        metrics_dict = self.calc_metrics(
                            X.iloc[test_idx], y.iloc[test_idx], best_model, None, status_callback=update_status
                        )
                        fold_metrics.append(metrics_dict)
                    except Exception as e:
                        logger.error(f"❌ Fold {fold_idx} failed: {e}")
                        # Add NaN metrics to maintain consistent structure
                        fold_metrics.append(self._get_nan_metrics_dict())

                    progress.advance(task)

            # Aggregate metrics across folds (mean & std)
            if fold_metrics:
                aggregated = {}
                for m in fold_metrics[0].keys():
                    if m == "calibration_curve":
                        # Weighted average of calibration curves across folds
                        aggregated[m] = _aggregate_calibration_curves(
                            [fold[m] for fold in fold_metrics if fold.get(m) is not None]
                        )
                    else:
                        # Scalar metrics: mean & std
                        values = [fold[m] for fold in fold_metrics if m in fold and not np.isnan(fold[m])]
                        aggregated[m] = {
                            "mean": float(np.mean(values)) if values else float("nan"),
                            "std": float(np.std(values)) if values else float("nan"),
                        }
                results["datasets"][metadata.name]["metrics"] = aggregated
                logger.info(f"✅ Finished {metadata.name} with aggregated metrics")
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

            try:
                model.fit(X_train, y_train)

                # Classification: always use log_loss
                if self.target_type == "classification":
                    y_pred = model.predict_proba(X_test)[:, 1]
                    return log_loss(y_test, y_pred)

                # Regression: support configurable tuning metric
                tuning_metric = getattr(self.cfg, "tuning_metric", "mse")

                if tuning_metric == "mse":
                    y_pred = model.predict(X_test)
                    return mean_squared_error(y_test, y_pred)

                elif tuning_metric == "crps":
                    from ..metrics.regression import crps_wrapper

                    # Use native prediction type for efficiency
                    prediction_type = get_model_prediction_type(model)

                    if prediction_type == "quantiles" and hasattr(model, "quantiles"):
                        y_pred_q = model.predict_quantiles(X_test)
                        quantiles = np.array(sorted(model.quantiles))
                        return crps_wrapper(y_test, y_pred_q, quantile_levels=quantiles)
                    else:
                        # Use fewer samples for tuning efficiency
                        tuning_samples = getattr(self.cfg, "tuning_sample_size", 500)
                        y_pred_samples = model.predict_samples(X_test, n_samples=tuning_samples)
                        return crps_wrapper(y_test, y_pred_samples, quantile_levels=None)

                else:
                    raise ValueError(f"Unknown tuning_metric: {tuning_metric}")

            except Exception as e:
                logger.warning(f"⚠️ Warning: Fitting or prediction failed: {e}")
                return float("inf")

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
        """Save intermediate results to avoid losing progress.

        Uses YAML format for native NaN/inf support that can be loaded back.
        """
        os.makedirs("benchmarks/results/custom/", exist_ok=True)
        with open(f"benchmarks/results/custom/{self.model_cfg.name}.yaml", "w") as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
