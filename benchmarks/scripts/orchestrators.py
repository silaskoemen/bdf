import json
import os
from dataclasses import asdict

import classification_metrics
import numpy as np
import optuna
import regression_metrics
from factory import ModelFactory
from omegaconf import OmegaConf
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler
from utils import DatasetMetadata, available_classification_datasets, available_regression_datasets


class Orchestrator:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.target_type = self.model_cfg.target_type
        self.standardize_target = self.model_cfg.get("standardize_target", "none")

    def _is_compatible(self, dataset_metadata: DatasetMetadata) -> bool:
        """Check if model can handle this dataset's target domain."""

        compatible_domains = self.model_cfg["compatible_target_domains"]
        return dataset_metadata.target_domain.value in compatible_domains

    def _maybe_apply_target_standardization(self, y_train):
        """Apply target standardization based on config."""
        if self.standardize_target == "no" or self.target_type != "regression":
            return y_train, None

        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train).ravel()

        return y_train_scaled, scaler

    def _get_score_metric(self, metadata: DatasetMetadata):
        if self.target_type == "regression":
            return make_scorer(mean_squared_error, greater_is_better=False)
        else:
            # Discern binary from multi-class
            if metadata.target_domain.value == "binary":
                return make_scorer(log_loss, greater_is_better=False)
            else:
                return make_scorer(log_loss, greater_is_better=False)

    def run(self):
        model_cls = ModelFactory.get(self.model_cfg)
        results = {"model_config": OmegaConf.to_container(self.model_cfg, resolve=True), "datasets": {}}

        dataset_iterator = (
            available_regression_datasets if self.target_type == "regression" else available_classification_datasets
        )

        # Create optuna storage directory
        os.makedirs("../../results/optuna/", exist_ok=True)

        for metadata, X, y in dataset_iterator():
            # NEW: Compatibility check
            if not self._is_compatible(metadata):
                print(f"⏭️ Skipping {metadata.name}: incompatible target domain {metadata.target_domain.value}")
                continue

            print(f"🔬 Processing {metadata.name}")
            results["datasets"][metadata.name] = {"metadata": metadata.to_dict(), "metrics": {}, "best_params": {}}

            X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.25, random_state=self.cfg.seed, shuffle=True)

            # NEW: Apply standardization
            y_train_proc, scaler = self._maybe_apply_target_standardization(y_train)

            # Tune model
            def objective(trial):
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

                model = model_cls(**self.model_cfg.fixed_init_kwargs, **iter_init_kwargs)
                score_metric = mean_squared_error if self.target_type == "regression" else log_loss

                return np.mean(
                    CVS(model, X_train, y_train_proc, cv=3, scoring=make_scorer(score_metric, greater_is_better=False))
                )

            study_name = f"{self.model_cfg.name}-{metadata.name}"
            storage_name = f"sqlite:///../../results/optuna/{study_name}.db"

            # Clean up existing study - only if storage exists
            try:
                optuna.delete_study(study_name=study_name, storage=storage_name)
            except KeyError:
                pass  # Study doesn't exist yet, that's fine
            except Exception as e:
                print(f"⚠️  Warning: Could not delete existing study: {e}")

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                direction="minimize",
                load_if_exists=False,  # Changed to minimize since we're minimizing error
            )
            study.optimize(objective, n_trials=self.cfg.n_trials)

            # Evaluate on test set
            best_model = model_cls(**self.model_cfg.fixed_init_kwargs, **study.best_params)
            best_model.fit(X_train, y_train_proc)

            # NEW: Store best params
            results["datasets"][metadata.name]["best_params"] = study.best_params

            # Calculate metrics (handle inverse transform if needed)
            metrics_dict = self._calc_metrics(X_test, y_test, best_model, scaler)
            results["datasets"][metadata.name]["metrics"] = metrics_dict

            # NEW: Save intermediate results
            self._save_results(results)

        return results

    def _calc_metrics(self, X, y, model, scaler=None) -> dict:
        """Calculate metrics with optional inverse standardization."""
        do_reg = self.target_type == "regression"
        point_metrics = regression_metrics.POINT_METRICS if do_reg else classification_metrics.POINT_METRICS
        proba_metrics = (
            regression_metrics.PROBABILISTIC_METRICS if do_reg else classification_metrics.PROBABILISTIC_METRICS
        )

        metric_dict = {}
        y_pred = model.predict(X)

        # Inverse transform if we standardized test targets
        if scaler is not None and self.standardize_target in ["only", "both"]:
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        for m, m_func in point_metrics.items():
            metric_dict[m] = float(m_func(y, y_pred))

        if self.model_cfg.probabilistic:
            y_pred_samples = model.predict_samples(X, n_samples=self.cfg.n_samples)

            # Inverse transform samples if needed
            if scaler is not None and self.standardize_target in ["only", "both"]:
                y_pred_samples = scaler.inverse_transform(y_pred_samples)

            for m, m_func in proba_metrics.items():
                metric_dict[m] = float(m_func(y, y_pred_samples))

        return metric_dict

    def _save_results(self, results):
        """Save intermediate results to avoid losing progress."""
        os.makedirs("../../results/", exist_ok=True)
        with open(f"../../results/{self.model_cfg.name}.json", "w") as f:
            json.dump(results, f, indent=2)
