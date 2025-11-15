import json
import os

import classification_metrics
import numpy as np
import optuna
import regression_metrics
from factory import ModelFactory
from omegaconf import OmegaConf
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import train_test_split as TTS
from utils import available_classification_datasets, available_regression_datasets


class Orchestrator:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.target_type = self.model_cfg.target_type

    def run(self):
        # Instantiate model from config
        model_cls = ModelFactory.get(self.model_cfg)
        print("‼️pwd", os.getcwd())

        results = {}

        dataset_iterator = (
            available_regression_datasets if self.target_type == "regression" else available_classification_datasets
        )

        for dataset_name, X, y in dataset_iterator():
            results[dataset_name] = {}
            # Split dataset into train and test sets
            X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.25, random_state=self.cfg.seed, shuffle=True)

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
                        iter_init_kwargs[name] = trial.gugest_categorical(name, args["categories"])
                    elif args["type"] == "uniform":
                        if args.get("log", False):
                            iter_init_kwargs[name] = trial.suggest_loguniform(name, args["low"], args["high"])
                        else:
                            iter_init_kwargs[name] = trial.suggest_uniform(name, args["low"], args["high"])
                    else:
                        raise ValueError(f"Parameter type '{args['type']}' unknown!")
                if self.model_cfg.get("tunable_params") is not None:
                    print("👀 Ayo hier gibt's echt n params dict!! ")
                    iter_params = {}
                    # Here add the parameters of the `params` dict given to the BDF constructor, depending on the distribution
                model = model_cls(**self.model_cfg.fixed_init_kwargs, **iter_init_kwargs)
                score_metric = mean_squared_error if self.target_type == "regression" else log_loss
                return np.mean(
                    CVS(model, X_train, y_train, cv=3, scoring=make_scorer(score_metric, greater_is_better=False))
                )

            study_name = f"{self.model_cfg.name}-{dataset_name}"
            storage_name = "sqlite:///" + study_name + ".db"
            try:
                optuna.delete_study(study_name, storage=storage_name)
            except Exception as e:
                print(e)
            study = optuna.create_study(
                study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=False
            )
            study.optimize(objective, n_trials=self.cfg.n_trials)

            # Evaluate on test set
            best_model = model_cls(**self.model_cfg.fixed_init_kwargs, **study.best_params)
            best_model.fit(X_train, y_train)

            results[dataset_name].update(self._calc_metrics(X_test, y_test, best_model))

        # Write to ./results/ as <model_name>.json, already contains everything
        # NOTE: Need to re-route optuna to benchmarks folder, specific name, delete .db etc.
        # with open(f"../../results/{self.model_cfg.name}.json", "w") as f:
        #     json.dump(results, f, indent=4)
        print(results)

    def _calc_metrics(self, X, y, model) -> dict:
        """Calculate metrics given the (best) fitted model.
        Uses config to determine whether probabilistic metrics are available
        and if so, how samples are obtained.
        """
        do_reg = self.target_type == "regression"
        point_metrics = regression_metrics.POINT_METRICS if do_reg else classification_metrics.POINT_METRICS
        proba_metrics = (
            regression_metrics.PROBABILISTIC_METRICS if do_reg else classification_metrics.PROBABILISTIC_METRICS
        )
        metric_dict = {}
        y_pred = model.predict(X)
        for m, m_func in point_metrics.items():
            metric_dict[m] = m_func(y, y_pred)
        if self.model_cfg.probabilistic:
            y_pred_samples = model.predict_samples(X)
            for m, m_func in proba_metrics.items():
                metric_dict[m] = m_func(y, y_pred_samples)
        return metric_dict
