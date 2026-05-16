"""Scoring Method Ablation Study for JMLR Submission.

Dedicated study investigating when NLE (Bayesian marginal likelihood) outperforms
NLL-based scoring for tree split selection in BDF.

We characterize the regime in which NLE-based split scoring is preferable to
NLL-based scoring with standard model-selection corrections (AIC, BIC), varying
sample size, noise/imbalance, and evaluating on real UCI data.

Usage:
    pixi run python benchmarks/scoring_method_study.py              # Full study
    pixi run python benchmarks/scoring_method_study.py regression   # Regression only
    pixi run python benchmarks/scoring_method_study.py classification  # Classification only
    pixi run python benchmarks/scoring_method_study.py --quick      # Quick test run
    pixi run python benchmarks/scoring_method_study.py --core       # Core experiments only
    pixi run python benchmarks/scoring_method_study.py --robustness # Robustness experiments only
    pixi run python benchmarks/scoring_method_study.py --real-data  # Real data experiments only

Note: LOO-CV is deferred to the KDE/non-conjugate setting, where NLE is unavailable;
      it is not meaningful to compare against NLE on conjugate likelihoods.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scoringrules
from scipy import stats
from sklearn.datasets import (
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman3,
    make_moons,
    make_regression,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFClassifier, BDFRegressor
from benchmarks.pipeline.data import load as load_pipeline_dataset

# Suppress warnings during experiments
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
RESULTS_DIR = Path(__file__).parent / "results" / "scoring_method"
PLOTS_DIR = Path(__file__).parent / "plots" / "scoring_method"
TABLES_DIR = Path(__file__).parent / "tables" / "scoring_method"

# Create directories
for d in [RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    (d / "regression").mkdir(exist_ok=True)
    (d / "classification").mkdir(exist_ok=True)
    (d / "real_data").mkdir(exist_ok=True)

# Experiment configuration
SEED = 42
N_SEEDS = 10  # Sufficient for Friedman test with α=0.05
SAMPLE_SIZES = [100, 250, 500, 1000, 2500]  # Removed 50 (too small)
TEST_FRACTION = 0.2

# Quick mode configuration (for testing)
QUICK_N_SEEDS = 3
QUICK_SAMPLE_SIZES = [100, 500]

# Scoring method configurations
# NOTE: nll_loo excluded - Rust implementation is O(n²), not O(n) like Python
SCORING_CONFIGS = {
    "nle": {"score_method": "nle", "score_correction": None},
    "nll": {"score_method": "nll", "score_correction": None},
    "nll_aic": {"score_method": "nll", "score_correction": "aic"},
    "nll_bic": {"score_method": "nll", "score_correction": "bic"},
}

# Model hyperparameters (fixed, not ablated)
MODEL_CONFIG = {
    "n_trees": 50,
    "max_depth": 50,
    "min_samples_leaf": 10,
    "alpha": 0.01,
    "gamma": 0.1,
    "delta": 0.01,
}

# Regression distribution config
REG_DIST_PARAMS = {
    "mu_mu": "auto",
    "sigma_mu": "auto",
    "sigma_mu_auto_scale": 1.0,
    "use_posterior_predictive": True,
}

# Classification distribution config
CLAS_DIST_PARAMS = {
    "alpha_p": 1.0,  # Uniform Beta(1,1) prior
    "beta_p": 1.0,
    "use_posterior_predictive": True,
}

# Noise levels for regression (relative to target std)
NOISE_LEVELS = {
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
}

# Class imbalance levels for classification
IMBALANCE_LEVELS = {
    "balanced": 0.5,
    "moderate": 0.3,
    "severe": 0.1,
}

# DGPs for core study (3 per task — minimum for credible CD diagram)
REGRESSION_DGPS = ["friedman1", "linear", "friedman3"]
CLASSIFICATION_DGPS = ["make_classification", "moons", "circles"]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# REAL DATA LOADERS
# ============================================================================


def load_real_regression_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load real regression datasets from pre-processed UCI parquet files."""
    dataset_names = [
        "yacht_hydrodynamics",  # n=308
        "boston_housing",  # n=506
        "energy_efficiency",  # n=768
        "concrete_strength",  # n=1030
        "wine_quality",  # n≈1600
    ]
    datasets = {}
    for name in dataset_names:
        try:
            _, X_df, y_series = load_pipeline_dataset(name)
            datasets[name] = (X_df.to_numpy(), y_series.to_numpy())
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
    logger.info(f"Loaded {len(datasets)} real regression datasets")
    return datasets


def load_real_classification_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load real classification datasets from pre-processed UCI parquet files."""
    dataset_names = [
        "heart_disease",  # n=303
        "ionosphere",  # n=351
        "breast_cancer_wisconsin",  # n=569
        "credit_approval",  # n=690
        "pima_diabetes",  # n=768
    ]
    datasets = {}
    for name in dataset_names:
        try:
            _, X_df, y_series = load_pipeline_dataset(name)
            datasets[name] = (X_df.to_numpy(), y_series.to_numpy().astype(int))
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
    logger.info(f"Loaded {len(datasets)} real classification datasets")
    return datasets


# ============================================================================
# DATA GENERATING PROCESSES
# ============================================================================


def generate_regression_data(
    dgp: str,
    n_samples: int,
    noise_level: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate regression data with specified noise level.

    Returns X_train, X_test, y_train, y_test.
    """
    rng = np.random.default_rng(seed)
    noise_scale = NOISE_LEVELS[noise_level]

    if dgp == "friedman1":
        X, y = make_friedman1(n_samples=n_samples, n_features=10, noise=0, random_state=seed)
        y_std = np.std(y)
        y = y + rng.normal(0, noise_scale * y_std, size=y.shape)

    elif dgp == "linear":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            noise=0,
            random_state=seed,
        )
        y_std = np.std(y)
        y = y + rng.normal(0, noise_scale * y_std, size=y.shape)

    elif dgp == "friedman3":
        X, y = make_friedman3(n_samples=n_samples, noise=0, random_state=seed)
        y_std = np.std(y)
        y = y + rng.normal(0, noise_scale * y_std, size=y.shape)

    else:
        raise ValueError(f"Unknown regression DGP: {dgp}")

    return train_test_split(X, y, test_size=TEST_FRACTION, random_state=seed)


def generate_classification_data(
    dgp: str,
    n_samples: int,
    imbalance_level: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate classification data with specified class imbalance.

    Returns X_train, X_test, y_train, y_test.
    """
    minority_ratio = IMBALANCE_LEVELS[imbalance_level]

    if dgp == "make_classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[1 - minority_ratio, minority_ratio],
            flip_y=0.05,
            class_sep=1.0,
            random_state=seed,
        )

    elif dgp == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=seed)
        X, y = _apply_imbalance(X, y, minority_ratio, seed)

    elif dgp == "circles":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=seed)
        X, y = _apply_imbalance(X, y, minority_ratio, seed)

    else:
        raise ValueError(f"Unknown classification DGP: {dgp}")

    return train_test_split(X, y, test_size=TEST_FRACTION, random_state=seed, stratify=y)


def _apply_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    minority_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample majority class to achieve target minority ratio."""
    if minority_ratio >= 0.5:
        return X, y

    rng = np.random.default_rng(seed)
    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    n_minority = int(len(y) * minority_ratio)
    n_majority = len(y) - n_minority

    if len(class_0_idx) > len(class_1_idx):
        majority_idx = rng.choice(class_0_idx, size=n_majority, replace=False)
        minority_idx = class_1_idx[:n_minority] if len(class_1_idx) >= n_minority else class_1_idx
    else:
        majority_idx = rng.choice(class_1_idx, size=n_majority, replace=False)
        minority_idx = class_0_idx[:n_minority] if len(class_0_idx) >= n_minority else class_0_idx

    all_idx = np.concatenate([majority_idx, minority_idx])
    rng.shuffle(all_idx)

    return X[all_idx], y[all_idx]


# ============================================================================
# METRICS
# ============================================================================


def compute_regression_metrics(
    model: BDFRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute regression metrics including CRPSS against climatological baseline."""
    metrics = {}

    # Point predictions
    y_pred = model.predict(X_test)
    metrics["rmse"] = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    # Probabilistic predictions
    try:
        samples = model.predict_samples(X_test, n_samples=500)
        metrics["crps"] = float(np.mean(scoringrules.crps_ensemble(y_test, samples)))
    except Exception:
        metrics["crps"] = np.nan

    try:
        quantiles = model.predict_quantiles(X_test, q=[0.05, 0.95], n_samples=500)
        lower, upper = quantiles[:, 0], quantiles[:, 1]
        metrics["coverage_90"] = float(np.mean((y_test >= lower) & (y_test <= upper)))
        metrics["interval_width_90"] = float(np.mean(upper - lower))
    except Exception:
        metrics["coverage_90"] = np.nan
        metrics["interval_width_90"] = np.nan

    # CRPSS with climatological (Gaussian fit to y_train) baseline
    if y_train is not None and not np.isnan(metrics.get("crps", np.nan)):
        clim_mu = float(np.mean(y_train))
        clim_sigma = float(np.std(y_train)) + 1e-8
        z = (y_test - clim_mu) / clim_sigma
        # Closed-form CRPS for Gaussian: sigma * (2*phi(z) + z*(2*Phi(z)-1) - 1/sqrt(pi))
        clim_crps = float(
            np.mean(clim_sigma * (2 * stats.norm.pdf(z) + z * (2 * stats.norm.cdf(z) - 1) - 1 / np.sqrt(np.pi)))
        )
        metrics["crpss"] = float(1.0 - metrics["crps"] / clim_crps) if clim_crps > 0 else np.nan
    else:
        metrics["crpss"] = np.nan

    return metrics


def compute_classification_metrics(
    model: BDFClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics including BSS against train-prior baseline."""
    metrics = {}

    y_prob = model.predict_proba(X_test)
    y_prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

    # Log loss
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob_pos, eps, 1 - eps)
    metrics["log_loss"] = float(-np.mean(y_test * np.log(y_prob_clipped) + (1 - y_test) * np.log(1 - y_prob_clipped)))

    # Brier score
    metrics["brier_score"] = float(np.mean((y_prob_pos - y_test) ** 2))

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score

        metrics["auroc"] = float(roc_auc_score(y_test, y_prob_pos))
    except Exception:
        metrics["auroc"] = np.nan

    # ECE
    metrics["ece"] = _compute_ece(y_test, y_prob_pos, n_bins=10)

    # Brier Skill Score with train-prior (p = y_train.mean()) as reference forecast
    if y_train is not None:
        p_ref = float(np.mean(y_train))
        brier_ref = float(np.mean((y_test - p_ref) ** 2))
        metrics["bss"] = float(1.0 - metrics["brier_score"] / brier_ref) if brier_ref > 0 else np.nan
    else:
        metrics["bss"] = np.nan

    return metrics


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            ece += np.sum(mask) / n * np.abs(bin_accuracy - bin_confidence)

    return float(ece)


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================


@dataclass
class ExperimentResult:
    """Single experiment result."""

    task: str
    dgp: str
    n_samples: int
    condition: str  # noise_level or imbalance_level
    scoring_method: str
    seed: int
    metrics: dict[str, float]
    fit_time: float
    predict_time: float


@dataclass
class StudyResults:
    """Full study results."""

    task: str
    config: dict[str, Any]
    results: list[ExperimentResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for r in self.results:
            row = {
                "task": r.task,
                "dgp": r.dgp,
                "n_samples": r.n_samples,
                "condition": r.condition,
                "scoring": r.scoring_method,
                "seed": r.seed,
                "fit_time": r.fit_time,
                "predict_time": r.predict_time,
            }
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: Path):
        """Save results to parquet."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        logger.info(f"Saved results to {path}")

    @classmethod
    def load(cls, path: Path, task: str) -> "StudyResults":
        """Load results from parquet."""
        df = pd.read_parquet(path)
        results = cls(task=task, config={})
        for _, row in df.iterrows():
            metrics = {
                k: row[k]
                for k in row.index
                if k not in ["task", "dgp", "n_samples", "condition", "scoring", "seed", "fit_time", "predict_time"]
            }
            results.results.append(
                ExperimentResult(
                    task=row["task"],
                    dgp=row["dgp"],
                    n_samples=row["n_samples"],
                    condition=row["condition"],
                    scoring_method=row["scoring"],
                    seed=row["seed"],
                    metrics=metrics,
                    fit_time=row["fit_time"],
                    predict_time=row["predict_time"],
                )
            )
        return results


def run_regression_experiment(
    dgp: str,
    n_samples: int,
    noise_level: str,
    scoring_name: str,
    seed: int,
) -> ExperimentResult:
    """Run a single regression experiment."""
    X_train, X_test, y_train, y_test = generate_regression_data(dgp, n_samples, noise_level, seed)

    dist_params = REG_DIST_PARAMS.copy()
    dist_params.update(SCORING_CONFIGS[scoring_name])

    model = BDFRegressor(dist="NormalMuNormal", params=dist_params, **MODEL_CONFIG)  # ty:ignore[invalid-argument-type]

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    metrics = compute_regression_metrics(model, X_test, y_test, y_train=y_train)
    predict_time = time.time() - start_time

    return ExperimentResult(
        task="regression",
        dgp=dgp,
        n_samples=n_samples,
        condition=noise_level,
        scoring_method=scoring_name,
        seed=seed,
        metrics=metrics,
        fit_time=fit_time,
        predict_time=predict_time,
    )


def run_classification_experiment(
    dgp: str,
    n_samples: int,
    imbalance_level: str,
    scoring_name: str,
    seed: int,
) -> ExperimentResult:
    """Run a single classification experiment."""
    X_train, X_test, y_train, y_test = generate_classification_data(dgp, n_samples, imbalance_level, seed)

    dist_params = CLAS_DIST_PARAMS.copy()
    dist_params.update(SCORING_CONFIGS[scoring_name])

    model = BDFClassifier(
        dist="BetaABBernoulli", params=dist_params, **MODEL_CONFIG  # ty:ignore[invalid-argument-type]
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    metrics = compute_classification_metrics(model, X_test, y_test, y_train=y_train)
    predict_time = time.time() - start_time

    return ExperimentResult(
        task="classification",
        dgp=dgp,
        n_samples=n_samples,
        condition=imbalance_level,
        scoring_method=scoring_name,
        seed=seed,
        metrics=metrics,
        fit_time=fit_time,
        predict_time=predict_time,
    )


# ============================================================================
# STUDY RUNNERS
# ============================================================================


def run_core_study(
    task: str,
    n_seeds: int = N_SEEDS,
    sample_sizes: list[int] | None = None,
    resume: bool = True,
) -> StudyResults:
    """Run core study: sample size × scoring method (fixed noise/imbalance).

    This is the PRIMARY analysis for the paper.
    """
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES

    results_path = RESULTS_DIR / task / "core_results.parquet"

    if resume and results_path.exists():
        logger.info(f"Loading existing results from {results_path}")
        results = StudyResults.load(results_path, task)
        completed = {(r.dgp, r.n_samples, r.condition, r.scoring_method, r.seed) for r in results.results}
    else:
        results = StudyResults(task=task, config={"type": "core", "sample_sizes": sample_sizes, "n_seeds": n_seeds})
        completed = set()

    dgps = REGRESSION_DGPS if task == "regression" else CLASSIFICATION_DGPS
    # Fixed condition for core study
    condition = "medium" if task == "regression" else "balanced"
    scoring_methods = list(SCORING_CONFIGS.keys())

    total = len(dgps) * len(sample_sizes) * len(scoring_methods) * n_seeds
    remaining = total - len(completed)

    logger.info(f"Core {task} study: {remaining} experiments remaining of {total}")
    pbar = tqdm(total=remaining, desc=f"Core {task}")

    run_fn = run_regression_experiment if task == "regression" else run_classification_experiment

    for dgp in dgps:
        for n_samples in sample_sizes:
            for scoring_name in scoring_methods:
                for seed in range(SEED, SEED + n_seeds):
                    key = (dgp, n_samples, condition, scoring_name, seed)
                    if key in completed:
                        continue

                    try:
                        result = run_fn(dgp, n_samples, condition, scoring_name, seed)
                        results.results.append(result)
                    except Exception as e:
                        logger.warning(f"Experiment failed {key}: {e}")

                    pbar.update(1)

            # Checkpoint after each sample size
            results.save(results_path)

    pbar.close()
    return results


def run_robustness_study(
    task: str,
    n_seeds: int = 10,
    resume: bool = True,
) -> StudyResults:
    """Run robustness study: vary noise/imbalance at fixed n=500.

    This is SECONDARY analysis for the paper.
    """
    results_path = RESULTS_DIR / task / "robustness_results.parquet"
    fixed_n = 500

    if resume and results_path.exists():
        logger.info(f"Loading existing results from {results_path}")
        results = StudyResults.load(results_path, task)
        completed = {(r.dgp, r.n_samples, r.condition, r.scoring_method, r.seed) for r in results.results}
    else:
        results = StudyResults(task=task, config={"type": "robustness", "n_samples": fixed_n, "n_seeds": n_seeds})
        completed = set()

    dgps = REGRESSION_DGPS if task == "regression" else CLASSIFICATION_DGPS
    conditions = list(NOISE_LEVELS.keys()) if task == "regression" else list(IMBALANCE_LEVELS.keys())
    scoring_methods = list(SCORING_CONFIGS.keys())

    total = len(dgps) * len(conditions) * len(scoring_methods) * n_seeds
    remaining = total - len(completed)

    logger.info(f"Robustness {task} study: {remaining} experiments remaining of {total}")
    pbar = tqdm(total=remaining, desc=f"Robustness {task}")

    run_fn = run_regression_experiment if task == "regression" else run_classification_experiment

    for dgp in dgps:
        for condition in conditions:
            # Drop moons/circles × severe imbalance: minority_ratio=0.1 → ~10 minority
            # samples at n=500 is too few for reliable stratified splits.
            if dgp in ("moons", "circles") and condition == "severe":
                continue
            for scoring_name in scoring_methods:
                for seed in range(SEED, SEED + n_seeds):
                    key = (dgp, fixed_n, condition, scoring_name, seed)
                    if key in completed:
                        continue

                    try:
                        result = run_fn(dgp, fixed_n, condition, scoring_name, seed)
                        results.results.append(result)
                    except Exception as e:
                        logger.warning(f"Experiment failed {key}: {e}")

                    pbar.update(1)

            # Checkpoint
            results.save(results_path)

    pbar.close()
    return results


def run_real_data_study(
    task: str,
    n_folds: int = 5,
    n_seeds: int = 3,
    resume: bool = True,
) -> StudyResults:
    """Run real data validation: 5-fold CV × 5 seeds.

    Required for JMLR submission.
    """
    results_path = RESULTS_DIR / "real_data" / f"{task}_results.parquet"

    if resume and results_path.exists():
        logger.info(f"Loading existing results from {results_path}")
        results = StudyResults.load(results_path, task)
        completed = {(r.dgp, r.n_samples, r.condition, r.scoring_method, r.seed) for r in results.results}
    else:
        results = StudyResults(task=task, config={"type": "real_data", "n_folds": n_folds, "n_seeds": n_seeds})
        completed = set()

    # Load datasets
    if task == "regression":
        datasets = load_real_regression_datasets()
        kfold_cls = KFold
    else:
        datasets = load_real_classification_datasets()
        kfold_cls = StratifiedKFold

    scoring_methods = list(SCORING_CONFIGS.keys())

    total = len(datasets) * len(scoring_methods) * n_folds * n_seeds
    remaining = total - len(completed)

    logger.info(f"Real data {task} study: {remaining} experiments remaining of {total}")
    pbar = tqdm(total=remaining, desc=f"Real data {task}")

    for dataset_name, (X, y) in datasets.items():
        n_samples = len(y)

        for scoring_name in scoring_methods:
            for seed in range(SEED, SEED + n_seeds):
                kfold = kfold_cls(n_splits=n_folds, shuffle=True, random_state=seed)

                for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                    condition = f"fold_{fold_idx}"
                    key = (dataset_name, n_samples, condition, scoring_name, seed)
                    if key in completed:
                        continue

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    dist_params = (REG_DIST_PARAMS if task == "regression" else CLAS_DIST_PARAMS).copy()
                    dist_params.update(SCORING_CONFIGS[scoring_name])

                    try:
                        if task == "regression":
                            model = BDFRegressor(
                                dist="NormalMuNormal",
                                params=dist_params,
                                **MODEL_CONFIG,  # ty:ignore[invalid-argument-type]
                            )
                            start_time = time.time()
                            model.fit(X_train, y_train)
                            fit_time = time.time() - start_time

                            start_time = time.time()
                            metrics = compute_regression_metrics(model, X_test, y_test, y_train=y_train)
                            predict_time = time.time() - start_time
                        else:
                            model = BDFClassifier(
                                dist="BetaABBernoulli",
                                params=dist_params,
                                **MODEL_CONFIG,  # ty:ignore[invalid-argument-type]
                            )
                            start_time = time.time()
                            model.fit(X_train, y_train)
                            fit_time = time.time() - start_time

                            start_time = time.time()
                            metrics = compute_classification_metrics(model, X_test, y_test, y_train=y_train)
                            predict_time = time.time() - start_time

                        results.results.append(
                            ExperimentResult(
                                task=task,
                                dgp=dataset_name,
                                n_samples=n_samples,
                                condition=condition,
                                scoring_method=scoring_name,
                                seed=seed,
                                metrics=metrics,
                                fit_time=fit_time,
                                predict_time=predict_time,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Experiment failed {key}: {e}")

                    pbar.update(1)

            # Checkpoint after each scoring method
            results.save(results_path)

    pbar.close()
    return results


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================


def compute_statistical_tests(
    df: pd.DataFrame,
    metric: str,
    groupby: list[str] | None = None,
) -> dict[str, Any]:
    """Compute Friedman test and pairwise Wilcoxon tests."""
    results = {}

    if groupby is None:
        groupby = ["n_samples"]

    for group_vals, group_df in df.groupby(groupby):
        group_key = str(group_vals) if isinstance(group_vals, tuple) else str(group_vals)

        pivot_df = group_df.pivot_table(
            index=["dgp", "condition", "seed"],
            columns="scoring",
            values=metric,
            aggfunc="first",
        ).dropna()

        if len(pivot_df) < 3:
            continue

        scoring_methods = pivot_df.columns.tolist()

        # Friedman test
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot_df[s].values for s in scoring_methods])
            results[f"{group_key}_friedman"] = {
                "statistic": float(friedman_stat),
                "p_value": float(friedman_p),
                "n_samples": len(pivot_df),
            }
        except Exception as e:
            logger.warning(f"Friedman test failed for {group_key}: {e}")

        # Average ranks for CD diagram — higher skill score is better, so rank descending
        ranks = pivot_df.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean().to_dict()
        results[f"{group_key}_avg_ranks"] = avg_ranks

        # Pairwise Wilcoxon tests (NLE vs others)
        if "nle" in scoring_methods:
            for other in scoring_methods:
                if other == "nle":
                    continue
                try:
                    # Skill scores are higher-is-better: test whether NLE > others
                    stat, pval = stats.wilcoxon(
                        pivot_df["nle"].values,
                        pivot_df[other].values,
                        alternative="greater",
                    )
                    cliffs_d = _cliffs_delta(pivot_df["nle"].values, pivot_df[other].values)
                    results[f"{group_key}_nle_vs_{other}"] = {
                        "wilcoxon_stat": float(stat),
                        "p_value": float(pval),
                        "cliffs_delta": float(cliffs_d),
                    }
                except Exception as e:
                    logger.warning(f"Wilcoxon test failed for {group_key} nle vs {other}: {e}")

    return results


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size.

    Interpretation:
    - |d| < 0.147: negligible
    - 0.147 <= |d| < 0.33: small
    - 0.33 <= |d| < 0.474: medium
    - |d| >= 0.474: large
    """
    n_x, n_y = len(x), len(y)
    more = int(np.sum(x[:, None] > y[None, :]))
    less = int(np.sum(x[:, None] < y[None, :]))
    return (more - less) / (n_x * n_y)


def compute_nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute critical difference for Nemenyi test."""
    from scipy.stats import studentized_range

    q_alpha = studentized_range.ppf(1 - alpha, n_methods, np.inf)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    return cd


# ============================================================================
# PLOTTING
# ============================================================================


def plot_interaction(
    df: pd.DataFrame,
    metric: str,
    title: str,
    save_path: Path | None = None,
):
    """Plot metric vs sample size for each scoring method (THE key figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"nle": "#2ecc71", "nll": "#e74c3c", "nll_aic": "#9b59b6", "nll_bic": "#3498db"}
    markers = {"nle": "o", "nll": "s", "nll_aic": "^", "nll_bic": "D"}

    scoring_methods = df["scoring"].unique()
    sample_sizes = sorted(df["n_samples"].unique())

    for scoring in scoring_methods:
        means = []
        stds = []
        for n in sample_sizes:
            subset = df[(df["scoring"] == scoring) & (df["n_samples"] == n)]
            if metric in subset.columns and not subset[metric].isna().all():
                means.append(subset[metric].mean())
                stds.append(subset[metric].std())
            else:
                means.append(np.nan)
                stds.append(np.nan)

        ax.errorbar(
            sample_sizes,
            means,
            yerr=stds,
            label=scoring.upper(),
            color=colors.get(scoring, "#333333"),
            marker=markers.get(scoring, "o"),
            markersize=8,
            linewidth=2,
            capsize=4,
        )

    metric_label = {"crpss": "CRPSS (higher is better)", "bss": "BSS (higher is better)"}.get(metric, metric.upper())
    ax.set_xlabel("Sample Size (n)", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale("log")
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(n) for n in sample_sizes])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_cd_diagram(
    avg_ranks: dict[str, float],
    cd: float,
    title: str,
    save_path: Path | None = None,
):
    """Plot Critical Difference diagram."""
    fig, ax = plt.subplots(figsize=(10, 3))

    methods = list(avg_ranks.keys())
    ranks = [avg_ranks[m] for m in methods]

    sorted_idx = np.argsort(ranks)
    methods = [methods[i] for i in sorted_idx]
    ranks = [ranks[i] for i in sorted_idx]

    n_methods = len(methods)
    y_positions = np.arange(n_methods)

    ax.scatter(ranks, y_positions, s=100, zorder=3)

    for i, (method, rank) in enumerate(zip(methods, ranks)):
        ax.annotate(
            f"{method.upper()} ({rank:.2f})",
            (rank, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )

    ax.axhline(y=-0.5, color="black", linewidth=2)
    ax.plot([1, 1 + cd], [-0.3, -0.3], "k-", linewidth=2)
    ax.annotate(f"CD = {cd:.2f}", (1 + cd / 2, -0.1), ha="center", fontsize=10)

    for i, (m1, r1) in enumerate(zip(methods, ranks)):
        for j, (m2, r2) in enumerate(zip(methods[i + 1 :], ranks[i + 1 :]), i + 1):
            if abs(r1 - r2) < cd:
                ax.plot([r1, r2], [i, j], "k-", alpha=0.3, linewidth=1)

    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-1, n_methods)
    ax.set_xlabel("Average Rank (lower is better)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved CD diagram to {save_path}")

    plt.close()


def plot_robustness(
    df: pd.DataFrame,
    metric: str,
    condition_col: str,
    title: str,
    save_path: Path | None = None,
):
    """Plot metric by condition (noise or imbalance) for each scoring method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = sorted(df[condition_col].unique())
    scoring_methods = sorted(df["scoring"].unique())

    x = np.arange(len(conditions))
    width = 0.18

    colors = {"nle": "#2ecc71", "nll": "#e74c3c", "nll_aic": "#9b59b6", "nll_bic": "#3498db"}

    for i, scoring in enumerate(scoring_methods):
        means = []
        stds = []
        for cond in conditions:
            subset = df[(df["scoring"] == scoring) & (df[condition_col] == cond)]
            means.append(subset[metric].mean())
            stds.append(subset[metric].std())

        offset = (i - len(scoring_methods) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=scoring.upper(),
            color=colors.get(scoring, "#cccccc"),
            alpha=0.8,
            capsize=3,
        )

    metric_label = {"crpss": "CRPSS (higher is better)", "bss": "BSS (higher is better)"}.get(metric, metric.upper())
    ax.set_xlabel(condition_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in conditions])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved robustness plot to {save_path}")

    plt.close()


# ============================================================================
# TABLES
# ============================================================================


def generate_summary_tables(
    df: pd.DataFrame,
    task: str,
    primary_metric: str,
) -> dict[str, pd.DataFrame]:
    """Generate summary tables for the study."""
    tables = {}
    # Higher skill score is better
    higher_is_better = primary_metric in ("crpss", "bss")

    # Table 1: Mean ± std per scoring method per sample size (with fit_time)
    summary_rows = []
    for n_samples in sorted(df["n_samples"].unique()):
        row = {"n_samples": n_samples}
        for scoring in sorted(df["scoring"].unique()):
            subset = df[(df["n_samples"] == n_samples) & (df["scoring"] == scoring)]
            if primary_metric in subset.columns and not subset[primary_metric].isna().all():
                mean = subset[primary_metric].mean()
                std = subset[primary_metric].std()
                fit_mean = subset["fit_time"].mean()
                row[scoring] = f"{mean:.4f} ± {std:.4f}"
                row[f"{scoring}_fit_s"] = f"{fit_mean:.2f}"
            else:
                row[scoring] = "N/A"
                row[f"{scoring}_fit_s"] = "N/A"
        summary_rows.append(row)

    tables["summary_by_n"] = pd.DataFrame(summary_rows)

    # Table 2: Best scoring method per sample size
    best_rows = []
    for n_samples in sorted(df["n_samples"].unique()):
        subset = df[df["n_samples"] == n_samples]
        if primary_metric not in subset.columns or subset[primary_metric].isna().all():
            continue
        means = subset.groupby("scoring")[primary_metric].mean()
        if higher_is_better:
            best_method = means.idxmax()
            best_value = means.max()
            second_best = means.drop(best_method).max()
            improvement = (best_value - second_best) / abs(second_best) * 100 if second_best != 0 else 0
        else:
            best_method = means.idxmin()
            best_value = means.min()
            second_best = means.drop(best_method).min()
            improvement = (second_best - best_value) / second_best * 100 if second_best > 0 else 0

        fit_means = subset.groupby("scoring")["fit_time"].mean()
        best_rows.append(
            {
                "n_samples": n_samples,
                "best_method": best_method,
                "best_value": f"{best_value:.4f}",
                "improvement_%": f"{improvement:.2f}%",
                "fit_time_s": f"{fit_means[best_method]:.2f}",
            }
        )

    tables["best_by_n"] = pd.DataFrame(best_rows)

    # Table 3: Recommendation table (fixed — regime-based guidance)
    tables["recommendation"] = pd.DataFrame(
        [
            {
                "Regime": r"$n < 500$, conjugate distribution",
                "Recommendation": "NLE",
                "Rationale": "Marginal likelihood regularises splits; NLL overfits at small $n$",
            },
            {
                "Regime": r"$n \geq 500$, conjugate distribution",
                "Recommendation": "NLL + BIC",
                "Rationale": "Cheaper per split; asymptotically equivalent to NLE",
            },
            {
                "Regime": "Non-conjugate (KDE / SkewNormal)",
                "Recommendation": "LOO-CV",
                "Rationale": "NLE unavailable; plug-in NLL overfits bandwidth; see future work",
            },
        ]
    )

    return tables


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    """Save DataFrame as LaTeX table."""
    latex = df.to_latex(index=False, caption=caption, label=label, escape=False)
    with open(path, "w") as f:
        f.write(latex)
    logger.info(f"Saved LaTeX table to {path}")


def generate_merged_real_data_table(
    reg_results: StudyResults | None,
    clas_results: StudyResults | None,
) -> pd.DataFrame:
    """One combined real-data table with task column (reg/clas rows).

    Columns: task, dataset, n, scoring, CRPSS/BSS, fit_time, predict_time.
    Replaces separate regression and classification LaTeX tables.
    """
    rows = []
    for task, results, skill_col in [
        ("regression", reg_results, "crpss"),
        ("classification", clas_results, "bss"),
    ]:
        if results is None:
            continue
        df = results.to_dataframe()
        if skill_col not in df.columns:
            continue
        for dataset in df["dgp"].unique():
            for scoring in df["scoring"].unique():
                subset = df[(df["dgp"] == dataset) & (df["scoring"] == scoring)]
                n = int(subset["n_samples"].iloc[0]) if len(subset) > 0 else -1
                skill_mean = subset[skill_col].mean()
                skill_std = subset[skill_col].std()
                fit_mean = subset["fit_time"].mean()
                predict_mean = subset["predict_time"].mean()
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "n": n,
                        "scoring": scoring,
                        "skill_score": f"{skill_mean:.4f} ± {skill_std:.4f}",
                        "fit_time_s": f"{fit_mean:.2f}",
                        "predict_time_s": f"{predict_mean:.2f}",
                    }
                )
    return pd.DataFrame(rows)


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================


def analyze_results(results: StudyResults, task: str, study_type: str):
    """Run full analysis on results."""
    df = results.to_dataframe()
    # Skill scores are primary; raw metrics kept as supplementary
    primary_metric = "crpss" if task == "regression" else "bss"
    task_dir = task if study_type != "real_data" else "real_data"

    logger.info(f"Analyzing {study_type} {task} results...")

    # Statistical tests
    stat_tests = compute_statistical_tests(df, primary_metric, groupby=["n_samples"])
    with open(RESULTS_DIR / task_dir / f"{study_type}_statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    # Interaction plot (for core study)
    if study_type == "core":
        plot_interaction(
            df,
            primary_metric,
            f"{primary_metric.upper()} vs Sample Size ({task.title()})",
            PLOTS_DIR / task_dir / f"fig1_{task}_interaction.pdf",
        )

    # CD diagram (overall)
    pivot_df = df.pivot_table(
        index=["dgp", "condition", "seed"],
        columns="scoring",
        values=primary_metric,
        aggfunc="first",
    ).dropna()

    if len(pivot_df) > 0:
        # Higher skill score is better → rank descending
        ranks = pivot_df.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean().to_dict()
        n_methods = len(avg_ranks)
        n_datasets = len(pivot_df)
        cd = compute_nemenyi_cd(n_methods, n_datasets)

        plot_cd_diagram(
            avg_ranks,
            cd,
            f"Critical Difference Diagram ({task.title()} - {study_type})",
            PLOTS_DIR / task_dir / f"fig3_cd_diagram_{study_type}.pdf",
        )

    # Robustness plot (for robustness study)
    if study_type == "robustness":
        condition_label = "Noise Level" if task == "regression" else "Class Imbalance"
        plot_robustness(
            df,
            primary_metric,
            "condition",
            f"{primary_metric.upper()} by {condition_label} ({task.title()})",
            PLOTS_DIR / task_dir / f"fig4_{task}_robustness.pdf",
        )

    # Tables
    tables = generate_summary_tables(df, task, primary_metric)
    for name, table_df in tables.items():
        table_df.to_csv(RESULTS_DIR / task_dir / f"{study_type}_{name}.csv", index=False)
        save_latex_table(
            table_df,
            TABLES_DIR / task_dir / f"{study_type}_{name}.tex",
            caption=f"{task.title()} {study_type} results: {name}",
            label=f"tab:{task}_{study_type}_{name}",
        )

    logger.info(f"{study_type.title()} {task} analysis complete.")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Scoring Method Ablation Study for JMLR")
    parser.add_argument(
        "task",
        nargs="?",
        default="both",
        choices=["regression", "classification", "both"],
        help="Which task to run (default: both)",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test run with reduced seeds and sample sizes")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume from existing results")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis on existing results")
    parser.add_argument("--core", action="store_true", help="Run only core study (sample size × scoring)")
    parser.add_argument("--robustness", action="store_true", help="Run only robustness study (noise/imbalance)")
    parser.add_argument("--real-data", action="store_true", help="Run only real data validation")

    args = parser.parse_args()

    # Determine which studies to run
    run_core = args.core or (not args.robustness and not args.real_data)
    run_robustness = args.robustness or (not args.core and not args.real_data)
    run_real = args.real_data or (not args.core and not args.robustness)

    # If no specific study selected, run all
    if not args.core and not args.robustness and not args.real_data:
        run_core = run_robustness = run_real = True

    # Configuration
    if args.quick:
        n_seeds = QUICK_N_SEEDS
        sample_sizes = QUICK_SAMPLE_SIZES
        logger.info("Running in quick mode")
    else:
        n_seeds = N_SEEDS
        sample_sizes = SAMPLE_SIZES

    resume = not args.no_resume
    tasks = ["regression", "classification"] if args.task == "both" else [args.task]

    real_data_results: dict[str, StudyResults] = {}

    for task in tasks:
        # Core study
        if run_core:
            if not args.analyze_only:
                results = run_core_study(task, n_seeds, sample_sizes, resume)
            else:
                results_path = RESULTS_DIR / task / "core_results.parquet"
                if results_path.exists():
                    results = StudyResults.load(results_path, task)
                else:
                    logger.error(f"No core {task} results found for analysis")
                    continue
            analyze_results(results, task, "core")

        # Robustness study
        if run_robustness:
            if not args.analyze_only:
                results = run_robustness_study(task, n_seeds=10, resume=resume)
            else:
                results_path = RESULTS_DIR / task / "robustness_results.parquet"
                if results_path.exists():
                    results = StudyResults.load(results_path, task)
                else:
                    logger.error(f"No robustness {task} results found for analysis")
                    continue
            analyze_results(results, task, "robustness")

        # Real data study (n_seeds=3, n_folds=5 → 15 runs/dataset/method)
        if run_real:
            if not args.analyze_only:
                results = run_real_data_study(task, n_folds=5, n_seeds=3, resume=resume)
            else:
                results_path = RESULTS_DIR / "real_data" / f"{task}_results.parquet"
                if results_path.exists():
                    results = StudyResults.load(results_path, task)
                else:
                    logger.error(f"No real data {task} results found for analysis")
                    continue
            analyze_results(results, task, "real_data")
            real_data_results[task] = results

    # Merged real-data table (replaces separate reg/clas LaTeX tables)
    if run_real and real_data_results:
        merged_df = generate_merged_real_data_table(
            reg_results=real_data_results.get("regression"),
            clas_results=real_data_results.get("classification"),
        )
        if not merged_df.empty:
            merged_df.to_csv(RESULTS_DIR / "real_data" / "merged_real_data.csv", index=False)
            save_latex_table(
                merged_df,
                TABLES_DIR / "real_data" / "merged_real_data.tex",
                caption=(
                    "Real-data results (5-fold CV, 3 seeds). Skill score column is CRPSS for regression "
                    "and BSS for classification. Hyperparameters are held fixed across scoring methods "
                    "to isolate split-selection effects."
                ),
                label="tab:a2_real_data",
            )

    logger.info("Study complete!")


if __name__ == "__main__":
    main()
