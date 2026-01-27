"""Scoring Method Ablation Study.

Dedicated study investigating when NLE (Bayesian marginal likelihood) outperforms
NLL-based scoring for tree split selection in BDF.

Key research question:
"Under what conditions does NLE outperform NLL-based scoring, and by how much?"

Hypotheses:
1. NLE dominates at small n (n ≤ 250)
2. NLE and NLL converge at large n (n ≥ 1000)
3. NLE provides better calibration (lower ECE)
4. NLE is more robust to noise
5. NLE handles class imbalance better

Usage:
    pixi run python benchmarks/scoring_method_study.py              # Both tasks
    pixi run python benchmarks/scoring_method_study.py regression   # Regression only
    pixi run python benchmarks/scoring_method_study.py classification  # Classification only
    pixi run python benchmarks/scoring_method_study.py --quick      # Quick test run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scoringrules
from scipy import stats
from sklearn.datasets import (
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_moons,
    make_regression,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from bdf.tree_classes.bdf_regressor import BDFClassifier, BDFRegressor

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

# Experiment configuration
SEED = 42
N_SEEDS = 30  # For statistical power
SAMPLE_SIZES = [50, 100, 250, 500, 1000, 2500]
TEST_FRACTION = 0.2

# Quick mode configuration (for testing)
QUICK_N_SEEDS = 3
QUICK_SAMPLE_SIZES = [100, 500]

# Scoring method configurations
SCORING_CONFIGS = {
    "nle": {"score_method": "nle", "score_correction": None},
    "nll": {"score_method": "nll", "score_correction": None},
    "nll_aic": {"score_method": "nll", "score_correction": "aic"},
    "nll_bic": {"score_method": "nll", "score_correction": "bic"},
    "nll_loo": {"score_method": "nll", "score_correction": "loo_cv"},
}

# Model hyperparameters (fixed, not ablated)
MODEL_CONFIG = {
    "n_trees": 50,
    "max_depth": 50,
    "min_samples_leaf": 10,
    "reg_lambda": 0.01,
    "reg_gamma": 0.1,
    "reg_nu": 0.01,
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        # Add noise scaled to target std
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

    elif dgp == "friedman2":
        X, y = make_friedman2(n_samples=n_samples, noise=0, random_state=seed)
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
        # Generate balanced, then undersample
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
        return X, y  # Already balanced or minority is majority

    rng = np.random.default_rng(seed)

    # Identify classes
    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    # Determine which is minority (should have fewer samples after imbalancing)
    n_minority = int(len(y) * minority_ratio)
    n_majority = len(y) - n_minority

    # Undersample majority class
    if len(class_0_idx) > len(class_1_idx):
        # Class 0 is majority
        majority_idx = rng.choice(class_0_idx, size=n_majority, replace=False)
        minority_idx = class_1_idx[:n_minority] if len(class_1_idx) >= n_minority else class_1_idx
    else:
        # Class 1 is majority
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
) -> dict[str, float]:
    """Compute all regression metrics."""
    metrics = {}

    # Point predictions
    y_pred = model.predict(X_test)
    metrics["rmse"] = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    metrics["mae"] = float(np.mean(np.abs(y_test - y_pred)))

    # Probabilistic predictions
    try:
        # Get samples for CRPS
        samples = model.predict_samples(X_test, n_samples=500)
        metrics["crps"] = float(np.mean(scoringrules.crps_ensemble(y_test, samples)))
    except Exception:
        metrics["crps"] = np.nan

    try:
        # Coverage and interval metrics using quantiles
        quantiles = model.predict_quantiles(X_test, q=[0.05, 0.95], n_samples=500)
        lower, upper = quantiles[:, 0], quantiles[:, 1]
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        metrics["coverage_90"] = float(coverage)
        metrics["interval_width_90"] = float(np.mean(upper - lower))

        # Interval score (proper scoring rule for intervals)
        alpha = 0.1
        metrics["interval_score_90"] = float(
            np.mean(
                (upper - lower)
                + (2 / alpha) * (lower - y_test) * (y_test < lower)
                + (2 / alpha) * (y_test - upper) * (y_test > upper)
            )
        )
    except Exception:
        metrics["coverage_90"] = np.nan
        metrics["interval_width_90"] = np.nan
        metrics["interval_score_90"] = np.nan

    return metrics


def compute_classification_metrics(
    model: BDFClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute all classification metrics."""
    metrics = {}

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Handle binary case
    if y_prob.ndim == 2:
        y_prob_pos = y_prob[:, 1]
    else:
        y_prob_pos = y_prob

    # Point metrics
    metrics["accuracy"] = float(np.mean(y_pred == y_test))

    # Probabilistic metrics
    # Log loss (cross-entropy)
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

    # Calibration metrics
    metrics["ece"] = _compute_ece(y_test, y_prob_pos, n_bins=10)
    metrics["mce"] = _compute_mce(y_test, y_prob_pos, n_bins=10)

    return metrics


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include right boundary for last bin
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            ece += np.sum(mask) / n * np.abs(bin_accuracy - bin_confidence)

    return float(ece)


def _compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            mce = max(mce, np.abs(bin_accuracy - bin_confidence))

    return float(mce)


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================


@dataclass
class ExperimentResult:
    """Single experiment result."""

    task: str
    dgp: str
    n_samples: int
    noise_or_imbalance: str
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
                "condition": r.noise_or_imbalance,
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
                if k
                not in [
                    "task",
                    "dgp",
                    "n_samples",
                    "condition",
                    "scoring",
                    "seed",
                    "fit_time",
                    "predict_time",
                ]
            }
            results.results.append(
                ExperimentResult(
                    task=row["task"],
                    dgp=row["dgp"],
                    n_samples=row["n_samples"],
                    noise_or_imbalance=row["condition"],
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
    # Generate data
    X_train, X_test, y_train, y_test = generate_regression_data(dgp, n_samples, noise_level, seed)

    # Build distribution params with scoring config
    dist_params = REG_DIST_PARAMS.copy()
    dist_params.update(SCORING_CONFIGS[scoring_name])

    # Create and fit model
    model = BDFRegressor(dist="NormalMuNormal", params=dist_params, **MODEL_CONFIG)

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    metrics = compute_regression_metrics(model, X_test, y_test)
    predict_time = time.time() - start_time

    return ExperimentResult(
        task="regression",
        dgp=dgp,
        n_samples=n_samples,
        noise_or_imbalance=noise_level,
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
    # Generate data
    X_train, X_test, y_train, y_test = generate_classification_data(dgp, n_samples, imbalance_level, seed)

    # Build distribution params with scoring config
    dist_params = CLAS_DIST_PARAMS.copy()
    dist_params.update(SCORING_CONFIGS[scoring_name])

    # Create and fit model
    model = BDFClassifier(dist="BetaABBernoulli", params=dist_params, **MODEL_CONFIG)

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    metrics = compute_classification_metrics(model, X_test, y_test)
    predict_time = time.time() - start_time

    return ExperimentResult(
        task="classification",
        dgp=dgp,
        n_samples=n_samples,
        noise_or_imbalance=imbalance_level,
        scoring_method=scoring_name,
        seed=seed,
        metrics=metrics,
        fit_time=fit_time,
        predict_time=predict_time,
    )


def run_regression_study(
    n_seeds: int = N_SEEDS,
    sample_sizes: list[int] | None = None,
    resume: bool = True,
) -> StudyResults:
    """Run full regression scoring method study."""
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES

    results_path = RESULTS_DIR / "regression" / "raw_results.parquet"

    # Try to resume from existing results
    if resume and results_path.exists():
        logger.info(f"Loading existing results from {results_path}")
        results = StudyResults.load(results_path, "regression")
        completed = {(r.dgp, r.n_samples, r.noise_or_imbalance, r.scoring_method, r.seed) for r in results.results}
    else:
        results = StudyResults(
            task="regression",
            config={
                "sample_sizes": sample_sizes,
                "n_seeds": n_seeds,
                "scoring_configs": list(SCORING_CONFIGS.keys()),
                "noise_levels": list(NOISE_LEVELS.keys()),
                "dgps": ["friedman1", "linear"],
            },
        )
        completed = set()

    dgps = ["friedman1", "linear"]
    noise_levels = list(NOISE_LEVELS.keys())
    scoring_methods = list(SCORING_CONFIGS.keys())

    total_experiments = len(dgps) * len(sample_sizes) * len(noise_levels) * len(scoring_methods) * n_seeds
    remaining = total_experiments - len(completed)

    logger.info(f"Regression study: {remaining} experiments remaining of {total_experiments}")

    pbar = tqdm(total=remaining, desc="Regression experiments")

    for dgp in dgps:
        for n_samples in sample_sizes:
            for noise_level in noise_levels:
                for scoring_name in scoring_methods:
                    for seed in range(SEED, SEED + n_seeds):
                        key = (dgp, n_samples, noise_level, scoring_name, seed)
                        if key in completed:
                            continue

                        try:
                            result = run_regression_experiment(dgp, n_samples, noise_level, scoring_name, seed)
                            results.results.append(result)
                        except Exception as e:
                            logger.warning(f"Experiment failed {key}: {e}")

                        pbar.update(1)

                # Save checkpoint after each n_samples × noise_level block
                results.save(results_path)

    pbar.close()
    return results


def run_classification_study(
    n_seeds: int = N_SEEDS,
    sample_sizes: list[int] | None = None,
    resume: bool = True,
) -> StudyResults:
    """Run full classification scoring method study."""
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES

    results_path = RESULTS_DIR / "classification" / "raw_results.parquet"

    # Try to resume from existing results
    if resume and results_path.exists():
        logger.info(f"Loading existing results from {results_path}")
        results = StudyResults.load(results_path, "classification")
        completed = {(r.dgp, r.n_samples, r.noise_or_imbalance, r.scoring_method, r.seed) for r in results.results}
    else:
        results = StudyResults(
            task="classification",
            config={
                "sample_sizes": sample_sizes,
                "n_seeds": n_seeds,
                "scoring_configs": list(SCORING_CONFIGS.keys()),
                "imbalance_levels": list(IMBALANCE_LEVELS.keys()),
                "dgps": ["make_classification", "moons", "circles"],
            },
        )
        completed = set()

    dgps = ["make_classification", "moons", "circles"]
    imbalance_levels = list(IMBALANCE_LEVELS.keys())
    # Skip LOO-CV for classification (slow without fast implementation)
    scoring_methods = [s for s in SCORING_CONFIGS.keys() if s != "nll_loo"]

    total_experiments = len(dgps) * len(sample_sizes) * len(imbalance_levels) * len(scoring_methods) * n_seeds
    remaining = total_experiments - len(completed)

    logger.info(f"Classification study: {remaining} experiments remaining of {total_experiments}")

    pbar = tqdm(total=remaining, desc="Classification experiments")

    for dgp in dgps:
        for n_samples in sample_sizes:
            for imbalance_level in imbalance_levels:
                for scoring_name in scoring_methods:
                    for seed in range(SEED, SEED + n_seeds):
                        key = (dgp, n_samples, imbalance_level, scoring_name, seed)
                        if key in completed:
                            continue

                        try:
                            result = run_classification_experiment(dgp, n_samples, imbalance_level, scoring_name, seed)
                            results.results.append(result)
                        except Exception as e:
                            logger.warning(f"Experiment failed {key}: {e}")

                        pbar.update(1)

                # Save checkpoint
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

        # Pivot to get scoring methods as columns
        # Each row is a unique (dgp, condition, seed) combination
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

        # Average ranks for CD diagram
        ranks = pivot_df.rank(axis=1, ascending=True)  # Lower is better
        avg_ranks = ranks.mean().to_dict()
        results[f"{group_key}_avg_ranks"] = avg_ranks

        # Pairwise Wilcoxon tests (NLE vs others)
        if "nle" in scoring_methods:
            for other in scoring_methods:
                if other == "nle":
                    continue
                try:
                    stat, pval = stats.wilcoxon(
                        pivot_df["nle"].values,
                        pivot_df[other].values,
                        alternative="less",  # NLE < other (lower is better)
                    )
                    # Cliff's delta effect size
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
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    more = int(np.sum(x[:, None] > y[None, :]))
    less = int(np.sum(x[:, None] < y[None, :]))
    return (more - less) / (n_x * n_y)


def compute_nemenyi_cd(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Compute critical difference for Nemenyi test."""
    # q_alpha values for Nemenyi test (from tables)
    # Approximate using Studentized range distribution
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

    colors = {
        "nle": "#2ecc71",
        "nll": "#e74c3c",
        "nll_aic": "#9b59b6",
        "nll_bic": "#3498db",
        "nll_loo": "#f39c12",
    }

    markers = {
        "nle": "o",
        "nll": "s",
        "nll_aic": "^",
        "nll_bic": "D",
        "nll_loo": "v",
    }

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

        means = np.array(means)
        stds = np.array(stds)

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

    ax.set_xlabel("Sample Size (n)", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale("log")
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(n) for n in sample_sizes])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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

    # Sort by rank
    sorted_idx = np.argsort(ranks)
    methods = [methods[i] for i in sorted_idx]
    ranks = [ranks[i] for i in sorted_idx]

    n_methods = len(methods)
    y_positions = np.arange(n_methods)

    # Plot ranks
    ax.scatter(ranks, y_positions, s=100, zorder=3)

    # Add method labels
    for i, (method, rank) in enumerate(zip(methods, ranks)):
        ax.annotate(
            f"{method.upper()} ({rank:.2f})",
            (rank, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )

    # Draw CD bar
    ax.axhline(y=-0.5, color="black", linewidth=2)
    ax.plot([1, 1 + cd], [-0.3, -0.3], "k-", linewidth=2)
    ax.annotate(f"CD = {cd:.2f}", (1 + cd / 2, -0.1), ha="center", fontsize=10)

    # Draw connections for methods not significantly different
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
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved CD diagram to {save_path}")

    plt.close()


def plot_calibration_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    save_path: Path | None = None,
):
    """Plot boxplot of calibration metric by scoring method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scoring_methods = sorted(df["scoring"].unique())
    data = [df[df["scoring"] == s][metric].dropna().values for s in scoring_methods]

    colors = {
        "nle": "#2ecc71",
        "nll": "#e74c3c",
        "nll_aic": "#9b59b6",
        "nll_bic": "#3498db",
        "nll_loo": "#f39c12",
    }

    bp = ax.boxplot(data, tick_labels=[s.upper() for s in scoring_methods], patch_artist=True)

    for patch, method in zip(bp["boxes"], scoring_methods):
        patch.set_facecolor(colors.get(method, "#cccccc"))
        patch.set_alpha(0.7)

    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved boxplot to {save_path}")

    plt.close()


def plot_condition_interaction(
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
    width = 0.15

    colors = {
        "nle": "#2ecc71",
        "nll": "#e74c3c",
        "nll_aic": "#9b59b6",
        "nll_bic": "#3498db",
        "nll_loo": "#f39c12",
    }

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

    ax.set_xlabel(condition_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in conditions])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved interaction plot to {save_path}")

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

    # Table 1: Mean ± std per scoring method per sample size
    summary_rows = []
    for n_samples in sorted(df["n_samples"].unique()):
        row = {"n_samples": n_samples}
        for scoring in sorted(df["scoring"].unique()):
            subset = df[(df["n_samples"] == n_samples) & (df["scoring"] == scoring)]
            if primary_metric in subset.columns:
                mean = subset[primary_metric].mean()
                std = subset[primary_metric].std()
                row[scoring] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[scoring] = "N/A"
        summary_rows.append(row)

    tables["summary_by_n"] = pd.DataFrame(summary_rows)

    # Table 2: Best scoring method per sample size
    best_rows = []
    for n_samples in sorted(df["n_samples"].unique()):
        subset = df[df["n_samples"] == n_samples]
        means = subset.groupby("scoring")[primary_metric].mean()
        best_method = means.idxmin()
        best_value = means.min()
        second_best = means.drop(best_method).min()
        improvement = (second_best - best_value) / second_best * 100

        best_rows.append(
            {
                "n_samples": n_samples,
                "best_method": best_method,
                "best_value": f"{best_value:.4f}",
                "improvement_%": f"{improvement:.2f}%",
            }
        )

    tables["best_by_n"] = pd.DataFrame(best_rows)

    return tables


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    """Save DataFrame as LaTeX table."""
    latex = df.to_latex(index=False, caption=caption, label=label, escape=False)
    with open(path, "w") as f:
        f.write(latex)
    logger.info(f"Saved LaTeX table to {path}")


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================


def analyze_regression_results(results: StudyResults):
    """Run full analysis on regression results."""
    df = results.to_dataframe()
    primary_metric = "crps"
    task_dir = "regression"

    logger.info("Analyzing regression results...")

    # Statistical tests
    stat_tests = compute_statistical_tests(df, primary_metric, groupby=["n_samples"])
    with open(RESULTS_DIR / task_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    # Plots
    plot_interaction(
        df,
        primary_metric,
        "CRPS vs Sample Size (Regression)",
        PLOTS_DIR / task_dir / "interaction_crps.png",
    )

    # CD diagram (overall)
    pivot_df = df.pivot_table(
        index=["dgp", "condition", "seed", "n_samples"],
        columns="scoring",
        values=primary_metric,
        aggfunc="first",
    ).dropna()

    if len(pivot_df) > 0:
        ranks = pivot_df.rank(axis=1, ascending=True)
        avg_ranks = ranks.mean().to_dict()
        n_methods = len(avg_ranks)
        n_datasets = len(pivot_df)
        cd = compute_nemenyi_cd(n_methods, n_datasets)

        plot_cd_diagram(
            avg_ranks,
            cd,
            "Critical Difference Diagram (Regression)",
            PLOTS_DIR / task_dir / "cd_diagram_overall.png",
        )

    # Calibration
    plot_calibration_boxplot(
        df,
        "coverage_90",
        "90% Coverage by Scoring Method (Regression)",
        PLOTS_DIR / task_dir / "calibration_coverage.png",
    )

    # Noise interaction
    plot_condition_interaction(
        df,
        primary_metric,
        "condition",
        "CRPS by Noise Level (Regression)",
        PLOTS_DIR / task_dir / "noise_interaction.png",
    )

    # Tables
    tables = generate_summary_tables(df, "regression", primary_metric)
    for name, table_df in tables.items():
        table_df.to_csv(RESULTS_DIR / task_dir / f"{name}.csv", index=False)

    logger.info("Regression analysis complete.")


def analyze_classification_results(results: StudyResults):
    """Run full analysis on classification results."""
    df = results.to_dataframe()
    primary_metric = "log_loss"
    task_dir = "classification"

    logger.info("Analyzing classification results...")

    # Statistical tests
    stat_tests = compute_statistical_tests(df, primary_metric, groupby=["n_samples"])
    with open(RESULTS_DIR / task_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    # Plots
    plot_interaction(
        df,
        primary_metric,
        "Log Loss vs Sample Size (Classification)",
        PLOTS_DIR / task_dir / "interaction_logloss.png",
    )

    # CD diagram (overall)
    pivot_df = df.pivot_table(
        index=["dgp", "condition", "seed", "n_samples"],
        columns="scoring",
        values=primary_metric,
        aggfunc="first",
    ).dropna()

    if len(pivot_df) > 0:
        ranks = pivot_df.rank(axis=1, ascending=True)
        avg_ranks = ranks.mean().to_dict()
        n_methods = len(avg_ranks)
        n_datasets = len(pivot_df)
        cd = compute_nemenyi_cd(n_methods, n_datasets)

        plot_cd_diagram(
            avg_ranks,
            cd,
            "Critical Difference Diagram (Classification)",
            PLOTS_DIR / task_dir / "cd_diagram_overall.png",
        )

    # Calibration
    plot_calibration_boxplot(
        df,
        "ece",
        "ECE by Scoring Method (Classification)",
        PLOTS_DIR / task_dir / "calibration_ece.png",
    )

    # Imbalance interaction
    plot_condition_interaction(
        df,
        primary_metric,
        "condition",
        "Log Loss by Class Imbalance (Classification)",
        PLOTS_DIR / task_dir / "imbalance_interaction.png",
    )

    # Tables
    tables = generate_summary_tables(df, "classification", primary_metric)
    for name, table_df in tables.items():
        table_df.to_csv(RESULTS_DIR / task_dir / f"{name}.csv", index=False)

    logger.info("Classification analysis complete.")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Scoring Method Ablation Study")
    parser.add_argument(
        "task",
        nargs="?",
        default="both",
        choices=["regression", "classification", "both"],
        help="Which task to run (default: both)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with reduced seeds and sample sizes",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from existing results",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on existing results",
    )

    args = parser.parse_args()

    # Configuration
    if args.quick:
        n_seeds = QUICK_N_SEEDS
        sample_sizes = QUICK_SAMPLE_SIZES
        logger.info("Running in quick mode")
    else:
        n_seeds = N_SEEDS
        sample_sizes = SAMPLE_SIZES

    resume = not args.no_resume

    # Run experiments
    if args.task in ["regression", "both"]:
        if not args.analyze_only:
            results = run_regression_study(n_seeds, sample_sizes, resume)
        else:
            results_path = RESULTS_DIR / "regression" / "raw_results.parquet"
            if results_path.exists():
                results = StudyResults.load(results_path, "regression")
            else:
                logger.error("No regression results found for analysis")
                return

        analyze_regression_results(results)

    if args.task in ["classification", "both"]:
        if not args.analyze_only:
            results = run_classification_study(n_seeds, sample_sizes, resume)
        else:
            results_path = RESULTS_DIR / "classification" / "raw_results.parquet"
            if results_path.exists():
                results = StudyResults.load(results_path, "classification")
            else:
                logger.error("No classification results found for analysis")
                return

        analyze_classification_results(results)

    logger.info("Study complete!")


if __name__ == "__main__":
    main()
