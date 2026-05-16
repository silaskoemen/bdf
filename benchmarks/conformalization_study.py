"""Conformalization study: adaptive vs non-adaptive uncertainty under conformal correction.

Key question: does conformalization fix marginal but not conditional coverage?

Expected result:
- CatBoostUncertainty: undercovers globally *and* conditionally
- ConfCatBoost (additive): fixes global coverage but fails conditionally (non-adaptive residuals)
- CQRCatBoost (CQR): fixes global coverage and partially improves conditional coverage,
  but inherits CatBoost's limited local adaptivity
- BDFNormal: better conditional coverage even without conformalization
- ConfBDFNormal: near-nominal conditional coverage across all uncertainty deciles

Design:
- Same DGPs as synthetic_dgp_benchmark.py
- Fixed hyperparameters (no Optuna tuning — conformal wrappers internalize calibration)
- 10-fold CV, all folds evaluated (no held-out tuning fold)
- Conformal calibration is internal to each wrapper (train/cal split inside fit())
- Output: benchmarks/results/conformalization/conformalization_results.json
"""

import json
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from loguru import logger
from sklearn.model_selection import KFold
from tqdm import tqdm

from .metrics.regression import crps_wrapper
from .pipeline.synthetic_dgps import DGP_REGISTRY
from .synthetic_dgp_benchmark import DGPS_TO_RUN, N_UNCERTAINTY_BINS, SEED

N_POSTERIOR_SAMPLES = 500
N_SPLITS = 10

OUTPUT_DIR = Path("benchmarks/results/conformalization")

# Model sets by environment — used to detect which env we are in and suffix outputs
_CORE_MODEL_NAMES = {"BDFNormal", "ConfBDFNormal", "BDFKDE", "ConfBDFKDE"}
_BENCH_MODEL_NAMES = {"CatBoostUncertainty", "ConfCatBoost", "CQRCatBoost", "ConformalRF"}


# =============================================================================
# Model registry
# =============================================================================


def get_conformalization_models() -> dict[str, dict[str, Any]]:
    """Build model configs available in the current environment."""
    available: dict[str, dict[str, Any]] = {}

    # BDF (raw — no conformalization)
    try:
        from bdf.tree_classes.bdf_regressor import BDFRegressor

        available["BDFNormal"] = {
            "class": BDFRegressor,
            "init_kwargs": {"dist": "NormalMuNormal", "random_state": SEED, "n_trees": 50, "min_samples_leaf": 10},
            "params": {"mu_mu": "auto", "score_correction": "bic"},
            "probabilistic": True,
        }
        logger.info("✓ BDFNormal available")
    except ImportError:
        logger.info("✗ BDF not available")

    # BDF KDE (raw — no conformalization)
    try:
        from bdf.tree_classes.bdf_regressor import BDFRegressor

        available["BDFKDE"] = {
            "class": BDFRegressor,
            "init_kwargs": {"dist": "KDE", "random_state": SEED, "n_trees": 50, "min_samples_leaf": 10},
            "params": {"score_correction": "bic"},
            "probabilistic": True,
        }
        logger.info("✓ BDFKDE available")
    except ImportError:
        pass

    # ConfBDF (additive split-conformal on BDF)
    try:
        from bdf.tree_classes.bdf_regressor import BDFRegressor

        from .models.bdf_wrappers import ConformalizedSamplesWrapper

        available["ConfBDFNormal"] = {
            "class": ConformalizedSamplesWrapper,
            "init_kwargs": {
                "base_model_class": BDFRegressor,
                "base_model_init_kwargs": {
                    "dist": "NormalMuNormal",
                    "random_state": SEED,
                    "n_trees": 50,
                    "min_samples_leaf": 10,
                },
                "base_model_params": {"mu_mu": "auto", "score_correction": "bic"},
                "cal_fraction": 0.2,
                "random_state": SEED,
            },
            "params": {},
            "probabilistic": True,
        }
        logger.info("✓ ConfBDFNormal available")
    except ImportError:
        logger.info("✗ ConfBDFNormal not available")

    # ConfBDF KDE (additive split-conformal on BDF KDE)
    try:
        from bdf.tree_classes.bdf_regressor import BDFRegressor

        from .models.bdf_wrappers import ConformalizedSamplesWrapper

        available["ConfBDFKDE"] = {
            "class": ConformalizedSamplesWrapper,
            "init_kwargs": {
                "base_model_class": BDFRegressor,
                "base_model_init_kwargs": {
                    "dist": "KDE",
                    "random_state": SEED,
                    "n_trees": 50,
                    "min_samples_leaf": 10,
                },
                "base_model_params": {"score_correction": "bic"},
                "cal_fraction": 0.2,
                "random_state": SEED,
            },
            "params": {},
            "probabilistic": True,
        }
        logger.info("✓ ConfBDFKDE available")
    except ImportError:
        logger.info("✗ ConfBDFKDE not available")

    # CatBoost with uncertainty (raw)
    try:
        from .models.wrappers import CatBoostUncertaintyWrapper

        available["CatBoostUncertainty"] = {
            "class": CatBoostUncertaintyWrapper,
            "init_kwargs": {
                "loss_function": "RMSEWithUncertainty",
                "random_seed": SEED,
                "verbose": 0,
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
            },
            "params": {},
            "probabilistic": True,
        }
        logger.info("✓ CatBoostUncertainty available")
    except ImportError:
        logger.info("✗ CatBoostUncertainty not available (run in bench-models environment)")

    # ConfCatBoost (additive split-conformal on CatBoost)
    try:
        from .models.wrappers import ConformalizedCatBoostWrapper

        available["ConfCatBoost"] = {
            "class": ConformalizedCatBoostWrapper,
            "init_kwargs": {
                "random_state": SEED,
                "loss_function": "RMSEWithUncertainty",
                "verbose": 0,
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
            },
            "params": {},
            "probabilistic": True,
        }
        logger.info("✓ ConfCatBoost available")
    except ImportError:
        logger.info("✗ ConfCatBoost not available")

    # CQR-CatBoost (adaptive CQR on CatBoost uncertainty)
    try:
        from .models.wrappers import CQRCatBoostWrapper

        available["CQRCatBoost"] = {
            "class": CQRCatBoostWrapper,
            "init_kwargs": {
                "alpha": 0.10,
                "random_state": SEED,
                "loss_function": "RMSEWithUncertainty",
                "verbose": 0,
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
            },
            "params": {},
            "probabilistic": True,
        }
        logger.info("✓ CQRCatBoost available")
    except ImportError:
        logger.info("✗ CQRCatBoost not available")

    # ConformalRF (baseline conformal method)
    try:
        from .models.wrappers import ConformalizedRFWrapper

        available["ConformalRF"] = {
            "class": ConformalizedRFWrapper,
            "init_kwargs": {"random_state": SEED, "n_estimators": 100},
            "params": {},
            "probabilistic": True,
        }
        logger.info("✓ ConformalRF available")
    except ImportError:
        logger.info("✗ ConformalRF not available")

    return available


# =============================================================================
# Fold evaluation
# =============================================================================


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Train and evaluate one model on one fold.

    Returns:
        (metrics_dict, fit_time_seconds)
    """
    cls = config["class"]
    init_kwargs = config.get("init_kwargs", {})
    params = config.get("params", {})

    np.random.seed(SEED)

    if params:
        model = cls(**init_kwargs, params=params)
    else:
        model = cls(**init_kwargs)

    t0 = time()
    model.fit(X_train, y_train)
    fit_time = time() - t0

    y_pred = model.predict(X_test)
    y_samples = model.predict_samples(X_test, n_samples=N_POSTERIOR_SAMPLES)
    y_pred_std = np.std(y_samples, axis=-1)

    metrics: dict[str, Any] = {
        "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        "crps": float(crps_wrapper(y_test, y_samples)),
    }

    # Global coverage and interval width at 50 / 90 / 95 %
    for width in [0.50, 0.90, 0.95]:
        alpha = 1 - width
        lo = np.quantile(y_samples, alpha / 2, axis=-1)
        hi = np.quantile(y_samples, 1 - alpha / 2, axis=-1)
        metrics[f"global_coverage_{int(width * 100)}"] = float(np.mean((y_test >= lo) & (y_test <= hi)))
        metrics[f"global_width_{int(width * 100)}"] = float(np.mean(hi - lo))

    # Conditional calibration by predicted-uncertainty decile
    bin_edges = np.percentile(y_pred_std, np.linspace(0, 100, N_UNCERTAINTY_BINS + 1))
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) >= 2:
        bin_indices = np.digitize(y_pred_std, bin_edges[1:-1])

        cond_cov_90: list[float] = []
        cond_width_90: list[float] = []
        spread_pred_std: list[float] = []
        spread_rmse: list[float] = []

        for b in range(N_UNCERTAINTY_BINS):
            mask = bin_indices == b
            if mask.sum() < 5:
                cond_cov_90.append(float("nan"))
                cond_width_90.append(float("nan"))
                spread_pred_std.append(float("nan"))
                spread_rmse.append(float("nan"))
                continue

            y_test_b = y_test[mask]
            y_samp_b = y_samples[mask]
            y_pred_b = y_pred[mask]

            lo90 = np.quantile(y_samp_b, 0.05, axis=-1)
            hi90 = np.quantile(y_samp_b, 0.95, axis=-1)
            cond_cov_90.append(float(np.mean((y_test_b >= lo90) & (y_test_b <= hi90))))
            cond_width_90.append(float(np.mean(hi90 - lo90)))
            spread_pred_std.append(float(np.mean(y_pred_std[mask])))
            spread_rmse.append(float(np.sqrt(np.mean((y_test_b - y_pred_b) ** 2))))

        # Pad to N_UNCERTAINTY_BINS if bin_edges collapsed
        for lst in (cond_cov_90, cond_width_90, spread_pred_std, spread_rmse):
            while len(lst) < N_UNCERTAINTY_BINS:
                lst.append(float("nan"))

        metrics["cond_cov_90_by_bin"] = cond_cov_90
        metrics["cond_width_90_by_bin"] = cond_width_90
        metrics["spread_pred_std_by_bin"] = spread_pred_std
        metrics["spread_rmse_by_bin"] = spread_rmse

    return metrics, fit_time


# =============================================================================
# Result aggregation
# =============================================================================


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Average metrics across folds, handling list-valued metrics with nanmean."""
    if not fold_metrics:
        return {}

    aggregated: dict[str, Any] = {}
    all_keys = set().union(*[m.keys() for m in fold_metrics])

    for key in all_keys:
        raw = [m[key] for m in fold_metrics if key in m]
        if not raw:
            continue

        if isinstance(raw[0], list):
            arr = np.array(raw, dtype=float)
            aggregated[key] = {
                "mean": np.nanmean(arr, axis=0).tolist(),
                "std": np.nanstd(arr, axis=0).tolist(),
            }
        else:
            vals = [v for v in raw if not np.isnan(float(v))]
            if vals:
                aggregated[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return aggregated


# =============================================================================
# Main benchmark loop
# =============================================================================


def _env_suffix(models: dict[str, Any]) -> str:
    """Return file suffix based on which models are present.

    _core  — only BDF and ConfBDF (default pixi environment)
    _full  — includes benchmark-env models (CatBoost variants, ConformalRF)
    """
    model_set = set(models.keys())
    has_bench = bool(model_set & _BENCH_MODEL_NAMES)
    return "" if has_bench else "_core"


def run_conformalization_study(
    dgps: list[dict],
    models: dict[str, dict[str, Any]],
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the conformalization study across all DGPs and models.

    Automatically detects the environment from available models and appends
    ``_core`` to output filenames when running without benchmark-env models
    (mirrors the ``env_suffix`` convention in synthetic_dgp_benchmark.py).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _env_suffix(models)

    all_results: dict[str, Any] = {}

    for dgp_spec in dgps:
        dgp_name = dgp_spec["name"]
        dgp_kwargs = dgp_spec.get("kwargs", {})

        logger.info(f"\n{'='*80}")
        logger.info(f"DGP: {dgp_name}")
        logger.info(f"{'='*80}")

        dataset = DGP_REGISTRY[dgp_name](**dgp_kwargs, seed=SEED)
        X, y = dataset.X, dataset.y

        dgp_results: dict[str, Any] = {
            "dgp_name": dgp_name,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "models": {},
        }

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(enumerate(kf.split(X)))

        for model_name, config in models.items():
            logger.info(f"\n  Model: {model_name}")

            fold_metrics: list[dict[str, Any]] = []
            fold_times: list[float] = []

            for fold_idx, (train_idx, test_idx) in tqdm(folds, desc=f"  {model_name} folds"):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                try:
                    metrics, fit_time = evaluate_fold(X_train, y_train, X_test, y_test, model_name, config)
                    fold_metrics.append(metrics)
                    fold_times.append(fit_time)
                except Exception as e:
                    logger.error(f"    Fold {fold_idx} failed for {model_name}: {e}")
                    continue

            if fold_metrics:
                aggregated = aggregate_fold_metrics(fold_metrics)
                dgp_results["models"][model_name] = {
                    "fold_metrics": fold_metrics,
                    "aggregated_metrics": aggregated,
                    "mean_fit_time": float(np.mean(fold_times)),
                }

                cov90 = aggregated.get("global_coverage_90", {}).get("mean", float("nan"))
                crps_val = aggregated.get("crps", {}).get("mean", float("nan"))
                logger.info(f"    Global cov@90%: {cov90:.3f}  |  CRPS: {crps_val:.4f}")

        all_results[dgp_name] = dgp_results

        # Save per-DGP results incrementally
        with open(output_dir / f"{dgp_name}_conformalization{suffix}.json", "w") as f:
            json.dump(dgp_results, f, indent=2)

    # Save combined results
    combined_path = output_dir / f"conformalization_results{suffix}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved combined results to {combined_path}")

    env_label = "core models" if suffix == "_core" else "full comparison"
    logger.info(f"Environment: {env_label} — {', '.join(models.keys())}")
    if suffix == "_core":
        logger.info("Tip: run in bench-models env to add CatBoost conformal variants")

    return all_results


# =============================================================================
# Result loading and merging (for post-hoc analysis across environments)
# =============================================================================


def load_and_merge_conformalization_results(
    output_dir: Path = OUTPUT_DIR,
    dgp_names: list[str] | None = None,
) -> dict[str, Any]:
    """Load and merge conformalization results from core and full environment runs.

    Mirrors the ``load_dgp_results`` pattern in compare_synthetic_results.py:
    - ``*_conformalization.json``      — full run (bench-models env)
    - ``*_conformalization_core.json`` — core run (default env, BDF only)

    When both exist for a DGP, models from the core file are merged in unless
    the full file already contains that model (full takes precedence).

    Args:
        output_dir: Directory containing the conformalization JSON files.
        dgp_names: DGP names to load.  Defaults to DGPS_TO_RUN names.

    Returns:
        Merged dict keyed by dgp_name → {dgp_name, models, ...}
    """
    if dgp_names is None:
        dgp_names = [s["name"] for s in DGPS_TO_RUN]

    all_results: dict[str, Any] = {}

    for dgp_name in dgp_names:
        full_path = output_dir / f"{dgp_name}_conformalization.json"
        core_path = output_dir / f"{dgp_name}_conformalization_core.json"

        full_data: dict[str, Any] | None = None
        core_data: dict[str, Any] | None = None

        if full_path.exists():
            with open(full_path) as f:
                full_data = json.load(f)
            logger.info(f"  {dgp_name}: loaded full results")

        if core_path.exists():
            with open(core_path) as f:
                core_data = json.load(f)
            logger.info(f"  {dgp_name}: loaded core results")

        if full_data is None and core_data is None:
            logger.warning(f"  {dgp_name}: no conformalization results found — skipping")
            continue

        if full_data is None:
            all_results[dgp_name] = core_data  # type: ignore[assignment]
            continue

        if core_data is None:
            all_results[dgp_name] = full_data
            continue

        # Merge: full takes precedence, add core-only models
        merged = dict(full_data)
        if "models" not in merged:
            merged["models"] = {}
        for model_name, model_data in core_data.get("models", {}).items():
            if model_name not in merged["models"]:
                merged["models"][model_name] = model_data
                logger.info(f"    Merged {model_name} from core results into {dgp_name}")
            else:
                logger.debug(f"    {model_name} already in full results for {dgp_name} — skipping core version")
        all_results[dgp_name] = merged

    return all_results


# =============================================================================
# Entry point
# =============================================================================


if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("Conformalization Study")
    logger.info("=" * 80)

    model_configs = get_conformalization_models()

    if not model_configs:
        logger.error("No models available. Run in bench-models environment.")
        raise SystemExit(1)

    logger.info(f"Models: {', '.join(model_configs.keys())}")
    logger.info(f"DGPs:   {len(DGPS_TO_RUN)}")

    run_conformalization_study(
        dgps=DGPS_TO_RUN,
        models=model_configs,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Conformalization study complete!")
    logger.info("=" * 80)
