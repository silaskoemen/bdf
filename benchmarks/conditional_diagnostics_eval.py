"""Evaluate conditional regression diagnostics without rerunning Optuna.

This script reuses stored benchmark best parameters, reconstructs the same
deterministic CV folds as ``benchmarks.run``, and computes prediction-level
diagnostics aggregated by predicted-uncertainty bins.

It is intentionally split from plotting so it can be run separately in the BDF
and all-model benchmark environments. Model imports are lazy: BDF classes are
imported only for BDF configs, while non-BDF wrappers are imported only for
non-BDF configs.

Examples:
    # Default/BDF environment
    pixi run python -m benchmarks.conditional_diagnostics_eval --models BDF

    # Benchmark/all-model environment
    pixi run -e benchmark python -m benchmarks.conditional_diagnostics_eval \
        --models qrf drf conflgbm ngboost_reg catbunc_reg --env-tag benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import scoringrules
from loguru import logger
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from benchmarks.metrics.regression import get_percentile_from_prediction, precompute_percentiles
from benchmarks.pipeline.data import DatasetMetadata, available_regression_datasets
from benchmarks.pipeline.orchestrators import get_model_prediction_type, get_model_quantiles
from benchmarks.utils.yaml_loader import aggregate_bdf_models, load_yaml_result

RESULTS_DIR = Path("benchmarks/results/regression")
OUTPUT_DIR = Path("benchmarks/results/conditional_diagnostics")
CONFIG_DIR = Path("benchmarks/configs/model")

DEFAULT_BDF_MODELS = [
    "bdf_normalmunormal",
    "bdf_kde",
    "bdf_gammamvlambdapoisson",
    "bdf_freqstudentt",
]

DEFAULT_MODELS = ["BDF", "qrf", "drf", "conflgbm", "ngboost_reg", "catbunc_reg"]


def _jsonable(value: Any) -> Any:
    """Convert numpy/scalar values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_model_cfg(model_name: str) -> OmegaConf:
    cfg_path = CONFIG_DIR / f"{model_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def _load_stored_result(model_name: str) -> dict[str, Any]:
    result_path = RESULTS_DIR / f"res_{model_name}.yaml"
    if not result_path.exists():
        raise FileNotFoundError(f"Stored result not found: {result_path}")
    return load_yaml_result(result_path)


def _split_best_params(model_cfg: OmegaConf, best_params: dict[str, Any]) -> dict[str, Any]:
    """Recreate the tuned init kwargs produced by the benchmark orchestrator."""
    tuned_init_kwargs: dict[str, Any] = {}
    tuned_params: dict[str, Any] = {}

    tunable_init = model_cfg.get("tunable_init_kwargs", {}) or {}
    tunable_params = model_cfg.get("tunable_params", {}) or {}

    for key, value in best_params.items():
        if key in tunable_init:
            tuned_init_kwargs[key] = value
        elif key in tunable_params:
            tuned_params[key] = value
        else:
            # Keep unknown keys as init kwargs. This is useful for older results
            # whose config changed slightly after the benchmark was run.
            tuned_init_kwargs[key] = value

    combined_params: dict[str, Any] = {}
    fixed_params = model_cfg.get("fixed_params", None)
    if fixed_params:
        combined_params.update(OmegaConf.to_container(fixed_params, resolve=True))
    if tuned_params:
        combined_params.update(tuned_params)
    if combined_params:
        tuned_init_kwargs["params"] = combined_params

    return tuned_init_kwargs


def _get_model_class(model_cfg: OmegaConf) -> type:
    """Lazy import the appropriate model factory for the requested config."""
    class_name = str(model_cfg.class_name)
    if class_name.startswith("BDF"):
        from benchmarks.models.bdf_factory import ModelFactory
    else:
        from benchmarks.models.model_factory import ModelFactory

    return ModelFactory.get(model_cfg)


def _is_compatible(model_cfg: OmegaConf, metadata: DatasetMetadata) -> bool:
    domains = list(model_cfg.get("compatible_target_domains", []))
    return metadata.target_domain.value in domains


def _prediction_arrays(model: Any, X: Any, sample_size: int) -> tuple[str, np.ndarray, np.ndarray | None, dict | None]:
    prediction_type = get_model_prediction_type(model)
    quantile_levels = get_model_quantiles(model)
    if prediction_type == "quantiles":
        y_pred_native = model.predict_quantiles(X)
        precomputed = None
    else:
        y_pred_native = model.predict_samples(X, n_samples=sample_size)
        precomputed = precompute_percentiles(y_pred_native)
    return prediction_type, np.asarray(y_pred_native), quantile_levels, precomputed


def _per_observation_crps(
    y_true: np.ndarray,
    y_pred_native: np.ndarray,
    prediction_type: str,
    quantile_levels: np.ndarray | None,
) -> np.ndarray:
    if prediction_type == "quantiles":
        if quantile_levels is None:
            raise ValueError("Quantile levels are required for quantile CRPS")
        return np.asarray(scoringrules.crps_quantile(y_true, y_pred_native, quantile_levels), dtype=float)
    return np.asarray(scoringrules.crps_ensemble(y_true, y_pred_native), dtype=float)


def _extract_percentiles(
    y_pred_native: np.ndarray,
    prediction_type: str,
    quantile_levels: np.ndarray | None,
    precomputed: dict | None,
) -> dict[str, np.ndarray]:
    return {
        "q05": get_percentile_from_prediction(y_pred_native, 5.0, prediction_type, quantile_levels, precomputed),
        "q25": get_percentile_from_prediction(y_pred_native, 25.0, prediction_type, quantile_levels, precomputed),
        "q50": get_percentile_from_prediction(y_pred_native, 50.0, prediction_type, quantile_levels, precomputed),
        "q75": get_percentile_from_prediction(y_pred_native, 75.0, prediction_type, quantile_levels, precomputed),
        "q95": get_percentile_from_prediction(y_pred_native, 95.0, prediction_type, quantile_levels, precomputed),
    }


def _make_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.nanpercentile(values, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.zeros(values.shape[0], dtype=int)
    idx = np.digitize(values, edges[1:-1], right=False)
    return np.clip(idx, 0, n_bins - 1)


def _summarize_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: dict[str, np.ndarray],
    uncertainty: np.ndarray,
    crps: np.ndarray | None,
    n_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    width90 = q["q95"] - q["q05"]
    iqr = q["q75"] - q["q25"]
    covered90 = (y_true >= q["q05"]) & (y_true <= q["q95"])
    err = y_true - y_pred
    alpha = 0.10
    interval_score90 = (
        width90 + (2 / alpha) * np.maximum(q["q05"] - y_true, 0) + (2 / alpha) * np.maximum(y_true - q["q95"], 0)
    )

    bin_idx = _make_bins(uncertainty, n_bins)
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": b + 1,
                    "n": 0,
                    "coverage_90": None,
                    "coverage_error_90": None,
                    "mean_iqr": None,
                    "mean_width_90": None,
                    "interval_score_90": None,
                    "rmse": None,
                    "mae": None,
                    "crps": None,
                }
            )
            continue

        row = {
            "bin": b + 1,
            "n": n,
            "coverage_90": float(np.mean(covered90[mask])),
            "coverage_error_90": float(np.mean(covered90[mask]) - 0.90),
            "mean_uncertainty": float(np.mean(uncertainty[mask])),
            "mean_iqr": float(np.mean(iqr[mask])),
            "mean_width_90": float(np.mean(width90[mask])),
            "interval_score_90": float(np.mean(interval_score90[mask])),
            "rmse": float(np.sqrt(np.mean(err[mask] ** 2))),
            "mae": float(np.mean(np.abs(err[mask]))),
            "crps": float(np.mean(crps[mask])) if crps is not None else None,
        }
        rows.append(row)

    global_summary = {
        "n": int(y_true.shape[0]),
        "coverage_90": float(np.mean(covered90)),
        "coverage_error_90": float(np.mean(covered90) - 0.90),
        "mean_iqr": float(np.mean(iqr)),
        "mean_width_90": float(np.mean(width90)),
        "interval_score_90": float(np.mean(interval_score90)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "crps": float(np.mean(crps)) if crps is not None else None,
    }
    return rows, global_summary


def _weighted_mean(values: list[float | None], weights: list[int]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    if not pairs:
        return None
    vals = np.array([p[0] for p in pairs], dtype=float)
    wts = np.array([p[1] for p in pairs], dtype=float)
    return float(np.average(vals, weights=wts))


def _aggregate_fold_summaries(folds: list[dict[str, Any]], n_bins: int) -> dict[str, Any]:
    global_rows = [f["global"] for f in folds]
    global_weights = [int(g["n"]) for g in global_rows]
    global_summary = {
        key: _weighted_mean([g.get(key) for g in global_rows], global_weights)
        for key in [
            "coverage_90",
            "coverage_error_90",
            "mean_iqr",
            "mean_width_90",
            "interval_score_90",
            "rmse",
            "mae",
            "crps",
        ]
    }
    global_summary["n"] = int(sum(global_weights))

    bin_summaries = []
    for b in range(n_bins):
        bin_rows = [f["bins"][b] for f in folds if len(f["bins"]) > b]
        weights = [int(row["n"]) for row in bin_rows]
        out = {
            "bin": b + 1,
            "n": int(sum(weights)),
        }
        for key in [
            "coverage_90",
            "coverage_error_90",
            "mean_uncertainty",
            "mean_iqr",
            "mean_width_90",
            "interval_score_90",
            "rmse",
            "mae",
            "crps",
        ]:
            out[key] = _weighted_mean([row.get(key) for row in bin_rows], weights)
        bin_summaries.append(out)

    return {"global": global_summary, "bins": bin_summaries}


def _resolve_model_for_dataset(
    requested_model: str,
    dataset_name: str,
    bdf_selection_map: dict[str, str],
) -> str:
    if requested_model == "BDF":
        selected = bdf_selection_map.get(dataset_name)
        if selected is None:
            raise KeyError(f"No selected BDF variant found for dataset {dataset_name}")
        return selected
    return requested_model


def evaluate_model_dataset(
    display_model: str,
    source_model: str,
    metadata: DatasetMetadata,
    X: Any,
    y: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    model_cfg = _load_model_cfg(source_model)
    if not _is_compatible(model_cfg, metadata):
        logger.info(f"Skipping {display_model}/{metadata.name}: incompatible target domain")
        return None

    stored = _load_stored_result(source_model)
    ds_result = stored.get("datasets", {}).get(metadata.name)
    if ds_result is None:
        logger.warning(f"Skipping {display_model}/{metadata.name}: no stored results for {source_model}")
        return None

    best_params = ds_result.get("best_params", {})
    if not best_params:
        logger.warning(f"Skipping {display_model}/{metadata.name}: no stored best_params for {source_model}")
        return None

    model_cls = _get_model_class(model_cfg)
    fixed_init_kwargs = OmegaConf.to_container(model_cfg.get("fixed_init_kwargs", {}) or {}, resolve=True)
    tuned_init_kwargs = _split_best_params(model_cfg, best_params)
    init_kwargs = {**fixed_init_kwargs, **tuned_init_kwargs}

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(cv.split(X, y))[1:]
    if args.max_folds is not None:
        splits = splits[: args.max_folds]

    folds = []
    source_model_display = source_model if display_model == source_model else f"{display_model} <- {source_model}"
    logger.info(f"Evaluating {source_model_display} on {metadata.name} ({len(splits)} folds)")

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        np.random.seed(args.seed + fold_idx)
        model = model_cls(**init_kwargs)
        start = time()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds = time() - start

        X_test = X.iloc[test_idx]
        y_test = np.asarray(y.iloc[test_idx], dtype=float)
        y_pred = np.asarray(model.predict(X_test), dtype=float)
        prediction_type, y_pred_native, quantile_levels, precomputed = _prediction_arrays(
            model, X_test, args.sample_size
        )
        q = _extract_percentiles(y_pred_native, prediction_type, quantile_levels, precomputed)

        if args.bin_on == "iqr":
            uncertainty = q["q75"] - q["q25"]
        elif args.bin_on == "width90":
            uncertainty = q["q95"] - q["q05"]
        elif args.bin_on == "pred_std":
            if prediction_type == "samples":
                uncertainty = np.std(y_pred_native, axis=1, ddof=1)
            else:
                uncertainty = (q["q95"] - q["q05"]) / 3.289707253902945
        else:
            raise ValueError(f"Unknown bin_on value: {args.bin_on}")

        crps = (
            None if args.skip_crps else _per_observation_crps(y_test, y_pred_native, prediction_type, quantile_levels)
        )
        bins, global_summary = _summarize_bins(y_test, y_pred, q, uncertainty, crps, args.n_bins)
        folds.append(
            {
                "fold": fold_idx,
                "n": int(y_test.shape[0]),
                "fit_seconds": float(fit_seconds),
                "prediction_type": prediction_type,
                "bins": bins,
                "global": global_summary,
            }
        )

    if not folds:
        return None

    return {
        "source_model": source_model,
        "best_params": _jsonable(best_params),
        "tuning": _jsonable(ds_result.get("tuning", {})),
        "metadata": _jsonable(ds_result.get("metadata", metadata.to_dict())),
        "folds": _jsonable(folds),
        "summary": _jsonable(_aggregate_fold_summaries(folds, args.n_bins)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model config names, or BDF pseudo-model.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset subset.")
    parser.add_argument(
        "--bdf-models", nargs="+", default=DEFAULT_BDF_MODELS, help="BDF variants used for BDF selection."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-name", default=None, help="Optional output file name. Defaults to env-tag/models.")
    parser.add_argument("--env-tag", default="default", help="Label for this partial run, e.g. default or benchmark.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--bin-on", choices=["iqr", "width90", "pred_std"], default="iqr")
    parser.add_argument(
        "--max-folds", type=int, default=None, help="Debug/pilot: evaluate only the first k eval folds."
    )
    parser.add_argument("--skip-crps", action="store_true", help="Skip per-observation CRPS to make a pilot cheaper.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    needs_bdf_selection = "BDF" in args.models
    bdf_selection_map: dict[str, str] = {}
    if needs_bdf_selection:
        _, bdf_selection_map = aggregate_bdf_models(
            results_dir=RESULTS_DIR,
            bdf_models=args.bdf_models,
            selection_metric="crps",
            use_tuning_value=True,
            datasets=args.datasets,
        )

    output: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "env_tag": args.env_tag,
            "models": args.models,
            "datasets": args.datasets,
            "seed": args.seed,
            "n_splits": args.n_splits,
            "sample_size": args.sample_size,
            "n_bins": args.n_bins,
            "bin_on": args.bin_on,
            "skip_crps": args.skip_crps,
            "bdf_selection_map": bdf_selection_map,
        },
        "models": {},
    }

    for requested_model in args.models:
        output["models"].setdefault(requested_model, {"datasets": {}})

    for metadata, X, y in available_regression_datasets():
        if args.datasets is not None and metadata.name not in args.datasets:
            continue
        for requested_model in args.models:
            try:
                source_model = _resolve_model_for_dataset(requested_model, metadata.name, bdf_selection_map)
                result = evaluate_model_dataset(requested_model, source_model, metadata, X, y, args)
            except ImportError as exc:
                logger.warning(f"Skipping {requested_model}/{metadata.name}: missing dependency: {exc}")
                continue
            except Exception as exc:
                logger.exception(f"Failed {requested_model}/{metadata.name}: {exc}")
                continue
            if result is not None:
                output["models"][requested_model]["datasets"][metadata.name] = result

    output = _jsonable(output)
    if args.output_name:
        output_path = args.output_dir / args.output_name
    else:
        model_tag = "-".join(str(m).replace("_", "") for m in args.models)
        output_path = args.output_dir / f"conditional_{args.env_tag}_{model_tag}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved conditional diagnostics to {output_path}")


if __name__ == "__main__":
    main()
