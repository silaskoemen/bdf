from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scoringrules
from loguru import logger
from sklearn import metrics

# Type alias for prediction type
PredictionType = Literal["samples", "quantiles"]

# Standard percentile levels needed across metrics (0-100 scale)
# Derived from: 50% interval (25, 75), 90% interval (5, 95), 95% interval (2.5, 97.5)
# Plus median (50) and levels needed by pica/wis
STANDARD_PERCENTILES = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 80.0, 90.0, 95.0, 97.5, 99.0, 99.5]


@dataclass
class MetricSpec:
    """Specification for a probabilistic metric."""

    func: Callable
    accepts: tuple[PredictionType, ...]
    requires_quantiles: tuple[float, ...] | None = None  # For quantile-specific metrics

    def __call__(self, y_true, y_pred, **kwargs):
        return self.func(y_true, y_pred, **kwargs)


def has_required_quantiles(quantile_levels: np.ndarray, required: list[float], tolerance: float = 1e-6) -> bool:
    """Check if required quantile levels are present in the model's quantiles.

    Args:
        quantile_levels: Available quantile levels (0-1 scale)
        required: Required quantile levels (0-1 scale)
        tolerance: Floating point tolerance for comparison

    Returns:
        True if all required quantiles are present
    """
    for q in required:
        if not np.any(np.abs(quantile_levels - q) < tolerance):
            return False
    return True


def get_quantile_index(quantile_levels: np.ndarray, target: float, tolerance: float = 1e-6) -> int:
    """Get the index of a quantile level in the sorted quantile levels array.

    Args:
        quantile_levels: Available quantile levels (0-1 scale), assumed sorted
        target: Target quantile level (0-1 scale)
        tolerance: Floating point tolerance for comparison

    Returns:
        Index of the target quantile

    Raises:
        ValueError if target quantile is not found
    """
    matches = np.where(np.abs(quantile_levels - target) < tolerance)[0]
    if len(matches) == 0:
        raise ValueError(f"Quantile {target} not found in quantile levels")
    return int(matches[0])


def precompute_percentiles(
    y_pred_samples: np.ndarray, percentiles: list[float] | None = None
) -> dict[float, np.ndarray]:
    """Pre-compute multiple percentiles at once from sample predictions.

    Args:
        y_pred_samples: Sample predictions of shape (n_obs, n_samples)
        percentiles: List of percentiles to compute (0-100 scale). Defaults to STANDARD_PERCENTILES.

    Returns:
        Dict mapping percentile level (0-100) to arrays of shape (n_obs,)
    """
    if percentiles is None:
        percentiles = STANDARD_PERCENTILES

    # Compute all percentiles in one vectorized call
    percentile_arrays = np.percentile(y_pred_samples, percentiles, axis=1)

    # Return as dict for easy lookup
    return {p: percentile_arrays[i] for i, p in enumerate(percentiles)}


def get_percentile_from_prediction(
    y_pred: np.ndarray,
    percentile: float,
    prediction_type: PredictionType,
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> np.ndarray:
    """Extract a percentile from either samples or quantile predictions.

    Args:
        y_pred: Predictions array of shape (n_obs, n_samples) or (n_obs, n_quantiles)
        percentile: Percentile to extract (0-100 scale)
        prediction_type: Either "samples" or "quantiles"
        quantile_levels: Quantile levels (0-1 scale) if prediction_type is "quantiles"
        precomputed: Optional dict of pre-computed percentiles (from precompute_percentiles)

    Returns:
        Array of shape (n_obs,) with the requested percentile for each observation
    """
    # Use pre-computed if available (samples only)
    if precomputed is not None and percentile in precomputed:
        return precomputed[percentile]

    if prediction_type == "samples":
        return np.percentile(y_pred, percentile, axis=1)
    else:  # quantiles
        if quantile_levels is None:
            raise ValueError("quantile_levels must be provided for quantile predictions")
        # Interpolate between quantile predictions
        target_q = percentile / 100.0
        return np.array([np.interp(target_q, quantile_levels, y_pred[i]) for i in range(y_pred.shape[0])])


# Implement 'safe' versions of mse, mae, rmse, mape that handle inf/nan predictions
def _safe_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE that returns NaN if any predictions are infinite.

    This is better than clamping to a large value, as NaN can be properly
    serialized/deserialized in YAML and clearly indicates a failed prediction.
    """
    if np.any(np.isinf(y_pred)):
        return float("nan")
    square = (y_true - y_pred) ** 2
    return float(np.mean(square))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE that returns NaN if any predictions are infinite."""
    if np.any(np.isinf(y_pred)):
        return float("nan")
    abs_error = np.abs(y_true - y_pred)
    return float(np.mean(abs_error))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE that returns NaN if any predictions are infinite."""
    mse = _safe_mse(y_true, y_pred)
    if np.isnan(mse):
        return float("nan")
    return float(np.sqrt(mse))


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE that returns NaN if any predictions are infinite."""
    if np.any(np.isinf(y_pred)):
        return float("nan")
    ape = np.abs((y_true - y_pred) / y_true)
    # Handle division by zero separately from inf predictions
    safe_ape = np.where(np.isfinite(ape), ape, np.nan)
    if np.all(np.isnan(safe_ape)):
        return float("nan")
    return float(np.nanmean(safe_ape)) * 100.0


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> float:
    """Compute the quantile loss for a specific quantile.

    Works with both samples and quantiles. For quantiles, requires the specific quantile level.
    """
    if quantile_levels is not None:
        # Check if required quantile is present
        if not has_required_quantiles(quantile_levels, [quantile]):
            logger.warning(f"quantile_loss skipped: quantile {quantile} not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        idx = get_quantile_index(quantile_levels, quantile)
        y_pred_quantile = y_pred[:, idx]
    else:
        # Sample-based calculation - use pre-computed if available
        percentile = quantile * 100
        if precomputed and percentile in precomputed:
            y_pred_quantile = precomputed[percentile]
        else:
            y_pred_quantile = np.percentile(y_pred, percentile, axis=1)

    errors = y_true - y_pred_quantile
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def dawid_sebastiani_score(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6, quantile_levels: np.ndarray | None = None
) -> float:
    """Compute the Dawid-Sebastiani score for probabilistic regression predictions.

    Note: Only works with samples, not quantiles. quantile_levels parameter is ignored.
    """
    means = np.mean(y_pred, axis=1)
    variances = np.var(y_pred, axis=1, ddof=1) + eps  # Add eps for numerical stability
    ds_scores = ((y_true - means) ** 2) / variances + np.log(variances)
    return np.mean(ds_scores)


def crps_ensemble_wrapper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """CRPS for sample-based predictions. y_pred shape: (n_obs, n_samples)"""
    return float(np.mean(scoringrules.crps_ensemble(y_true, y_pred)))


def crps_quantile_wrapper(y_true: np.ndarray, y_pred: np.ndarray, quantile_levels: np.ndarray) -> float:
    """CRPS for quantile-based predictions. y_pred shape: (n_obs, n_quantiles)"""
    return float(np.mean(scoringrules.crps_quantile(y_true, y_pred, quantile_levels)))


def crps_wrapper(y_true: np.ndarray, y_pred: np.ndarray, quantile_levels: np.ndarray | None = None) -> float:
    """Unified CRPS wrapper that handles both samples and quantiles.

    Args:
        y_true: True values
        y_pred: Predictions (samples or quantiles)
        quantile_levels: Quantile levels if y_pred contains quantiles, None for samples

    Returns:
        Mean CRPS score
    """
    if quantile_levels is not None:
        # Quantile-based CRPS
        return crps_quantile_wrapper(y_true, y_pred, quantile_levels)
    else:
        # Sample-based CRPS
        return crps_ensemble_wrapper(y_true, y_pred)


def sharpness(y_pred: np.ndarray, quantile_levels: np.ndarray | None = None) -> float:
    """Compute sharpness as the average standard deviation of the predictive distributions.

    Note: Only works with samples, not quantiles. quantile_levels parameter is ignored.
    """
    return np.mean(np.std(y_pred, ddof=1, axis=1))


def ci_width_50(y_pred: np.ndarray, quantile_levels: np.ndarray | None = None) -> float:
    """Compute the average width of the 50% prediction interval.

    Works with both samples and quantiles. For quantiles, requires levels 0.25 and 0.75.
    """
    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [0.25, 0.75]):
            logger.warning("ci_width_50 skipped: quantiles 0.25 and 0.75 not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, 0.25)
        upper_idx = get_quantile_index(quantile_levels, 0.75)
        lower_bound = y_pred[:, lower_idx]
        upper_bound = y_pred[:, upper_idx]
    else:
        # Sample-based calculation
        lower_bound = np.percentile(y_pred, 25, axis=1)
        upper_bound = np.percentile(y_pred, 75, axis=1)

    return np.mean(upper_bound - lower_bound)


def ci_width_div_sigma_50(y_true: np.ndarray, y_pred: np.ndarray, quantile_levels: np.ndarray | None = None) -> float:
    """Compute the average width of the 50% prediction interval divided by standard deviation.

    Works with both samples and quantiles. For quantiles, requires levels 0.25 and 0.75.
    """
    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [0.25, 0.75]):
            logger.warning("ci_width_div_sigma_50 skipped: quantiles 0.25 and 0.75 not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, 0.25)
        upper_idx = get_quantile_index(quantile_levels, 0.75)
        lower_bound = y_pred[:, lower_idx]
        upper_bound = y_pred[:, upper_idx]
    else:
        # Sample-based calculation
        lower_bound = np.percentile(y_pred, 25, axis=1)
        upper_bound = np.percentile(y_pred, 75, axis=1)

    interval_width = upper_bound - lower_bound
    sigma = np.std(y_true, ddof=1)
    return np.mean(interval_width / sigma)


def ci_width_90(y_pred: np.ndarray, quantile_levels: np.ndarray | None = None) -> float:
    """Compute the average width of the 90% prediction interval.

    Works with both samples and quantiles. For quantiles, requires levels 0.05 and 0.95.
    """
    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [0.05, 0.95]):
            logger.warning("ci_width_90 skipped: quantiles 0.05 and 0.95 not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, 0.05)
        upper_idx = get_quantile_index(quantile_levels, 0.95)
        lower_bound = y_pred[:, lower_idx]
        upper_bound = y_pred[:, upper_idx]
    else:
        # Sample-based calculation
        lower_bound = np.percentile(y_pred, 5, axis=1)
        upper_bound = np.percentile(y_pred, 95, axis=1)

    return np.mean(upper_bound - lower_bound)


def ci_width_div_sigma_90(y_true: np.ndarray, y_pred: np.ndarray, quantile_levels: np.ndarray | None = None) -> float:
    """Compute the average width of the 90% prediction interval divided by standard deviation.

    Works with both samples and quantiles. For quantiles, requires levels 0.05 and 0.95.
    """
    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [0.05, 0.95]):
            logger.warning("ci_width_div_sigma_90 skipped: quantiles 0.05 and 0.95 not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, 0.05)
        upper_idx = get_quantile_index(quantile_levels, 0.95)
        lower_bound = y_pred[:, lower_idx]
        upper_bound = y_pred[:, upper_idx]
    else:
        # Sample-based calculation
        lower_bound = np.percentile(y_pred, 5, axis=1)
        upper_bound = np.percentile(y_pred, 95, axis=1)

    interval_width = upper_bound - lower_bound
    sigma = np.std(y_true, ddof=1)
    return np.mean(interval_width / sigma)


def pica(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    levels=[0.5, 0.8, 0.9, 0.95, 0.975, 0.99],
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> float:
    """Prediction Interval Coverage Accuracy (PICA).

    Average absolute deviation between nominal confidence levels and empirical coverage.
    Works with both samples and quantiles. For quantiles, requires all quantile levels
    corresponding to alpha/2 and 1-alpha/2 for each confidence level.
    """
    n_levels = len(levels)

    # For quantiles, check if all required quantiles are present
    if quantile_levels is not None:
        required_quantiles = []
        for level in levels:
            alpha = 1.0 - level
            required_quantiles.extend([alpha / 2.0, 1.0 - alpha / 2.0])

        if not has_required_quantiles(quantile_levels, required_quantiles):
            logger.warning(
                f"pica skipped: required quantiles {required_quantiles} not all present in model predictions"
            )
            return float("nan")

    total_deviation = 0.0
    for level in levels:
        alpha = 1.0 - level
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0

        if quantile_levels is not None:
            # Quantile-based calculation - direct indexing
            lower_idx = get_quantile_index(quantile_levels, lower_q)
            upper_idx = get_quantile_index(quantile_levels, upper_q)
            lower_bound = y_pred[:, lower_idx]
            upper_bound = y_pred[:, upper_idx]
        else:
            # Sample-based calculation - use pre-computed if available
            lower_p = lower_q * 100
            upper_p = upper_q * 100
            if precomputed and lower_p in precomputed and upper_p in precomputed:
                lower_bound = precomputed[lower_p]
                upper_bound = precomputed[upper_p]
            else:
                lower_bound = np.percentile(y_pred, lower_p, axis=1)
                upper_bound = np.percentile(y_pred, upper_p, axis=1)

        empirical_coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        total_deviation += abs(empirical_coverage - level)

    return total_deviation / n_levels


def coverage_90(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> float:
    """Compute the 90% prediction interval coverage.

    Works with both samples and quantiles. For quantiles, requires levels 0.05 and 0.95.
    """
    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [0.05, 0.95]):
            logger.warning("coverage_90 skipped: quantiles 0.05 and 0.95 not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, 0.05)
        upper_idx = get_quantile_index(quantile_levels, 0.95)
        lower_bound = y_pred[:, lower_idx]
        upper_bound = y_pred[:, upper_idx]
    else:
        # Sample-based calculation - use pre-computed if available
        if precomputed and 5.0 in precomputed and 95.0 in precomputed:
            lower_bound = precomputed[5.0]
            upper_bound = precomputed[95.0]
        else:
            lower_bound = np.percentile(y_pred, 5, axis=1)
            upper_bound = np.percentile(y_pred, 95, axis=1)

    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return float(coverage)


def coverage_50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> float:
    """Compute the 50% prediction interval coverage.

    Works with both samples and quantiles. For quantiles, requires levels 0.25 and 0.75.
    """
    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [0.25, 0.75]):
            logger.warning("coverage_50 skipped: quantiles 0.25 and 0.75 not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, 0.25)
        upper_idx = get_quantile_index(quantile_levels, 0.75)
        lower_bound = y_pred[:, lower_idx]
        upper_bound = y_pred[:, upper_idx]
    else:
        # Sample-based calculation - use pre-computed if available
        if precomputed and 25.0 in precomputed and 75.0 in precomputed:
            lower_bound = precomputed[25.0]
            upper_bound = precomputed[75.0]
        else:
            lower_bound = np.percentile(y_pred, 25, axis=1)
            upper_bound = np.percentile(y_pred, 75, axis=1)

    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return float(coverage)


def coverage_95(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> float:
    """Compute the 95% prediction interval coverage.

    Works with both samples and quantiles. For quantiles, requires levels 0.025 and 0.975.
    """
    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [0.025, 0.975]):
            logger.warning("coverage_95 skipped: quantiles 0.025 and 0.975 not present in model predictions")
            return float("nan")
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, 0.025)
        upper_idx = get_quantile_index(quantile_levels, 0.975)
        lower_bound = y_pred[:, lower_idx]
        upper_bound = y_pred[:, upper_idx]
    else:
        # Sample-based calculation - use pre-computed if available
        if precomputed and 2.5 in precomputed and 97.5 in precomputed:
            lower_bound = precomputed[2.5]
            upper_bound = precomputed[97.5]
        else:
            lower_bound = np.percentile(y_pred, 2.5, axis=1)
            upper_bound = np.percentile(y_pred, 97.5, axis=1)

    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return float(coverage)


# TODO: Return underpred/overpred/spread to plot decomposition
def interval_score_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05,
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute interval score.

    Works with both samples and quantiles. For quantiles, requires alpha/2 and 1-alpha/2 levels.
    """
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    if quantile_levels is not None:
        # Check if required quantiles are present
        if not has_required_quantiles(quantile_levels, [lower_q, upper_q]):
            logger.warning(
                f"interval_score skipped: quantiles {lower_q:.4f} and {upper_q:.4f} not present in model predictions"
            )
            return np.full(y_true.shape[0], float("nan"))
        # Quantile-based calculation - direct indexing
        lower_idx = get_quantile_index(quantile_levels, lower_q)
        upper_idx = get_quantile_index(quantile_levels, upper_q)
        lower = y_pred[:, lower_idx]
        upper = y_pred[:, upper_idx]
    else:
        # Sample-based calculation - use pre-computed if available
        lower_p = lower_q * 100
        upper_p = upper_q * 100
        if precomputed and lower_p in precomputed and upper_p in precomputed:
            lower = precomputed[lower_p]
            upper = precomputed[upper_p]
        else:
            lower = np.percentile(y_pred, lower_p, axis=1)
            upper = np.percentile(y_pred, upper_p, axis=1)

    width = upper - lower
    # If y_true is larger than upper, underprediction; if smaller than lower, overprediction
    underpredicted = (y_true - upper) * (y_true > upper)
    overpredicted = (lower - y_true) * (y_true < lower)
    penalty = (2 / alpha) * (overpredicted + underpredicted)
    return width + penalty


def pit_ks_statistic(y_true: np.ndarray, y_pred: np.ndarray, quantile_levels: np.ndarray | None = None) -> float:
    """Compute the Kolmogorov-Smirnov statistic for the probability integral transform.

    Note: Only works with samples, not quantiles. quantile_levels parameter is ignored.
    """
    y_true = np.asarray(y_true)
    pit = np.mean(y_pred <= y_true[:, None], axis=1)
    pit_sorted = np.sort(pit)
    n = pit_sorted.size
    ecdf = np.arange(1, n + 1) / n
    ks = np.max(np.maximum(ecdf - pit_sorted, pit_sorted - (np.arange(n) / n)))
    return float(ks)


def weighted_interval_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile_levels: np.ndarray | None = None,
    precomputed: dict[float, np.ndarray] | None = None,
) -> float:
    """Compute weighted interval score.

    Works with both samples and quantiles. For quantiles, requires 0.5 (median) and
    all alpha/2 and 1-alpha/2 quantiles for alphas: [0.01, 0.025, 0.05, 0.1, 0.2, 0.5].
    """
    alphas = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.5])

    # For quantiles, check if all required quantiles are present
    if quantile_levels is not None:
        required_quantiles = [0.5]  # median
        for alpha in alphas:
            required_quantiles.extend([alpha / 2.0, 1.0 - alpha / 2.0])

        if not has_required_quantiles(quantile_levels, required_quantiles):
            logger.warning(
                f"weighted_interval_score skipped: required quantiles {required_quantiles} not all present in model predictions"
            )
            return float("nan")

        # Get median using direct indexing
        median_idx = get_quantile_index(quantile_levels, 0.5)
        median = y_pred[:, median_idx]
    else:
        # Sample-based calculation - use pre-computed if available
        if precomputed and 50.0 in precomputed:
            median = precomputed[50.0]
        else:
            median = np.median(y_pred, axis=1)

    wis_terms = 0.5 * np.abs(y_true - median)
    for alpha in alphas:
        wis_terms += (alpha / 2.0) * interval_score_samples(y_true, y_pred, alpha, quantile_levels, precomputed)

    return float(np.mean(wis_terms) / (len(alphas) + 0.5))


REG_POINT_METRICS = {
    "mse": _safe_mse,
    "mae": _safe_mae,
    "rmse": _safe_rmse,
    "r2": metrics.r2_score,
    # "msle": metrics.mean_squared_log_error,
    # "mape": _safe_mape,
}

REG_PROB_METRICS = {
    # MAIN METRICS
    "crps": MetricSpec(
        func=crps_wrapper,
        accepts=("samples", "quantiles"),
    ),
    "pica": MetricSpec(
        func=pica,
        accepts=("samples", "quantiles"),
    ),
    # COVERAGES
    "coverage_50": MetricSpec(
        func=coverage_50,
        accepts=("samples", "quantiles"),
    ),
    "coverage_90": MetricSpec(
        func=coverage_90,
        accepts=("samples", "quantiles"),
    ),
    "coverage_95": MetricSpec(
        func=coverage_95,
        accepts=("samples", "quantiles"),
    ),
    # INTERVAL SCORES
    "interval_score_50": MetricSpec(
        func=lambda y_true, y_pred, quantile_levels=None, precomputed=None: np.mean(
            interval_score_samples(y_true, y_pred, alpha=0.5, quantile_levels=quantile_levels, precomputed=precomputed)
        ),
        accepts=("samples", "quantiles"),
    ),
    "interval_score_90": MetricSpec(
        func=lambda y_true, y_pred, quantile_levels=None, precomputed=None: np.mean(
            interval_score_samples(y_true, y_pred, alpha=0.1, quantile_levels=quantile_levels, precomputed=precomputed)
        ),
        accepts=("samples", "quantiles"),
    ),
    "interval_score_95": MetricSpec(
        func=lambda y_true, y_pred, quantile_levels=None, precomputed=None: np.mean(
            interval_score_samples(y_true, y_pred, alpha=0.05, quantile_levels=quantile_levels, precomputed=precomputed)
        ),
        accepts=("samples", "quantiles"),
    ),
    "weighted_interval_score": MetricSpec(
        func=weighted_interval_score,
        accepts=("samples", "quantiles"),
    ),
    # ADDITIONAL METRICS
    "sharpness": MetricSpec(
        func=lambda _, y_pred, quantile_levels=None: sharpness(y_pred, quantile_levels=quantile_levels),
        accepts=("samples", "quantiles"),
    ),
    "ci_width_50": MetricSpec(
        func=lambda _, y_pred, quantile_levels=None: ci_width_50(y_pred, quantile_levels=quantile_levels),
        accepts=("samples", "quantiles"),
    ),
    "ci_width_90": MetricSpec(
        func=lambda _, y_pred, quantile_levels=None: ci_width_90(y_pred, quantile_levels=quantile_levels),
        accepts=("samples", "quantiles"),
    ),
    # "ci_width_div_sigma_50": MetricSpec(
    #     func=ci_width_div_sigma_50,
    #     accepts=("samples", "quantiles"),
    # ),
    # "ci_width_div_sigma_90": MetricSpec(
    #     func=ci_width_div_sigma_90,
    #     accepts=("samples", "quantiles"),
    # ),
    "pit_ks_statistic": MetricSpec(
        func=pit_ks_statistic,
        accepts=("samples",),
    ),
    # "quantile_loss_50": MetricSpec(
    #     func=lambda y_true, y_pred, quantile_levels=None, precomputed=None: quantile_loss(
    #         y_true, y_pred, 0.5, quantile_levels=quantile_levels, precomputed=precomputed
    #     ),
    #     accepts=("samples", "quantiles"),
    # ),
    # "quantile_loss_90": MetricSpec(
    #     func=lambda y_true, y_pred, quantile_levels=None, precomputed=None: quantile_loss(
    #         y_true, y_pred, 0.9, quantile_levels=quantile_levels, precomputed=precomputed
    #     ),
    #     accepts=("samples", "quantiles"),
    # ),
    # "quantile_loss_95": MetricSpec(
    #     func=lambda y_true, y_pred, quantile_levels=None, precomputed=None: quantile_loss(
    #         y_true, y_pred, 0.95, quantile_levels=quantile_levels, precomputed=precomputed
    #     ),
    #     accepts=("samples", "quantiles"),
    # ),
    "dawid_sebastiani_score": MetricSpec(
        func=dawid_sebastiani_score,
        accepts=("samples",),
    ),
}
