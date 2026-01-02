import numpy as np
import scoringrules
from sklearn import metrics


# Implement 'safe' versions of mse, mae, rmse, mape that sets inf predictions to large finite values
def _safe_preds(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.copy(y_pred)
    y_pred[np.isinf(y_pred)] = np.finfo(np.float64).max
    return y_pred


def _safe_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = _safe_preds(y_pred)
    square = (y_true - y_pred) ** 2
    safe_square = np.where(np.isfinite(square), square, np.finfo(np.float64).max)
    return float(np.mean(safe_square))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = _safe_preds(y_pred)
    abs_error = np.abs(y_true - y_pred)
    safe_abs_error = np.where(np.isfinite(abs_error), abs_error, np.finfo(np.float64).max)
    return float(np.mean(safe_abs_error))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(_safe_mse(y_true, y_pred))


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = _safe_preds(y_pred)
    ape = np.abs((y_true - y_pred) / y_true)
    safe_ape = np.where(np.isfinite(ape), ape, np.finfo(np.float64).max)
    return float(np.mean(safe_ape)) * 100.0


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute the quantile loss for a specific quantile."""
    # Extract the predicted quantile from the samples
    y_pred_quantile = np.percentile(y_pred, quantile * 100, axis=1)
    errors = y_true - y_pred_quantile
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def dawid_sebastiani_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Dawid-Sebastiani score for probabilistic regression predictions."""
    means = np.mean(y_pred, axis=1)
    variances = np.var(y_pred, axis=1, ddof=1) + eps  # Add eps for numerical stability
    ds_scores = ((y_true - means) ** 2) / variances + np.log(variances)
    return np.mean(ds_scores)


def crps_wrapper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the CRPS for probabilistic regression predictions."""
    return np.mean(scoringrules.crps_ensemble(y_true, y_pred))  # type: ignore


def sharpness(y_pred: np.ndarray) -> float:
    """Compute sharpness as the average standard deviation of the predictive distributions."""
    return np.mean(np.std(y_pred, ddof=1, axis=1))


def ci_width_50(y_pred: np.ndarray) -> float:
    """Compute the average width of the 50% prediction interval."""
    lower_bound = np.percentile(y_pred, 25, axis=1)
    upper_bound = np.percentile(y_pred, 75, axis=1)
    return np.mean(upper_bound - lower_bound)


def ci_width_div_sigma_50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the average width of the 50% prediction interval divided by standard deviation."""
    lower_bound = np.percentile(y_pred, 25, axis=1)
    upper_bound = np.percentile(y_pred, 75, axis=1)
    interval_width = upper_bound - lower_bound
    sigma = np.std(y_true, ddof=1)
    return np.mean(interval_width / sigma)


def ci_width_90(y_pred: np.ndarray) -> float:
    """Compute the average width of the 90% prediction interval."""
    lower_bound = np.percentile(y_pred, 5, axis=1)
    upper_bound = np.percentile(y_pred, 95, axis=1)
    return np.mean(upper_bound - lower_bound)


def ci_width_div_sigma_90(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the average width of the 90% prediction interval divided by standard deviation."""
    lower_bound = np.percentile(y_pred, 5, axis=1)
    upper_bound = np.percentile(y_pred, 95, axis=1)
    interval_width = upper_bound - lower_bound
    sigma = np.std(y_true, ddof=1)
    return np.mean(interval_width / sigma)


def pica(y_true: np.ndarray, y_pred: np.ndarray, levels=[0.5, 0.8, 0.9, 0.95, 0.975, 0.99]) -> float:
    """Prediction Interval Coverage Accuracy (PICA).

    Average absolute deviation between nominal confidence levels and empirical coverage.
    """
    n_levels = len(levels)
    total_deviation = 0.0
    for level in levels:
        alpha = 1.0 - level
        lower_p = (alpha / 2.0) * 100
        upper_p = (1.0 - alpha / 2.0) * 100

        lower_bound = np.percentile(y_pred, lower_p, axis=1)
        upper_bound = np.percentile(y_pred, upper_p, axis=1)

        empirical_coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        total_deviation += abs(empirical_coverage - level)

    return total_deviation / n_levels


def coverage_90(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the 90% prediction interval coverage."""
    lower_bound = np.percentile(y_pred, 5, axis=1)
    upper_bound = np.percentile(y_pred, 95, axis=1)
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return coverage


def coverage_50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the 50% prediction interval coverage."""
    lower_bound = np.percentile(y_pred, 25, axis=1)
    upper_bound = np.percentile(y_pred, 75, axis=1)
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return coverage


def coverage_95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the 95% prediction interval coverage."""
    lower_bound = np.percentile(y_pred, 2.5, axis=1)
    upper_bound = np.percentile(y_pred, 97.5, axis=1)
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return coverage


# TODO: Return underpred/overpred/spread to plot decomposition
def interval_score_samples(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    lower = np.percentile(y_pred, (alpha / 2) * 100, axis=1)
    upper = np.percentile(y_pred, (1 - alpha / 2) * 100, axis=1)
    width = upper - lower
    # If y_true is larger than upper, underprediction; if smaller than lower, overprediction
    underpredicted = (y_true - upper) * (y_true > upper)
    overpredicted = (lower - y_true) * (y_true < lower)
    penalty = (2 / alpha) * (overpredicted + underpredicted)
    return width + penalty


def pit_ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    pit = np.mean(y_pred <= y_true[:, None], axis=1)
    pit_sorted = np.sort(pit)
    n = pit_sorted.size
    ecdf = np.arange(1, n + 1) / n
    ks = np.max(np.maximum(ecdf - pit_sorted, pit_sorted - (np.arange(n) / n)))
    return float(ks)


def weighted_interval_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    alphas = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.5])
    median = np.median(y_pred, axis=1)
    wis_terms = 0.5 * np.abs(y_true - median)
    for alpha in alphas:
        wis_terms += (alpha / 2.0) * interval_score_samples(y_true, y_pred, alpha)
    return float(np.mean(wis_terms) / (len(alphas) + 0.5))


REG_POINT_METRICS = {
    "mse": _safe_mse,
    "mae": _safe_mae,
    "rmse": _safe_rmse,
    "r2": metrics.r2_score,
    # "msle": metrics.mean_squared_log_error,
    "mape": _safe_mape,
}

REG_PROB_METRICS = {
    "crps": crps_wrapper,
    "pica": pica,
    "coverage_50": coverage_50,
    "coverage_90": coverage_90,
    "coverage_95": coverage_95,
    "interval_score_50": lambda y_true, y_pred: np.mean(interval_score_samples(y_true, y_pred, alpha=0.5)),
    "interval_score_90": lambda y_true, y_pred: np.mean(interval_score_samples(y_true, y_pred, alpha=0.1)),
    "interval_score_95": lambda y_true, y_pred: np.mean(interval_score_samples(y_true, y_pred, alpha=0.05)),
    "weighted_interval_score": weighted_interval_score,
    "sharpness": lambda _, y_pred: sharpness(y_pred),
    "ci_width_50": lambda _, y_pred: ci_width_50(y_pred),
    "ci_width_90": lambda _, y_pred: ci_width_90(y_pred),
    "ci_width_div_sigma_50": ci_width_div_sigma_50,
    "ci_width_div_sigma_90": ci_width_div_sigma_90,
    "pit_ks_statistic": pit_ks_statistic,
    "quantile_loss_50": lambda y_true, y_pred: quantile_loss(y_true, y_pred, 0.5),
    "quantile_loss_90": lambda y_true, y_pred: quantile_loss(y_true, y_pred, 0.9),
    "quantile_loss_95": lambda y_true, y_pred: quantile_loss(y_true, y_pred, 0.95),
    "dawid_sebastiani_score": dawid_sebastiani_score,
}
