import numpy as np
import properscoring
from sklearn import metrics


def crps_wrapper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the CRPS for probabilistic regression predictions."""
    return np.mean(properscoring.crps_ensemble(y_true, y_pred))  # type: ignore


def pica(
    y_true: np.ndarray, y_pred: np.ndarray, levels=[0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975]
) -> float:
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


def coverage_95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the 95% prediction interval coverage."""
    lower_bound = np.percentile(y_pred, 2.5, axis=1)
    upper_bound = np.percentile(y_pred, 97.5, axis=1)
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return coverage


def interval_score(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the interval score for the given alpha level."""
    lower_bound = np.percentile(y_pred, (alpha / 2) * 100, axis=1)
    upper_bound = np.percentile(y_pred, (1 - alpha / 2) * 100, axis=1)
    interval_width = upper_bound - lower_bound
    penalty = (2 / alpha) * (
        (lower_bound - y_true) * (y_true < lower_bound) + (y_true - upper_bound) * (y_true > upper_bound)
    )
    score = np.mean(interval_width + penalty)
    return score


def weighted_interval_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the weighted interval score (WIS).

    Approximates CRPS using a set of prediction intervals.
    Weights are alpha/2 as per Bracher et al. (2021).
    """
    # Alphas corresponding to the intervals we want to evaluate
    # e.g. alpha=0.1 corresponds to 90% interval (5% to 95%)
    alphas = np.array([0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975])

    wis = 0.0

    # 1. Add contribution from the median (absolute error)
    # This corresponds to the limit as alpha -> 1
    median_pred = np.median(y_pred, axis=1)
    wis += 0.5 * np.abs(y_true - median_pred)

    # 2. Add weighted interval scores
    for alpha in alphas:
        # Weight w_k = alpha / 2
        weight = alpha / 2.0
        score = interval_score(y_true, y_pred, alpha)
        wis += weight * score

    # Normalize by sum of weights to keep scale comparable?
    # Standard WIS definition is just the sum (approximating the integral).
    # Often normalized by (1 / (K + 0.5)) to make it comparable to MAE.

    return np.mean(wis) / (len(alphas) + 0.5)


REG_POINT_METRICS = {
    "mse": metrics.mean_squared_error,
    "mae": metrics.mean_absolute_error,
    "rmse": metrics.root_mean_squared_error,
    "r2": metrics.r2_score,
    "msle": metrics.mean_squared_log_error,
    "mape": metrics.mean_absolute_percentage_error,
}
REG_PROB_METRICS = {
    "crps": crps_wrapper,
    "pica": pica,
    "coverage_90": coverage_90,
    "coverage_95": coverage_95,
    "interval_score_90": lambda y_true, y_pred: interval_score(y_true, y_pred, alpha=0.1),
    "interval_score_95": lambda y_true, y_pred: interval_score(y_true, y_pred, alpha=0.05),
    "weighted_interval_score": weighted_interval_score,
}
