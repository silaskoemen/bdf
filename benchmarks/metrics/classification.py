from typing import Callable

import numpy as np
from sklearn import metrics


def custom_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score for binary classification."""
    return float(np.mean((y_prob - y_true) ** 2))


def custom_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute the Expected Calibration Error (ECE) for binary classification."""
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence_in_bin)
    return ece


def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration curve data for later plotting/aggregation.

    Uses uniform binning strategy so bins align across folds for averaging.

    Returns:
        dict with keys:
            - prob_true: fraction of positives per bin (n_bins,)
            - prob_pred: mean predicted probability per bin (n_bins,)
            - bin_counts: number of samples per bin (n_bins,) for weighted averaging
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1] if i < n_bins - 1 else 1.0 + 1e-8  # Include 1.0 in last bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_counts[i] = np.sum(in_bin)

        if bin_counts[i] > 0:
            prob_true[i] = np.mean(y_true[in_bin])
            prob_pred[i] = np.mean(y_prob[in_bin])
        else:
            prob_true[i] = np.nan
            prob_pred[i] = np.nan

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


CLAS_POINT_METRICS: dict[str, Callable] = {
    "auroc": metrics.roc_auc_score,
    "accuracy": lambda y, y_pred: metrics.accuracy_score(y, np.round(y_pred)),  # Ensure accuracy gets labels
    "f1": lambda y, y_pred: metrics.f1_score(y, np.round(y_pred)),
    "recall": lambda y, y_pred: metrics.recall_score(y, np.round(y_pred)),
    "precision": lambda y, y_pred: metrics.precision_score(y, np.round(y_pred)),
    "log_loss": metrics.log_loss,  # Log loss handles probabilities naturally
    "brier": custom_brier,
    "ece": custom_expected_calibration_error,
}

CLAS_PROB_METRICS: dict[str, Callable] = {}
