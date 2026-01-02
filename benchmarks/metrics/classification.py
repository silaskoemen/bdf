from typing import Callable

import numpy as np

# from mapie.metrics.calibration import expected_calibration_error  # mapie now in bench-models dependencies
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
