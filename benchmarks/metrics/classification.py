from typing import Callable

import numpy as np
import properscoring
from sklearn import metrics


def custom_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier score for binary classification."""
    return float(np.mean((y_prob - y_true) ** 2))


CLAS_POINT_METRICS: dict[str, Callable] = {
    "auroc": metrics.roc_auc_score,
    "accuracy": lambda y, y_pred: metrics.accuracy_score(y, np.round(y_pred)),  # Ensure accuracy gets labels
    "f1": lambda y, y_pred: metrics.f1_score(y, np.round(y_pred)),
    "recall": lambda y, y_pred: metrics.recall_score(y, np.round(y_pred)),
    "precision": lambda y, y_pred: metrics.precision_score(y, np.round(y_pred)),
    "log_loss": metrics.log_loss,  # Log loss handles probabilities naturally
    "brier": custom_brier,
}

CLAS_PROB_METRICS: dict[str, Callable] = {}
