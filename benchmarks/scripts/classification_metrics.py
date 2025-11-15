from typing import Callable

import properscoring
from sklearn import metrics

POINT_METRICS: dict[str, Callable] = {
    "auroc": metrics.roc_auc_score,
    "accuracy": metrics.accuracy_score,
    "f1": metrics.f1_score,
    "recall": metrics.recall_score,
    "precision": metrics.precision_score,
    "log_loss": metrics.log_loss,
}

PROBABILISTIC_METRICS: dict[str, Callable] = {"brier": properscoring.brier_score}
