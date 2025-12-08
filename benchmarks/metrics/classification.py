from typing import Callable

import properscoring
from sklearn import metrics

CLAS_POINT_METRICS: dict[str, Callable] = {
    "auroc": metrics.roc_auc_score,
    "accuracy": metrics.accuracy_score,
    "f1": metrics.f1_score,
    "recall": metrics.recall_score,
    "precision": metrics.precision_score,
    "log_loss": metrics.log_loss,
}

CLAS_PROB_METRICS: dict[str, Callable] = {"brier": properscoring.brier_score}
