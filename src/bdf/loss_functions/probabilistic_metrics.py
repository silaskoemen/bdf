""" Implement probabilistic metrics for evaluating distributional/probabilistic
predictions"""
import numpy as np


def pica(
    y_true: np.ndarray,
    y_pred: dict[float, np.ndarray],
    levels: list[float] = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99],
) -> float:
    """Calculate the PICA (Prediction Interval Coverage Accuracy) metric.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        levels (list[float]): List of prediction interval levels.

    Returns:
        float: The PICA score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return -1
    # pica_scores: list[float] = []
    # for level in levels:
    #     if not (0 < level < 1):
    #         raise ValueError(f"Level {level} is not in the range (0, 1).")
    #     pica_scores.append(
    #         np.mean((y_true >= y_pred[f"{level}_lower"]) & (y_true <= y_pred[f"{level}_upper"]))
    #     )
    # return np.mean(pica_scores)
