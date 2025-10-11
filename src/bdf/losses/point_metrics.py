from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


def pointwise_metrics(y_true, y_pred):
    """Calculate pointwise metrics for regression tasks.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: Dictionary containing MAE, MSE, and RMSE.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }
