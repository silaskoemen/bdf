import properscoring
from sklearn import metrics

REG_POINT_METRICS = {
    "mse": metrics.mean_squared_error,
    "mae": metrics.mean_absolute_error,
    "rmse": metrics.root_mean_squared_error,
    "r2": metrics.r2_score,
    "msle": metrics.mean_squared_log_error,
    "mape": metrics.mean_absolute_percentage_error,
}
REG_PROB_METRICS = {
    "crps": properscoring.crps_ensemble,
}
