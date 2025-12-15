# Create point metrics plots for regression benchmarks
from pathlib import Path

from .utils.aggregate_results import make_bdf_reg

JSON_ROOT = Path("benchmarks/results")

# 1. Avg rank of RMSE
# Could include others like concrete strength etc.
DATASETS = [
    "realestate",
    "boston_housing",
    "abalone_age",
    "parkinsons_updrs",
    "wine_quality",
]
MODELS = [
    "bdf_normalmunormal",
    "bdf_kde",
    "bdf_gammamvlambdaexponential",
    "bdf_gammamvlambdapoisson",
    "bdf_normalmeanpseudoalphaskewnormal",
]

if __name__ == "__main__":
    # Load all results
    make_bdf_reg(
        results_dir=JSON_ROOT,
        models=MODELS,
        datasets=DATASETS,
        tuning_metric="mse",
    )
    make_bdf_reg(
        results_dir=JSON_ROOT,
        models=MODELS,
        datasets=DATASETS,
        tuning_metric="crps",
    )
