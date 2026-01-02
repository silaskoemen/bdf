# Create point metrics plots for regression benchmarks
from pathlib import Path

from .utils.aggregate_results import average_rmse_rank, load_all_results, rel_to_best

JSON_ROOT = Path("benchmarks/results/custom/")

# 1. Avg rank of RMSE
# Could include others like concrete strength etc.
DATASETS = ["breast_cancer", "boston_housing_classification", "titanic"]
MODELS = [
    "bdf_betamvbernoulli",
    "rf_clas",
    "knn_clas",
    "calrf_clas",
    "lgbm_clas",
    "ngboost_clas",
]

if __name__ == "__main__":
    # Load all results
    df = load_all_results(JSON_ROOT, models=MODELS)
    # Filter to datasets and models of interest
    df = df.filter(df["dataset"].is_in(DATASETS) & df["model"].is_in(MODELS))

    # Calculate rel to best WIS and DSS if available, print for table in paper
    rel_to_best_log_loss_df = rel_to_best(df, metric="log_loss")
    print("Relative to best log loss:")
    print(rel_to_best_log_loss_df)
    rel_to_best_brier_df = rel_to_best(df, metric="brier")
    print("Relative to best Brier score:")
    print(rel_to_best_brier_df)

    # Make 1x2 subplot of PICA + PIT KS stat for all probabilistic models
    # Dots per dataset (legend with dataset names), x-axis model, y axis metrics (separate)
    # red diamond for average over datasets
    from .utils.plotting import plot_auroc_and_ece

    plot_auroc_and_ece(
        df,
        models=MODELS,
        datasets=DATASETS,
        save_path=Path("benchmarks/plots/classification_auroc_ece.png"),
    )
