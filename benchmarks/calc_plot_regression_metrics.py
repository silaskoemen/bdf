# Create point metrics plots for regression benchmarks
from pathlib import Path

from .utils.aggregate_results import average_rmse_rank, load_all_results, rel_to_best

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
ALL_MODELS = [
    # "bdf_normalmunormal",
    # "bdf_kde",
    "bdf_reg_mse",
    "de_reg",
    "lgbm_reg",
    "lgbm_qreg",
    "mq_reg",
    "ngboost_reg",
    "rf_reg",
]

PROB_MODELS = [
    "bdf_reg_crps",
    "lgbm_qreg",
    "mq_reg",
    "ngboost_reg",
]

if __name__ == "__main__":
    from .utils.plotting import plot_average_rmse_rank

    # Load all results
    df = load_all_results(JSON_ROOT, models=ALL_MODELS)
    # Filter to datasets and models of interest
    df = df.filter(df["dataset"].is_in(DATASETS) & df["model"].is_in(ALL_MODELS))
    # Compute average RMSE rank
    avg_rank_df = average_rmse_rank(df)
    # Plot
    plot_average_rmse_rank(avg_rank_df, save_path=Path("benchmarks/plots/regression_avg_rmse_rank.png"))

    # Compute relative-to-best RMSE per dataset and model, plot model average
    from .utils.plotting import plot_rel_to_best

    rel_to_best_df = rel_to_best(df, metric="rmse")
    plot_rel_to_best(rel_to_best_df, metric="rmse", save_path=Path("benchmarks/plots/regression_rel_to_best_rmse.png"))

    # Now for probabilistic models and CRPS
    df = load_all_results(JSON_ROOT, models=PROB_MODELS)
    # Filter to datasets and models of interest
    df = df.filter(df["dataset"].is_in(DATASETS) & df["model"].is_in(PROB_MODELS))
    # Plot relative-to-best CRPS per dataset and model, plot model average
    rel_to_best_crps_df = rel_to_best(df, metric="crps")
    print(rel_to_best_crps_df)
    plot_rel_to_best(
        rel_to_best_crps_df, metric="crps", save_path=Path("benchmarks/plots/regression_rel_to_best_crps.png")
    )

    # Calculate rel to best WIS and DSS if available, print for table in paper
    rel_to_best_wis_df = rel_to_best(df, metric="weighted_interval_score")
    print("Relative to best WIS:")
    print(rel_to_best_wis_df)
    rel_to_best_dss_df = rel_to_best(df, metric="dawid_sebastiani_score")
    print("Relative to best DSS:")
    print(rel_to_best_dss_df)

    # Make 1x2 subplot of PICA + PIT KS stat for all probabilistic models
    # Dots per dataset (legend with dataset names), x-axis model, y axis metrics (separate)
    # red diamond for average over datasets
    from .utils.plotting import plot_pica_and_pit

    plot_pica_and_pit(
        df,
        models=PROB_MODELS,
        datasets=DATASETS,
        save_path=Path("benchmarks/plots/regression_pica_pit.png"),
    )

    from .utils.plotting import plot_coverage_vs_interval_score_grid

    plot_coverage_vs_interval_score_grid(
        df,
        datasets=DATASETS,
        models=PROB_MODELS,
        save_path=Path("benchmarks/plots/regression_coverage_vs_interval_score.png"),
    )
