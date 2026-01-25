# Effect of Number of Trees (n_trees) Analysis

Investigation of how ensemble size affects BDF performance, calibration, and computational cost.

## Purpose

This benchmark answers:
1. **How many trees are needed for convergence?** (identifying the plateau)
2. **What is the performance-cost tradeoff?** (diminishing returns)
3. **Does variance reduction follow law of large numbers?** (theoretical expectation)

## Experimental Design

### Methodology

For each value of `n_trees`:
1. **Tune** other hyperparameters on fold 0 (train/val split)
2. **Evaluate** with those hyperparameters on folds 1-9
3. **Report** mean ± std across 9 evaluation folds

This approach ensures fair comparison: all n_trees values use optimally-tuned companion hyperparameters.

### n_trees Grid

```python
N_TREES_GRID = [5, 10, 25, 50, 100, 200]
```

Covers orders of magnitude to observe convergence behavior.

### Data Generating Processes

- **friedman1**: 10 features (5 informative), noise=1.0
- **friedman2**: 4 features, multiplicative interactions, noise=1.0
- **friedman3**: 4 features, arctan function, noise=0.1
- **make_regression**: 10 features (5 informative), linear, noise=10.0

Fixed sample size: n=2000

### Hyperparameter Tuning

For each n_trees value, tune on fold 0 using Optuna (50 trials, CRPS objective):
- `reg_lambda`: [1e-5, 0.1] (log scale)
- `reg_gamma`: [1e-4, 1.0] (log scale)
- `reg_nu`: [1e-4, 1.0] (log scale)
- `min_samples_leaf`: [5, 50]
- `subsample`: [0.5, 1.0]
- `colsample`: [0.5, 1.0]
- `eta`: [0.001, 0.1] (log scale)
- `sigma_mu_auto_scale`: [0.01, 10.0] (log scale)
- `score_method`: {nll, nle}

## Metrics Tracked

### Point Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

### Probabilistic Metrics
- CRPS (Continuous Ranked Probability Score)
- Coverage@90% (Empirical coverage of 90% prediction intervals)
- Interval Score@90% (Sharpness + miscoverage penalty)
- PICA (Prediction Interval Coverage Accuracy)

### Computational Cost
- **Fit time**: Mean ± std across folds (seconds)

## Usage

```bash
# Run the analysis
pixi run python benchmarks/effect_n_trees.py

# Results will be saved to:
# - benchmarks/results/effect_n_trees/effect_n_trees.json
# - benchmarks/plots/effect_n_trees/*.png
```

## Outputs

### JSON Structure

```json
{
  "config": {
    "n_trees_grid": [5, 10, 25, 50, 100, 200],
    "n_folds": 10,
    "datasets": ["friedman1", "friedman2", "friedman3", "make_regression"]
  },
  "experiments": {
    "friedman1": {
      "n_trees_results": {
        "5": {
          "aggregated_metrics": {
            "crps": {"mean": 0.xx, "std": 0.xx},
            "rmse": {"mean": 0.xx, "std": 0.xx},
            ...
          },
          "mean_fit_time": 0.xx,
          "std_fit_time": 0.xx,
          "best_init_kwargs": {...},
          "best_params": {...}
        },
        ...
      }
    },
    ...
  }
}
```

### Plots Generated

All plots are 2x2 grids (one subplot per DGP) with n_trees on x-axis (log scale).

| Plot File | Y-axis | Error Bars |
|-----------|--------|------------|
| `rmse_vs_n_trees.png` | RMSE | ± 1 std |
| `mae_vs_n_trees.png` | MAE | ± 1 std |
| `r2_vs_n_trees.png` | R² | ± 1 std |
| `crps_vs_n_trees.png` | CRPS | ± 1 std |
| `coverage_90_vs_n_trees.png` | Coverage@90% | ± 1 std |
| `interval_score_90_vs_n_trees.png` | Interval Score@90% | ± 1 std |
| `pica_vs_n_trees.png` | PICA | ± 1 std |
| `fit_time_vs_n_trees.png` | Fit Time (s) | ± 1 std (log-log scale) |

## Expected Patterns

### Performance Metrics (CRPS, RMSE, etc.)
- **Initial improvement**: Rapid improvement from 5 → 25 trees
- **Plateau**: Diminishing returns after ~50 trees
- **Convergence**: Minimal change 100 → 200 trees

### Calibration Metrics (Coverage, PICA)
- Coverage should stabilize near nominal level (0.90) as n_trees increases
- PICA should decrease (better calibration) with more trees

### Fit Time
- **Linear scaling**: Doubling n_trees should approximately double fit time
- Slope on log-log plot ≈ 1.0 (confirms linearity)

## Interpretation Guide

1. **Sufficient n_trees**: Identify where metric curves flatten (e.g., 50 trees)
2. **Optimal n_trees**: Balance performance gain vs computational cost
3. **Variance reduction**: Error bars should narrow with more trees (ensemble averaging)
4. **Calibration convergence**: Coverage/PICA should stabilize at larger n_trees

## Comparison with Other Methods

For publication, compare convergence rate to:
- **Random Forest**: Should plateau similarly
- **NGBoost**: May need fewer trees due to boosting
- **Gradient Boosting**: Different convergence (sequential vs parallel)

## Extension for Classification

Replace:
- `BDFRegressor` → `BDFClassifier`
- `NormalMuNormal` → `BetaMvBernoulli`
- DGPs → `make_classification`, etc.
- Metrics → Brier score, log-loss, calibration error

## Notes

- Each n_trees value gets independently tuned hyperparameters (not shared)
- Tuning on fold 0 means 9 evaluation folds, sufficient for stable estimates
- Log scale x-axis makes it easier to see early gains vs later plateau
- Saved Optuna studies allow post-hoc analysis of tuning process
