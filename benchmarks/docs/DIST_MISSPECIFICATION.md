# Distribution Misspecification Study

## Motivation

BDF's core claim is that choosing the right distributional assumption improves predictive performance. This study provides direct evidence by measuring the cost of misspecification: what happens when the assumed distribution is wrong?

## Experimental Design

### DGPs x Distributions Cross-Design

5 synthetic DGPs, each the "home" of one BDF distribution variant:

| DGP | Home Distribution | Key Property |
|-----|------------------|-------------|
| Gaussian Heteroscedastic | NormalMuNormal | `y = 2x1 + sin(3x2) + N(0, σ(x)²)` |
| Poisson Count | GammaMVLambdaPoisson | `y ~ Poisson(exp(0.5 + 0.8x1 + 0.3x2))` |
| Exponential Waiting-Time | GammaMVLambdaExponential | `y ~ Exp(rate = 0.5 + 0.3x1 + 0.2x2)` |
| Heavy-Tailed | FrequentistStudentT | `y = x1² + x2 + t(df=3) * 0.5` |
| Multimodal Mixture | KDE | `y ~ 0.5*N(2x, σ²) + 0.5*N(2x + 2sin(2πx), σ²)` |

All 25 cells (5 DGPs x 5 distributions) are attempted. 7 cells are domain-incompatible (Poisson/Exponential on data with negative values, plus Exponential on count data which includes zeros).

### Protocol

- **CV**: 10-fold, fold 0 for Optuna tuning (50 trials on CRPS), folds 1-9 for evaluation
- **Seeds**: 5 complete repetitions → 45 evaluation observations per cell
- **Tuning**: Independent per (DGP, distribution) pair
- **n_samples**: 5000 per DGP

### Metrics

**Primary**: CRPS, PIT-KS, PICA, Coverage@90%
**Ground truth**: gt_mean_rmse, gt_variance_rmse

### Statistical Tests

- Friedman test per DGP row (are distributions significantly different?)
- Nemenyi post-hoc (which pairs differ?)
- Wilcoxon signed-rank: HOME vs each other distribution

## Usage

```bash
# Full study (~hours)
python -m benchmarks.dist_misspecification_study

# Smoke test (~minutes)
python -m benchmarks.dist_misspecification_study --quick

# Generate plots and tables (after study completes)
python -m benchmarks.plot_misspecification_results
```

## Outputs

- `benchmarks/results/dist_misspecification/results.parquet` — tidy DataFrame
- `benchmarks/results/dist_misspecification/summary.csv` — aggregated summary
- `benchmarks/plots/dist_misspecification/misspecification_heatmap_relative.pdf` — main figure
- `benchmarks/plots/dist_misspecification/pit_calibration_panel.pdf` — calibration diagnostic
- `benchmarks/plots/dist_misspecification/coverage_comparison.pdf` — coverage bar charts
- `benchmarks/tables/dist_misspecification/misspecification_crps.tex` — LaTeX table
- `benchmarks/results/dist_misspecification/statistical_tests.json` — significance tests

## Expected Findings

1. **Diagonal dominance**: HOME distribution achieves best CRPS for its DGP
2. **KDE as universal second-best**: nonparametric, adapts to any shape
3. **Domain mismatch is fatal**: Poisson/Exponential cannot run on real-valued data
4. **Student-t degrades gracefully**: approaches Normal when data is Gaussian
5. **Misspecification hurts uncertainty more than mean**: CRPS gaps larger than RMSE gaps
