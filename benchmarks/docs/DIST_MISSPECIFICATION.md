# Distribution Misspecification Study

## Motivation

BDF's core claim is that choosing the right distributional assumption improves predictive performance. This study provides direct evidence by measuring the cost of misspecification: what happens when the assumed distribution is wrong?

## Experimental Design

### DGPs x Distributions Cross-Design

6 synthetic DGPs, each the "home" of one BDF distribution variant, crossed with 7 distribution models (42 cells total):

| DGP | Home Distribution | Key Property |
|-----|------------------|-------------|
| Gaussian Heteroscedastic | NormalMuNormal | `y = 2x1 + sin(3x2) + N(0, σ(x)²)` |
| Poisson Count | GammaMVLambdaPoisson | `y ~ Poisson(exp(0.5 + 0.8x1 + 0.3x2))` |
| Exponential Waiting-Time | GammaMVLambdaExponential | `y ~ Exp(rate = 0.5 + 0.3x1 + 0.2x2)` |
| Heavy-Tailed | FrequentistStudentT | `y = x1² + x2 + t(df=3) * 0.5` |
| Multimodal Mixture | KDE | `y ~ 0.5*N(2x, σ²) + 0.5*N(2x + 2sin(2πx), σ²)` |
| Skewed Heteroscedastic | NormalMeanPseudoAlphaSkewNormal | `y ~ SkewNormal(α=5, loc=f(x), scale=σ(x))` |

**Distribution models**: NormalMuNormal, NormalMuInvGammaSigmaNormal (NIG), NormalMeanPseudoAlphaSkewNormal (Skew-Normal), GammaMVLambdaPoisson, GammaMVLambdaExponential, FrequentistStudentT, KDE.

9 cells are domain-incompatible (Poisson/Exponential on data with negative values or zeros) and are skipped.

### Protocol

- **CV**: 10-fold, fold 0 for Optuna tuning (50 trials on CRPS), folds 1-9 for evaluation
- **Seeds**: 5 complete repetitions → 45 evaluation observations per cell
- **Tuning**: Independent per (DGP, distribution) pair
- **n_samples**: 5000 per DGP

### Scoring Defaults

- **Conjugate models** (Normal, NIG, Poisson, Exponential): NLE (Bayesian evidence) or NLL with BIC, tuned via Optuna
- **Non-conjugate models** (Skew-Normal, Student-t, KDE): NLL + BIC (fixed). LOO-CV is avoided for non-conjugate distributions as they lack fast closed-form LOOCV (KDE is an exception where LOO-CV may be tuned)

### Metrics

**Primary**: CRPS, PIT-KS, PICA, Coverage@90%
**Ground truth**: gt_mean_rmse, gt_variance_rmse
**Tree complexity**: avg_depth, avg_nodes (logged per cell and stored in results)

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

**Results:**
- `benchmarks/results/dist_misspecification/results.parquet` — tidy DataFrame
- `benchmarks/results/dist_misspecification/summary.csv` — aggregated summary
- `benchmarks/results/dist_misspecification/statistical_tests.json` — significance tests

**Main paper figures:**
- `misspecification_heatmap_relative.pdf` — relative CRPS heatmap (NxM)
- `degradation_summary.pdf` — per-DGP dot plot of misspecification cost
- `cd_diagrams/` — Nemenyi critical difference diagrams per DGP

**Appendix figures:**
- `misspecification_heatmap_absolute.pdf` — absolute CRPS heatmap
- `pit_calibration_panel.pdf` — PIT histogram grid (NxM)
- `coverage_comparison.pdf` — coverage bar charts

**Tables:**
- `benchmarks/tables/dist_misspecification/misspecification_crps.tex` — CRPS LaTeX table
- `benchmarks/tables/dist_misspecification/tree_complexity.tex` — avg depth / nodes per cell

## Expected Findings

1. **Diagonal dominance**: HOME distribution achieves best CRPS for its DGP
2. **KDE as universal second-best**: nonparametric, adapts to any shape
3. **Domain mismatch is fatal**: Poisson/Exponential cannot run on real-valued data
4. **Student-t degrades gracefully**: approaches Normal when data is Gaussian
5. **Misspecification hurts uncertainty more than mean**: CRPS gaps larger than RMSE gaps
6. **NIG handles heteroscedasticity**: joint (μ, σ²) inference adapts scale per leaf
7. **Skew-Normal robustness**: non-conjugate model works reliably with NLL + BIC scoring
