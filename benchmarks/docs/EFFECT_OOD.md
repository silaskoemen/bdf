# Out-of-Distribution (OOD) / Covariate Shift Analysis

Investigates how distributional regression models behave under covariate shift, focusing on whether uncertainty estimates appropriately increase in OOD regions.

## Research Questions

1. **Does uncertainty increase in OOD regions?** (desirable for reliable UQ)
2. **Does coverage degrade gracefully or catastrophically?**
3. **Which models are most robust to distribution shift?**

## Experimental Design

### Covariate Shift Setup

Training and test data are generated from known Friedman functions, but with different feature ranges:

| Shift Level | Train Range | Test Range | Overlap | OOD (extrapolation) |
|-------------|-------------|------------|---------|---------------------|
| `none` | [0, 1] | [0, 1] | 100% | 0% |
| `mild` | [0, 0.85] | [0.15, 1.0] | 70% | 15% |
| `moderate` | [0, 0.7] | [0.3, 1.0] | 40% | 30% |
| `strong` | [0, 0.55] | [0.45, 1.0] | 10% | 45% |

### DGPs

**Friedman 1**: `10*sin(π*x₁*x₂) + 20*(x₃-0.5)² + 10*x₄ + 5*x₅`
- 10 features (5 informative + 5 noise)
- Noise: σ = 1.0

**Friedman 2**: `√(x₁² + (x₂*x₃ - 1/(x₂*x₄))²)`
- 4 features (all informative)
- Noise: σ = 125.0 (scaled to original)

**Friedman 3**: `arctan((x₂*x₃ - 1/(x₂*x₄)) / x₁)`
- 4 features (all informative)
- Noise: σ = 0.1

### Methodology

1. Generate training data from restricted feature range
2. Tune hyperparameters on fold 0 (within training range)
3. Evaluate on folds 1-9 with test data from shifted range
4. Compute metrics **separately** for:
   - **Overlap region**: Test points in intersection of train/test ranges
   - **OOD region**: Test points outside training range (extrapolation)

## Key Metrics

### Region-Specific Metrics

| Metric | Description |
|--------|-------------|
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `mean_uncertainty` | Mean predicted std dev |
| `crps` | Continuous Ranked Probability Score |
| `coverage_90` | Empirical coverage of 90% intervals |
| `ci_width_90` | Mean width of 90% confidence intervals |

### Aggregated Metric

**Uncertainty Ratio (OOD / Overlap)**: Key metric for UQ quality
- Ratio > 1: Model correctly increases uncertainty in OOD regions
- Ratio ≈ 1: Model overconfident in OOD regions (bad)
- Ratio >> 1: May indicate overly conservative predictions

## Usage

```bash
# Run in default environment (BDF + RandomForest)
pixi run python benchmarks/effect_out_of_distribution.py

# Run in bench-models environment (full comparison)
pixi run -e bench-models python benchmarks/effect_out_of_distribution.py
```

## Outputs

### Results Directory: `benchmarks/results/effect_ood/`

| File | Description |
|------|-------------|
| `{dgp}_{shift}_results_core.json` | Per-experiment results (core models) |
| `{dgp}_{shift}_results.json` | Per-experiment results (full comparison) |
| `all_results_core.json` | Combined results (core models) |
| `all_results.json` | Combined results (full comparison) |

### Plots Directory: `benchmarks/plots/effect_ood/`

| Plot | Description |
|------|-------------|
| `uncertainty_ratio_by_shift.png` | Uncertainty ratio vs shift level (1x3 grid per DGP) |
| `coverage_by_shift.png` | Coverage in OOD vs overlap regions |
| `rmse_ood_by_shift.png` | RMSE in OOD region vs shift level |

## Interpretation Guide

### Uncertainty Ratio Plot
- **Ideal**: Ratio increases with shift level (more uncertainty when extrapolating)
- **Bayesian models** (BDF with NLE): Should show increasing ratio
- **Conformal methods**: May show flat ratio (marginal coverage guarantee)

### Coverage Degradation Plot
- **Graceful**: Coverage slowly decreases in OOD, stays near nominal in overlap
- **Catastrophic**: Coverage drops sharply even for mild shifts

### RMSE Plot
- Shows predictive accuracy degradation
- Useful for comparing models' extrapolation robustness

## Statistical Tests

The `compare_ood_results.py` script runs the following significance tests using fold-level data (9 folds per cell):

| Test | Purpose | Applied to |
|------|---------|-----------|
| Friedman (Iman-Davenport) | Do models differ significantly? | Per (DGP, shift) on OOD CRPS |
| Nemenyi post-hoc | Which model pairs differ? | After significant Friedman |
| Pairwise Wilcoxon + Holm | Best model vs each other | Per (DGP, shift) on OOD CRPS |
| One-sample Wilcoxon (one-sided) | Is uncertainty ratio > 1? | Per (DGP, shift, model) |

Only probabilistic models (with CRPS) are included in Friedman/Wilcoxon tests.
Outputs: `tables/statistical_tests.json` and `tables/statistical_tests.md`.

## Expected Results

For well-calibrated probabilistic models:
1. **Coverage in overlap** should stay near 90% across all shift levels
2. **Coverage in OOD** may drop, but should not collapse
3. **Uncertainty ratio** should increase with shift level
4. **RMSE ratio** (OOD/overlap) should also increase, but less than uncalibrated models

## Known Limitations

**BDF uncertainty does not always increase in OOD regions.** Unlike Gaussian Processes, which have an explicit distance-aware covariance structure, BDF's uncertainty comes from Bayesian leaf posteriors conditioned on the data falling into each leaf. Under strong covariate shift, OOD points may land in leaves that happen to contain sufficient training data (from the overlap region), producing uncertainty estimates comparable to in-distribution predictions. This is a structural property of tree-based methods: they partition the feature space into discrete regions and cannot extrapolate uncertainty beyond the partition boundaries.

Specifically:
- **GP** consistently shows uncertainty ratio > 1 across all shifts (significant at p < 0.01)
- **BDF** shows significant uncertainty increase only for mild shifts on some DGPs (e.g., friedman3), but fails to increase uncertainty under moderate-to-strong shift for most DGPs
- This limitation is shared with other tree-based distributional methods and is not unique to BDF

The main text should discuss this as a known trade-off: BDF provides better calibrated predictive distributions in-distribution and on mildly shifted data, while GP provides better OOD uncertainty awareness at the cost of scalability and in-distribution flexibility.

## Aggregate Analysis

For publication-ready tables and combined plots across `_core` and full results:

```bash
pixi run python benchmarks/compare_ood_results.py
```

This script:
1. Loads both `all_results_core.json` and `all_results.json`
2. Combines results (prefers full env when duplicates exist)
3. Generates summary tables (CSV + Markdown):
   - `crps_overall.csv` - CRPS comparison (all regions)
   - `crps_ood.csv` - CRPS in OOD region only
   - `coverage_ood.csv` - Coverage @ 90% in OOD region
   - `uncertainty_ratio.csv` - Key UQ quality metric
   - `model_rankings.csv` - Best model per (DGP, shift)
4. Generates aggregate plots:
   - `heatmap_crps_overall.png` - Heatmap across all conditions
   - `heatmap_crps_ood.png` - OOD-specific heatmap
   - `uncertainty_ratio_strong_shift.png` - Bar chart for strong shift
   - `coverage_degradation.png` - Coverage vs shift level
   - `crps_comparison_strong_shift.png` - In-dist vs OOD comparison

Output locations:
- `benchmarks/results/effect_ood/tables/`
- `benchmarks/plots/effect_ood/aggregate/`

## Connection to Theory

This experiment tests **epistemic uncertainty quantification**:
- **Aleatoric uncertainty** (noise) should be constant across regions
- **Epistemic uncertainty** (model uncertainty) should increase in OOD regions

Bayesian methods (BDF with NLE scoring) explicitly model epistemic uncertainty through the posterior, so should show appropriate uncertainty increase. Frequentist methods may struggle.

## Total Experiments

- 3 DGPs × 4 shift levels × 4+ models × 9 eval folds = ~430+ model fits
- Plus 50 Optuna trials × 12 tuning runs = 600 tuning fits
