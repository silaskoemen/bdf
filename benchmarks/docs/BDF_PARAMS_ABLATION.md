# BDF Parameter Ablation Study

Systematic one-at-a-time (OAT) ablation study investigating the sensitivity of BDF performance to its key hyperparameters. Supports both **regression** and **classification** tasks.

## Purpose

This benchmark answers key questions for a publication:
1. **Do the regularization terms matter?** (comparing on vs off)
2. **How sensitive is performance to parameter values?** (sensitivity curves)
3. **Does NLE outperform NLL+BIC?** (scoring method comparison with statistical tests)
4. **Do parameters behave differently for classification?** (regression vs classification comparison)

## Parameters Ablated

| Parameter | Description | Grid Values |
|-----------|-------------|-------------|
| `alpha` | Prior split probability penalty | 0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0 |
| `gamma` | Multiplicity correction | 0, 1e-3, 0.01, 0.1, 0.5, 1.0 |
| `delta` | Depth penalty | 0, 1e-3, 0.01, 0.1, 0.5, 1.0 |
| `min_samples_leaf` | Minimum samples per leaf | 5, 10, 25, 50 |
| Scoring method | NLE vs NLL+BIC vs NLL | nle, nll_bic, nll |

## Default Values

When varying one parameter, others are held at:
- `alpha`: 0.01
- `gamma`: 0.1
- `delta`: 0.01
- `min_samples_leaf`: 10
- `n_trees`: 50

### Regression Distribution
- `NormalMuNormal` with `mu_mu="auto"`, `sigma_mu="auto"`, `sigma_mu_auto_scale=1.0`

### Classification Distribution
- `BetaABBernoulli` with `alpha_p=1.0`, `beta_p=1.0` (uniform Beta(1,1) prior)

## Data Generating Processes

### Regression DGPs
- **friedman1**: 5 informative features, nonlinear interactions, noise=1.0
- **friedman2**: 4 features, multiplicative interactions, noise=1.0
- **friedman3**: 4 features, arctan function, noise=0.1
- **make_regression**: 5 informative of 10 features, linear, noise=10.0

### Classification DGPs
- **make_classification**: 5 informative of 10 features, 2 clusters/class, 5% label noise
- **moons**: Two interleaving half circles, noise=0.2
- **circles**: Two concentric circles, noise=0.1, factor=0.5

## Metrics

### Regression
**Primary metric**: CRPS (Continuous Ranked Probability Score)

**Additional metrics**:
- Point: RMSE, MAE
- Probabilistic: Coverage@90%, Interval Score@90%

### Classification
**Primary metric**: Log Loss

**Additional metrics**:
- Brier Score
- AUROC
- ECE (Expected Calibration Error)
- Accuracy

## Experimental Design

- **OAT approach**: Vary one parameter at a time, hold others at defaults
- **Seeds**: 10 random seeds per configuration for confidence intervals
- **Sample size**: n=1000 per DGP
- **Train/test split**: 80/20 (stratified for classification)

Total experiments per task type: ~1,150 (4 DGPs × ~23 param values × 10 seeds + 120 scoring comparisons)

## Usage

```bash
# Run both regression and classification ablation studies
pixi run effect-params

# Run regression only
pixi run effect-params-reg

# Run classification only
pixi run effect-params-clas

# Alternative: direct python invocation
pixi run python benchmarks/effect_bdf_params.py both          # Both
pixi run python benchmarks/effect_bdf_params.py regression    # Regression only
pixi run python benchmarks/effect_bdf_params.py classification # Classification only
```

## Configuration

Edit constants at the top of `effect_bdf_params.py`:
- `N_SEEDS`: Number of random seeds (default: 10)
- `N_SAMPLES`: Dataset size (default: 1000)
- `ABLATION_GRIDS`: Parameter grids to sweep
- `DEFAULT_PARAMS`: Default values when varying other params
- `DEFAULT_DIST_PARAMS`: Regression distribution parameters
- `DEFAULT_CLAS_DIST_PARAMS`: Classification distribution parameters

## Outputs

### Directory Structure

```
benchmarks/
├── results/effect_bdf_params/
│   ├── ablation_alpha.json
│   ├── ablation_gamma.json
│   ├── ablation_delta.json
│   ├── ablation_min_samples_leaf.json
│   ├── ablation_scoring.json
│   ├── tables/
│   │   ├── best_values.csv
│   │   ├── scoring_comparison.csv
│   │   └── sensitivity_ranking.csv
│   └── classification/
│       ├── ablation_*.json
│       └── tables/
├── plots/effect_bdf_params/
│   ├── sensitivity_*.png
│   ├── combined_sensitivity.png
│   ├── scoring_comparison.png
│   ├── sensitivity_heatmap.png
│   └── classification/
│       ├── clas_sensitivity_*.png
│       ├── clas_combined_sensitivity.png
│       ├── clas_scoring_comparison.png
│       └── clas_sensitivity_heatmap.png
```

### Regression Plots

| File | Description |
|------|-------------|
| `sensitivity_{param}.png` | 2x2 grid showing param sensitivity per DGP |
| `combined_sensitivity.png` | All params in one figure, lines for each DGP |
| `scoring_comparison.png` | Bar plot comparing NLE vs NLL+BIC vs NLL |
| `sensitivity_heatmap.png` | Heatmap of % improvement over default |
| `combined_sensitivity_{metric}.png` | Alternative metrics (RMSE, coverage) |

### Classification Plots

| File | Description |
|------|-------------|
| `clas_sensitivity_{param}.png` | 2x2 grid showing param sensitivity per DGP |
| `clas_combined_sensitivity.png` | All params in one figure, lines for each DGP |
| `clas_scoring_comparison.png` | Bar plot comparing NLE vs NLL+BIC vs NLL |
| `clas_sensitivity_heatmap.png` | Heatmap of % improvement over default |

### Tables

| File | Description |
|------|-------------|
| `best_values.csv` | Best parameter value per DGP with improvement % |
| `scoring_comparison.csv` | Scoring method comparison with Wilcoxon p-values |
| `sensitivity_ranking.csv` | Parameters ranked by sensitivity |

## Resumability

The script saves results after each parameter ablation. If interrupted, rerunning will:
1. Load existing results from the appropriate results directory
2. Skip completed ablations
3. Continue with remaining parameters

To regenerate plots from existing results, simply rerun after all ablations complete.

## Statistical Testing

For scoring method comparison:
- **Wilcoxon signed-rank test**: NLE vs NLL (paired, one-sided)
- Reports p-values for hypothesis: NLE < NLL (lower is better for both CRPS and Log Loss)

## Expected Results

### Regression
- NLE expected to outperform NLL for small sample sizes
- Regularization parameters should show U-shaped curves with optimal ranges
- `min_samples_leaf` trades off bias/variance

### Classification
- **NLE expected to show larger improvement** for classification because:
  - Bernoulli is the canonical distribution for binary outcomes
  - Discrete outcomes benefit more from Bayesian model selection
  - Posterior predictive integrates parameter uncertainty naturally
- ECE (calibration) should be better with NLE scoring

## Interpretation Guide

1. **Flat sensitivity curves**: Parameter has little effect, default is fine
2. **U-shaped curves**: There's an optimal range, extreme values hurt
3. **Monotonic curves**: Clear trend, consider adjusting default
4. **Large CI bands**: High variance across seeds, need more data/seeds
5. **Regression vs Classification differences**: Parameters may have different optimal ranges for different task types
