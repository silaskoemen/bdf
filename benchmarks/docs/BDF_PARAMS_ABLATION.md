# BDF Parameter Ablation Study

Systematic one-at-a-time (OAT) ablation study investigating the sensitivity of BDF performance to its key hyperparameters.

## Purpose

This benchmark answers key questions for a publication:
1. **Do the regularization terms matter?** (comparing on vs off)
2. **How sensitive is performance to parameter values?** (sensitivity curves)
3. **Does NLE outperform NLL+BIC?** (scoring method comparison with statistical tests)

## Parameters Ablated

| Parameter | Description | Grid Values |
|-----------|-------------|-------------|
| `reg_lambda` | Prior split probability penalty | 0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0 |
| `reg_gamma` | Multiplicity correction | 0, 1e-3, 0.01, 0.1, 0.5, 1.0 |
| `reg_nu` | Depth penalty | 0, 1e-3, 0.01, 0.1, 0.5, 1.0 |
| `min_samples_leaf` | Minimum samples per leaf | 5, 10, 25, 50 |
| Scoring method | NLE vs NLL+BIC vs NLL | nle, nll_bic, nll |

## Default Values

When varying one parameter, others are held at:
- `reg_lambda`: 0.01
- `reg_gamma`: 0.1
- `reg_nu`: 0.01
- `min_samples_leaf`: 10
- `n_trees`: 50
- Distribution: `NormalMuNormal` with `mu_mu="auto"`, `sigma_mu="auto"`

## Data Generating Processes

- **friedman1**: 5 informative features, nonlinear interactions, noise=1.0
- **friedman2**: 4 features, multiplicative interactions, noise=1.0
- **friedman3**: 4 features, arctan function, noise=0.1
- **make_regression**: 5 informative of 10 features, linear, noise=10.0

## Metrics

**Primary metric**: CRPS (Continuous Ranked Probability Score)

**Additional metrics**:
- Point: RMSE, MAE
- Probabilistic: Coverage@90%, Interval Score@90%

## Experimental Design

- **OAT approach**: Vary one parameter at a time, hold others at defaults
- **Seeds**: 10 random seeds per configuration for confidence intervals
- **Sample size**: n=1000 per DGP
- **Train/test split**: 80/20

Total experiments: ~1,150 (4 DGPs × ~23 param values × 10 seeds + 120 scoring comparisons)

## Usage

```bash
# Run the full ablation study
pixi run python benchmarks/effect_bdf_params.py

# The script will:
# 1. Run all experiments (saves intermediate results)
# 2. Generate plots in benchmarks/plots/effect_bdf_params/
# 3. Generate summary tables in benchmarks/results/effect_bdf_params/tables/
```

## Configuration

Edit constants at the top of `effect_bdf_params.py`:
- `N_SEEDS`: Number of random seeds (default: 10)
- `N_SAMPLES`: Dataset size (default: 1000)
- `ABLATION_GRIDS`: Parameter grids to sweep
- `DEFAULT_PARAMS`: Default values when varying other params

## Outputs

### Plots

| File | Description |
|------|-------------|
| `sensitivity_{param}.png` | 2x2 grid showing param sensitivity per DGP |
| `combined_sensitivity.png` | All params in one figure, lines for each DGP |
| `scoring_comparison.png` | Bar plot comparing NLE vs NLL+BIC vs NLL |
| `sensitivity_heatmap.png` | Heatmap of % improvement over default |
| `combined_sensitivity_{metric}.png` | Alternative metrics (RMSE, coverage) |

### Tables

| File | Description |
|------|-------------|
| `best_values.csv` | Best parameter value per DGP with improvement % |
| `scoring_comparison.csv` | Scoring method comparison with Wilcoxon p-values |
| `sensitivity_ranking.csv` | Parameters ranked by sensitivity |

## Resumability

The script saves results after each parameter ablation. If interrupted, rerunning will:
1. Load existing results from `benchmarks/results/effect_bdf_params/`
2. Skip completed ablations
3. Continue with remaining parameters

To regenerate plots from existing results, simply rerun after all ablations complete.

## Extension to Classification

The script is designed to be extensible for classification. Key changes needed:
- Add `BDFClassifier` with `BetaMvBernoulli` distribution
- Classification DGPs (e.g., `make_classification`)
- Classification metrics (accuracy, log-loss, Brier score)

NLE is expected to show larger improvement for classification because Bernoulli is the canonical distribution for binary outcomes.

## Statistical Testing

For scoring method comparison:
- **Wilcoxon signed-rank test**: NLE vs NLL (paired, one-sided)
- Reports p-values for hypothesis: NLE < NLL

## Interpretation Guide

1. **Flat sensitivity curves**: Parameter has little effect, default is fine
2. **U-shaped curves**: There's an optimal range, extreme values hurt
3. **Monotonic curves**: Clear trend, consider adjusting default
4. **Large CI bands**: High variance across seeds, need more data/seeds
