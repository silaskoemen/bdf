# Scoring Method Ablation Study

Dedicated study investigating when NLE (Bayesian marginal likelihood) outperforms NLL-based scoring for tree split selection in BDF.

## Research Question

**"Under what conditions does NLE outperform NLL-based scoring, and by how much?"**

## Motivation

This study addresses a key methodological claim of BDF: that using Bayesian marginal likelihood (NLE) for split selection provides better generalization than plug-in negative log-likelihood (NLL), especially on small samples.

The parameter ablation study (`effect_bdf_params.py`) treats scoring as one of many hyperparameters. This study makes scoring the **central focus** with rigorous statistical testing suitable for JMLR publication.

## Hypotheses

| # | Hypothesis | Mechanism |
|---|------------|-----------|
| H1 | NLE dominates at small n (≤250) | Automatic Occam's razor via marginal likelihood |
| H2 | NLE and NLL converge at large n (≥1000) | Data overwhelms prior |
| H3 | NLE provides better calibration | Posterior predictive integrates parameter uncertainty |
| H4 | NLE is more robust to noise | Marginal likelihood penalizes overfitting |
| H5 | NLE handles class imbalance better | Prior regularizes extreme probabilities |
| H6 | No computational overhead | Closed-form for conjugate distributions |

## Experimental Design

### Dimensions

| Dimension | Values | Purpose |
|-----------|--------|---------|
| Sample size (n) | 50, 100, 250, 500, 1000, 2500 | Test H1, H2 |
| Scoring method | nle, nll, nll_aic, nll_bic, nll_loo | Compare all approaches |
| Noise level (regression) | low (σ=0.5), medium (σ=1.0), high (σ=2.0) | Test H4 |
| Class imbalance | balanced (0.5), moderate (0.3), severe (0.1) | Test H5 |
| Seeds | 30 | Statistical power |

### Total Experiments

- **Regression**: 6 × 5 × 3 × 2 × 30 = 5,400 experiments
- **Classification**: 6 × 4 × 3 × 3 × 30 = 6,480 experiments (no LOO-CV)

### Data Generating Processes

**Regression** (NormalMuNormal):
- `friedman1`: Nonlinear, 5 of 10 features informative
- `linear`: Linear, 5 of 10 features informative

**Classification** (BetaABBernoulli):
- `make_classification`: Linear boundary, 5% label noise
- `moons`: Nonlinear boundary
- `circles`: Concentric boundary

### Scoring Configurations

```python
SCORING_CONFIGS = {
    "nle":     {"score_method": "nle", "score_correction": None},
    "nll":     {"score_method": "nll", "score_correction": None},
    "nll_aic": {"score_method": "nll", "score_correction": "aic"},
    "nll_bic": {"score_method": "nll", "score_correction": "bic"},
    "nll_loo": {"score_method": "nll", "score_correction": "loo_cv"},
}
```

Note: `nll_loo` is only used for regression (NormalMuNormal has fast LOO-CV).

## Metrics

### Regression

| Metric | Type | Description |
|--------|------|-------------|
| **CRPS** | Probabilistic | Primary metric - full distribution quality |
| RMSE | Point | Root mean squared error |
| MAE | Point | Mean absolute error |
| Coverage@90% | Calibration | Fraction of true values in 90% interval |
| Interval Width@90% | Calibration | Sharpness |
| PIT KS-stat | Calibration | Kolmogorov-Smirnov test for PIT uniformity |

### Classification

| Metric | Type | Description |
|--------|------|-------------|
| **Log Loss** | Probabilistic | Primary metric - cross-entropy |
| Brier Score | Probabilistic | Quadratic scoring rule |
| AUROC | Discrimination | Ranking ability |
| ECE | Calibration | Expected Calibration Error |
| MCE | Calibration | Maximum Calibration Error |
| Accuracy | Point | Classification accuracy |

## Statistical Analysis

### Primary: Friedman Test + Nemenyi Post-hoc

For each sample size:
1. Friedman test across all scoring methods
2. Nemenyi post-hoc test
3. Critical Difference (CD) diagram

### Secondary

| Analysis | Test | Purpose |
|----------|------|---------|
| NLE vs each NLL | Wilcoxon signed-rank | Pairwise comparison |
| Effect sizes | Cliff's delta | Magnitude of difference |
| Noise interaction | 2-way comparison | Test H4 |
| Imbalance interaction | 2-way comparison | Test H5 |

## Usage

```bash
# Full study (both regression and classification)
pixi run scoring-study

# Regression only
pixi run scoring-study-reg

# Classification only
pixi run scoring-study-clas

# Quick test (reduced seeds and sample sizes)
pixi run scoring-study-quick

# Analyze existing results only
pixi run python -m benchmarks.scoring_method_study --analyze-only

# Start fresh (don't resume)
pixi run python -m benchmarks.scoring_method_study --no-resume
```

## Outputs

### Directory Structure

```
benchmarks/
├── results/scoring_method/
│   ├── regression/
│   │   ├── raw_results.parquet      # All experiment results
│   │   ├── summary_by_n.csv         # Aggregated by sample size
│   │   ├── best_by_n.csv            # Best method per sample size
│   │   └── statistical_tests.json   # Friedman, Wilcoxon, ranks
│   └── classification/
│       └── ... (same structure)
├── plots/scoring_method/
│   ├── regression/
│   │   ├── interaction_crps.png     # THE key figure
│   │   ├── cd_diagram_overall.png   # Critical difference
│   │   ├── calibration_coverage.png
│   │   └── noise_interaction.png
│   └── classification/
│       ├── interaction_logloss.png
│       ├── cd_diagram_overall.png
│       ├── calibration_ece.png
│       └── imbalance_interaction.png
└── tables/scoring_method/
    └── ... (LaTeX tables)
```

### Key Figures

**Figure 1: Interaction Plot** (primary result)
- X-axis: Sample size (log scale)
- Y-axis: Primary metric (CRPS or Log Loss)
- Lines: One per scoring method with error bars

**Figure 2: Critical Difference Diagram**
- Average ranks across all experiments
- Nemenyi significance bars
- Groups methods that are not significantly different

**Figure 3: Calibration**
- Boxplots of ECE (classification) or coverage (regression)
- By scoring method

## Resumability

The study saves checkpoints after each configuration block. If interrupted:
1. Results are preserved in `raw_results.parquet`
2. Rerunning will skip completed experiments
3. Use `--no-resume` to start fresh

## Expected Results

| Sample Size | Expected Winner | Expected Magnitude |
|-------------|-----------------|-------------------|
| n = 50 | NLE | 5-15% improvement |
| n = 100 | NLE | 3-8% improvement |
| n = 250 | NLE | 1-5% improvement |
| n = 500 | NLE ≈ NLL_BIC | <2% difference |
| n ≥ 1000 | No significant difference | - |

## Differences from Parameter Ablation Study

| Aspect | Parameter Ablation | Scoring Method Study |
|--------|-------------------|---------------------|
| Focus | Multiple parameters | Scoring only |
| Sample sizes | Fixed (n=1000) | Variable (50-2500) |
| Scoring methods | 3 | 5 |
| Statistical tests | Wilcoxon | Friedman + Nemenyi + CD |
| Calibration metrics | Coverage | ECE, MCE, PIT |
| Conditions | Fixed | Noise/imbalance varied |
| Seeds | 10 | 30 |

## Interpretation Guide

1. **Interaction plot shape**: Converging lines = NLE advantage diminishes with n
2. **CD diagram**: Methods connected by bars are not significantly different
3. **Cliff's delta**: |d| < 0.147 small, < 0.33 medium, ≥ 0.33 large effect
4. **Calibration**: NLE should show lower ECE and coverage closer to 90%

## Citation

If using this study design in publications, please cite the BDF paper and reference this benchmark.
