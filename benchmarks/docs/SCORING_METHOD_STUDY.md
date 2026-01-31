# Scoring Method Ablation Study

Dedicated study investigating when NLE (Bayesian marginal likelihood) outperforms NLL-based scoring for tree split selection in BDF. Designed for JMLR submission with rigorous statistical testing.

## Research Question

> "Under what conditions does NLE (Bayesian marginal likelihood) outperform NLL-based scoring for tree split selection, and by how much?"

This is a **core contribution** of BDF and novel because:
- Most tree methods use information gain/Gini (classification) or variance reduction (regression)
- BART uses different Bayesian framework (priors on tree structure)
- NGBoost/GAMLSS use NLL without corrections
- **No existing work systematically compares NLE vs NLL+BIC/AIC at split-finding level**

## Hypotheses

| # | Hypothesis | Mechanism |
|---|------------|-----------|
| H1 | NLE dominates at small n (n ≤ 500) | Automatic Occam's razor via marginal likelihood |
| H2 | NLE and NLL converge at large n (n ≥ 1000) | Data overwhelms prior |
| H3 | NLE provides better calibration | Posterior predictive integrates parameter uncertainty |
| H4 | NLE is more robust to noise | Marginal likelihood penalizes overfitting |
| H5 | NLE handles class imbalance better | Prior regularizes extreme probabilities |
| H6 | No computational overhead | Closed-form for conjugate distributions |

## Technical Notes

### LOO-CV Implementation Status

| Component | LOO-CV Speed | Notes |
|-----------|--------------|-------|
| **Python NormalMuNormal** | O(n) | Analytical formulas in `_loo_cv_log_likelihood()` |
| **Rust NormalMuNormal** | O(n²) | Naive loop in `loo_cv_nll()`, no analytical shortcut |
| **Python BetaABBernoulli** | O(n²) | `_has_fast_loo_cv = False` |
| **Rust BetaABBernoulli** | O(n²) | No fast implementation |

**Decision:** LOO-CV (`nll_loo`) is **excluded** from this study because:
- Rust's `score_split_from_stats()` returns `None` for LOO-CV
- Falls back to naive O(n²) implementation creating n arrays of size n-1
- For n=2500, this is ~6.25M array operations per split candidate
- BIC provides similar complexity penalty with O(1) cost

### Scoring Methods

```python
SCORING_CONFIGS = {
    "nle":     {"score_method": "nle", "score_correction": None},      # Bayesian (the claim)
    "nll":     {"score_method": "nll", "score_correction": None},      # Baseline
    "nll_aic": {"score_method": "nll", "score_correction": "aic"},     # 2k penalty
    "nll_bic": {"score_method": "nll", "score_correction": "bic"},     # k·log(n) penalty
    # "nll_loo": EXCLUDED - Rust implementation is O(n²)
}
```

### NLE Support by Distribution

| Distribution | NLE Support | Use Case |
|--------------|-------------|----------|
| NormalMuNormal | Yes | Regression |
| NormalMuInvGammaSigma | Yes | Regression (unknown variance) |
| BetaABBernoulli | Yes | Binary classification |
| BetaMVBernoulli | Yes | Binary classification |
| PoissonGammaAB | Yes | Count regression |
| ExponentialGammaAB | Yes | Survival/duration |
| KDE | No | Nonparametric |

## Experimental Design

### Study 1: Core Comparison (Primary Analysis)

**Research question:** Does NLE advantage depend on sample size?

| Dimension | Values | Notes |
|-----------|--------|-------|
| Sample sizes | [100, 250, 500, 1000, 2500] | Test H1, H2 |
| Scoring methods | [nle, nll, nll_aic, nll_bic] | 4 methods |
| Seeds | 20 | Sufficient for Friedman test (α=0.05) |
| DGPs | 2 per task | friedman1/linear (reg), make_classification/moons (clas) |
| Noise (regression) | medium (σ=1.0) | Fixed |
| Imbalance (classification) | balanced (0.5) | Fixed |

**Experiment count:** 5 × 4 × 20 × 2 = **800 per task** (1,600 total)

### Study 2: Robustness Analysis (Secondary)

**Research question:** Is NLE more robust to noise/imbalance?

| Dimension | Values | Notes |
|-----------|--------|-------|
| Sample size | 500 | Fixed |
| Seeds | 10 | |
| Noise levels (regression) | [low=0.5, medium=1.0, high=2.0] | Test H4 |
| Imbalance (classification) | [balanced=0.5, moderate=0.3, severe=0.1] | Test H5 |

**Experiment count:** 3 × 4 × 10 × 2 = **240 per task** (480 total)

### Study 3: Real Data Validation (Required for JMLR)

**Research question:** Do synthetic results generalize to real data?

**Regression datasets:**
| Dataset | n | p | Source | Why included |
|---------|---|---|--------|--------------|
| California Housing | 20,640 | 8 | sklearn | Large n baseline |
| Diabetes | 442 | 10 | sklearn | Small n case |

**Classification datasets:**
| Dataset | n | p | Source | Why included |
|---------|---|---|--------|--------------|
| Breast Cancer | 569 | 30 | sklearn | Small n, imbalanced |

**Protocol:** 5-fold CV × 5 seeds = 25 runs per config

**Experiment count:** ~1,000 experiments (extendable with OpenML datasets)

### Total Experiment Budget

| Component | Experiments | Est. Runtime |
|-----------|-------------|--------------|
| Core (synthetic) | 1,600 | 2-3 hours |
| Robustness | 480 | 1 hour |
| Real data | 1,000 | 3 hours |
| **Total** | **~3,080** | **~7 hours** |

## Data Generating Processes

### Regression DGPs

```python
REGRESSION_DGPS = {
    "friedman1": {
        # y = 10*sin(πx₁x₂) + 20*(x₃-0.5)² + 10*x₄ + 5*x₅ + ε
        "generator": make_friedman1,
        "n_features": 10,  # 5 informative, 5 noise
    },
    "linear": {
        # y = Xβ + ε, sparse coefficients
        "generator": make_regression,
        "n_features": 10,
        "n_informative": 5,
    },
}

NOISE_LEVELS = {"low": 0.5, "medium": 1.0, "high": 2.0}  # relative to y_std
```

### Classification DGPs

```python
CLASSIFICATION_DGPS = {
    "make_classification": {
        "generator": make_classification,
        "n_features": 10,
        "n_informative": 5,
        "n_redundant": 2,
        "flip_y": 0.05,  # 5% label noise
    },
    "moons": {
        "generator": make_moons,
        "noise": 0.2,
    },
}

IMBALANCE_LEVELS = {"balanced": 0.5, "moderate": 0.3, "severe": 0.1}
```

## Model Configuration

Fixed hyperparameters (not ablated):

```python
MODEL_CONFIG = {
    "n_trees": 50,
    "max_depth": 50,
    "min_samples_leaf": 10,
    "reg_lambda": 0.01,
    "reg_gamma": 0.1,
    "reg_nu": 0.01,
}

REG_DIST_CONFIG = {
    "dist": "NormalMuNormal",
    "params": {
        "mu_mu": "auto",
        "sigma_mu": "auto",
        "use_posterior_predictive": True,
    },
}

CLAS_DIST_CONFIG = {
    "dist": "BetaABBernoulli",
    "params": {
        "alpha_p": 1.0,  # Uniform Beta(1,1) prior
        "beta_p": 1.0,
        "use_posterior_predictive": True,
    },
}
```

## Metrics

### Primary Metrics

| Task | Primary Metric | Why |
|------|----------------|-----|
| Regression | **CRPS** | Gold standard for probabilistic forecasting |
| Classification | **Log Loss** | Gold standard for probability calibration |

### Secondary Metrics

**Regression:**
- RMSE (point prediction quality)
- Coverage@90% (calibration)
- Interval Width@90% (sharpness)

**Classification:**
- Brier Score (proper scoring rule)
- ECE (calibration)
- AUROC (discrimination)

## Statistical Analysis Plan

### Primary Analysis: Sample Size × Scoring Interaction

```
For each sample size n:
1. Friedman test across all 4 scoring methods
2. If p < 0.05:
   - Post-hoc Nemenyi test
   - Critical Difference (CD) diagram
3. Pairwise Wilcoxon signed-rank: NLE vs each NLL variant
4. Effect size: Cliff's delta
```

### Aggregated Analysis

```
Overall (pooled across all n):
1. Average rank across all experiments
2. CD diagram showing significant differences
3. Win/tie/loss table: NLE vs each competitor
```

### Effect Size Interpretation (Cliff's delta)

| |d| | Interpretation |
|-----|----------------|
| < 0.147 | Negligible |
| 0.147 - 0.33 | Small |
| 0.33 - 0.474 | Medium |
| > 0.474 | Large |

## Usage

```bash
# Full study (both regression and classification)
pixi run python benchmarks/scoring_method_study.py

# Regression only
pixi run python benchmarks/scoring_method_study.py regression

# Classification only
pixi run python benchmarks/scoring_method_study.py classification

# Quick test (reduced seeds and sample sizes)
pixi run python benchmarks/scoring_method_study.py --quick

# Run specific study types
pixi run python benchmarks/scoring_method_study.py --core        # Core study only
pixi run python benchmarks/scoring_method_study.py --robustness  # Robustness only
pixi run python benchmarks/scoring_method_study.py --real-data   # Real data only

# Analyze existing results only
pixi run python benchmarks/scoring_method_study.py --analyze-only

# Start fresh (don't resume)
pixi run python benchmarks/scoring_method_study.py --no-resume
```

## Outputs

### Directory Structure

```
benchmarks/
├── scoring_method_study.py           # Main script
├── results/scoring_method/
│   ├── regression/
│   │   ├── core_results.parquet      # Core study experiments
│   │   ├── robustness_results.parquet
│   │   ├── core_summary_by_n.csv
│   │   ├── core_best_by_n.csv
│   │   └── core_statistical_tests.json
│   ├── classification/
│   │   └── ... (same structure)
│   └── real_data/
│       ├── regression_results.parquet
│       └── classification_results.parquet
├── plots/scoring_method/
│   ├── regression/
│   │   ├── fig1_regression_interaction.pdf
│   │   ├── fig3_cd_diagram_core.pdf
│   │   └── fig4_regression_robustness.pdf
│   ├── classification/
│   │   └── ... (same structure)
│   └── real_data/
│       └── ...
└── tables/scoring_method/
    ├── regression/
    │   ├── core_summary_by_n.tex
    │   └── core_best_by_n.tex
    └── classification/
        └── ...
```

### Key Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| **Fig 1** | CRPS vs n (regression), lines with error bars | Show NLE advantage at small n |
| **Fig 2** | Log Loss vs n (classification), lines | Same pattern for classification |
| **Fig 3** | CD diagram (pooled across all experiments) | Statistical significance |
| **Fig 4** | Noise/imbalance robustness grouped bars | Secondary claims |
| **Fig 5** | Real data results (grouped bar chart) | External validity |

### Key Tables

| Table | Content |
|-------|---------|
| **Table 1** | Mean±std for each scoring method at each n (synthetic) |
| **Table 2** | Real data results (per dataset, best in bold) |
| **Table 3** | Statistical tests: Friedman p-values, effect sizes |
| **Table 4** | Computational cost comparison |

## Resumability

The study saves checkpoints after each configuration block. If interrupted:
1. Results are preserved in `.parquet` files
2. Rerunning will skip completed experiments
3. Use `--no-resume` to start fresh

## Expected Results

| Sample Size | Expected Winner | Expected Magnitude |
|-------------|-----------------|-------------------|
| n = 100 | NLE | 5-15% improvement |
| n = 250 | NLE | 3-8% improvement |
| n = 500 | NLE | 1-5% improvement |
| n = 1000 | NLE ≈ NLL_BIC | <2% difference |
| n = 2500 | No significant difference | - |

### Claims to Support

| Claim | Expected Evidence | How to Phrase |
|-------|-------------------|---------------|
| NLE dominates at small n | Significant at n ≤ 500 | "NLE provides X% lower CRPS at n=100" |
| Convergence at large n | No sig. diff at n ≥ 1000 | "Methods converge as n→∞" |
| Better calibration | Lower ECE for NLE | "NLE yields better-calibrated probabilities" |
| Robust to noise | Smaller CRPS increase | "NLE degrades more gracefully under noise" |
| No computational overhead | fit_time ≈ across methods | "Bayesian scoring adds negligible cost" |

## Potential Reviewer Concerns

| Concern | Mitigation |
|---------|-----------|
| "Only synthetic data" | Added real datasets (Study 3) |
| "Why not compare to BART/NGBoost?" | Not the point—we compare within BDF's scoring options. Cite prior work showing BDF competitive with baselines. |
| "Limited to conjugate distributions" | Acknowledge as limitation; note conjugate models cover most use cases |
| "Why no LOO-CV?" | Explain Rust implementation is O(n²); BIC is close approximation (both penalize complexity) |
| "Sample size range too narrow" | n=100 represents realistic small-data regime; n=2500 is where most methods converge |

## Interpretation Guide

1. **Interaction plot shape**: Converging lines = NLE advantage diminishes with n
2. **CD diagram**: Methods connected by bars are not significantly different
3. **Cliff's delta**: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ 0.474 large
4. **Calibration**: NLE should show lower ECE and coverage closer to 90%

## Differences from Parameter Ablation Study

| Aspect | Parameter Ablation | Scoring Method Study |
|--------|-------------------|---------------------|
| Focus | Multiple parameters | Scoring only |
| Sample sizes | Fixed (n=1000) | Variable (100-2500) |
| Scoring methods | 3 | 4 (nll_loo excluded) |
| Statistical tests | Wilcoxon | Friedman + Nemenyi + CD |
| Studies | Single | Core + Robustness + Real Data |
| Conditions | Fixed | Noise/imbalance varied |
| Seeds | 10 | 20 (core), 10 (robustness) |
| Real data | No | Yes (JMLR requirement) |
