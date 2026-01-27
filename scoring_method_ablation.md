Scoring Method Ablation Study: Design Plan

  Research Question

  "Under what conditions does NLE (Bayesian marginal likelihood)
  outperform NLL-based scoring for tree split selection, and by how much?"

  ---
  1. Experimental Dimensions
  ┌──────────────────────┬────────────────────────┬──────────────────────┐
  │      Dimension       │         Values         │      Rationale       │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ Sample size (n)      │ 50, 100, 250, 500,     │ Core hypothesis: NLE │
  │                      │ 1000, 2500             │  shines at small n   │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │                      │ nle, nll, nll_aic,     │ Compare Bayesian vs  │
  │ Scoring method       │ nll_bic, nll_loo_cv    │ all frequentist      │
  │                      │                        │ corrections          │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ Signal-to-noise      │ Low (σ=0.5), Medium    │ NLE may handle noise │
  │ (regression)         │ (σ=1.0), High (σ=2.0)  │  differently         │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ Class imbalance      │ Balanced (0.5),        │ Prior matters more   │
  │ (classification)     │ Moderate (0.3), Severe │ when imbalanced      │
  │                      │  (0.1)                 │                      │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │                      │ Dense (all             │ Tests Occam's razor  │
  │ Feature sparsity     │ informative), Sparse   │ effect               │
  │                      │ (20% informative)      │                      │
  ├──────────────────────┼────────────────────────┼──────────────────────┤
  │ Seeds                │ 30                     │ Statistical power    │
  │                      │                        │ for Friedman test    │
  └──────────────────────┴────────────────────────┴──────────────────────┘
  Total experiments:
  - Regression: 6 × 5 × 3 × 2 × 30 = 5,400
  - Classification: 6 × 5 × 3 × 2 × 30 = 5,400

  ---
  2. Data Generating Processes

  2.1 Regression DGPs (NormalMuNormal)

  REGRESSION_DGPS = {
      # Canonical smooth function
      "friedman1": {
          "type": "friedman1",
          "n_features": 10,  # 5 informative
          "noise_levels": [0.5, 1.0, 2.0],
      },
      # Linear (model well-specified)
      "linear": {
          "type": "make_regression",
          "n_features": 10,
          "n_informative": 5,  # or 2 for sparse
          "noise_levels": [0.5, 1.0, 2.0],
      },
      # Nonlinear interactions
      "friedman2": {
          "type": "friedman2",
          "noise_levels": [0.5, 1.0, 2.0],  # scaled relative to y range
      },
  }

  2.2 Classification DGPs (BetaABBernoulli)

  CLASSIFICATION_DGPS = {
      # Linearly separable
      "linear_sep": {
          "type": "make_classification",
          "n_features": 10,
          "n_informative": 5,
          "class_sep": 1.0,
          "flip_y": 0.05,  # 5% label noise
          "weights": [0.5, 0.3, 0.1],  # balance levels
      },
      # Nonlinear boundary
      "moons": {
          "type": "make_moons",
          "noise": 0.2,
          "weights": [0.5, 0.3, 0.1],  # via undersampling
      },
      # Concentric
      "circles": {
          "type": "make_circles",
          "noise": 0.1,
          "factor": 0.5,
          "weights": [0.5, 0.3, 0.1],
      },
  }

  ---
  3. Scoring Configurations

  SCORING_CONFIGS = {
      # Bayesian (the claim)
      "nle": {
          "score_method": "nle",
          "score_correction": None,  # ignored for NLE
      },
      # Frequentist baselines
      "nll": {
          "score_method": "nll",
          "score_correction": None,  # no correction (baseline)
      },
      "nll_aic": {
          "score_method": "nll",
          "score_correction": "aic",  # 2k penalty
      },
      "nll_bic": {
          "score_method": "nll",
          "score_correction": "bic",  # k·log(n) penalty
      },
      "nll_loo": {
          "score_method": "nll",
          "score_correction": "loo_cv",  # leave-one-out
      },
  }

  ---
  4. Model Configuration

  Fixed hyperparameters (not ablated):

  MODEL_CONFIG = {
      "n_trees": 50,
      "max_depth": 50,
      "min_samples_leaf": 10,
      "reg_lambda": 0.01,
      "reg_gamma": 0.1,
      "reg_nu": 0.01,
  }

  # Distribution configs
  REG_DIST_CONFIG = {
      "dist": "NormalMuNormal",
      "params": {
          "mu_mu": "auto",
          "sigma_mu": "auto",
          "sigma_mu_auto_scale": 1.0,
          "use_posterior_predictive": True,
      },
  }

  CLAS_DIST_CONFIG = {
      "dist": "BetaABBernoulli",
      "params": {
          "alpha_p": 1.0,  # Uniform prior
          "beta_p": 1.0,
          "use_posterior_predictive": True,
      },
  }

  ---
  5. Metrics

  5.1 Regression Metrics
  Metric: CRPS
  Type: Probabilistic
  Purpose: Primary: full distribution quality
  ────────────────────────────────────────
  Metric: NLL (test)
  Type: Probabilistic
  Purpose: Log-likelihood on held-out data
  ────────────────────────────────────────
  Metric: RMSE
  Type: Point
  Purpose: Predictive accuracy
  ────────────────────────────────────────
  Metric: Coverage@90%
  Type: Calibration
  Purpose: Interval validity
  ────────────────────────────────────────
  Metric: Interval Width@90%
  Type: Calibration
  Purpose: Sharpness
  ────────────────────────────────────────
  Metric: PIT KS-statistic
  Type: Calibration
  Purpose: Uniformity of probability integral transform
  5.2 Classification Metrics
  ┌─────────────┬────────────────┬──────────────────────────────┐
  │   Metric    │      Type      │           Purpose            │
  ├─────────────┼────────────────┼──────────────────────────────┤
  │ Log Loss    │ Probabilistic  │ Primary: probability quality │
  ├─────────────┼────────────────┼──────────────────────────────┤
  │ Brier Score │ Probabilistic  │ Quadratic scoring rule       │
  ├─────────────┼────────────────┼──────────────────────────────┤
  │ ECE         │ Calibration    │ Expected Calibration Error   │
  ├─────────────┼────────────────┼──────────────────────────────┤
  │ MCE         │ Calibration    │ Maximum Calibration Error    │
  ├─────────────┼────────────────┼──────────────────────────────┤
  │ AUROC       │ Discrimination │ Ranking ability              │
  ├─────────────┼────────────────┼──────────────────────────────┤
  │ Accuracy    │ Point          │ Decision accuracy            │
  └─────────────┴────────────────┴──────────────────────────────┘
  ---
  6. Statistical Analysis

  6.1 Primary Analysis: Sample Size × Scoring Interaction

  Key figure: Interaction plot showing metric vs. sample size, with lines
  for each scoring method.

  Hypothesis tests per sample size:
  - Friedman test across all 5 scoring methods
  - Nemenyi post-hoc with Critical Difference (CD) diagram
  - Effect size: Cliff's delta for NLE vs NLL_BIC (the strongest
  competitor)

  6.2 Secondary Analyses
  Analysis: Overall ranking
  Test: Friedman + Nemenyi
  Output: CD diagram
  ────────────────────────────────────────
  Analysis: NLE vs each NLL variant
  Test: Wilcoxon signed-rank
  Output: p-values + effect sizes
  ────────────────────────────────────────
  Analysis: Sample size threshold
  Test: Changepoint detection
  Output: n* where NLE ≈ NLL_BIC
  ────────────────────────────────────────
  Analysis: Noise robustness
  Test: 2-way ANOVA
  Output: Interaction: scoring × noise
  ────────────────────────────────────────
  Analysis: Imbalance effect
  Test: 2-way ANOVA
  Output: Interaction: scoring × imbalance
  6.3 Calibration Analysis

  - Reliability diagrams: Per scoring method, averaged across DGPs
  - PIT histograms: Should be uniform for well-calibrated models
  - ECE boxplots: Per scoring method, showing variability

  ---
  7. Computational Cost Analysis

  Track and report:

  TIMING_METRICS = {
      "fit_time_seconds": "Wall-clock time for model.fit()",
      "predict_time_seconds": "Wall-clock time for model.predict(X_test)",
      "total_splits_evaluated": "Number of candidate splits scored",
  }

  Analysis:
  - Box plot of fit time by scoring method (expect NLE ≈ NLL for
  conjugate)
  - Scaling: fit time vs. sample size (should be similar across methods)

  ---
  8. Expected Results & Claims
  Claim: NLE dominates at small n
  Expected Evidence: Significant improvement at n ≤ 250, p < 0.01
  ────────────────────────────────────────
  Claim: Convergence at large n
  Expected Evidence: No significant difference at n ≥ 1000
  ────────────────────────────────────────
  Claim: Better calibration
  Expected Evidence: Lower ECE for NLE across all n
  ────────────────────────────────────────
  Claim: Robust to noise
  Expected Evidence: Smaller performance degradation for NLE as σ
  increases
  ────────────────────────────────────────
  Claim: Handles imbalance
  Expected Evidence: Larger NLE advantage when classes imbalanced
  ────────────────────────────────────────
  Claim: No computational overhead
  Expected Evidence: fit_time(NLE) ≈ fit_time(NLL_BIC)
  ---
  9. Output Structure

  benchmarks/
  ├── scoring_method_study.py           # Main script
  ├── docs/SCORING_METHOD_STUDY.md      # This document
  ├── results/scoring_method/
  │   ├── regression/
  │   │   ├── raw_results.parquet       # All experiment results
  │   │   ├── summary_by_n.csv          # Aggregated by sample size
  │   │   ├── statistical_tests.json    # Friedman, Wilcoxon, etc.
  │   │   └── timing.csv                # Computational costs
  │   └── classification/
  │       └── ... (same structure)
  ├── plots/scoring_method/
  │   ├── regression/
  │   │   ├── interaction_crps.pdf      # THE key figure
  │   │   ├── cd_diagram_overall.pdf    # Critical difference
  │   │   ├── cd_diagram_n50.pdf        # Per sample size
  │   │   ├── cd_diagram_n1000.pdf
  │   │   ├── calibration_reliability.pdf
  │   │   ├── calibration_pit.pdf
  │   │   ├── calibration_ece_boxplot.pdf
  │   │   ├── noise_interaction.pdf
  │   │   └── timing_comparison.pdf
  │   └── classification/
  │       ├── interaction_logloss.pdf
  │       ├── cd_diagram_overall.pdf
  │       ├── imbalance_interaction.pdf
  │       ├── calibration_ece_boxplot.pdf
  │       └── ...
  └── tables/scoring_method/
      ├── main_results.tex              # LaTeX table for paper
      ├── statistical_tests.tex
      └── timing.tex

  ---
  10. Key Figures for Paper

  Figure 1: Sample Size × Scoring Interaction (THE money figure)

  ┌─────────────────────────────────────────────────────┐
  │  CRPS vs Sample Size (Regression)                   │
  │                                                     │
  │  0.8│ ╲                                             │
  │     │  ╲  nll                                       │
  │  0.6│   ╲___________                                │
  │     │    ╲ nll_aic                                  │
  │  0.4│     ╲_________                                │
  │     │      ╲ nll_bic                                │
  │  0.2│       ╲_______───────────────                 │
  │     │        ╲ nle ─────────────────                │
  │  0.0├────┬────┬────┬────┬────┬────┬                 │
  │     50  100  250  500  1000 2500                    │
  │                Sample Size (n)                      │
  └─────────────────────────────────────────────────────┘

  Figure 2: Critical Difference Diagram

  Shows average rank across all experiments with Nemenyi significance
  bars.

  Figure 3: Calibration (ECE) by Scoring Method

  Box plots showing NLE has consistently lower ECE.

  ---
  11. Implementation Notes

  11.1 Handling LOO-CV

  NormalMuNormal has _has_fast_loo_cv = True, but BetaABBernoulli does
  not. For classification, LOO-CV will be slower. Consider:
  - Using nll_bic as primary NLL competitor for classification
  - Or implementing fast LOO-CV for Beta-Bernoulli (closed-form exists)

  11.2 Class Imbalance Implementation

  For moons/circles with imbalance:
  def create_imbalanced(X, y, minority_ratio=0.3):
      """Undersample majority class to achieve target ratio."""
      ...

  11.3 Parallelization

  # Use joblib for parallel execution across seeds
  from joblib import Parallel, delayed

  results = Parallel(n_jobs=-1)(
      delayed(run_single_experiment)(config, seed)
      for config in configs
      for seed in range(N_SEEDS)
  )

  ---
  12. Timeline Estimate
  Phase: Setup
  Tasks: Create script skeleton, DGP generators, metric functions
  ────────────────────────────────────────
  Phase: Run
  Tasks: Execute ~10,800 experiments (parallelized)
  ────────────────────────────────────────
  Phase: Analysis
  Tasks: Statistical tests, generate figures/tables
  ────────────────────────────────────────
  Phase: Documentation
  Tasks: Write results section, interpret findings
  ---
  13. Differences from Current Ablation Study
  Aspect: Focus
  Current (effect_bdf_params.py): Scoring is one of many params
  Proposed: Scoring is THE focus
  ────────────────────────────────────────
  Aspect: Sample sizes
  Current (effect_bdf_params.py): Fixed n=1000
  Proposed: Varies: 50-2500
  ────────────────────────────────────────
  Aspect: Scoring methods
  Current (effect_bdf_params.py): 3 (nle, nll_bic, nll)
  Proposed: 5 (adds aic, loo_cv)
  ────────────────────────────────────────
  Aspect: Statistical tests
  Current (effect_bdf_params.py): Wilcoxon only
  Proposed: Friedman + Nemenyi + CD
  ────────────────────────────────────────
  Aspect: Calibration
  Current (effect_bdf_params.py): Coverage only
  Proposed: ECE, MCE, PIT, reliability
  ────────────────────────────────────────
  Aspect: Noise/imbalance
  Current (effect_bdf_params.py): Fixed
  Proposed: Varied systematically
  ────────────────────────────────────────
  Aspect: Seeds
  Current (effect_bdf_params.py): 10
  Proposed: 30
  ---
