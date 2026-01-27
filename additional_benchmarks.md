Critical Gaps for JMLR

  ---
  1. Statistical Significance Testing (HIGH PRIORITY)

  Missing entirely. JMLR papers require formal statistical comparisons.

  Plan:
  - Implement Friedman test across all datasets per metric
  - Use Nemenyi post-hoc test with Critical Difference (CD) diagrams
  - For pairwise comparisons: Wilcoxon signed-rank test (BDF vs each baseline)
  - Report p-values and effect sizes (e.g., Cliff's delta)
  - Package: critdd or scipy.stats + custom CD plotting

  Rationale: Reviewers will immediately ask "is BDF significantly better?" You need formal tests, not just mean ±
  std.

  ---
  3. Distribution Selection Study (HIGH PRIORITY)

  Missing. Key question: "When should I use which distribution?"

  Plan:
  - Compare NormalMuNormal, KDE, BayesianKDE, SkewNormal, StudentT across:
    - Symmetric vs skewed targets
    - Light vs heavy-tailed targets
    - Unimodal vs multimodal targets
    - Different target domains (real, positive, integer)
  - Synthetic DGPs with known ground truth distributions
  - Metric: KL divergence or Wasserstein distance to true distribution (where known)

  ---
  4. Scoring Method Comparison (MEDIUM-HIGH PRIORITY) -> OR just add param ablation study for Classification too -> nle
  likely more impactful.

  Add betaMVBernoulli auto mean and auto scale (although var is constrained here too) s.t. params comparable to NormalMuNormal.

  Not systematically tested. This is a key methodological contribution.

  Plan:
  - Compare nle (marginal likelihood) vs nll + corrections on:
    - Small sample sizes (where Bayesian should shine)
    - Large sample sizes (convergence behavior)
    - Misspecified models (wrong distribution assumption)
  - Show when NLE outperforms NLL+BIC and vice versa
  - Include aic, bic, loo_cv, kfold_cv in comparison

  ---
  7. Calibration Deep-Dive (MEDIUM PRIORITY)

  Metrics exist but no dedicated analysis.

  Plan:
  - Reliability diagrams (expected vs observed coverage) for multiple intervals
  - PIT histograms should be uniform - show they are
  - Calibration under distribution shift (train on one domain, test on another)
  - Recalibration analysis: Does BDF need post-hoc calibration? Compare to baselines that do.
  - Include Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
