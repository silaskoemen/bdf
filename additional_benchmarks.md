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
  4. Scoring Method Comparison (MEDIUM-HIGH PRIORITY)

  Not systematically tested. This is a key methodological contribution.

  Plan:
  - Compare nle (marginal likelihood) vs nll + corrections on:
    - Small sample sizes (where Bayesian should shine)
    - Large sample sizes (convergence behavior)
    - Misspecified models (wrong distribution assumption)
  - Show when NLE outperforms NLL+BIC and vice versa
  - Include aic, bic, loo_cv, kfold_cv in comparison

  ---
  6. Computational Complexity Analysis (MEDIUM PRIORITY)

  Basic timing exists, but no formal analysis.

  Plan:
  - Theoretical complexity: State O(n × d × n_trees × ...)
  - Empirical scaling:
    - Fix d, vary n: log-log plot of fit time vs n (find slope)
    - Fix n, vary d: log-log plot of fit time vs d
    - Vary n_trees: linear scaling verification
  - Memory profiling: Peak memory vs (n, d)
  - Comparison: Wall-clock time vs RF, NGBoost, BART, GP at matched accuracy

prompt:
Let's add a rigorous complexity analysis (run time) for my BDF. I       
already note tuning and fit times versus comparison models, meaning     
I can easily compare final fit times of tuned models, but an            
analysis comparing effect of d, n and n_trees might make sense to       
do (see additional_benchmarks.md point 6.). I assume it's fine to       
use NormalMuNormal distribution at default values (or tune for each     
setting??) and have a clear json result, markdown table and             
comparison plots. Evaluate

  ---
  7. Calibration Deep-Dive (MEDIUM PRIORITY)

  Metrics exist but no dedicated analysis.

  Plan:
  - Reliability diagrams (expected vs observed coverage) for multiple intervals
  - PIT histograms should be uniform - show they are
  - Calibration under distribution shift (train on one domain, test on another)
  - Recalibration analysis: Does BDF need post-hoc calibration? Compare to baselines that do.
  - Include Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
