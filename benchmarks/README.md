# BDF Benchmark Suite

Comprehensive empirical evaluation of Bayesian Distributional Forests. The suite covers real-data performance, synthetic ground-truth validation, ablation studies, complexity analysis, and convergence rate theory — everything needed for a publication-quality assessment.

## Quick Reference

| What you want to do | Command | Script |
|---------------------|---------|--------|
| Run BDF on custom model configs | `pixi run bench-bdf` | `run.py` |
| Full regression suite | `pixi run reg-suite` | `run_regression_suite.py` |
| Compute regression metrics + plots | `pixi run reg` | `calc_plot_regression_metrics.py` |
| Compute classification metrics + plots | `pixi run clas` | `calc_plot_classification_metrics.py` |
| Synthetic DGP benchmark | `pixi run synth` | `synthetic_dgp_benchmark.py` |
| Convergence rate study (BDF vs RF) | `pixi run python -m benchmarks.convergence_rate_study` | `convergence_rate_study.py` |
| Complexity analysis | `pixi run complexity` | `complexity_analysis.py` |
| Parameter ablation | `pixi run effect-params` | `effect_bdf_params.py` |
| Scoring method study | `pixi run scoring-study` | `scoring_method_study.py` |
| Distribution misspecification | `pixi run dist-study` | `dist_misspecification_study.py` |
| OOD / covariate shift | `pixi run ood` | `effect_out_of_distribution.py` |
| Sample size effect | `pixi run sample-size` | `effect_sample_size.py` |
| Noise features robustness | `pixi run noise-study` | `effect_noise_features.py` |
| Noise features plots/tables | `pixi run noise-output` | `plot_noise_features_results.py` |
| Tree prior comparison | `pixi run effect-tree-prior` | `effect_tree_prior.py` |
| Effect of n_trees | (direct invocation) | `effect_n_trees.py` |
| Label noise (classification) | `pixi run label-noise-clas` | `label_noise_clas.py` |
| Beta prior effect (classification) | `pixi run prior-clas` | `effect_prior_clas.py` |
| Quick out-of-box performance | `pixi run perf` | `performance.py` |

The `bench-models` pixi environment adds external baselines (NGBoost, LightGBM, XGBoost, CatBoost, etc.) that are not in the default environment.

## Studies Overview

### 1. Main Benchmarks (Real Data)

**Scripts**: `run.py`, `run_regression_suite.py`, `calc_plot_regression_metrics.py`, `calc_plot_classification_metrics.py`

Hydra-based benchmarks on 20+ UCI/OpenML regression and classification datasets. Models are configured via YAML files in `configs/model/`. The pipeline tunes hyperparameters with Optuna on fold 0 and evaluates on folds 1–9.

Metrics output includes Friedman tests, Nemenyi post-hoc analysis, critical difference diagrams, and LaTeX tables — publication-ready.

**Doc**: [`docs/MAIN_BENCHMARK.md`](docs/MAIN_BENCHMARK.md)

### 2. Synthetic DGP Benchmark

**Script**: `synthetic_dgp_benchmark.py`

Five synthetic DGPs with known ground truth (heteroscedastic sinusoidal, step function, bimodal mixture, heavy-tailed, sparse sampling). Evaluates MSE against true mean, variance estimation error, CRPS, interval coverage, and PIT calibration. Compares BDF (Normal + KDE) vs Random Forest, with optional NGBoost and Conformal LightGBM.

**Doc**: [`docs/SYNTHETIC_BENCHMARKS.md`](docs/SYNTHETIC_BENCHMARKS.md)

### 3. Convergence Rate Study

**Script**: `convergence_rate_study.py`

Compares empirical learning curves of BDF and Random Forest across 17 DGPs stratified by smoothness class (smooth/non-smooth/mixed) at d=1, 5, 10. Measures MSE vs true mean at n=50 to 10,000 and fits convergence exponents via log-log regression. BDF shows steeper learning curves, especially at d=5 and d=10.

**Doc**: [`docs/CONVERGENCE_RATES.md`](docs/CONVERGENCE_RATES.md)

### 4. Computational Complexity Analysis

**Script**: `complexity_analysis.py`

Empirical scaling analysis measuring fit time and memory as functions of sample size (n), feature count (d), and ensemble size (T). Fits power-law exponents in log-log space. Compares BDF vs Random Forest vs Gaussian Process.

**Doc**: [`docs/COMPLEXITY_ANALYSIS.md`](docs/COMPLEXITY_ANALYSIS.md)

### 5. Parameter Ablation

**Script**: `effect_bdf_params.py`

One-at-a-time ablation of BDF regularization parameters: alpha, gamma, delta, min_samples_leaf, score_method. Runs for both regression and classification, producing sensitivity plots and summary tables.

**Doc**: [`docs/BDF_PARAMS_ABLATION.md`](docs/BDF_PARAMS_ABLATION.md)

### 6. Scoring Method Study

**Script**: `scoring_method_study.py`

Dedicated comparison of NLE (Bayesian marginal likelihood) vs NLL-based scoring with various corrections (BIC, AIC, LOO-CV, K-fold CV). Tests six hypotheses about when each scoring method dominates.

**Doc**: [`docs/SCORING_METHOD_STUDY.md`](docs/SCORING_METHOD_STUDY.md)

### 7. Distribution Misspecification

**Scripts**: `dist_misspecification_study.py`, `plot_misspecification_results.py`

5 DGPs x 5 BDF distributions = 25-cell cross-test. Each DGP is the "home" of one distribution; measures how much performance degrades when the assumed distribution is wrong. Generates misspecification heatmaps, PIT histograms, and LaTeX tables.

**Doc**: [`docs/DIST_MISSPECIFICATION.md`](docs/DIST_MISSPECIFICATION.md)

### 8. Out-of-Distribution / Covariate Shift

**Scripts**: `effect_out_of_distribution.py`, `compare_ood_results.py`

Covariate shift analysis on Friedman 1–3 datasets with four shift levels (none/mild/moderate/strong). Tests whether BDF's uncertainty estimates increase appropriately in OOD regions.

**Doc**: [`docs/EFFECT_OOD.md`](docs/EFFECT_OOD.md)

### 9. Effect of Ensemble Size

**Script**: `effect_n_trees.py`

Varies n_trees from 5 to 200 on four synthetic datasets, tracking both point and probabilistic metrics to identify the plateau.

**Doc**: [`docs/EFFECT_N_TREES.md`](docs/EFFECT_N_TREES.md)

### 10. Effect of Sample Size

**Script**: `effect_sample_size.py`

Varies n from 100 to 5,000 on Friedman 1–3 and make_regression. Per-fold Optuna tuning. Compares BDF vs Random Forest (no GP).

### 11. Noise Features Robustness

**Script**: `effect_noise_features.py` | **Plots**: `plot_noise_features_results.py`

Adds 0–500 pure noise features to Friedman 1–3 and make_regression datasets. Evaluates CRPS degradation across 5 probabilistic models (BDFNormal, BDFKDE, ConformalRF, NGBoost, BART) with Optuna tuning on CRPS (tune on fold 0, evaluate on folds 1–9). Tracks feature selection behavior for all models by counting split feature usage across their internal tree structures. Run `pixi run noise-study` for the experiment, `pixi run noise-output` for plots and LaTeX tables.

### 12. Tree Prior Comparison

**Script**: `effect_tree_prior.py`

Compares three tree structure prior formulations (LINEAR, DEFER/CART, BERNOULLI) with separately tuned alpha/delta.

### 13. Classification-Specific Studies

**Scripts**: `effect_prior_clas.py`, `label_noise_clas.py`

- **Beta prior effect**: Sweeps prior mean/variance on Bernoulli classification; generates heatmaps.
- **Label noise**: Varies noise rate 0.0–0.5 across sample sizes 200–5,000; compares BDF, RF, NGBoost.

## Directory Structure

```
benchmarks/
├── README.md                          # This file
│
├── configs/                           # Hydra YAML configuration
│   ├── config.yaml                    # Global settings (seed, folds, tuning metric)
│   └── model/                         # Per-model configs (~30 YAML files)
│       ├── bdf_normal_mu_normal.yaml
│       ├── random_forest.yaml
│       ├── lightgbm.yaml
│       └── ...
│
├── pipeline/                          # Core benchmark infrastructure
│   ├── orchestrators.py               # Hydra orchestrators (tune fold 0, eval folds 1-9)
│   ├── data.py                        # Dataset loaders (20+ regression/classification)
│   └── synthetic_dgps.py             # Synthetic DGP registry (9 DGPs with ground truth)
│
├── models/                            # Model wrappers and factories
│   ├── wrappers.py                    # Baseline wrappers (RF, LightGBM, XGBoost, NGBoost, ...)
│   ├── model_factory.py               # Generic factory from YAML config
│   └── bdf_factory.py                 # BDF-specific factory with auto-resolved params
│
├── metrics/                           # Evaluation metrics
│   ├── regression.py                  # CRPS, NLL, quantile loss, coverage, PIT, PICA
│   └── classification.py             # Accuracy, AUROC, F1, ECE, calibration curves
│
├── utils/                             # Shared utilities
│   ├── statistical_tests.py           # Friedman, Nemenyi, Wilcoxon, win/tie/loss
│   ├── plotting.py                    # Critical difference diagrams, heatmaps, PIT plots
│   ├── synthetic_plotting.py          # Ground truth overlays, conditional densities
│   ├── latex_tables.py                # Publication-ready LaTeX table generation
│   ├── yaml_loader.py                 # YAML result aggregation
│   ├── aggregate_results.py           # Cross-fold aggregation
│   └── benchmark_utils.py             # Logging setup
│
├── docs/                              # Study documentation (one per study)
│   ├── MAIN_BENCHMARK.md
│   ├── SYNTHETIC_BENCHMARKS.md
│   ├── CONVERGENCE_RATES.md
│   ├── COMPLEXITY_ANALYSIS.md
│   ├── BDF_PARAMS_ABLATION.md
│   ├── SCORING_METHOD_STUDY.md
│   ├── DIST_MISSPECIFICATION.md
│   ├── EFFECT_OOD.md
│   └── EFFECT_N_TREES.md
│
├── results/                           # Output data (gitignored)
│   ├── custom/                        # Main benchmark YAML results
│   ├── regression_suite/              # Regression suite results
│   ├── synthetic_dgp/                 # Synthetic DGP results
│   ├── convergence_rate/              # Convergence rate JSON + auto-generated report
│   ├── complexity_analysis/           # Complexity JSON + auto-generated report
│   ├── effect_sample_size/
│   ├── dist_misspecification/
│   └── optuna/                        # Optuna study databases
│
├── plots/                             # Output figures (gitignored)
│   ├── regression/                    # CD diagrams, metric plots
│   ├── classification/
│   ├── synthetic_dgp/
│   ├── convergence_rate/
│   ├── complexity_analysis/
│   └── ...
│
├── tables/                            # Generated LaTeX tables
│   ├── dist_misspecification/
│   └── scoring_method/
│
├── run.py                             # Main benchmark entry point
├── run_regression_suite.py            # Regression suite entry point
├── calc_plot_regression_metrics.py    # Regression analysis + plots
├── calc_plot_classification_metrics.py
├── synthetic_dgp_benchmark.py
├── convergence_rate_study.py
├── complexity_analysis.py
├── effect_bdf_params.py
├── scoring_method_study.py
├── dist_misspecification_study.py
├── effect_out_of_distribution.py
├── effect_sample_size.py
├── effect_n_trees.py
├── effect_noise_features.py
├── effect_tree_prior.py
├── effect_prior_clas.py
├── label_noise_clas.py
├── performance.py
├── synthetic_data.py
├── load_save_datasets.py
├── make_bdf_reg.py
├── compare_synthetic_results.py
├── compare_ood_results.py
├── plot_misspecification_results.py
└── plot_synthetic_results.py
```

## Common Patterns

### Tune-Evaluate Protocol

All studies that involve model comparison follow the same protocol:
1. **Fold 0**: Tune hyperparameters with Optuna (50 trials, TPE sampler)
2. **Folds 1–9**: Evaluate with the tuned hyperparameters
3. **Aggregate**: Mean +/- std across evaluation folds

This avoids data leakage while giving honest uncertainty estimates.

### Result Formats

- **Main benchmarks**: YAML files in `results/custom/` (one per model), loaded by `calc_plot_*_metrics.py`
- **Study-specific**: JSON files in `results/{study_name}/`, each study also generates an auto-report (markdown)
- **Plots**: PNG at 300 DPI in `plots/{study_name}/`
- **Tables**: LaTeX files in `tables/{study_name}/`

### Model Configuration

Models are configured at two levels:
- **init_kwargs**: Constructor arguments (n_trees, max_depth, etc.)
- **params dict**: Distribution-specific parameters (mu_mu, sigma_mu, etc.) — BDF only

Both can have fixed and tunable subsets. The `configs/model/` YAML files define the tuning ranges.

### Environments

- **Default** (`pixi run`): BDF + sklearn baselines (RandomForest, GaussianProcess)
- **bench-models** (`pixi run -e bench-models`): Adds NGBoost, LightGBM, XGBoost, CatBoost, QuantileForest, ConformalRF, etc.
- **test** (`pixi run -e test`): For running pytest

## Adding a New Study

1. Create `benchmarks/my_study.py` following the pattern of existing scripts
2. Add a pixi task in `pixi.toml` under `[tasks]`
3. Write documentation in `benchmarks/docs/MY_STUDY.md`
4. Results go in `benchmarks/results/my_study/`, plots in `benchmarks/plots/my_study/`
