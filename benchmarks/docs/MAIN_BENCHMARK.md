# Main Benchmark System

This document describes the comprehensive benchmark suite that compares BDF against state-of-the-art baseline models across multiple real-world datasets for both regression and classification tasks.

## Overview

The benchmark system uses **Hydra** for configuration management and **Optuna** for hyperparameter tuning. It provides:

- **Automated model comparison** across 10+ real-world datasets
- **Fair hyperparameter tuning** using Optuna TPE sampler (50 trials per dataset)
- **Comprehensive evaluation** with point and probabilistic metrics
- **Cross-validation** with 10-fold CV (fold 0 for tuning, folds 1-9 for evaluation)
- **Reproducible results** with git tracking and JSON exports

## Quick Start

```bash
# Run single model on all regression datasets
pixi run bench-bdf model=bdf_kde

# Run regression suite (all models, all datasets)
pixi run -e benchmark reg-suite-models

# Run classification suite (all models, all datasets)
pixi run -e benchmark clas-suite-models

# Override configuration
pixi run bench-bdf model=bdf_normalmunormal n_trials=100 seed=42
```

**Note**: Most models require the `benchmark` environment which includes optional dependencies (NGBoost, LightGBM, etc.).

## Architecture

### Configuration System (Hydra)

The benchmark uses hierarchical YAML configs:

```
benchmarks/configs/
├── config.yaml           # Main config (seed, n_splits, n_trials, etc.)
└── model/                # Model-specific configs
    ├── bdf_kde.yaml
    ├── bdf_normalmunormal.yaml
    ├── ngboost_reg.yaml
    ├── lgbm_reg.yaml
    ├── rf_reg.yaml
    └── ...
```

### Workflow

1. **Dataset Loading**: Automatically loads all compatible datasets from `data/raw/`
2. **Hyperparameter Tuning**: Optuna optimizes on fold 0 using validation split
3. **Cross-Validation**: Evaluates best hyperparameters on folds 1-9
4. **Metric Computation**: Calculates both point and probabilistic metrics
5. **Results Export**: Saves to `benchmarks/results/` as JSON with metadata

## Configuration Reference

### Main Config (`config.yaml`)

```yaml
defaults:
  - _self_
  - model: bdf_kde  # Model config to use

seed: 1234              # Random seed for reproducibility
n_splits: 10            # Number of CV folds (fold 0 = tuning)
n_trials: 50            # Optuna trials per dataset
sample_size: 1000       # Posterior samples for probabilistic metrics

# Tuning configuration
tuning_metric: crps     # Metric to optimize: "mse" or "crps"
tuning_sample_size: 100 # Samples for CRPS during tuning (lower for speed)

logging:
  level: DEBUG          # Logging level
```

**Override examples:**
```bash
# Use MSE for tuning instead of CRPS
pixi run bench-bdf tuning_metric=mse

# Increase tuning trials
pixi run bench-bdf n_trials=100

# Change random seed
pixi run bench-bdf seed=42
```

### Model Config Structure

Each model config in `benchmarks/configs/model/` has this structure:

```yaml
name: bdf_kde
target_type: regression  # or "classification"
class_name: BDFRegressor
probabilistic: true

# Target domains this model can handle
compatible_target_domains:
  - real
  - positive_real
  - nonnegative_real
  - integer
  - nonnegative_integer
  - positive_integer

# Fixed initialization arguments (not tuned)
fixed_init_kwargs:
  dist: KDE
  random_state: 1234
  verbose: 0

# Tunable initialization arguments (hyperparameters)
tunable_init_kwargs:
  n_trees:
    type: int
    low: 15
    high: 100
  alpha:
    type: float
    low: 0.00001
    high: 0.1
    log: true
  min_samples_leaf:
    type: int
    low: 10
    high: 100
  subsample:
    type: float
    low: 0.5
    high: 1.0

# Tunable params dict (for BDF models)
tunable_params:
  bandwidth:
    type: categorical
    categories: [scott, silverman]
  score_correction:
    type: categorical
    categories: [loo_cv, bic, null]

# Fixed params dict (not tuned)
fixed_params:
  kernel: gaussian
  kde_backend: switch
```

**Hyperparameter types:**
- `int`: Integer with `low`/`high` bounds, optional `log: true` for log scale
- `float`: Float with `low`/`high` bounds, optional `log: true` for log scale
- `categorical`: List of discrete choices

## Available Models

### Regression Models

#### BDF Variants
- **`bdf_kde`**: KDE distribution (most flexible, non-parametric)
- **`bdf_normalmunormal`**: Normal distribution with Bayesian mean
- **`bdf_normalmeanpseudoalphaskewnormal`**: Skew-Normal distribution
- **`bdf_gammamvlambdapoisson`**: Poisson distribution (count data)
- **`bdf_gammamvlambdaexponential`**: Exponential distribution (waiting times)

#### Baseline Models
- **`rf_reg`**: Random Forest (sklearn)
- **`rf_qreg`**: Quantile Random Forest
- **`ngboost_reg`**: NGBoost with Normal/Exponential distributions
- **`lgbm_reg`**: LightGBM (point predictions)
- **`lgbm_qreg`**: LightGBM Quantile Regression
- **`conflgbm`**: Conformalized LightGBM (uncertainty via conformal prediction)
- **`confrf`**: Conformalized Random Forest
- **`gp_reg`**: Gaussian Process Regression (sklearn)
- **`bayesridge_reg`**: Bayesian Ridge Regression
- **`bart_reg`**: Bayesian Additive Regression Trees
- **`catbunc_reg`**: CatBoost with uncertainty
- **`de_reg`**: Density Estimation (various backends)
- **`knnkde`**: KNN-based KDE
- **`treeffuser`**: Tree-based diffusion model

### Classification Models

#### BDF Variants
- **`bdf_betamvbernoulli`**: Beta-Bernoulli for binary classification

#### Baseline Models
- **`rf_clas`**: Random Forest Classifier
- **`lgbm_clas`**: LightGBM Classifier
- **`ngboost_clas`**: NGBoost Classifier
- **`gp_clas`**: Gaussian Process Classifier
- **`knn_clas`**: K-Nearest Neighbors
- **`calrf_clas`**: Calibrated Random Forest

## Datasets

### Regression Datasets (10)

| Dataset | Samples | Features | Target Domain | Description |
|---------|---------|----------|---------------|-------------|
| `abalone_age` | 4177 | 8 | positive_integer | Predict abalone age from physical measurements |
| `bike_sharing` | 731 | 11 | positive_integer | Daily bike rental counts |
| `boston_housing` | 506 | 13 | positive_real | Median housing prices in Boston |
| `combined_cycle_power_plant` | 9568 | 4 | positive_real | Power plant energy output |
| `concrete_strength` | 1030 | 8 | positive_real | Concrete compressive strength |
| `energy_efficiency` | 768 | 8 | positive_real | Building heating load |
| `kin8nm` | 8192 | 8 | positive_real | Kinematics of robot arm |
| `parkinsons_updrs` | 5875 | 20 | positive_real | Parkinson's disease progression |
| `realestate` | 414 | 6 | positive_real | House prices |
| `superconductor` | 21263 | 81 | positive_real | Critical temperature of superconductors |
| `wine_quality` | 1599 | 11 | positive_integer | Wine quality ratings |

**Target domains:**
- `real`: Any real number (-∞, +∞)
- `positive_real`: Strictly positive (0, +∞)
- `nonnegative_real`: Non-negative [0, +∞)
- `positive_integer`: Positive integers {1, 2, 3, ...}
- `nonnegative_integer`: Non-negative integers {0, 1, 2, ...}

### Classification Datasets (3)

| Dataset | Samples | Features | Target Domain | Description |
|---------|---------|----------|---------------|-------------|
| `breast_cancer_wisconsin` | 569 | 30 | binary | Cancer diagnosis (malignant/benign) |
| `titanic` | 891 | 6 | binary | Survival prediction |
| `boston_housing_classification` | 506 | 13 | binary | Above/below median price |

## Metrics

### Regression Metrics

#### Point Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

#### Probabilistic Metrics
- **CRPS**: Continuous Ranked Probability Score (lower is better)
- **Interval Score**: Proper scoring rule for prediction intervals (50%, 90%, 95%)
- **Weighted Interval Score**: Combined interval score across multiple levels
- **Sharpness**: Average standard deviation of predictive distributions
- **CI Width**: Average width of prediction intervals (50%, 90%)
- **PICA**: Prediction Interval Coverage Average
- **PIT KS Statistic**: Kolmogorov-Smirnov test for calibration
- **Dawid-Sebastiani Score**: Proper scoring rule based on mean and variance

#### Calibration Metrics (dict-valued, for plotting)
- **Coverage Curve**: Empirical vs nominal coverage at levels [10%, 20%, ..., 90%, 95%, 99%]
- **PIT Histogram**: Probability Integral Transform histogram for calibration assessment

### Classification Metrics

#### Point Metrics
- **Accuracy**: Classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

#### Probabilistic Metrics
- **Log Loss**: Negative log-likelihood
- **Brier Score**: Mean squared error of probabilities
- **ECE**: Expected Calibration Error
- **MCE**: Maximum Calibration Error
- **Calibration Curve**: Predicted vs observed probabilities

## Output Structure

Results are saved to `benchmarks/results/` with the following structure:

```json
{
  "metadata": {
    "timestamp": "2024-01-25T12:00:00Z",
    "git_commit": "abc123...",
    "git_dirty": false,
    "config": {...}
  },
  "dataset_name": {
    "dataset_metadata": {
      "name": "boston_housing",
      "target_domain": "positive_real",
      "n_samples": 506,
      "n_features": 13
    },
    "best_params": {
      "n_trees": 75,
      "alpha": 0.001,
      ...
    },
    "tuning_time": 125.3,
    "fold_results": [
      {
        "fold": 1,
        "fit_time": 2.5,
        "predict_time": 0.1,
        "point_metrics": {
          "mse": 15.3,
          "rmse": 3.9,
          "mae": 2.8,
          "r2": 0.85
        },
        "probabilistic_metrics": {
          "crps": 1.8,
          "pica": 0.03,
          "interval_score_90": 5.2,
          "sharpness": 2.1,
          ...
        },
        "calibration_metrics": {
          "coverage_curve": {
            "levels": [0.1, 0.2, ..., 0.9, 0.95, 0.99],
            "empirical": [0.11, 0.19, ..., 0.88, 0.94, 0.98]
          },
          "pit_histogram": {
            "bin_counts": [12, 15, 11, ...],
            "n_bins": 20,
            "n_samples": 200
          }
        }
      },
      ...
    ],
    "aggregated_metrics": {
      "mse": {"mean": 15.5, "std": 1.2, "values": [...]},
      "crps": {"mean": 1.9, "std": 0.2, "values": [...]},
      "coverage_curve": {
        "levels": [0.1, 0.2, ..., 0.9, 0.95, 0.99],
        "empirical": {
          "0.5": [0.48, 0.51, ...],
          "0.9": [0.88, 0.91, ...]
        }
      },
      ...
    }
  }
}
```

## Model Compatibility

Models are automatically matched to datasets based on `compatible_target_domains`:

```python
# Example: BDF-KDE can handle most regression targets
compatible_target_domains: [
  real,
  positive_real,
  nonnegative_real,
  integer,
  nonnegative_integer,
  positive_integer
]

# Example: Poisson BDF is specialized for counts
compatible_target_domains: [
  nonnegative_integer,
  positive_integer
]
```

The orchestrator automatically skips incompatible model-dataset pairs.

## Advanced Usage

### Running Specific Models on Specific Datasets

The orchestrator automatically runs all compatible datasets, but you can modify the dataset registry in `benchmarks/pipeline/data.py`:

```python
# Temporarily comment out datasets you don't want to run
REGRESSION_DATASET_REGISTRY = {
    "boston_housing": _load_boston_housing,
    # "superconductor": _load_superconductor,  # Skip large dataset
}
```

### Custom Hyperparameter Ranges

Edit the model config to adjust search space:

```yaml
# Narrow search space for faster tuning
tunable_init_kwargs:
  n_trees:
    type: int
    low: 25    # was 15
    high: 50   # was 100
```

### Parallel Execution

Optuna uses parallel trial execution. The orchestrator processes datasets sequentially but can be parallelized externally:

```bash
# Run different models in parallel (different terminals)
pixi run -e benchmark bench-bdf model=bdf_kde &
pixi run -e benchmark bench-bdf model=ngboost_reg &
pixi run -e benchmark bench-bdf model=rf_reg &
```

### Debugging Failed Runs

Check the Optuna study database:

```bash
# View study in SQLite browser
sqlite3 benchmarks/results/optuna/<study_name>.db
```

Enable verbose logging:

```bash
pixi run bench-bdf logging.level=DEBUG
```

## Performance Considerations

### Tuning Time

Expected time per dataset (50 trials):
- **BDF-KDE**: ~30-60 min (most expensive)
- **BDF-Normal**: ~10-20 min
- **NGBoost**: ~5-10 min
- **Random Forest**: ~2-5 min

**Tips to speed up:**
- Reduce `n_trials` (e.g., 20 for quick tests)
- Reduce `tuning_sample_size` for CRPS (e.g., 50 instead of 100)
- Use `tuning_metric=mse` instead of `crps` (no sampling needed)

### Memory Usage

Large datasets (e.g., superconductor with 21k samples) can require 4-8GB RAM for:
- Storing posterior samples (1000 samples × n_test_samples)
- KDE computations (especially with many features)

Reduce `sample_size` if memory is constrained:
```bash
pixi run bench-bdf sample_size=500
```

## Timing Analysis

The analysis script (`calc_plot_regression_metrics.py`) computes training speed comparisons using **geometric mean speedup ratios** across datasets.

### What's Computed

For each baseline model M and each dataset where both BDF and M have timing data:

```
ratio_i = time_M(dataset_i) / time_BDF(dataset_i)
speedup = geometric_mean(ratio_1, ..., ratio_k)
```

A speedup > 1 means BDF is faster. The geometric mean is used because it's robust to scale (a single large dataset doesn't dominate) and symmetric (inverting the ratio inverts the result cleanly).

### Two Comparison Tables

| Table | Control Model | Purpose |
|-------|--------------|---------|
| `speedup_selected.tex` | BDF (best distribution per dataset) | Matches the prediction quality tables — same model selection |
| `speedup_normal.tex` | BDF (Normal, default config) | Shows speed of the default/fastest BDF configuration |

The default distribution for the second table is configured via `BDF_DEFAULT_DIST` in the script (set to `bdf_normalmunormal`).

### Timing Fields

Two fields are extracted from each YAML result file:

| Field | Type | Description |
|-------|------|-------------|
| `fitting_times` | `list[float]` | Per-fold training times (folds 1-9), averaged to get mean fit time |
| `tuning_time_seconds` | `float` | Total Optuna hyperparameter search time (fold 0) |

Both fields must be present and correctly typed — models with mismatched types raise a `TypeError` rather than silently converting.

### Output

Each table contains per-model rows sorted by fit speedup:

```
Model        | Fit Speedup | N  | Tune Speedup | N
-------------|-------------|----|--------------|---
BART         | 11.9x       | 11 | 11.1x        | 11
CatBoostUnc  | 2.32x       | 11 | 7.46x        | 11
NGBoost      | 2.16x       | 11 | 0.64x        | 11
ConfRF       | 1.00x       | 11 | 0.56x        | 11
```

N = number of shared datasets used for comparison. Models with `---` lack `fitting_times` lists in their YAML files.

## Comparison with Synthetic DGP Benchmark

The main benchmark (`run.py`) differs from the synthetic DGP benchmark (`synthetic_dgp_benchmark.py`):

| Feature | Main Benchmark | Synthetic DGP |
|---------|---------------|---------------|
| **Datasets** | Real-world (10 regression, 3 classification) | Synthetic with known ground truth (5 DGPs) |
| **Metrics** | Point + probabilistic | Point + probabilistic + ground truth errors |
| **Purpose** | Model comparison for practical use | Theoretical evaluation and method validation |
| **Plotting** | External analysis scripts | Built-in ground truth overlays |
| **Configuration** | Hydra YAML configs | Hardcoded in script |

Use main benchmark for **practical model selection**. Use synthetic benchmark for **algorithmic development and validation**.

## Troubleshooting

### "Model X not available"
Install benchmark environment:
```bash
pixi install -e benchmark
pixi run -e benchmark bench-bdf
```

### "Dataset Y not found"
Ensure data files exist in `data/raw/`. Run data download script if available:
```bash
pixi run -e benchmark load-data
```

### Optuna "Study already exists"
The orchestrator automatically cleans up studies. If manual intervention needed:
```python
import optuna
optuna.delete_study(study_name="study_name", storage="sqlite:///path/to/db")
```

### CRPS computation slow
Reduce `tuning_sample_size` or use `tuning_metric=mse`:
```bash
pixi run bench-bdf tuning_sample_size=50
# or
pixi run bench-bdf tuning_metric=mse
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{bdf_benchmark,
  title = {Bayesian Distributional Forest Benchmark Suite},
  author = {...},
  year = {2024},
  url = {https://github.com/...}
}
```

## Statistical Considerations

### Limitations of 9-Fold Evaluation

The benchmark uses 10-fold CV with fold 0 reserved for hyperparameter tuning and folds 1-9 for evaluation. This provides **9 evaluation points per configuration**, which has statistical implications:

| What We Can Report | What We Cannot Claim |
|--------------------|---------------------|
| Mean metric values | Proper 95% confidence intervals |
| Standard error (SE = σ/√9) | Statistical significance at α=0.05 |
| Rough variance estimates | Effect sizes with tight bounds |

**Why this matters:** With only 9 observations, a t-distribution 95% CI requires t₀.₉₇₅,₈ ≈ 2.31, resulting in very wide intervals (±2.31 × SE). These intervals are often wider than meaningful differences between methods.

### Recommended Reporting

For honest reporting from the main benchmark:

```
CRPS: 1.82 ± 0.15 (mean ± SE, 9-fold CV)
```

**Do not claim** "95% CI" unless you have substantially more samples.

### For Rigorous Statistical Claims

The **[Scoring Method Study](./SCORING_METHOD_STUDY.md)** provides proper statistical rigor with:
- **20 seeds** for core experiments (sufficient for Friedman test at α=0.05)
- **Friedman test** with Nemenyi post-hoc for multiple comparisons
- **Wilcoxon signed-rank** tests for pairwise comparisons
- **Cliff's delta** effect sizes with standard interpretation

Use the scoring method study for formal hypothesis testing about NLE vs NLL scoring. Use the main benchmark for practical model comparison and selection.

### Post-Hoc Analysis

The following can be computed post-hoc from saved per-fold values (no need to re-run experiments):
- Bonferroni-corrected p-values
- Effect sizes (Cohen's d, Cliff's delta)
- Critical difference diagrams
- Win/tie/loss tables

Per-fold metric values, fitting times, and tuning information are preserved in the JSON output for downstream analysis.

## See Also

- [Synthetic DGP Benchmark](./SYNTHETIC_BENCHMARK.md) - Ground truth validation
- [Scoring Method Study](./SCORING_METHOD_STUDY.md) - Statistical comparison of NLE vs NLL scoring
- [Model Development Guide](../../CONTRIBUTING.md) - Adding new models
- [Hydra Documentation](https://hydra.cc/) - Configuration system
- [Optuna Documentation](https://optuna.org/) - Hyperparameter optimization
