# Synthetic DGP Benchmarks

## Overview

This module provides rigorous, JMLR-quality synthetic benchmarks for evaluating distributional regression methods. Unlike real-world benchmarks, synthetic data has **known ground truth** for the conditional distribution p(y|x), enabling direct evaluation of:

- **Mean function estimation accuracy**: MSE against true E[Y|X=x]
- **Variance function estimation**: Error in predicted Var[Y|X=x]
- **Distributional calibration**: Whether predictive intervals match nominal coverage
- **Local uncertainty**: Calibration stratified by covariate regions
- **Tail behavior**: Performance on heavy-tailed or skewed distributions

## Data Generating Processes (DGPs)

### 1. Heteroscedastic Sinusoidal

**Purpose**: Tests ability to adapt uncertainty estimates to changing variance.

**Model**:
```
y = sin(x) + ε
ε ~ N(0, σ²(x))
```

where σ²(x) increases at x = π:
- For x < π: σ(x) = noise_scale × √|x|
- For x ≥ π: σ(x) = noise_scale × √(4|x|)

**Key challenge**: Detect variance change point and provide wider intervals in high-variance region.

**Expected behavior**:
- Good models: Intervals widen after x=π
- Poor models: Constant-width intervals, undercoverage in high-variance region

---

### 2. Step Function

**Purpose**: Tests handling of discontinuities and avoiding over-smoothing.

**Model**:
```
y = f_step(x) + ε
ε ~ N(0, σ²)
```

where f_step(x) is piecewise constant with `k` discrete levels.

**Key challenge**: Capture sharp discontinuities without spurious smoothing.

**Expected behavior**:
- Good models: Sharp transitions, constant predictions within steps
- Poor models: Smooth transitions, bias near change points

---

### 3. Bimodal Mixture

**Purpose**: Tests capacity to capture multimodal conditional distributions.

**Model**:
```
p(y|x) = π(x) N(y; x, σ²) + (1-π(x)) N(y; x + c, σ²)
π(x) = 0.1 + 0.8x  for x ∈ [0,1]
```

**Key challenge**: Represent two modes with input-dependent mixing probability.

**Expected behavior**:
- Good models: Wide intervals capturing both modes, bimodal predictive densities
- Poor models: Unimodal predictions, poor CRPS

---

### 4. Heavy-Tailed Noise

**Purpose**: Tests robustness to outliers and tail modeling.

**Model**:
```
y = x₁² + x₂ + ε
ε ~ t_df × scale
```

where t_df is Student-t with df degrees of freedom (default df=3).

**Key challenge**: Provide appropriate uncertainty for fat-tailed distributions.

**Expected behavior**:
- Good models: Wider intervals than Gaussian assumption would suggest
- Poor models: Undercoverage, failure on outliers

---

### 5. Sparse Sampling

**Purpose**: Tests uncertainty quantification in data-sparse regions.

**Model**:
```
y = sin(x) + ε
ε ~ N(0, σ²)
```

but x is sampled non-uniformly (e.g., p(x) ∝ 1 - x/max(x)).

**Key challenge**: Increase uncertainty where data is sparse.

**Expected behavior**:
- Good models: Wider intervals in sparse regions
- Poor models: Overconfident predictions in low-data regions

---

## Experimental Protocol

### Cross-Validation Strategy

Follows the established pattern in this codebase:

1. **10-fold split** with fixed random seed (42)
2. **Fold 0**: Hyperparameter tuning (internal 3-fold CV for objective)
3. **Folds 1-9**: Evaluation with best parameters from fold 0

This avoids:
- Data leakage (tuning samples not used in evaluation)
- Excessive computational cost (tuning once vs. 10 times)

### Hyperparameter Tuning

- **Optimizer**: Optuna with TPE sampler
- **Trials**: 75 per DGP-model pair
- **Objective**: MSE on internal CV
- **Storage**: SQLite databases in `benchmarks/results/optuna/`

### Metrics Computed

**Point Prediction**:
- MSE, MAE, RMSE, R²

**Distributional**:
- CRPS (Continuous Ranked Probability Score)
- Interval scores (50%, 90%, 95%)
- Coverage (50%, 90%, 95%)
- Quantile losses (5%, 10%, 25%, 50%, 75%, 90%, 95%)
- PIT (Probability Integral Transform) KS statistic
- PICA (PIT-based calibration metric)

**Ground Truth** (unique to synthetic):
- `gt_mean_mse`: MSE against true mean function E[Y|X]
- `gt_mean_mae`: MAE against true mean function
- `gt_variance_mse`: MSE against true variance function Var[Y|X]
- `gt_variance_mae`: MAE against true variance function
- `gt_variance_rel_error`: Relative error in variance estimation

---

## Usage

The benchmark system automatically detects which models are available in your environment:
- **Default environment**: Runs BDF models + sklearn baseline only
- **Bench-models environment**: Runs full comparison including NGBoost, LightGBM, etc.

### 1. Run Benchmarks

The benchmark system has **two output types**:
1. **Individual diagnostic plots**: Generated during the benchmark run (model-specific)
2. **Comparison plots**: Generated separately from saved metrics (cross-model)

#### Quick Start (Core Models Only)

Run benchmarks with BDF models + RandomForest baseline:

```bash
pixi run python -m benchmarks.synthetic_dgp_benchmark
```

This evaluates:
- BDFNormal (Bayesian Normal-Normal)
- BDFKDE (Kernel Density Estimation)
- RandomForest (sklearn baseline)

**Outputs**:
- Metrics: `benchmarks/results/synthetic_dgp/{dgp_name}_results_core.json`
- Individual plots: `benchmarks/plots/synthetic_dgp/{dgp_name}/{model}_*.png`
  - `{model}_predictions.png`: Ground truth overlay with prediction intervals
  - `{model}_conditional_densities.png`: Conditional densities at selected x values
  - `{model}_local_calibration.png`: Coverage by covariate region

#### Full Comparison (With External Baselines)

For complete evaluation including NGBoost and other baselines, run in the bench-models environment:

```bash
pixi run -e benchmark python -m benchmarks.synthetic_dgp_benchmark
```

This adds:
- NGBoost (parametric distributional boosting)
- (other models available in bench-models environment)

**Outputs**: Same structure as above but with `_results.json` suffix (no `_core`)

**Timing**: Expect ~2-4 hours depending on hardware and number of models.

**Additional output files**:
- `all_results[_core].json`: Combined results across all DGPs
- `benchmarks/results/optuna/synthetic_{dgp}_{model}.db`: Hyperparameter tuning history

### 2. Generate Comparison Plots

After running benchmarks, generate cross-model comparison plots and tables:

```bash
pixi run python -m benchmarks.compare_synthetic_results
```

**Key advantage**: This script is **environment-agnostic** - it only loads metrics from JSON, so you can run it in any environment regardless of which models were benchmarked.

The comparison script automatically:
- Detects available results (merges full and core results if both exist)
- Loads all available DGP results
- Generates cross-model visualizations

**Output plots** (in `benchmarks/plots/synthetic_dgp/summary/`):
- Comparison tables: Metric rankings across DGPs
- Summary visualizations: Performance heatmaps, critical difference diagrams
- Markdown tables: Ready for inclusion in papers/reports

### Recommended Workflow

```bash
# During development: fast iteration with core models
pixi run python -m benchmarks.synthetic_dgp_benchmark  # Runs benchmark + individual plots
pixi run python -m benchmarks.compare_synthetic_results  # Generate comparisons

# Before publication: full comparison
pixi run -e benchmark python -m benchmarks.synthetic_dgp_benchmark  # All models
pixi run python -m benchmarks.compare_synthetic_results  # Cross-model comparisons
```

**Why this design?**
- Individual plots generated during benchmark → Solves environment compatibility (NGBoost plots need NGBoost installed)
- Comparison plots from JSON only → Fast iteration on presentation without re-running benchmarks
- Clean separation → Diagnostic plots vs publication figures

The core-only benchmark is much faster and sufficient for:
- Testing DGP implementations
- Debugging BDF model issues
- Iterating on hyperparameter ranges
- Quick sanity checks

Run the full benchmark when you need:
- Complete baseline comparisons for papers
- Validating BDF performance claims
- Generating publication figures

### 3. Customize DGPs

To add or modify DGPs, edit `benchmarks/pipeline/synthetic_dgps.py`:

```python
def generate_my_dgp(n_samples: int, seed: int, **kwargs) -> SyntheticDataset:
    """Your custom DGP."""
    # Generate X, y
    # Define ground truth functions
    return SyntheticDataset(
        X=X, y=y,
        dgp_type=DGPType.MY_TYPE,
        name="my_dgp",
        ground_truth=GroundTruthFunctions(...),
        ...
    )

# Register
DGP_REGISTRY["my_dgp"] = generate_my_dgp
```

Then update `DGPS_TO_RUN` in `synthetic_dgp_benchmark.py`.

---

## Interpreting Results

### What to Look For

**BDF should excel at**:
- `gt_variance_mse`: Accurate variance estimation (vs NGBoost's parametric assumption)
- Coverage near nominal (0.50, 0.90, 0.95) across DGPs
- CRPS competitive with or better than NGBoost
- Handling multimodality (bimodal_mixture) via KDE distribution

**Failure modes to check**:
- Undercoverage (<0.90 for 90% intervals) → overconfident
- Overcoverage (>0.95 for 90% intervals) → too conservative
- Poor `gt_mean_mse` → trees not fitting mean well
- High `gt_variance_rel_error` → uncertainty calibration broken

### JMLR-Quality Analysis

For publication:

1. **Report mean ± std** across 9 evaluation folds
2. **Statistical tests**: Use Wilcoxon signed-rank for pairwise BDF vs baseline
3. **Critical difference diagrams**: Rank methods across DGPs
4. **Visualizations**: Include ground truth overlays for at least 2 DGPs
5. **Ablations**: Vary DGP parameters (e.g., noise_scale, df, n_steps) to show robustness

---

## Implementation Details

### Code Structure

```
benchmarks/
├── pipeline/
│   └── synthetic_dgps.py              # DGP generators with ground truth
├── utils/
│   └── synthetic_plotting.py          # Plotting utilities
├── synthetic_dgp_benchmark.py         # Main benchmark script (generates individual plots)
├── compare_synthetic_results.py       # Comparison/summary plots (from JSON only)
└── SYNTHETIC_BENCHMARKS.md            # This file
```

### Ground Truth Functions

Each `SyntheticDataset` includes a `GroundTruthFunctions` object with:

```python
@dataclass
class GroundTruthFunctions:
    mean_fn: Callable[[np.ndarray], np.ndarray]      # E[Y|X=x]
    variance_fn: Callable[[np.ndarray], np.ndarray]  # Var[Y|X=x]
    quantile_fn: Callable[[np.ndarray, float], np.ndarray]  # Q_τ[Y|X=x]
    density_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]  # p(y|x)
    sample_fn: Callable[[np.ndarray, int], np.ndarray]  # Sample from p(Y|X=x)
```

These functions enable:
- Computing true mean/variance for any x
- Plotting ground truth on top of predictions
- Measuring distributional distance (KL, Wasserstein) when tractable

### Reproducibility

All stochasticity is controlled:
- `SEED = 42` for data generation
- `random_state=SEED` in model constructors
- `shuffle=True, random_state=SEED` in KFold
- `seed=SEED` in Optuna samplers

Re-running the benchmark should produce identical results (modulo floating-point errors).

---

## Comparison to Real-World Benchmarks

| Aspect | Real-World | Synthetic |
|--------|-----------|-----------|
| **Ground truth** | Unknown | Known p(y\|x) |
| **Metrics** | MSE, CRPS | + GT mean/variance errors |
| **Sample size** | Fixed | Can vary systematically |
| **Interpretability** | Complex relationships | Controlled, interpretable |
| **Generalizability** | High | Lower (toy problems) |
| **Purpose** | Overall performance | Specific capability testing |

**Recommendation**: Use both. Real-world benchmarks demonstrate practical utility; synthetic benchmarks provide diagnostic insight into model behavior.

---

## Future Extensions

### Additional DGPs

Potential additions for comprehensive evaluation:

1. **Skewed distributions**: Log-normal or Gamma noise
2. **Change points in mean**: Abrupt mean shifts
3. **Interactions**: y = x₁ × x₂ + ε with heteroscedasticity
4. **High-dimensional**: Sparse additive models with many features
5. **Time series**: Autoregressive structure

### Advanced Metrics

- **Wasserstein distance**: Between predicted and true distributions
- **KL divergence**: If both densities are tractable
- **Energy score**: Multivariate extension of CRPS
- **Log-likelihood**: Against true density (when available)

### Ablations

- **Effect of sample size**: Run each DGP at n ∈ {500, 1000, 2000, 5000}
- **Effect of noise level**: Vary noise_scale systematically
- **Effect of dimensionality**: Add uninformative features

---

## Contact

For questions or contributions related to synthetic benchmarks:
- Check existing DGPs in `benchmarks/pipeline/synthetic_dgps.py`
- See examples in `benchmarks/synthetic_data.py` (original notebook)
- Follow patterns from `benchmarks/effect_sample_size.py` for structure

**Maintained as part of the BDF project for JMLR submission.**
