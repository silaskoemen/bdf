# Computational Complexity Analysis

Rigorous empirical analysis of BDF's computational complexity, measuring how training time and memory scale with dataset dimensions and model parameters.

## Purpose

This benchmark addresses key questions for a publication:
1. **What is BDF's empirical complexity?** (scaling exponents via log-log regression)
2. **How does BDF compare to baselines?** (Random Forest, Gaussian Process)
3. **Does the Rust backend provide efficiency gains?** (comparison with pure-Python alternatives)
4. **What are the memory requirements?** (peak memory profiling)

## Theoretical Complexity

| Model | Training Complexity | Notes |
|-------|---------------------|-------|
| BDF | O(n × d × T × depth) | Tree ensemble with Bayesian inference |
| Random Forest | O(n × d × T × depth) | Standard tree ensemble |
| Gaussian Process | O(n³) | Kernel matrix inversion dominates |

Where:
- n = number of samples
- d = number of features
- T = number of trees
- depth = maximum tree depth

GP's cubic complexity makes it prohibitively slow for n > 5000, providing a clear reference point for BDF's scalability advantage.

## Scaling Experiments

### 1. Sample Size Scaling (n)
- **Fixed**: d=10 features, T=50 trees
- **Varied**: n ∈ {100, 250, 500, 1000, 2500, 5000, 10000}
- **Expected**: Near-linear scaling O(n^α) where α ≈ 1.0-1.2

### 2. Feature Scaling (d)
- **Fixed**: n=2000 samples, T=50 trees
- **Varied**: d ∈ {5, 10, 20, 50, 100, 200}
- **Expected**: Linear scaling O(d^β) where β ≈ 1.0

### 3. Tree Scaling (T)
- **Fixed**: n=2000 samples, d=10 features
- **Varied**: T ∈ {5, 10, 25, 50, 100, 150, 200}
- **Expected**: Linear scaling O(T^γ) where γ ≈ 1.0

Note: Gaussian Process is excluded from tree scaling (no ensemble parameter).

## Models Compared

| Model | Distribution | Scoring | Notes |
|-------|--------------|---------|-------|
| BDF | NormalMuNormal | NLL+BIC | Default configuration |
| Random Forest | N/A | MSE | sklearn, single-threaded for fair comparison |
| Gaussian Process | RBF kernel | MLE | sklearn, limited to n ≤ 5000 |

## Configuration

### BDF Settings
```python
BDF_CONFIG = {
    "dist": "NormalMuNormal",
    "n_trees": 50,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "alpha": 0.001,
    "gamma": 0.1,
    "delta": 0.01,
    "subsample": 0.9,
    "colsample": 0.9,
}
```

### Measurement Settings
- **Timing**: 5 repetitions per configuration, report mean ± std
- **Memory**: Peak memory via `tracemalloc`
- **Seed**: Fixed at 42 for reproducibility

## Usage

```bash
# Run the full complexity analysis
pixi run complexity

# Alternative: direct invocation
pixi run python -m benchmarks.complexity_analysis
```

## Outputs

### Directory Structure

```
benchmarks/
├── results/complexity_analysis/
│   ├── complexity_analysis.json    # Full results
│   └── COMPLEXITY_ANALYSIS.md      # Auto-generated report
└── plots/complexity_analysis/
    ├── bdf_scaling.png             # BDF 3-panel scaling plot
    ├── randomforest_scaling.png    # RF 3-panel scaling plot
    ├── comparison_n_scaling.png    # n-scaling comparison
    ├── comparison_d_scaling.png    # d-scaling comparison
    └── comparison_tree_scaling.png # Tree scaling comparison
```

### JSON Results Structure

```json
{
  "metadata": {
    "timestamp": "2024-01-27T12:00:00",
    "seed": 42,
    "n_repeats": 5
  },
  "config": {
    "n_samples_grid": [100, 250, 500, ...],
    "n_features_grid": [5, 10, 20, ...],
    "n_trees_grid": [5, 10, 25, ...]
  },
  "models": {
    "BDF": {
      "n_scaling": {
        "parameter_values": [100, 250, ...],
        "estimated_slope": 1.05,
        "r_squared": 0.998,
        "timing": {"100": {"mean_time": 0.15, "std_time": 0.02, ...}},
        "memory": {"100": {"peak_memory_mb": 5.2, ...}}
      },
      "d_scaling": {...},
      "tree_scaling": {...}
    },
    "RandomForest": {...},
    "GaussianProcess": {...}
  }
}
```

### Plots

| Plot | Description |
|------|-------------|
| `{model}_scaling.png` | 3-panel plot showing n, d, and tree scaling for single model |
| `comparison_n_scaling.png` | All models on same plot for sample size scaling |
| `comparison_d_scaling.png` | All models on same plot for feature scaling |
| `comparison_tree_scaling.png` | BDF vs RF tree scaling (GP excluded) |

Each comparison plot includes:
- Left panel: Fit time (log-log scale) with error bars and fitted slopes
- Right panel: Peak memory usage (log-log scale)

### Auto-Generated Report

The script generates `COMPLEXITY_ANALYSIS.md` with:
- Scaling exponent table
- Timing tables for each dimension
- Memory usage summary
- Conclusions

## Interpretation

### Scaling Exponents

The slope in log-log space indicates the scaling exponent:
- **slope ≈ 1.0**: Linear scaling O(n)
- **slope ≈ 2.0**: Quadratic scaling O(n²)
- **slope ≈ 3.0**: Cubic scaling O(n³)

R² values close to 1.0 indicate good fit to power-law scaling.

### Expected Results

1. **BDF vs RF**: Similar scaling exponents (both tree-based), BDF slightly slower due to Bayesian inference overhead
2. **BDF vs GP**: BDF vastly more scalable; GP shows O(n³) while BDF shows O(n)
3. **Memory**: BDF uses more memory than RF due to storing distribution parameters per node
4. **Tree scaling**: Both BDF and RF should show linear scaling with number of trees

### What "Lower Slope" Means

A lower slope is **better** - it means the algorithm scales more efficiently:
- BDF slope 0.9 vs RF slope 1.1 → BDF becomes relatively faster as n increases
- This can happen if Rust optimizations reduce per-sample overhead

## Extending the Analysis

To add new models:

1. Add model config at the top of `complexity_analysis.py`
2. Create a factory function: `create_X_factory(n_estimators=None)`
3. Add to the `models` dict in `main()`
4. Update color/marker dicts in `plot_scaling_comparison()`
5. Update markdown report generation if needed

## Limitations

- **Single-threaded**: RF configured with `n_jobs=1` for fair comparison
- **GP sample limit**: Capped at n=5000 due to O(n³) complexity
- **No tuning**: Fixed hyperparameters to isolate complexity from optimization
- **Synthetic data**: Uses `make_regression` which may not reflect real-world patterns

## References

- BDF theoretical complexity analysis: See main paper Section X
- Random Forest complexity: Breiman (2001)
- Gaussian Process complexity: Rasmussen & Williams (2006)
