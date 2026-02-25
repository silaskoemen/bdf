# Convergence Rate Study

Empirical comparison of learning curves (MSE vs training set size) for BDF and Random Forest across synthetic DGPs stratified by target function smoothness and dimensionality.

## Purpose

This study compares how quickly BDF and Random Forest reduce prediction error as training data grows. BDF exhibits steeper empirical learning curves than RF, particularly in moderate dimensions (d=5, d=10), across all smoothness classes. The exact mechanism — whether driven by the Bayesian prior, evidence-based scoring, or tree structure penalty — remains an open question for future work.

The study answers:
1. **Does BDF converge faster than RF on smooth functions?** (sinusoidal, polynomial, Friedman)
2. **Does BDF match or beat RF on non-smooth functions?** (step, XOR indicator)
3. **How does dimensionality interact with convergence?** (curse of dimensionality)

## Experimental Design

### Metric

MSE is computed against the **true mean function** f(x), not against noisy observations y. This isolates the estimation error (bias^2 + variance) from irreducible noise:

```
MSE = E[(f_hat(x) - f(x))^2]
```

A fixed test set of n=2000 points (same seed offset per configuration) is used for stable MSE estimation.

### DGPs by Smoothness Class

#### Smooth (C^inf or high-order smooth)

| DGP | Function | Informative features | Noise sigma |
|-----|----------|---------------------|---------|
| `sinusoidal_d{d}` | sum_j sin(2pi(j+1)x_j)/(j+1) for j=0..2 | 3 | 0.1 |
| `polynomial_d{d}` | Mean of Chebyshev-like cubics in first 3 features | 3 | 0.1 |
| `friedman1_d{d}` | 10 sin(pi x1 x2) + 20(x3-0.5)^2 + 10 x4 + 5 x5 | 5 | 1.0 |

Friedman #1 is only included for d >= 5.

#### Non-smooth (discontinuous)

| DGP | Function | Informative features | Noise sigma |
|-----|----------|---------------------|---------|
| `step_d{d}` | Additive piecewise-constant with 5 levels per feature | 3 | 0.1 |
| `indicator_d{d}` | XOR: 2 (I(x1>0.5) XOR I(x2>0.5)) | 2 | 0.1 |

#### Mixed (smooth + discontinuity)

| DGP | Function | Informative features | Noise sigma |
|-----|----------|---------------------|---------|
| `mixed_d{d}` | sin(2pi x1) + 2 I(x2>0.5) | 2 | 0.1 |

### Sample Sizes

n in {50, 100, 200, 500, 1000, 2000, 5000, 10000}

This range spans two orders of magnitude, giving enough leverage for reliable slope estimation in log-log space.

### Dimensionalities

d in {1, 5, 10}

At d=1 the problem is easy and differences in rates are clearest. At d=10, the curse of dimensionality kicks in and all methods slow down — the question is whether BDF's advantage grows or shrinks in higher dimensions.

### Seeds

5 seeds (42, 123, 456, 789, 1024) for error bars on MSE estimates.

## Models Compared

| Model | Configuration | Notes |
|-------|--------------|-------|
| BDF | NormalMuNormal, 100 trees, NLE+BIC, auto priors | `min_samples_leaf` = max(5, n/100) |
| Random Forest | 100 trees, max_features=sqrt | `min_samples_leaf` = max(5, n/100) |

All models use fixed (non-tuned) hyperparameters. This is intentional: the goal is to compare convergence rates (slopes), not absolute MSE levels. Tuning would improve both models but obscure the rate comparison by introducing an optimization confound.

## Convergence Rate Analysis

### What the Slope Means

We fit log(MSE) = a + b log(n) via OLS. The slope b is the convergence rate exponent:

| Slope b | Interpretation |
|---------|----------------|
| -1.0 | MSE halves when n doubles — parametric rate O(1/n) |
| -0.8 | Faster than typical nonparametric but slower than parametric |
| -0.5 | Typical 1D nonparametric rate O(1/sqrt(n)) |
| -0.3 | Slow convergence (high-d or misspecified) |
| 0.0 | No improvement with more data (severe misspecification) |

More negative = faster convergence = better.

## Usage

```bash
# Run the full study
pixi run python -m benchmarks.convergence_rate_study

# Quick test (modify SAMPLE_SIZES and SEEDS in script to reduce)
pixi run python -c "
from benchmarks.convergence_rate_study import evaluate_single, build_dgp_suite
dgp = build_dgp_suite()[0]  # sinusoidal_d1
for model in ['BDF', 'RandomForest']:
    r = evaluate_single(dgp, model, 500, 42)
    print(f'{model}: MSE={r[\"mse\"]:.4f}')
"
```

### Estimated Runtime

| Component | Time |
|-----------|------|
| BDF (17 DGPs x 8 sizes x 5 seeds) | ~3-6 hours |
| RF (same) | ~15-30 min |
| **Total** | **~3-6 hours** |

BDF dominates runtime due to Bayesian split-finding at large n. Results are saved incrementally, so the study can be interrupted and inspected at any point.

## Outputs

### Directory Structure

```
benchmarks/
├── results/convergence_rate/
│   ├── convergence_results.json    # Full results (saved incrementally)
│   └── CONVERGENCE_RATES.md        # Auto-generated summary table
└── plots/convergence_rate/
    ├── individual/
    │   ├── sinusoidal_d1.png       # Per-DGP log-log plot
    │   ├── step_d1.png
    │   └── ...
    ├── grid_d1.png                 # Grid: rows=smoothness, cols=DGPs (d=1)
    ├── grid_d5.png                 # Same for d=5
    ├── grid_d10.png                # Same for d=10
    └── rate_comparison.png         # Bar chart: rates by smoothness class
```

### JSON Results Structure

```json
{
  "metadata": {
    "timestamp": "...",
    "seeds": [42, 123, 456, 789, 1024],
    "sample_sizes": [50, 100, 200, 500, 1000, 2000, 5000, 10000],
    "dimensionalities": [1, 5, 10],
    "test_n": 2000
  },
  "dgps": {
    "sinusoidal_d1": {
      "description": "Additive sinusoidal (d=1)",
      "smoothness_class": "smooth",
      "dimensionality": 1,
      "noise_std": 0.1,
      "models": {
        "BDF": {
          "n_values": [50, 100, ...],
          "mean_mse": [0.05, 0.02, ...],
          "std_mse": [0.01, 0.005, ...],
          "all_mse": [[...], [...]],
          "rate": {
            "slope": -0.85,
            "intercept": 2.1,
            "r_squared": 0.98,
            "slope_se": 0.04
          }
        },
        "RandomForest": {...}
      }
    }
  }
}
```

### Plots

| Plot | Description |
|------|-------------|
| `individual/{dgp}.png` | Log-log MSE vs n for one DGP, both models with fitted regression lines and error bars |
| `grid_d{d}.png` | Panel grid: rows are smoothness classes (smooth / non-smooth / mixed), columns are individual DGPs within that class. Each panel shows both models. |
| `rate_comparison.png` | Bar chart of mean convergence rate (negated slope) grouped by smoothness class, one bar per model |

### Auto-Generated Report

`CONVERGENCE_RATES.md` contains a summary table with fitted slopes and standard errors for every (DGP, model) pair.

## Interpretation Guide

### Reading the Log-Log Plots

- **Steeper downward slope** = faster convergence = better
- **Parallel lines** = same rate, different constant (one model uniformly better/worse)
- **Diverging lines** = different rates (the steeper one wins asymptotically)
- **Dashed lines** = fitted OLS trend; solid with markers = actual data

### What to Look For

1. **Smooth DGPs**: BDF lines should be steeper than RF
2. **Non-smooth DGPs**: Both models should have similar slopes (though BDF may still outperform)
3. **Mixed DGPs**: Intermediate behavior
4. **Increasing d**: All slopes should flatten (curse of dimensionality). The gap between BDF and RF tends to grow at d=5 and d=10.
5. **R^2 values**: Low R^2 (< 0.9) suggests the power-law model is a poor fit, which can happen at small n ranges or with non-monotone MSE curves

### Caveats on Interpretation

The steeper learning curves observed for BDF could stem from multiple interacting mechanisms:
- The Bayesian prior shrinking leaf predictions toward the grand mean
- Evidence-based (NLE) scoring selecting better split points
- The tree structure penalty (alpha, gamma, delta) providing better regularization

Isolating these individual contributions is left for future work. The results should be presented as empirical learning curves rather than evidence for a specific theoretical claim.

## Limitations

- **Fixed hyperparameters**: No tuning means absolute MSE levels are not optimized. The study measures *rates* (slopes), which are more robust to hyperparameter choices but not immune.
- **Additive DGPs**: Most smooth DGPs are additive in the first few features. Tree methods can exploit additivity via axis-aligned splits, which may inflate their rates relative to truly multivariate smooth functions.
- **No interaction effects in non-smooth DGPs**: The step function is additive. The XOR indicator has a two-way interaction but no higher-order structure.
- **Noise level fixed**: The noise sigma is the same across sample sizes. At very small n, signal-to-noise ratio is low and all methods may be dominated by noise, flattening the rate curve.

## Extending the Analysis

### Adding a New DGP

Add a `_make_*` factory function in the DGP Definitions section and include it in `build_dgp_suite()`. The function must:
1. Accept a dimensionality `d` parameter
2. Return a `DGP` dataclass with `name` ending in `_d{d}`
3. Provide a `mean_fn(X) -> y` that accepts (n, d) arrays

### Adding a New Model

1. Write a factory function that accepts `n_train` and returns a fitted-compatible model
2. Add it to `MODEL_FACTORIES`
3. Add color and marker entries to `COLORS` and `MARKERS`

### Running a Subset

Edit the constants at the top of `convergence_rate_study.py`:

```python
SEEDS = [42]                        # Single seed for quick test
SAMPLE_SIZES = [100, 500, 2000]     # Fewer sizes
DIMENSIONALITIES = [1]              # 1D only
```

## References

- Minimax rates for nonparametric regression: Stone (1982)
- Random Forest consistency: Scornet, Biau & Vert (2015)
- Bayesian tree shrinkage: BART — Chipman, George & McCulloch (2010)
