# BDF Computational Complexity Analysis

Generated: 2026-02-25T16:49:59.082002

## Summary

Empirical analysis of BDF vs Random Forest computational complexity across
three dimensions: sample size (n), feature count (d), and ensemble size (T).

## Theoretical Complexity

| Model | Training Complexity | Notes |
|-------|---------------------|-------|
| BDF | O(n x d x T) | Bayesian tree ensemble (Rust split finding) |
| Random Forest | O(n x d x T) | Standard tree ensemble (sklearn) |

Both scale identically in theory; BDF has a constant overhead from Bayesian
posterior updates and distributional scoring at each split.

## Empirical Scaling Exponents

| Model | n Scaling | d Scaling | T Scaling |
|-------|-----------|-----------|-----------|
| BDF | O(n^1.01) R²=0.998 | O(d^0.54) R²=0.982 | O(T^0.98) R²=0.998 |
| Random Forest | O(n^0.94) R²=0.971 | O(d^0.98) R²=0.998 | O(T^0.90) R²=0.990 |

## Overhead Ratio (BDF / Random Forest)

| Axis | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| n | 2.01x | 0.39 | 1.20x | 2.49x |
| d | 1.65x | 0.90 | 0.63x | 2.98x |
| T | 2.36x | 0.23 | 1.93x | 2.61x |

### Sample Size (n) Scaling

Fixed: n_features=10, n_trees=50

| n | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 100 | 0.045 +/- 0.006 | 0.037 +/- 0.000 | 1.20x |
| 250 | 0.100 +/- 0.009 | 0.052 +/- 0.000 | 1.92x |
| 500 | 0.195 +/- 0.014 | 0.084 +/- 0.001 | 2.32x |
| 1000 | 0.404 +/- 0.025 | 0.162 +/- 0.003 | 2.49x |
| 2500 | 1.015 +/- 0.009 | 0.456 +/- 0.001 | 2.22x |
| 5000 | 2.080 +/- 0.007 | 1.032 +/- 0.008 | 2.02x |
| 10000 | 4.753 +/- 0.136 | 2.485 +/- 0.077 | 1.91x |

### Feature Count (d) Scaling

Fixed: n_samples=2000, n_trees=50

| d | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 5 | 0.553 +/- 0.035 | 0.186 +/- 0.003 | 2.98x |
| 10 | 0.958 +/- 0.076 | 0.357 +/- 0.005 | 2.68x |
| 20 | 1.156 +/- 0.024 | 0.691 +/- 0.055 | 1.67x |
| 50 | 1.753 +/- 0.011 | 1.591 +/- 0.010 | 1.10x |
| 100 | 2.738 +/- 0.047 | 3.226 +/- 0.068 | 0.85x |
| 200 | 4.651 +/- 0.239 | 7.376 +/- 0.564 | 0.63x |

### Trees (T) Scaling

Fixed: n_samples=2000, n_features=10

| T | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 10 | 0.190 +/- 0.038 | 0.098 +/- 0.018 | 1.93x |
| 20 | 0.327 +/- 0.009 | 0.139 +/- 0.001 | 2.35x |
| 50 | 0.870 +/- 0.020 | 0.345 +/- 0.002 | 2.52x |
| 100 | 1.796 +/- 0.043 | 0.688 +/- 0.003 | 2.61x |
| 200 | 3.304 +/- 0.019 | 1.375 +/- 0.001 | 2.40x |

### Memory Usage

| Model | Peak Memory at n=10000 (MB) | Peak Memory at d=200 (MB) |
|-------|---|---|
| BDF | 22.8 | 7.7 |
| Random Forest | 0.8 | 1.7 |

## Configuration

```python
SEED = 42
N_REPEATS = 5
N_SAMPLES_GRID = [100, 250, 500, 1000, 2500, 5000, 10000]
N_FEATURES_GRID = [5, 10, 20, 50, 100, 200]
N_TREES_GRID = [10, 20, 50, 100, 200]
```
