# BDF Computational Complexity Analysis

Generated: 2026-02-25T13:30:15.022365

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
| BDF | O(n^1.03) R²=0.995 | O(d^0.56) R²=0.965 | O(T^0.85) R²=0.992 |
| Random Forest | O(n^0.94) R²=0.971 | O(d^0.96) R²=0.999 | O(T^0.99) R²=1.000 |

## Overhead Ratio (BDF / Random Forest)

| Axis | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| n | 2.49x | 0.74 | 1.28x | 3.64x |
| d | 1.88x | 0.98 | 0.83x | 3.33x |
| T | 3.28x | 0.63 | 2.63x | 4.46x |

### Sample Size (n) Scaling

Fixed: n_features=10, n_trees=50

| n | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 100 | 0.048 +/- 0.002 | 0.038 +/- 0.000 | 1.28x |
| 250 | 0.108 +/- 0.008 | 0.054 +/- 0.001 | 2.01x |
| 500 | 0.302 +/- 0.138 | 0.083 +/- 0.001 | 3.64x |
| 1000 | 0.485 +/- 0.093 | 0.172 +/- 0.012 | 2.83x |
| 2500 | 1.479 +/- 0.190 | 0.457 +/- 0.006 | 3.23x |
| 5000 | 2.388 +/- 0.259 | 1.042 +/- 0.010 | 2.29x |
| 10000 | 5.552 +/- 0.666 | 2.567 +/- 0.282 | 2.16x |

### Feature Count (d) Scaling

Fixed: n_samples=2000, n_trees=50

| d | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 5 | 0.564 +/- 0.060 | 0.186 +/- 0.003 | 3.03x |
| 10 | 1.164 +/- 0.201 | 0.350 +/- 0.008 | 3.33x |
| 20 | 1.246 +/- 0.030 | 0.653 +/- 0.008 | 1.91x |
| 50 | 1.914 +/- 0.034 | 1.601 +/- 0.012 | 1.20x |
| 100 | 3.145 +/- 0.092 | 3.214 +/- 0.042 | 0.98x |
| 200 | 5.421 +/- 0.474 | 6.517 +/- 0.076 | 0.83x |

### Trees (T) Scaling

Fixed: n_samples=2000, n_features=10

| T | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 10 | 0.314 +/- 0.092 | 0.070 +/- 0.000 | 4.46x |
| 20 | 0.439 +/- 0.045 | 0.140 +/- 0.001 | 3.14x |
| 50 | 1.133 +/- 0.154 | 0.347 +/- 0.002 | 3.27x |
| 100 | 2.012 +/- 0.534 | 0.692 +/- 0.005 | 2.91x |
| 200 | 3.621 +/- 0.155 | 1.375 +/- 0.002 | 2.63x |

### Memory Usage

| Model | Peak Memory at n=10000 (MB) | Peak Memory at d=200 (MB) |
|-------|---|---|
| BDF | 43.7 | 11.9 |
| Random Forest | 0.8 | 1.7 |

## Configuration

```python
SEED = 42
N_REPEATS = 5
N_SAMPLES_GRID = [100, 250, 500, 1000, 2500, 5000, 10000]
N_FEATURES_GRID = [5, 10, 20, 50, 100, 200]
N_TREES_GRID = [10, 20, 50, 100, 200]
```
