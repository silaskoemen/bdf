# BDF Computational Complexity Analysis

Generated: 2026-04-15T19:05:02.275731

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

### Fit Time

| Model | n Scaling | d Scaling | T Scaling |
|-------|-----------|-----------|-----------|
| BDF | O(n^0.87) R²=0.992 | O(d^0.62) R²=0.998 | O(T^0.95) R²=0.999 |
| Random Forest | O(n^0.97) R²=0.974 | O(d^0.98) R²=1.000 | O(T^1.00) R²=1.000 |

### Predict Time

| Model | n Scaling | d Scaling | T Scaling |
|-------|-----------|-----------|-----------|
| BDF | O(n^1.09) R²=1.000 | O(d^0.28) R²=0.851 | O(T^0.99) R²=0.999 |
| Random Forest | O(n^0.68) R²=0.936 | O(d^0.01) R²=0.651 | O(T^0.94) R²=1.000 |

## Overhead Ratio (BDF / Random Forest)

### Fit Overhead

| Axis | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| n | 1.73x | 0.36 | 1.23x | 2.31x |
| d | 1.23x | 0.57 | 0.58x | 2.20x |
| T | 1.93x | 0.11 | 1.79x | 2.07x |

### Predict Overhead

| Axis | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| n | 5.02x | 2.38 | 1.03x | 7.38x |
| d | 10.31x | 4.34 | 6.84x | 19.24x |
| T | 7.41x | 0.47 | 6.54x | 7.86x |

### Sample Size (n) Scaling

Fixed: n_features=10, n_trees=50

| n | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 100 | 0.024 +/- 0.001 | 0.015 +/- 0.000 | 1.66x |
| 250 | 0.048 +/- 0.001 | 0.021 +/- 0.000 | 2.31x |
| 500 | 0.071 +/- 0.002 | 0.035 +/- 0.000 | 2.04x |
| 1000 | 0.133 +/- 0.002 | 0.068 +/- 0.000 | 1.95x |
| 2500 | 0.318 +/- 0.007 | 0.200 +/- 0.000 | 1.59x |
| 5000 | 0.628 +/- 0.007 | 0.459 +/- 0.000 | 1.37x |
| 10000 | 1.319 +/- 0.014 | 1.076 +/- 0.003 | 1.23x |

### Feature Count (d) Scaling

Fixed: n_samples=2000, n_trees=50

| d | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 5 | 0.170 +/- 0.001 | 0.077 +/- 0.000 | 2.20x |
| 10 | 0.258 +/- 0.002 | 0.155 +/- 0.004 | 1.67x |
| 20 | 0.380 +/- 0.002 | 0.287 +/- 0.001 | 1.33x |
| 50 | 0.636 +/- 0.007 | 0.738 +/- 0.035 | 0.86x |
| 100 | 1.076 +/- 0.026 | 1.425 +/- 0.014 | 0.76x |
| 200 | 1.689 +/- 0.015 | 2.898 +/- 0.005 | 0.58x |

### Trees (T) Scaling

Fixed: n_samples=2000, n_features=10

| T | BDF Time (s) | RF Time (s) | Ratio |
|---|---|---|---|
| 10 | 0.063 +/- 0.005 | 0.031 +/- 0.000 | 2.07x |
| 20 | 0.120 +/- 0.005 | 0.061 +/- 0.000 | 1.98x |
| 50 | 0.304 +/- 0.034 | 0.152 +/- 0.001 | 2.01x |
| 100 | 0.540 +/- 0.004 | 0.302 +/- 0.001 | 1.79x |
| 200 | 1.092 +/- 0.006 | 0.606 +/- 0.002 | 1.80x |

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
