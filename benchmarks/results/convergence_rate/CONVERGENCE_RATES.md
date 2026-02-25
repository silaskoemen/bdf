# Convergence Rate Study: BDF vs RF vs GP

Generated: 2026-02-18T23:05:37.897473

## Hypothesis

BDF's Bayesian prior pulls leaf predictions toward the grand mean, smoothing
discontinuities between leaves. This should give BDF faster convergence rates
on smooth target functions (closer to GP) while remaining competitive with RF
on non-smooth (piecewise constant) targets.

## Summary of Convergence Rates

Rate = slope in log(MSE) vs log(n). More negative = faster convergence.

| DGP | Class | d | BDF rate | RF rate | GP rate |
|-----|-------|---|----------|---------|---------|
| friedman1_d10 | smooth | 10 | -0.230 (0.044) | -0.203 (0.045) | -0.581 (0.028) |
| friedman1_d5 | smooth | 5 | -0.186 (0.045) | -0.180 (0.050) | -0.340 (0.056) |
| indicator_d1 | non_smooth | 1 | -0.206 (0.038) | -0.753 (0.071) | 1.417 (0.218) |
| indicator_d10 | non_smooth | 10 | -0.534 (0.054) | -0.231 (0.015) | -0.220 (0.016) |
| indicator_d5 | non_smooth | 5 | -0.740 (0.040) | -0.409 (0.026) | -0.302 (0.020) |
| mixed_d1 | mixed | 1 | -0.398 (0.077) | -0.886 (0.065) | 1.814 (0.269) |
| mixed_d10 | mixed | 10 | -0.591 (0.115) | -0.413 (0.071) | -0.211 (0.021) |
| mixed_d5 | mixed | 5 | -0.562 (0.105) | -0.451 (0.075) | -0.350 (0.018) |
| polynomial_d1 | smooth | 1 | -1.030 (0.193) | -0.898 (0.223) | 1.195 (0.191) |
| polynomial_d10 | smooth | 10 | -0.386 (0.055) | -0.219 (0.039) | -0.836 (0.223) |
| polynomial_d5 | smooth | 5 | -0.342 (0.064) | -0.270 (0.050) | -0.979 (0.202) |
| sinusoidal_d1 | smooth | 1 | -0.937 (0.068) | -0.932 (0.105) | 1.620 (0.254) |
| sinusoidal_d10 | smooth | 10 | -0.301 (0.039) | -0.200 (0.033) | -0.154 (0.013) |
| sinusoidal_d5 | smooth | 5 | -0.261 (0.041) | -0.248 (0.049) | -0.467 (0.021) |
| step_d1 | non_smooth | 1 | -0.422 (0.087) | -1.211 (0.070) | 1.453 (0.232) |
| step_d10 | non_smooth | 10 | -0.419 (0.031) | -0.229 (0.023) | -0.010 (0.002) |
| step_d5 | non_smooth | 5 | -0.398 (0.031) | -0.293 (0.029) | -0.153 (0.028) |

## Interpretation

- If BDF rate is more negative than RF on smooth DGPs, the prior-smoothing hypothesis is supported.
- If BDF rate equals RF on non-smooth DGPs, the prior does not hurt on step functions.
- GP provides the theoretical ceiling for smooth function estimation.

## Configuration

- Seeds: [42, 123, 456, 789, 1024]
- Sample sizes: [50, 100, 200, 500, 1000, 2000, 5000, 10000]
- Dimensionalities: [1, 5, 10]
- Test set size: 2000
- GP max n: 3000
