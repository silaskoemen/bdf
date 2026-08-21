# Convergence Rate Study: BDF vs RF

Generated: 2026-08-20T09:04:58.823312

## Summary of Convergence Rates

Rate = slope in log(MSE) vs log(n). More negative = faster convergence.

| DGP | Class | d | BDF rate | RF rate |
|-----|-------|---|----------|---------|
| friedman1_d10 | smooth | 10 | -0.229 (0.044) | -0.203 (0.045) |
| friedman1_d5 | smooth | 5 | -0.186 (0.045) | -0.180 (0.050) |
| indicator_d1 | non_smooth | 1 | -0.204 (0.040) | -0.753 (0.071) |
| indicator_d10 | non_smooth | 10 | -0.506 (0.054) | -0.231 (0.015) |
| indicator_d5 | non_smooth | 5 | -0.499 (0.084) | -0.409 (0.026) |
| mixed_d1 | mixed | 1 | -0.400 (0.078) | -0.886 (0.065) |
| mixed_d10 | mixed | 10 | -0.588 (0.115) | -0.413 (0.071) |
| mixed_d5 | mixed | 5 | -0.562 (0.105) | -0.451 (0.075) |
| polynomial_d1 | smooth | 1 | -1.029 (0.192) | -0.898 (0.223) |
| polynomial_d10 | smooth | 10 | -0.387 (0.056) | -0.219 (0.039) |
| polynomial_d5 | smooth | 5 | -0.342 (0.065) | -0.270 (0.050) |
| sinusoidal_d1 | smooth | 1 | -0.934 (0.071) | -0.932 (0.105) |
| sinusoidal_d10 | smooth | 10 | -0.301 (0.039) | -0.200 (0.033) |
| sinusoidal_d5 | smooth | 5 | -0.261 (0.041) | -0.248 (0.049) |
| step_d1 | non_smooth | 1 | -0.423 (0.088) | -1.211 (0.070) |
| step_d10 | non_smooth | 10 | -0.421 (0.031) | -0.229 (0.023) |
| step_d5 | non_smooth | 5 | -0.399 (0.031) | -0.293 (0.029) |

## Configuration

- Seeds: [42, 123, 456, 789, 1024]
- Sample sizes: [50, 100, 200, 500, 1000, 2000, 5000, 10000]
- Dimensionalities: [1, 5, 10]
- Test set size: 2000
