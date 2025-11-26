"""#%%
import sympy as sym
#from scipy.special import gamma
from sympy import sqrt, pi, gamma

# g1, g2 are skewness and kurtosis parametesr (estimated from data), d and v are (reparamed) args of dist itself
g1, g2, d, v = sym.symbols('g1, g2, d, v', real=True)
# Just for definition, filled in entirely below
# b_v = sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2)

e1 = sym.Eq(
    d * sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2) * (v * (3-d**2) / (3 - v) - 3 * v / (v - 2) + 2 * d**2 * (sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2)) ** 2) *
    (v / (v - 2) - d**2 * (sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2)) ** 2) ** (-3/2),
    g1
)
e2 = sym.Eq(
    3 * v**2 / ((v - 2) * (v - 4)) - 4 * d**2 * (sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2))**2 * v * (3 - d**2) / (v - 3) + 6 * d**2 * (sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2)) ** 2 * v /
    (v - 2) - 3 * d**4 * (sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2)) ** 4 * (v / (v - 2) - d**2 * (sqrt(v/pi) * gamma(v/2 + 1/2) / gamma(v/2))**2) ** (-2) - 3,
    g2
)
#%%
sym.init_printing(use_unicode=True)
res = sym.solve([e1, e2], [d, v])
print(res)
"""

import math

# %%
import numpy as np


def estimate_skew_t_params(g1: float, g2: float) -> tuple[float, float]:
    """
    Estimates the parameters d and v of a standardized skew-t distribution
    from skewness (g1) and excess kurtosis (g2) using a fast,
    closed-form analytical approximation.

    This is suitable for performance-critical applications like tree-based models.

    Args:
        g1 (float): The sample skewness.
        g2 (float): The sample excess kurtosis.

    Returns:
        tuple[float, float]: A tuple containing the estimated (d, v).
    """
    # --- Step 1: Estimate degrees of freedom 'v' from kurtosis 'g2' ---
    # The kurtosis of a skew-t is primarily driven by 'v'.
    # This is a robust approximation derived from the moments of the t-distribution.
    # We must have g2 > 0 for v to be positive.
    if g2 <= 1e-6:
        # As kurtosis approaches 0 (that of a normal dist), v -> infinity.
        # We cap it at a large number.
        v = 1000.0
    else:
        v = 4 + 6 / g2

    # --- Step 2: Estimate skew parameter 'd' from skewness 'g1' and 'v' ---
    # With 'v' fixed, we can find an approximate inverse for skewness.
    # This approximation relates d to g1, correcting for v.

    # The relationship between g1 and d is complex, but this formula provides
    # a good estimate. It ensures that -1 < d < 1.
    c = (
        np.sqrt(v / (v - 2))
        * np.sqrt(np.pi)
        * (v - 2)
        / (v - 1)
        * (np.exp(math.lgamma((v - 1) / 2) - math.lgamma(v / 2)))
    )

    a = c * (3 * v - 8) / (3 * v - 6)

    # This solves an approximation of g1 = a * d / sqrt(1 + d**2) for d
    # which is more stable than a direct linear approximation.
    d_squared = (g1**2) / (a**2 * (1 - g1**2 / a**2))

    # Ensure d_squared is not negative due to numerical precision issues
    d_squared = max(0, d_squared)

    d = np.sqrt(d_squared) * np.sign(g1)

    # Clamp d to be within its valid range [-1, 1]
    d = np.clip(d, -0.999, 0.999)

    return d, v


# %%
# Target values from data
sample_skewness = -0.8
sample_kurtosis = 1

# Get the fast, closed-form estimates
d_est, v_est = estimate_skew_t_params(sample_skewness, sample_kurtosis)

print("Fast Closed-Form Approximation:")
print(f"  Estimated d: {d_est:.4f}")
print(f"  Estimated v: {v_est:.4f}")

# For comparison, let's see what a numerical solver would give
# (using only scipy and numpy for numerical values)
try:
    import numpy as np
    from scipy.optimize import root
    from scipy.special import gammaln

    def b_v(v):
        # Use log-gamma for numerical stability
        return np.sqrt(v / np.pi) * np.exp(gammaln(v / 2 + 0.5) - gammaln(v / 2))

    def moments(d, v):
        M1 = d * b_v(v)
        M2 = v / (v - 2)
        M3 = d * b_v(v) * (v * (3 - d**2) / (v - 3) - 3 * M2 + 2 * M1**2)
        M4 = (
            (3 * v**2 / ((v - 2) * (v - 4)))
            - (4 * d * b_v(v) * M3)
            + (6 * (d * b_v(v)) ** 2 * M2)
            - (3 * (d * b_v(v)) ** 4)
        )
        return M1, M2, M3, M4

    def skew_kurt_residuals(x, g1_target, g2_target):
        d, v = x
        if v <= 4.01:
            return [1e6, 1e6]
        M1, M2, M3, M4 = moments(d, v)
        variance = M2 - M1**2
        if variance <= 0:
            # Return large residuals to avoid invalid values
            return [1e6, 1e6]
        skewness = (M3 - 3 * M1 * M2 + 2 * M1**3) / variance ** (3 / 2)
        kurtosis = (M4 - 4 * M1 * M3 + 6 * M1**2 * M2 - 3 * M1**4) / variance**2 - 3
        return [skewness - g1_target, kurtosis - g2_target]

    solution = root(skew_kurt_residuals, [d_est, v_est], args=(sample_skewness, sample_kurtosis))
    if solution.success:
        d_sol, v_sol = solution.x
        print("\nNumerical Solver Solution (for comparison):")
        print(f"  Solver d: {d_sol:.4f}")
        print(f"  Solver v: {v_sol:.4f}")
    else:
        print(f"Numerical solver did not converge, {solution.x}")
except ImportError:
    print("\nScipy not installed, skipping numerical comparison.")
# %%
# from scipy.stats import skewnorm
# XI, OMEGA, ALPHA = 5, 2, 1.5
# n_samples = 1000
# data_skew = skewnorm.rvs(ALPHA, loc=XI, scale=OMEGA, size=n_samples)

# # Estimate the parameters of a skew-t distribution numerically from the data
# # Use the likelihood function and scipy.optimize to find the best fit
# from scipy.optimize import minimize

# print(f"Estimated parameters from data: d={d_est:.4f}, v={v_est:.4f}")

# # Compare to closed-form approximation
# d_cf, v_cf = estimate_skew_t_params(
#     np.mean(data_skew), np.var(data_skew, ddof=1) - 1
# )
# print(f"Closed-form approximation: d={d_cf:.4f}, v={v_cf:.4f}")
# %%
