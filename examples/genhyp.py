"""Generalized hyperbolic distribution (4th parameterization) example.
Tries to examine whether skewness/shape/tailness can be estimated from
3rd and 4th moments.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import kv


def loglik_genhyp(params: tuple | list | np.ndarray, x) -> np.ndarray:
    mu, delta, p, a, b = params
    if delta <= 0 or a <= 0 or np.abs(b) >= a:
        return -np.inf
    z = (x - mu) / delta

    return (
        -np.log(delta)
        + p / 2 * np.log(a**2 + b**2)
        - 0.5 * np.log(2 * np.pi)
        - (p - 1 / 2) * np.log(a + 1e-8)
        - np.log(kv(p, np.sqrt(a**2 + b**2)) + 1e-8)
        + b * z
        + np.log(kv(p - 1 / 2, a * np.sqrt(1 + z**2)) + 1e-8)
        - (1 / 2 * (1 / 2 - p)) * np.log(1 + z**2)
    )


def nll_genhyp(params, x):
    """Negative log-likelihood for the generalized hyperbolic distribution."""
    return -np.mean(loglik_genhyp(params, x))


def lik_genhyp(params, x, mean=False) -> np.ndarray | float:
    """Likelihood for the generalized hyperbolic distribution."""
    lik = np.exp(loglik_genhyp(params, x))
    if mean:
        return np.mean(lik)
    else:
        return lik


# %%
# Simulate data from normal, skewnormal, genhyp, uniform and fit genhyp to it.
from scipy.optimize import minimize
from scipy.stats import genhyperbolic, norm, skewnorm
from tqdm import tqdm


def simulate_data(n_mixtures=10, probs=[0.1, 0.5, 0.2, 0.2], samples_per_sim=100, seed=42):
    np.random.seed(seed)
    out_data = np.zeros((n_mixtures * samples_per_sim,))
    probs = np.array(probs) / sum(probs)
    rng = np.random.default_rng(seed)
    for i in tqdm(range(n_mixtures), "Simulating data"):
        dist = rng.choice(["normal", "skewnormal", "genhyp", "uniform"], p=probs)

        match dist:
            case "normal":
                mu = np.random.uniform(-10, 10)
                sigma = np.random.uniform(0.1, 50)
                data = norm.rvs(loc=mu, scale=sigma, size=samples_per_sim)
            case "skewnormal":
                mu = np.random.uniform(-10, 10)
                sigma = np.random.uniform(0.1, 50)
                skew = np.random.uniform(-10, 10)
                data = skewnorm.rvs(a=skew, loc=mu, scale=sigma, size=samples_per_sim)
            case "genhyp":
                p = np.random.uniform(-0.5, 1.5)  # lambda
                a = np.random.uniform(0.1, 10)  # alpha
                b = np.random.uniform(-a + 1e-3, a - 1e-3)  # beta
                loc = np.random.uniform(-5, 5)
                scale = np.random.uniform(0.5, 5)
                data = genhyperbolic.rvs(p, a, b, loc=loc, scale=scale, size=samples_per_sim)
            case "uniform":
                low = np.random.uniform(-10, 0)
                high = np.random.uniform(0, 10)
                data = np.random.uniform(low, high)
            case _:
                raise ValueError(f"Distribution '{dist}' not supported.")

        out_data[(i * samples_per_sim) : ((i + 1) * samples_per_sim)] = data

    return out_data


def calc_posterior_params(data: np.ndarray) -> tuple[float, float, float, float, float]:
    """Calculate the posterior params using scipy's minimize of the log likelihood given the data"""
    # Fit MLE, mu, delta, p, a, b
    mean, std = np.mean(data), np.std(data)
    res = minimize(
        nll_genhyp,
        np.array([mean, std, 1, 1.0, 0]),
        args=(data,),
        method="L-BFGS-B",
        bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-10, 10), (1e-6, 1000), (-1000, 1000)],
    )
    return res.x


# %%
data = simulate_data(n_mixtures=10, samples_per_sim=50, probs=[0.4, 2, 0.3, 0.8])
param_tuple = calc_posterior_params(data)
x_vals = np.arange(data.min(), data.max(), (data.max() - data.min()) / 1000)
plt.hist(data, density=True, color="dodgerblue", bins=35)
liks = lik_genhyp(param_tuple, x_vals)
plt.plot(x_vals, liks, color="crimson")
plt.show()

# %%
from time import time

# Compare minimize of loglik to `fit` directly
start_time = time()
params_min = calc_posterior_params(data)
end_time = time()
print(f"Scipy's `minimize` took {end_time - start_time:.2f}s")
start_time = time()
params_fit = genhyperbolic.fit(data)
params_fit = params_fit[2:] + (params_fit[0], params_fit[1])
end_time = time()
print(f"Scipy's `fit` took {end_time - start_time:.2f}s")
for n, m, f in zip(["mu", "sigma", "p", "a", "b"], params_min, params_fit):
    print(f"{n.capitalize()} | min: {m} | fit: {f}")

# %%
x_vals = np.arange(data.min(), data.max(), (data.max() - data.min()) / 1000)
plt.hist(data, density=True, color="dodgerblue", bins=35)
min_liks = lik_genhyp(params_min, x_vals)
plt.plot(x_vals, min_liks, color="crimson")
fit_liks = genhyperbolic(params_fit).pdf(x_vals)
plt.plot(x_vals, fit_liks, color="black")
plt.show()
# %%
