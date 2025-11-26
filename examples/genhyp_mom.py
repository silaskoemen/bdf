"""Generalized hyperbolic distribution (4th parameterization) example.
Tries to examine whether skewness/shape/tailness can be estimated from
3rd and 4th moments.
"""

import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
from scipy.special import kv


def loglik_genhyp(params: tuple | list | np.ndarray, x) -> float:
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


def lik_genhyp(params, x):
    """Likelihood for the generalized hyperbolic distribution."""
    return np.exp(loglik_genhyp(params, x))


# %%
# Simulate data from normal, skewnormal, genhyp, uniform and fit genhyp to it.
from scipy.optimize import minimize
from scipy.stats import genhyperbolic, norm, skewnorm
from tqdm import tqdm


def simulate_data(n_simulations=1000, samples_per_sim=100, seed=42):
    np.random.seed(seed)
    results = []
    for _ in tqdm(range(n_simulations)):
        dist = np.random.choice(["normal", "skewnormal", "genhyp"])

        match dist:
            case "normal":
                mu = np.random.uniform(-100, 100)
                sigma = np.random.uniform(0.1, 50)
                data = norm.rvs(loc=mu, scale=sigma, size=samples_per_sim)
            case "skewnormal":
                mu = np.random.uniform(-100, 100)
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

        # --- 2. Calculate Moments and Fit SHASH via MLE ---
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-6:
            continue

        skew = np.mean(((data - mean) / std) ** 3)
        kurt = np.mean(((data - mean) / std) ** 4) - 3

        # Fit MLE, mu, delta, p, a, b
        res = minimize(
            nll_genhyp,
            np.array([mean, std, 1, 1.0, 0]),
            args=(data,),
            method="L-BFGS-B",
            bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-10, 10), (1e-6, 1000), (-1000, 1000)],
        )

        if res.success:
            mu_mle, sigma_mle, p_mle, a_mle, b_mle = res.x
            results.append(
                {
                    "skew": skew,
                    "kurt": kurt,
                    "p_mle": p_mle,
                    "a_mle": a_mle,
                    "b_mle": b_mle,
                }
            )

    return pd.DataFrame(results)


# %%
empirical_df = simulate_data()
# %%
plt.scatter(
    empirical_df["kurt"], empirical_df["p_mle"], c=empirical_df["skew"], cmap="viridis", alpha=0.5, label="Empirical p"
)
kurt_space = np.linspace(empirical_df["kurt"].min(), empirical_df["kurt"].max(), 1000)
plt.xlabel("Sample Excess Kurtosis")
plt.xlim(-3, 5)
plt.ylabel("Fitted p (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.colorbar(label="Skew")
plt.show()
plt.scatter(
    empirical_df["kurt"], empirical_df["a_mle"], c=empirical_df["skew"], cmap="viridis", alpha=0.5, label="Empirical a"
)
kurt_space = np.linspace(empirical_df["kurt"].min(), empirical_df["kurt"].max(), 1000)
plt.xlabel("Sample Excess Kurtosis")
plt.xlim(-3, 5)
plt.ylabel("Fitted a (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Skew")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
plt.scatter(
    empirical_df["kurt"], empirical_df["b_mle"], c=empirical_df["skew"], cmap="viridis", alpha=0.5, label="Empirical b"
)
kurt_space = np.linspace(empirical_df["kurt"].min(), empirical_df["kurt"].max(), 1000)
plt.xlabel("Sample Excess Kurtosis")
plt.xlim(-3, 5)
plt.ylabel("Fitted b (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Skew")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
# %%
plt.scatter(
    empirical_df["skew"], empirical_df["p_mle"], c=empirical_df["kurt"], cmap="viridis", alpha=0.5, label="Empirical p"
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylabel("Fitted p (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.colorbar(label="Kurtosis")
plt.show()
plt.scatter(
    empirical_df["skew"], empirical_df["a_mle"], c=empirical_df["kurt"], cmap="viridis", alpha=0.5, label="Empirical a"
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylabel("Fitted a (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Kurtosis")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
plt.scatter(
    empirical_df["skew"], empirical_df["b_mle"], c=empirical_df["kurt"], cmap="viridis", alpha=0.5, label="Empirical b"
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylabel("Fitted b (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Kurtosis")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# %%
def skewness_kurtosis_to_b(skewness, kurtosis):
    """Convert skewness and kurtosis to b parameter."""
    return 1 / kurtosis * (skewness) ** 3


# Plot function over scatter plot
plt.scatter(
    empirical_df["skew"], empirical_df["b_mle"], c=empirical_df["kurt"], cmap="viridis", alpha=0.5, label="Empirical b"
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
for k in np.arange(0, 25, 5):
    plt.plot(skew_space, skewness_kurtosis_to_b(skew_space, k), label=f"Kurtosis = {k}", linestyle="--")
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylim(empirical_df["b_mle"].min(), empirical_df["b_mle"].max())
plt.ylabel("Fitted b (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Kurtosis")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
# %%
