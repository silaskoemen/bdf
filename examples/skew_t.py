""" Generalized hyperbolic distribution (4th parameterization) example.
Tries to examine whether skewness/shape/tailness can be estimated from
3rd and 4th moments.
"""
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
from scipy.special import beta


def loglik_skew_t(params: tuple | list | np.ndarray, x) -> float:
    mu, sigma, a, b = params
    if sigma <= 0 or a <= 0 or b <= 0:
        return np.inf
    z = (x - mu) / sigma

    return (
        -np.log(sigma)
        - (a + b - 1) * np.log(2)
        - np.log(beta(a, b) + 1e-7)
        - 0.5 * np.log(a + b)
        + (a + 0.5) * np.log(1 + z / (np.sqrt(a + b + z**2) + 1e-7))
        + (b + 0.5) * np.log(1 - z / (np.sqrt(a + b + z**2) + 1e-7))
    )


def nll_skew_t(params, x):
    """Negative log-likelihood for the generalized hyperbolic distribution."""
    return -np.mean(loglik_skew_t(params, x))


def lik_skew_t(params, x):
    """Likelihood for the generalized hyperbolic distribution."""
    return np.exp(loglik_skew_t(params, x))


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
            nll_skew_t,
            np.array([mean, std, 10, 10]),
            args=(data,),
            method="L-BFGS-B",
            bounds=[(-np.inf, np.inf), (1e-6, np.inf), (1e-6, 1e8), (1e-6, 1e8)],
        )

        if res.success:
            mu_mle, sigma_mle, a_mle, b_mle = res.x
            results.append(
                {
                    "skew": skew,
                    "kurt": kurt,
                    "a_mle": a_mle,
                    "b_mle": b_mle,
                }
            )

    return pd.DataFrame(results)


# %%
empirical_df = simulate_data()
# %%
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
# Instead of sampling from a diverse set of distributions, sample from
# skew t itself (meaning a and b are known) and plot a and b vs skew and kurtosis.
from scipy.stats import jf_skew_t


def simulate_skew_t_data(n_simulations=1000, samples_per_sim=100, seed=42):
    np.random.seed(seed)
    results = []
    for _ in tqdm(range(n_simulations)):
        mu = np.random.uniform(-100, 100)
        sigma = np.random.uniform(0.1, 50)
        a = np.random.uniform(1e-5, 1e3)
        b = np.random.uniform(1e-5, 1e3)
        data = jf_skew_t.rvs(a=a, b=b, loc=mu, scale=sigma, size=samples_per_sim)

        mean = np.mean(data)
        std = np.std(data, ddof=1)
        skew = np.mean(((data - mean) / std) ** 3)
        kurt = np.mean(((data - mean) / std) ** 4) - 3

        results.append({"skew": skew, "kurt": kurt, "a_mle": a, "b_mle": b})

    return pd.DataFrame(results)


# %%
empirical_df = simulate_skew_t_data()
# %%
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
# Investigate ratios of a/b vs skew and kurtosis.
plt.scatter(
    empirical_df["kurt"],
    empirical_df["a_mle"] / empirical_df["b_mle"],
    c=empirical_df["skew"],
    cmap="viridis",
    alpha=0.5,
    label="Empirical a",
)
kurt_space = np.linspace(empirical_df["kurt"].min(), empirical_df["kurt"].max(), 1000)
plt.xlabel("Sample Excess Kurtosis")
plt.xlim(-3, 5)
plt.ylabel("Fitted a/b (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Skew")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
plt.scatter(
    empirical_df["skew"],
    empirical_df["a_mle"] / empirical_df["b_mle"],
    c=empirical_df["kurt"],
    cmap="viridis",
    alpha=0.5,
    label="Empirical a",
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylabel("Fitted a/b (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Kurtosis")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
plt.scatter(
    empirical_df["kurt"],
    empirical_df["a_mle"] * empirical_df["b_mle"],
    c=empirical_df["skew"],
    cmap="viridis",
    alpha=0.5,
    label="Empirical a",
)
kurt_space = np.linspace(empirical_df["kurt"].min(), empirical_df["kurt"].max(), 1000)
plt.xlabel("Sample Excess Kurtosis")
plt.xlim(-3, 5)
plt.ylabel("Fitted a*b (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Skew")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
plt.scatter(
    empirical_df["skew"],
    empirical_df["a_mle"] * empirical_df["b_mle"],
    c=empirical_df["kurt"],
    cmap="viridis",
    alpha=0.5,
    label="Empirical a",
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylabel("Fitted a*b (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.colorbar(label="Kurtosis")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# %%
def skewness_kurtosis_to_adivb(skewness, kurtosis):
    """We can fit an exponential function to estimate the ratio of a/b from skewness
    and kurtosis. Moreover, we can fit an exponential function to estimate a from skewness
    and kurtosis, enabling us to solve for b as well.
    """
    return np.exp(np.pi * (1 + skewness) ** 3 / ((kurtosis + 3)))


plt.scatter(
    empirical_df["skew"],
    empirical_df["a_mle"] / empirical_df["b_mle"],
    c=empirical_df["kurt"],
    cmap="viridis",
    alpha=0.5,
    label="Empirical a",
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
for kurt in [0, 10, 25, 80]:
    plt.plot(skew_space, skewness_kurtosis_to_adivb(skew_space, kurt), label=f"Kurtosis = {kurt}", linestyle="--")
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylabel("Fitted a/b (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.ylim(0, 250)
plt.colorbar(label="Kurtosis")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


def skewness_to_a(skewness):
    """We can fit an exponential function to estimate a from skewness
    and kurtosis, enabling us to solve for b as well.
    """
    return np.exp(np.pi * (np.pi / 2 + skewness))


plt.scatter(
    empirical_df["skew"], empirical_df["a_mle"], c=empirical_df["kurt"], cmap="viridis", alpha=0.5, label="Empirical a"
)
skew_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 1000)
plt.plot(skew_space, skewness_to_a(skew_space), linestyle="--")
plt.xlabel("Sample Skew")
plt.xlim(-3, 5)
plt.ylabel("Fitted a (MLE)")
plt.title("Sample Skew vs. Fitted Delta")
plt.legend()
plt.ylim(0, 250)
plt.colorbar(label="Kurtosis")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
from math import lgamma

# %%
from scipy.special import gamma


def skewness_kurtosis_to_a_b(skewness, kurtosis):
    """We can fit an exponential function to estimate a from skewness
    and kurtosis, enabling us to solve for b as well.
    """
    a = skewness_to_a(skewness)
    a = np.clip(a, 1e-6, 1000)
    b = a / skewness_kurtosis_to_adivb(skewness, kurtosis)
    b = np.clip(b, 1e-6, 10)
    return a, b


def mu_from_mean_a_b_sigma(mean: float, a: float, b: float, sigma: float) -> float:
    """Calculate mu from mean, a and b."""
    log_gamma_frac = lgamma(a - 0.5) + lgamma(b - 0.5) - lgamma(a) - lgamma(b)
    gamma_frac = np.exp(log_gamma_frac)
    return mean - (sigma * (a - b) * np.sqrt(a + b) / 2 * gamma_frac)


def sigma_from_std_a_b(std, a, b):
    """Calculate sigma from std, a and b."""
    log_gamma_frac = lgamma(a - 0.5) + lgamma(b - 0.5) - lgamma(a) - lgamma(b)
    gamma_frac = np.exp(log_gamma_frac)
    return std / np.sqrt(1 + gamma_frac - gamma_frac**2)


def data_to_params(data):
    """Calculate a and b from data."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skew = np.mean(((data - mean) / std) ** 3)
    kurt = np.mean(((data - mean) / std) ** 4) - 3
    a, b = skewness_kurtosis_to_a_b(skew, kurt)
    sigma = sigma_from_std_a_b(std, a, b)
    mu = mu_from_mean_a_b_sigma(mean, a, b, sigma)
    return mu, sigma, a, b


def data_to_mle_params(data):
    """Calculate mu, sigma, a, b from data using MLE."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Fit MLE, mu, sigma, a, b
    res = minimize(
        nll_skew_t,
        np.array([mean, std, 10, 10]),
        args=(data,),
        method="L-BFGS-B",
        bounds=[(-np.inf, np.inf), (1e-6, np.inf), (1e-6, 1e8), (1e-6, 1e8)],
    )

    if res.success:
        mu_mle, sigma_mle, a_mle, b_mle = res.x
        return mu_mle, sigma_mle, a_mle, b_mle
    else:
        raise ValueError("MLE optimization failed.")


def data_to_nr_params(data, n_iter=10):
    """Calculate mu, sigma, a, b from data using Newton-Raphson.
    Uses newton-raphson with specified number of iterations using
    finite differences for first and second derivatives.

    Initial values are mean, std, 10, 10.
    """
    params = np.array([np.mean(data), np.std(data, ddof=1), 1, 1])
    for _ in range(n_iter):
        # Calculate gradient and hessian using finite differences
        # Update parameter vector by subtracting gradient divided by hessian
        grad = np.zeros_like(params)
        hess = np.zeros((len(params), len(params)))
        eps = 1e-6
        for i in range(len(params)):
            params_eps = params.copy()
            params_eps[i] += eps
            grad[i] = (nll_skew_t(params_eps, data) - nll_skew_t(params, data)) / eps

            # second derivative is defined as (f(x+h) - 2f(x) + f(x-h)) / h**2
            for j in range(len(params)):
                params_eps_j = params.copy()
                params_eps_j[j] += eps
                hess[i, j] = (
                    nll_skew_t(params_eps, data) - 2 * nll_skew_t(params, data) + nll_skew_t(params_eps_j, data)
                ) / (eps**2)
        # Update parameters
        params -= np.linalg.solve(hess, grad)

    mu, sigma, a, b = params
    if sigma <= 0 or a <= 0 or b <= 0:
        raise ValueError("Parameters must be positive.")
    return mu, sigma, a, b


MU, SIGMA, A, B = 5, 5, 2, 3
samples = jf_skew_t.rvs(a=A, b=B, loc=MU, scale=SIGMA, size=1000)
samples = np.random.normal(-10, 10, size=1000)  # For testing purposes, replace with actual samples
# mu_hat, sigma_hat, a_hat, b_hat = data_to_params(samples)
mu_hat_mle, sigma_hat_mle, a_hat_mle, b_hat_mle = data_to_mle_params(samples)
mu_hat_nr, sigma_hat_nr, a_hat_nr, b_hat_nr = data_to_nr_params(samples, 30)
print(f"True parameters: mu={MU}, sigma={SIGMA}, a={A}, b={B}")
print(
    f"MLE parameters: mu_hat_mle={mu_hat_mle:.3f}, sigma_hat_mle={sigma_hat_mle:.3f}, a_hat_mle={a_hat_mle:.3f}, b_hat_mle={b_hat_mle:.3f}"
)
print(
    f"NR parameters: mu_hat_nr={mu_hat_nr:.3f}, sigma_hat_nr={sigma_hat_nr:.3f}, a_hat_nr={a_hat_nr:.3f}, b_hat_nr={b_hat_nr:.3f}"
)

plt.hist(samples, bins=50, density=True, alpha=0.5, label="Samples")
x = np.linspace(samples.min(), samples.max(), 1000)
plt.plot(x, jf_skew_t.pdf(x, a=A, b=B, loc=MU, scale=SIGMA), label="True PDF", color="red")
plt.plot(
    x, jf_skew_t.pdf(x, a=a_hat_mle, b=b_hat_mle, loc=mu_hat_mle, scale=sigma_hat_mle), label="MLE PDF", color="blue"
)
plt.plot(x, jf_skew_t.pdf(x, a=a_hat_nr, b=b_hat_nr, loc=mu_hat_nr, scale=sigma_hat_nr), label="NR PDF", color="orange")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Skew t Distribution: True vs Estimated Parameters")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %%
