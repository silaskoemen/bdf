"""File to investigate various approximations of the parameters
of the sinh-arcsinh (SAS/SHASH) distribution based on moment estimates
or other relations.
"""

import matplotlib.pyplot as plt

# %%
import numpy as np
from scipy.special import kv
from scipy.stats import skewnorm


def c_epsilon_delta(z, epsilon, delta):
    return np.cosh(epsilon + delta * np.arcsinh(z))


def s_epsilon_delta(z, epsilon, delta):
    return np.sinh(epsilon + delta * np.arcsinh(z))


def loglik_shash(
    y: np.ndarray,
    mu: float = 0.0,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    delta: float = 1e-6,
) -> float:
    """Log-likelihood function for the sinh-arcsinh (SAS/SHASH) distribution.

    Args
    ----
    y : np.ndarray
        Input data.
    mu : float, optional
        Location parameter (default is 0.0).
    sigma : float, optional
        Scale parameter (default is 1.0).
    epsilon : float, optional
        Skewness parameter.
    delta : float, optional
        Tail heaviness parameter.

    Returns
    -------
    float
        Log-likelihood value.
    """
    if sigma <= 0:
        raise ValueError("Scale parameter sigma must be positive.")

    z = (y - mu) / sigma
    # Calculate the log-likelihood using the skew-normal distribution
    return (
        -np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        + np.log(delta)
        + np.log(c_epsilon_delta(z, epsilon, delta))
        - 0.5 * np.log(1 + z**2)
        - 0.5 * s_epsilon_delta(z, epsilon, delta) ** 2
    )


def nll_shash(
    params: np.ndarray,
    y: np.ndarray,
) -> float:
    """Negative log-likelihood function for the SAS/SHASH distribution.

    Args
    ----
    params : np.ndarray
        Parameters [mu, sigma, epsilon, delta].
    y : np.ndarray
        Input data.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    mu, sigma, epsilon, delta = params
    return -np.mean(loglik_shash(y, mu, sigma, epsilon, delta))


def approximate_parameters(y: np.ndarray) -> tuple[float, float, float, float]:
    """Approximate the parameters of the SAS/SHASH distribution based on the data.

    Idea is to recover mu and sigma afterwards from approximated epsilon and delta.
    Ideally, a closed-form solution is available, relating epsilon and delta to the
    3rd and 4th moments of the data (skewness and kurtosis).

    Args
    ----
    y : np.ndarray
        Input data.

    Returns
    -------
    tuple[float, float, float, float]
        Estimated parameters (mu, sigma, epsilon, delta).
    """
    mean = np.mean(y)
    std = np.std(y, ddof=1)  # Sample standard deviation
    skewness = np.mean(((y - mean) / std) ** 3)
    kurtosis = np.mean(((y - mean) / std) ** 4) - 3  # Excess kurtosis

    # Approximate epsilon and delta based on skewness and kurtosis
    epsilon = -np.arcsinh(skewness / (np.sqrt(1 + skewness**2)))
    delta = np.sqrt(kurtosis + 3) / (1 + np.abs(epsilon))

    mu = mean + np.sinh(epsilon / delta) * np.exp(0.25) / np.sqrt(np.pi * 8) * (
        kv((delta + 1) / (2 * delta), 0.25) + kv((delta - 1) / (2 * delta), 0.25)
    )

    sigma = np.sqrt(
        (
            np.cosh(2 * epsilon / delta)
            * np.exp(0.25)
            / np.sqrt(32 * np.pi)
            * (kv((delta + 1) / (2 * delta), 0.25) + kv((delta - 1) / (2 * delta), 0.25))
            - 0.5
            - (mu - mean) ** 2
        )
        * std**2
    )
    sigma = np.clip(sigma, 1e-6, None)  # Ensure sigma is positive

    return mu, sigma, epsilon, delta


# %%
# Simulate data from a skew-normal distribution
np.random.seed(42)
n_samples = 1000
ALPHA, XI, OMEGA = 5.0, 0.0, 5.0
data = skewnorm.rvs(ALPHA, loc=XI, scale=OMEGA, size=n_samples)

# Fit the SAS/SHASH distribution to the data
from scipy.optimize import minimize

initial_params = np.array([np.mean(data), np.std(data), 0.0, 1])
result = minimize(
    nll_shash,
    initial_params,
    args=(data,),
    method="L-BFGS-B",
    bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-np.inf, np.inf), (1e-6, np.inf)],
    options={"maxiter": 50},
)
fitted_params = result.x
print(
    f"Fitted parameters: mu={fitted_params[0]:.4f}, sigma={fitted_params[1]:.4f}, epsilon={fitted_params[2]:.4f}, delta={fitted_params[3]:.4f}"
)

approx_params = approximate_parameters(data)
print(
    f"Approximated parameters: mu={approx_params[0]:.4f}, sigma={approx_params[1]:.4f}, epsilon={approx_params[2]:.4f}, delta={approx_params[3]:.4f}"
)

# Plot the original data and the fitted distribution
x = np.linspace(min(data) - 3 * np.std(data), max(data) + 3 * np.std(data), 1000)
y_mle = np.exp(loglik_shash(x, *fitted_params))
y_approx = np.exp(loglik_shash(x, *approx_params))
plt.hist(data, bins=30, density=True, alpha=0.5, label="Data histogram")
plt.plot(x, y_mle, "r-", lw=2, label="Fitted SAS/SHASH distribution")
plt.plot(x, y_approx, "g--", lw=2, label="Approximated SAS/SHASH distribution")
plt.title("Fitted SAS/SHASH Distribution")
plt.xlabel("Data")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

import numpy as np

# %%
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genhyperbolic, skewnorm


def run_empirical_analysis(n_simulations=500, n_samples_per_sim=1000):
    """
    Runs a simulation to find the empirical relationship between moments
    and SHASH MLE parameters.
    """
    results = []

    print("Running simulations...")
    for i in range(n_simulations):
        if i % 50 == 0:
            print(f"  Simulation {i}/{n_simulations}...")

        # --- 1. Generate Data from a Random Distribution ---
        dist_type = np.random.choice(["skewt", "gh"])

        if dist_type == "skewt":
            # Random params for skew-t
            a = np.random.uniform(-10, 10)  # skew
            loc = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.5, 5)
            data = skewnorm.rvs(a, loc=loc, scale=scale, size=n_samples_per_sim)
        else:  # genhyperbolic
            p = np.random.uniform(-0.5, 1.5)  # lambda
            a = np.random.uniform(0.1, 10)  # alpha
            b = np.random.uniform(-a + 1e-3, a - 1e-3)  # beta
            loc = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.5, 5)
            data = genhyperbolic.rvs(p, a, b, loc=loc, scale=scale, size=n_samples_per_sim)

        # --- 2. Calculate Moments and Fit SHASH via MLE ---
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-6:
            continue

        skew = np.mean(((data - mean) / std) ** 3)
        kurt = np.mean(((data - mean) / std) ** 4) - 3

        # Fit MLE
        initial_params = np.array([mean, std, 0.0, 1.0])
        res = minimize(
            nll_shash,
            initial_params,
            args=(data,),
            method="L-BFGS-B",
            bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-5, 5), (1e-3, 10)],
        )

        if res.success:
            mu_mle, sigma_mle, eps_mle, delta_mle = res.x
            results.append({"skew": skew, "kurt": kurt, "epsilon_mle": eps_mle, "delta_mle": delta_mle})

    return pd.DataFrame(results)


def plot_empirical_results(df_results):
    """Plots the results of the empirical analysis."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Skewness vs. Epsilon
    axes[0].scatter(df_results["skew"], df_results["epsilon_mle"], alpha=0.5, s=10)
    axes[0].set_title("Sample Skewness vs. Fitted Epsilon")
    axes[0].set_xlabel("Sample Skewness")
    axes[0].set_ylabel("Fitted Epsilon (MLE)")

    # Add approximation line for comparison
    skew_range = np.linspace(df_results["skew"].min(), df_results["skew"].max(), 100)
    axes[0].plot(skew_range, -skew_range / 2.0, "r--", label="Approx: eps = -skew/2")
    axes[0].legend()

    # Kurtosis vs. Delta
    axes[1].scatter(df_results["kurt"], df_results["delta_mle"], alpha=0.5, s=10)
    axes[1].set_title("Sample Kurtosis vs. Fitted Delta")
    axes[1].set_xlabel("Sample Excess Kurtosis")
    axes[1].set_ylabel("Fitted Delta (MLE)")

    # Add approximation line for comparison
    kurt_range = np.linspace(df_results["kurt"].min(), df_results["kurt"].max(), 100)
    delta_approx = np.piecewise(
        kurt_range,
        [kurt_range < 0, kurt_range >= 0],
        [lambda k: np.sqrt(1 / (1 - np.sqrt(2 / 3 * -k))), lambda k: 1 + k / 6.0],
    )
    axes[1].plot(kurt_range, delta_approx, "r--", label="Approximation Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# --- Run the analysis ---
# Note: This can take a few minutes to run.
empirical_df = run_empirical_analysis(n_simulations=500)
plot_empirical_results(empirical_df)


# %%
def kurt_to_delta(kurtosis: float) -> float:
    """Approximate delta from excess kurtosis."""
    # return 1 / (
    #     .5 + np.abs(kurtosis) * 1 + kurtosis ** 2 * .1
    # ) + .4
    return (
        1
        / (
            1 / (1 + np.abs(kurtosis)) * np.sqrt(kurtosis**2 + 0.25)
            + np.abs(kurtosis) / (1 + np.abs(kurtosis)) * np.sqrt(np.abs(kurtosis) + 1)
        )
        + 0.25
    )
    return (
        1
        / (
            1 / (1 + np.abs(kurtosis)) * np.log(kurtosis**2 + 1)
            + np.abs(kurtosis) / (1 + np.abs(kurtosis)) * np.log(np.abs(kurtosis))
            + 1
        )
        + 0.25
    )


plt.plot(empirical_df["kurt"], empirical_df["delta_mle"], "o", alpha=0.5, markersize=3, label="Empirical Delta")
kurt_space = np.linspace(empirical_df["kurt"].min(), empirical_df["kurt"].max(), 100)
plt.plot(kurt_space, kurt_to_delta(kurt_space), "r--", label="Approximation: delta = 1 / (|kurtosis| + 1)", linewidth=2)
plt.xlabel("Sample Excess Kurtosis")
plt.ylabel("Fitted Delta (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# %%
# Given the function above, plot skewness vs epsilon, coloring the points by delta.
def skewness_delta_to_epsilon(
    skewness: float, delta: float, scale_transition: float = 10, scale_skew: float = 0.5
) -> float:
    """Approximate epsilon from skewness and delta."""
    skewness_tan = np.clip(skewness, -10, 10)  # Clip skewness to avoid extreme values
    skewness_tan = skewness_tan / 2.1 * np.pi
    delta_scaled = (delta) ** scale_transition
    trans = delta_scaled / (1 + delta_scaled)
    # NOTE: should be multiplicative s.t. delta = 1 results in epsilon = -skewness / 3
    # for higher deltas, should be negative tan / 3rd degree polynomial
    # for lower deltas, should be negative of tanh
    print(trans)
    return trans * (-3 * skewness**3) + (1 - trans) * ((-skewness / 3) ** (1 / 3))  # -np.tanh(skewness)


skewness_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 100)
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(
    empirical_df["skew"], empirical_df["epsilon_mle"], c=empirical_df["delta_mle"], cmap="viridis", alpha=0.5, s=10
)
# plot for delta in [.25, .4, 1.0, 1.5, 2.0
for delta in [0.25, 0.4, 1.0, 1.5, 2.0]:
    plt.plot(
        skewness_space, skewness_delta_to_epsilon(skewness_space, delta), label=f"Delta = {delta:.2f}", linestyle="--"
    )
plt.colorbar(label="Fitted Delta (MLE)")
plt.title("Sample Skewness vs. Fitted Epsilon (Colored by Delta)")
plt.xlabel("Sample Skewness")
plt.ylabel("Fitted Epsilon (MLE)")
plt.axhline(0, color="red", linestyle="--", label="Epsilon = 0")
plt.axvline(0, color="blue", linestyle="--", label="Skewness = 0")
plt.legend()
plt.ylim(-5, 5)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
# %%
