"""File to investigate various approximations of the parameters
of the sinh-arcsinh (SAS/SHASH) distribution based on moment estimates
or other relations.
"""
# %%
import matplotlib.pyplot as plt
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
        dist_type = np.random.choice(["skewnorm", "gh", "uniform", "normal", "laplace"])

        if dist_type == "skewnorm":
            # Random params for skew-t
            a = np.random.uniform(-10, 10)  # skew
            loc = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.5, 5)
            data = skewnorm.rvs(a, loc=loc, scale=scale, size=n_samples_per_sim)
        elif dist_type == "gh":  # genhyperbolic
            p = np.random.uniform(-0.5, 1.5)  # lambda
            a = np.random.uniform(0.1, 10)  # alpha
            b = np.random.uniform(-a + 1e-3, a - 1e-3)  # beta
            loc = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.5, 5)
            data = genhyperbolic.rvs(p, a, b, loc=loc, scale=scale, size=n_samples_per_sim)
        elif dist_type == "uniform":
            # Uniform distribution
            low = np.random.uniform(-10, 0)
            high = np.random.uniform(0, 10)
            data = np.random.uniform(low, high, n_samples_per_sim)
        elif dist_type == "normal":
            # Normal distribution
            mean = np.random.uniform(-5, 5)
            std = np.random.uniform(0.5, 5)
            data = np.random.normal(mean, std, n_samples_per_sim)
        elif dist_type == "laplace":
            # Laplace distribution
            loc = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.5, 5)
            data = np.random.laplace(loc, scale, n_samples_per_sim)

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
            bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-5, 5), (1e-3, 100)],
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
    kurtosis = np.abs(kurtosis + np.pi / 3 + 0.25)  # excess kurtosis, map s.t. peak is around 0
    return 1 / (np.log(kurtosis + 1)) + 0.25


plt.plot(empirical_df["kurt"], empirical_df["delta_mle"], "o", alpha=0.5, markersize=3, label="Empirical Delta")
kurt_space = np.linspace(empirical_df["kurt"].min(), empirical_df["kurt"].max(), 1000)
plt.plot(kurt_space, kurt_to_delta(kurt_space), "r--", label="Approximation: delta = 1 / (|kurtosis| + 1)", linewidth=2)
plt.xlabel("Sample Excess Kurtosis")
plt.xlim(-3, 5)
plt.ylabel("Fitted Delta (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# %%
# Now simulate data from SAS distribution directly and fit the parameters
# to see if the same pattern holds.
# Sampling can be done directly from standard normal samples via the transformation
# below
def sinh_arcsinh_pdf(x, xi, eta, epsilon, delta):
    """
    Apply the sinh-arcsinh transformation to a normal distribution.
    """
    z = (x - xi) / eta
    return (
        1
        / np.sqrt(2 * np.pi)
        * delta
        * c_epsilon_delta(z, epsilon, delta)
        / np.sqrt(1 + z**2)
        * np.exp(-0.5 * (s_epsilon_delta(z, epsilon, delta) ** 2))
    )


def run_empirical_sas_analysis(n_simulations=500, n_samples_per_sim=1000):
    """
    Runs a simulation to find the empirical relationship between moments
    and SAS MLE parameters.
    """
    results = []

    print("Running SAS simulations...")
    for i in range(n_simulations):
        if i % 50 == 0:
            print(f"  Simulation {i}/{n_simulations}...")

        # --- 1. Generate Data from SAS Distribution ---
        xi = np.random.uniform(-50, 50)
        eta = np.random.uniform(0.5, 50)
        epsilon = np.random.uniform(-50, 50)
        delta = np.random.uniform(0.25, 10)

        # Sample from standard normal and apply transformation
        z_samples = np.random.normal(size=n_samples_per_sim)
        data = sinh_arcsinh_pdf(z_samples, xi, eta, epsilon, delta)

        # --- 2. Calculate Moments and Fit SAS via MLE ---
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
            bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-50, 50), (1e-3, 100)],
        )

        if res.success:
            mu_mle, sigma_mle, eps_mle, delta_mle = res.x
            results.append({"skew": skew, "kurt": kurt, "epsilon_mle": eps_mle, "delta_mle": delta_mle})

    return pd.DataFrame(results)


# Run the empirical analysis for SAS distribution
sas_empirical_df = run_empirical_sas_analysis(n_simulations=500)
plot_empirical_results(sas_empirical_df)

# %%

plt.plot(sas_empirical_df["kurt"], sas_empirical_df["delta_mle"], "o", alpha=0.5, markersize=3, label="Empirical Delta")
kurt_space = np.linspace(sas_empirical_df["kurt"].min(), sas_empirical_df["kurt"].max(), 1000)
plt.plot(kurt_space, kurt_to_delta(kurt_space), "r--", label="Approximation: delta = 1 / (|kurtosis| + 1)", linewidth=2)
plt.xlabel("Sample Excess Kurtosis")
# plt.xlim(-3, 5)
plt.ylabel("Fitted Delta (MLE)")
plt.title("Sample Kurtosis vs. Fitted Delta")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# %%
# Given the function above, plot skewness vs epsilon, coloring the points by delta.
def skewness_delta_to_epsilon(x: np.ndarray, delta: float, tanh_scale: float = 5, cubic_scale=10) -> np.ndarray:
    """
    A function that smoothly transitions between three behaviors based on delta.

    - For delta > 1: Blends between a linear (-x) and a cubic (-x^3) function.
                     As delta increases, the behavior becomes more cubic.
    - For delta = 1: Behaves exactly like the line y = -x.
    - For delta < 1: Blends between a linear (-x) and a hyperbolic tangent (-tanh(x)).
                     As delta decreases towards 0, the behavior becomes more like -tanh(x).

    Args:
        x (np.ndarray): The input value(s).
        delta (float): The parameter controlling the function's behavior. Must be positive.

    Returns:
        np.ndarray: The transformed value(s).
    """
    if delta <= 0:
        raise ValueError("delta must be a positive number.")
    # Ensure x is a numpy array for vectorized operations
    x = np.asanyarray(x)

    if delta > 1:
        # For delta > 1, we blend between -x and -x^3.
        # The weight for the cubic part increases from 0 to 1 as delta goes from 1 to infinity.
        weight_cubic = 1 - (1 / delta**3)
        weight_linear = 1 - weight_cubic

        return weight_linear * (-x) + weight_cubic * (-delta * x**3)

    else:  # This handles both delta == 1 and delta < 1
        # For delta <= 1, we blend between -x and -tanh(x).
        # The weight for the tanh part increases from 0 to 1 as delta goes from 1 to 0.
        weight_tanh = 1 - delta**3
        weight_linear = 1 - weight_tanh

        return weight_linear * (-x) + weight_tanh * (-np.tanh(x))


skewness_space = np.linspace(empirical_df["skew"].min(), empirical_df["skew"].max(), 100)
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(
    empirical_df["skew"], empirical_df["epsilon_mle"], c=empirical_df["delta_mle"], cmap="viridis", alpha=0.5, s=10
)
# plot for delta in [.25, .4, 1.0, 1.5, 2.0
for delta in [0.25, 0.4, 1.0, 1.5, 2.0, 10, 20]:
    plt.plot(
        skewness_space, skewness_delta_to_epsilon(skewness_space, delta), label=f"Delta = {delta:.2f}", linestyle="--"
    )
plt.colorbar(label="Fitted Delta (MLE)")
plt.title("Sample Skewness vs. Fitted Epsilon (Colored by Delta)")
plt.xlabel("Sample Skewness")
plt.ylabel("Fitted Epsilon (MLE)")
plt.axhline(0, color="red", linestyle="--", label="Epsilon = 0")
plt.axvline(0, color="blue", linestyle="--", label="Skewness = 0")
# plt.legend()
plt.xlim(-2, 2)
plt.ylim(-5, 5)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %%
skewness_space = np.linspace(sas_empirical_df["skew"].min(), sas_empirical_df["skew"].max(), 1000)
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(
    sas_empirical_df["skew"],
    sas_empirical_df["epsilon_mle"],
    c=sas_empirical_df["delta_mle"],
    cmap="viridis",
    alpha=0.5,
    s=10,
)
# plot for delta in [.25, .4, 1.0, 1.5, 2.0]
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
# plt.legend()
# plt.xlim(-2, 2)
plt.ylim(-5, 5)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
# %%
# Sample from GH and fit SHASH
# 1. Define GH parameters for a skewed, heavy-tailed distribution
p_gh = -2.5  # Lambda
a_gh = 50  # Alpha (tail heaviness, smaller is heavier)
b_gh = -40  # Beta (skewness, |b| < a)
loc_gh = 2.0
scale_gh = 2.0
gh_params = (p_gh, a_gh, b_gh, loc_gh, scale_gh)
n_samples_gh = 2000

# 2. Generate samples
gh_data = genhyperbolic.rvs(*gh_params, size=n_samples_gh, random_state=123)

# 3. Fit the SHASH distribution to the GH data
initial_params_gh = np.array([np.mean(gh_data), np.std(gh_data), 0.0, 1.0])
res_gh = minimize(
    nll_shash,
    initial_params_gh,
    args=(gh_data,),
    method="L-BFGS-B",
    bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-20, 20), (1e-3, 50)],
)
fitted_shash_params = res_gh.x
# fitted_gh_params = genhyperbolic.fit(gh_data)

print("\n--- GH to SHASH Fit ---")
print(f"Original GH params (p, a, b, loc, scale): {gh_params}")
print(f"Fitted SHASH params (mu, sigma, eps, delta): {np.round(fitted_shash_params, 4)}")


# 4. Plot the results
plt.figure(figsize=(10, 6))
plt.hist(gh_data, bins=50, density=True, alpha=0.6, label="GH Samples Histogram")

# Create x-range for plotting PDFs
x_plot = np.linspace(gh_data.min(), gh_data.max(), 1000)

# Original GH PDF
gh_pdf = genhyperbolic.pdf(x_plot, *gh_params)
plt.plot(x_plot, gh_pdf, "-", lw=2, c="dodgerblue", label="Original GH PDF")
# Fitted GH PDF
# gh_pdf_fitted = genhyperbolic.pdf(x_plot, *fitted_gh_params)
# plt.plot(x_plot, gh_pdf_fitted, 'b--', lw=2, label="Fitted GH PDF")

# Fitted SHASH PDF
shash_pdf = np.exp(loglik_shash(x_plot, *fitted_shash_params))
plt.plot(x_plot, shash_pdf, "--", c="crimson", lw=2, label="Fitted SHASH PDF")

plt.title("Fitting SHASH Distribution to Generalized Hyperbolic Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# --- Combined Distribution Experiment ---

# 1. Define parameters and generate data from three distributions
n_samples_per_dist = 50
common_loc = 5.0
np.random.seed(420)  # for reproducibility

# Uniform distribution
unif_low = common_loc - 3
unif_high = common_loc + 3
data_unif = np.random.uniform(low=unif_low, high=unif_high, size=n_samples_per_dist)

# Skew-Normal distribution
a_sn = 4.0
scale_sn = 3.0
data_sn = skewnorm.rvs(a_sn, loc=common_loc, scale=scale_sn, size=n_samples_per_dist)

# Generalized Hyperbolic distribution
p_gh = -1.5
a_gh = 20.0
b_gh = 1.5
scale_gh = 2.0
data_gh = genhyperbolic.rvs(p_gh, a_gh, b_gh, loc=common_loc, scale=scale_gh, size=n_samples_per_dist)

# 2. Combine the data
combined_data = np.concatenate([data_unif, data_sn, data_gh])

# 3. Fit the SHASH distribution to the combined data
initial_params_comb = np.array([np.mean(combined_data), np.std(combined_data), 0.0, 1.0])
res_comb = minimize(
    nll_shash,
    initial_params_comb,
    args=(combined_data,),
    method="L-BFGS-B",
    bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-50, 50), (1e-3, 100)],
)
fitted_shash_params_comb = res_comb.x

print("\n--- Combined Data to SHASH Fit ---")
print(f"Fitted SHASH params (mu, sigma, eps, delta): {np.round(fitted_shash_params_comb, 4)}")

# 4. Plot the results
plt.figure(figsize=(10, 6))
plt.hist(
    combined_data,
    bins=20,
    density=True,
    alpha=0.6,
    label="Combined Data Histogram",
)

# Create x-range for plotting PDF
x_plot_comb = np.linspace(combined_data.min(), combined_data.max(), 1000)

# Fitted SHASH PDF
shash_pdf_comb = np.exp(loglik_shash(x_plot_comb, *fitted_shash_params_comb))
plt.plot(x_plot_comb, shash_pdf_comb, "--", c="crimson", lw=2, label="Fitted SHASH PDF")

plt.title("Fitting SHASH to a Mixture of Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# --- Uniform + Skew-Normal Experiment ---

# 1. Define parameters and generate data
n_samples_unif = 200
n_samples_sn = 50  # Set to 0 to have only uniform data
np.random.seed(1337)

# Uniform distribution
unif_low = -5
unif_high = 10
data_unif = np.random.uniform(low=unif_low, high=unif_high, size=n_samples_unif)

# Heavily skewed Skew-Normal distribution
a_sn = 15.0  # High skewness
loc_sn = 8.0  # Place it towards one end of the uniform range
scale_sn = 5.0
data_sn = skewnorm.rvs(a_sn, loc=loc_sn, scale=scale_sn, size=n_samples_sn)

# 2. Combine the data
combined_data = np.concatenate([data_unif, data_sn])

# 3. Fit the SHASH distribution to the combined data
initial_params_comb = np.array([np.mean(combined_data), np.std(combined_data), 0.0, 1.0])
res_comb = minimize(
    nll_shash,
    initial_params_comb,
    args=(combined_data,),
    method="L-BFGS-B",
    bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-50, 50), (1e-3, 100)],
)
fitted_shash_params_comb = res_comb.x

print("\n--- Uniform + Skew-Normal Data to SHASH Fit ---")
print(f"Fitted SHASH params (mu, sigma, eps, delta): {np.round(fitted_shash_params_comb, 4)}")

# 4. Plot the results
plt.figure(figsize=(10, 6))
plt.hist(
    combined_data,
    bins=30,
    density=True,
    alpha=0.6,
    label="Combined Data Histogram",
)

# Create x-range for plotting PDF
x_plot_comb = np.linspace(combined_data.min(), combined_data.max(), 1000)

# Fitted SHASH PDF
shash_pdf_comb = np.exp(loglik_shash(x_plot_comb, *fitted_shash_params_comb))
plt.plot(x_plot_comb, shash_pdf_comb, "--", c="crimson", lw=2, label="Fitted SHASH PDF")

plt.title("Fitting SHASH to Uniform Data with a Skewed Component")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
