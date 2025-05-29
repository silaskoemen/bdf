# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate some skewed data from gamma distribution
shape, scale = 3.0, 5.0
n_samples = 200
data = np.random.gamma(shape, scale, size=n_samples)
# data = np.array([-1, -1, 0, 1])

# Find MLE for skew normal parameters to get initial estimates
# For skew normal, we need location (xi), scale (omega), and shape (alpha)
alpha_mle, xi_mle, omega_mle = stats.skewnorm.fit(data)
print(f"MLE estimates: xi={xi_mle:.4f}, omega={omega_mle:.4f}, alpha={alpha_mle:.4f}")

# Define prior parameters for xi (location) and alpha (shape)
prior_mean_xi = 0.0
prior_std_xi = 100.0  # weak prior
prior_mean_alpha = 0.0
prior_std_alpha = 100.0  # weak prior

# Calculate posterior parameters
n = data.shape[0]
sample_mean = np.mean(data)
sample_var = np.var(data, ddof=1)

# Posterior mean for xi (Normal prior)
posterior_mean = (prior_mean_xi / prior_std_xi**2 + n * sample_mean / sample_var) / (
    1 / prior_std_xi**2 + n / sample_var
)

# Posterior for alpha, empirical Bayes approach, where lambda reflects stength of prior belief
lambda_alpha = 5
posterior_mean_alpha = n / (n + lambda_alpha) * alpha_mle + lambda_alpha / (n + lambda_alpha) * prior_mean_alpha  # type: ignore

posterior_mean_xi = posterior_mean - omega_mle * (
    posterior_mean_alpha / np.sqrt(1 + posterior_mean_alpha**2)
) * np.sqrt(2 / np.pi)
# Use omega_mle for scale parameter as it's not being updated
posterior_omega = omega_mle


def skewness(data):
    len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    return np.mean(((data - mean) / std_dev) ** 3)


skew = skewness(data)
gamma = np.clip(skew, -0.995, 0.995)
# if np.abs(skew) > 1:
#     gamma = np.clip(skew, -0.995, 0.995) ** np.abs(skew)
# else:
#     gamma = np.clip(skew, -0.995, 0.995)
print(f"Sample skewness: {skew:.4f}, clipped gamma: {gamma:.4f}")
delta = np.sign(gamma) * np.sqrt(
    np.pi / 2 * np.abs(gamma) ** (2 / 3) / (np.abs(gamma) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
)
posterior_mean_alpha = (delta / np.sqrt(1 - delta**2)) ** (1 / 3)
posterior_mean_alpha = n / (n + lambda_alpha) * posterior_mean_alpha + lambda_alpha / (n + lambda_alpha) * prior_mean_alpha  # type: ignore
posterior_omega = np.std(data) / np.sqrt(1 - 2 * delta**2 / np.pi)
posterior_mean_xi = posterior_mean - posterior_omega * delta * np.sqrt(2 / np.pi)

print(f"Posterior estimates: xi={posterior_mean_xi:.4f}, omega={posterior_omega:.4f}, alpha={posterior_mean_alpha:.4f}")
print(f"Mean MLE estimates: {xi_mle + omega_mle * (alpha_mle / np.sqrt(1 + alpha_mle**2)) * np.sqrt(2/np.pi):.4f}")  # type: ignore
print(f"Mean Posterior estimates: {posterior_mean_xi + posterior_omega * (posterior_mean_alpha / np.sqrt(1 + posterior_mean_alpha**2)) * np.sqrt(2/np.pi):.4f}")  # type: ignore
print(f"Mean data: {np.mean(data):.4f}")
# Plotting
plt.figure(figsize=(10, 6))

# Plot histogram of data
plt.hist(data, bins=30, density=True, alpha=0.6, color="skyblue", label="Data histogram")

# Plot MLE skew normal
x = np.linspace(min(data) - 1, max(data) + 1, 1000)
plt.plot(
    x,
    stats.skewnorm.pdf(x, alpha_mle, loc=xi_mle, scale=omega_mle),
    "r-",
    lw=2,
    label="MLE Skew Normal",
)

# Plot posterior skew normal
plt.plot(
    x,
    stats.skewnorm.pdf(x, posterior_mean_alpha, loc=posterior_mean_xi, scale=posterior_omega),
    "g-",
    lw=2,
    label="Posterior Skew Normal",
)
plt.plot(
    x,
    stats.norm.pdf(x, loc=data.mean(), scale=data.std()),
    "b--",
    lw=2,
    label="Normal Approximation",
)
plt.title("Skew Normal Distribution: Data, MLE, and Bayesian Posterior")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# Set user-defined parameters for skew normal data generation
XI = 10.0
OMEGA = 5.0
ALPHA = -30.0
n_samples = 5000

# Generate data from skew normal distribution
data_skew = stats.skewnorm.rvs(ALPHA, loc=XI, scale=OMEGA, size=n_samples, random_state=42)

# Calculate true mean of the skew normal distribution
true_mean = XI + OMEGA * (ALPHA / np.sqrt(1 + ALPHA**2)) * np.sqrt(2 / np.pi)
print(f"True skew normal mean: {true_mean:.4f}, parameters: xi={XI}, omega={OMEGA}, alpha={ALPHA}")

# Calculate sample statistics
sample_mean = np.mean(data_skew)
sample_var = np.var(data_skew, ddof=1)
skew = skewness(data_skew)
gamma = np.clip(skew, -0.995, 0.995)
print(f"Sample skewness: {skew:.4f}, clipped gamma: {gamma:.4f}")

# Define prior parameters
prior_mean_xi = 0.0
prior_std_xi = 100.0
prior_mean_alpha = 0.0
lambda_alpha = 50.0

# Calculate delta from gamma
delta = np.sign(gamma) * np.sqrt(
    np.pi / 2 * np.abs(gamma) ** (2 / 3) / (np.abs(gamma) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
)
print(delta)

# Calculate posterior parameters
posterior_mean_alpha = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 2)
posterior_mean_alpha = (
    n_samples / (n_samples + lambda_alpha) * posterior_mean_alpha
    + lambda_alpha / (n_samples + lambda_alpha) * prior_mean_alpha
)
posterior_omega = np.std(data_skew) / np.sqrt(1 - 2 * delta**2 / np.pi)

posterior_mean = (prior_mean_xi / prior_std_xi**2 + n * sample_mean / sample_var) / (
    1 / prior_std_xi**2 + n / sample_var
)
posterior_mean_xi = posterior_mean - posterior_omega * posterior_mean_alpha / np.sqrt(
    1 + posterior_mean_alpha**2
) * np.sqrt(2 / np.pi)

print(f"Posterior estimates: xi={posterior_mean_xi:.4f}, omega={posterior_omega:.4f}, alpha={posterior_mean_alpha:.4f}")
print(
    f"True mean: {XI + OMEGA * (ALPHA / np.sqrt(1 + ALPHA**2)) * np.sqrt(2/np.pi):.4f}, data mean: {sample_mean:.4f}, posterior mean: {posterior_mean_xi + posterior_omega * (posterior_mean_alpha / np.sqrt(1 + posterior_mean_alpha**2)) * np.sqrt(2/np.pi):.4f}"
)

# Plotting
plt.figure(figsize=(10, 6))

# Plot histogram of data
plt.hist(data_skew, bins=30, density=True, alpha=0.6, color="skyblue", label="Data histogram")

# Plot true skew normal
x = np.linspace(min(data_skew) - 5, max(data_skew) + 5, 1000)  # type: ignore
plt.plot(x, stats.skewnorm.pdf(x, ALPHA, loc=XI, scale=OMEGA), "k-", lw=2, label="True Skew Normal")

# Plot posterior skew normal
plt.plot(
    x,
    stats.skewnorm.pdf(x, posterior_mean_alpha, loc=posterior_mean_xi, scale=posterior_omega),
    "g-",
    lw=2,
    label="Posterior Skew Normal",
)

# Plot normal approximation
plt.plot(
    x,
    stats.norm.pdf(x, loc=sample_mean, scale=np.std(data_skew, ddof=1)),
    "b--",
    lw=2,
    label="Normal Approximation",
)

plt.title("Skew Normal Distribution: True vs Estimated")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
