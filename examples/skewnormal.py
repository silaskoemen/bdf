#%%
import numpy as np
from scipy.linalg import inv

import scipy.stats as stats
from scipy.special import ndtr  # Standard normal CDF
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate some skewed data from gamma distribution
shape, scale = 3.0, 5.0
n_samples = 10
data = np.random.gamma(shape, scale, size=n_samples)
data = np.array([-1, -1, 0, 1])

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
posterior_mean = (
    (prior_mean_xi / prior_std_xi**2 + n * sample_mean / sample_var) /
    (1 / prior_std_xi**2 + n / sample_var)
)


z = (data - sample_mean) / np.sqrt(sample_var)

phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

# Numerically stable approximation of Phi(z)
from scipy.stats import norm
Phi = norm.cdf(z)

# Avoid division by zero or very small values
Phi_safe = np.clip(Phi, 1e-10, 1 - 1e-10)

# Score and Fisher Information
score = np.sum((z * phi) / Phi_safe)
Fisher_info = np.sum((z**2 * phi**2) / (Phi_safe**2))

# Posterior for alpha, empirical Bayes approach, where lambda reflects stength of prior belief
lambda_alpha = 30
posterior_mean_alpha = n/(n + lambda_alpha) * alpha_mle + lambda_alpha/(n + lambda_alpha) * prior_mean_alpha  # type: ignore

posterior_mean_xi = posterior_mean - omega_mle * (posterior_mean_alpha / np.sqrt(1 + posterior_mean_alpha**2)) * np.sqrt(2/np.pi)
# Use omega_mle for scale parameter as it's not being updated
posterior_omega = omega_mle

print(f"Posterior estimates: xi={posterior_mean_xi:.4f}, omega={posterior_omega:.4f}, alpha={posterior_mean_alpha:.4f}")
print(f"Mean MLE estimates: {xi_mle + omega_mle * (alpha_mle / np.sqrt(1 + alpha_mle**2)) * np.sqrt(2/np.pi):.4f}")  # type: ignore
print(f"Mean Posterior estimates: {posterior_mean_xi + posterior_omega * (posterior_mean_alpha / np.sqrt(1 + posterior_mean_alpha**2)) * np.sqrt(2/np.pi):.4f}")  # type: ignore
print(f"Mean data: {np.mean(data):.4f}")
# Plotting
plt.figure(figsize=(10, 6))

# Plot histogram of data
plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label='Data histogram')

# Plot MLE skew normal
x = np.linspace(min(data)-1, max(data)+1, 1000)
plt.plot(x, stats.skewnorm.pdf(x, alpha_mle, loc=xi_mle, scale=omega_mle), 
         'r-', lw=2, label='MLE Skew Normal')

# Plot posterior skew normal
plt.plot(x, stats.skewnorm.pdf(x, posterior_mean_alpha, loc=posterior_mean_xi, scale=posterior_omega), 
         'g-', lw=2, label='Posterior Skew Normal')
plt.plot(x, stats.norm.pdf(x, loc=data.mean(), scale=data.std()),
         'b--', lw=2, label='Normal Approximation')
plt.title('Skew Normal Distribution: Data, MLE, and Bayesian Posterior')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
