#%%
import numpy as np
from scipy.linalg import inv

import scipy.stats as stats
from scipy.special import ndtr  # Standard normal CDF
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate some skewed data from gamma distribution
shape, scale = 2.0, 2.0
n_samples = 1000
data = np.random.gamma(shape, scale, size=n_samples)

# Scale the data to be more suitable for skew normal modeling
data = (data - np.mean(data)) / np.std(data)

# Define prior for alpha (shape parameter of skew normal)
alpha_prior_mean = 0.0
alpha_prior_var = 10.0  # weak prior

# Find MLE for skew normal parameters to get initial estimates
# For skew normal, we need location (xi), scale (omega), and shape (alpha)
alpha_mle, xi_mle, omega_mle = stats.skewnorm.fit(data)
print(f"MLE estimates: xi={xi_mle:.4f}, omega={omega_mle:.4f}, alpha={alpha_mle:.4f}")

#%%
# Compute posterior for alpha using the formula
# α* = α0 + Δᵀ Σ⁻¹ φₙ(0; 0, Σ) / Φₙ(0; 0, Σ)
# where Δ = ψ0 z, Σ = I + ΔΔᵀ

# Standardize the data using MLE estimates of location and scale
z = data

# Compute ψ0 (this is essentially the prior precision times the prior mean)
psi_0 = np.sqrt(alpha_prior_var)

# Compute Delta = ψ0 * z
Delta = psi_0 * z

# Compute Sigma = I + Delta * Delta^T
# Since Delta is a vector, Delta * Delta^T is a matrix
Sigma = np.eye(n_samples) + np.outer(Delta, Delta)

# Compute the multivariate normal PDF at 0
# φₙ(0; 0, Σ)
def multivariate_normal_pdf(x, mean, cov):
    n = len(x)
    det = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
    return (1 / np.sqrt((2 * np.pi) ** n * det)) * np.exp(exponent)

# Compute the multivariate normal CDF at 0
# Φₙ(0; 0, Σ)
# Note: For high dimensions, this is computationally expensive
# For simplicity, we'll use a Monte Carlo approximation
def multivariate_normal_cdf_mc(x, mean, cov, n_samples=10000):
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    return np.mean(np.all(samples <= x, axis=1))


# For univariate normal, the ratio φ(0)/Φ(0) has a known form
# For the simplified case, we'll use this approximation
pdf_cdf_ratio = stats.norm.pdf(0) / stats.norm.cdf(0)

# Calculate posterior alpha
alpha_posterior = alpha_prior_mean + Delta.T * pdf_cdf_ratio / Sigma

print(f"Prior alpha: {alpha_prior_mean}")
print(f"Posterior alpha: {alpha_posterior:.4f}")
print(f"MLE alpha: {alpha_mle:.4f}")

# Plot the data and fitted distributions
x = np.linspace(min(data), max(data), 1000)
prior_pdf = stats.skewnorm.pdf(x, alpha_prior_mean, loc=0, scale=1)
posterior_pdf = stats.skewnorm.pdf(x, alpha_posterior, loc=xi_mle, scale=omega_mle)
mle_pdf = stats.skewnorm.pdf(x, alpha_mle, loc=xi_mle, scale=omega_mle)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.6, label='Data')
plt.plot(x, prior_pdf, 'r-', label=f'Prior (α={alpha_prior_mean})')
plt.plot(x, posterior_pdf, 'g-', label=f'Posterior (α={alpha_posterior:.4f})')
plt.plot(x, mle_pdf, 'b--', label=f'MLE (α={alpha_mle:.4f})')
plt.legend()
plt.title('Skew Normal Distribution Fitting with Prior and Posterior')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.show()
# %%
# Essentially want to get posterior of xi to guide towards 0/mean and alpha to guide towards
# no skew. Still, want to see whether large amounts of data will correctly get to observed
# distribution and smaller amounts shift towards independent priors