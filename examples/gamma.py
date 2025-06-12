# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func


def bayesian_gamma_inference(data, alpha, prior_mean_shape, prior_mean_rate, plot=True):
    """
    Bayesian inference for Gamma distribution with known alpha and prior on mean.

    Parameters:
    - data: observed data (array-like)
    - alpha: known shape parameter of Gamma distribution
    - prior_mean_shape: shape parameter of prior Gamma distribution on mean
    - prior_mean_rate: rate parameter of prior Gamma distribution on mean

    Returns:
    - posterior_shape: shape parameter of posterior Gamma distribution for beta
    - posterior_rate: rate parameter of posterior Gamma distribution for beta
    """

    # Data statistics
    n = len(data)
    sum_data = np.sum(data)

    # Prior on mean follows Gamma(prior_mean_shape, prior_mean_rate)
    # This implies beta ~ InverseGamma, which we convert to Gamma for beta

    # Convert prior on mean to prior on beta
    # If mean ~ Gamma(a, b), then beta = alpha/mean ~ alpha * InverseGamma(a, b)
    # We moment-match this to a Gamma distribution for beta

    prior_mean_mean = prior_mean_shape / prior_mean_rate
    prior_mean_var = prior_mean_shape / (prior_mean_rate**2)

    # Mean and variance of beta = alpha/mean
    prior_beta_mean = alpha / prior_mean_mean
    prior_beta_var = (alpha**2 * prior_mean_var) / (prior_mean_mean**4)

    # Moment matching to Gamma distribution for beta
    prior_beta_rate = prior_beta_mean / prior_beta_var
    prior_beta_shape = prior_beta_mean * prior_beta_rate

    # Posterior parameters for beta
    # Gamma likelihood with known alpha gives conjugate update
    posterior_shape = prior_beta_shape + n * alpha
    posterior_rate = prior_beta_rate + sum_data

    # Posterior mean of beta
    posterior_beta_mean = posterior_shape / posterior_rate

    if plot:
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Prior and posterior distributions of beta
        beta_range = np.linspace(0.1, posterior_beta_mean * 3, 1000)

        prior_pdf = stats.gamma.pdf(beta_range, prior_beta_shape, scale=1 / prior_beta_rate)
        posterior_pdf = stats.gamma.pdf(beta_range, posterior_shape, scale=1 / posterior_rate)

        ax1.plot(beta_range, prior_pdf, "b--", label="Prior", linewidth=2)
        ax1.plot(beta_range, posterior_pdf, "r-", label="Posterior", linewidth=2)
        ax1.axvline(
            posterior_beta_mean, color="red", linestyle=":", label=f"Posterior Mean = {posterior_beta_mean:.3f}"
        )
        ax1.set_xlabel("β")
        ax1.set_ylabel("Density")
        ax1.set_title("Prior and Posterior Distributions of β")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Data histogram with fitted distributions
        ax2.hist(data, bins=20, density=True, alpha=0.7, color="lightblue", label="Observed Data")

        # Prior predictive (using prior mean of beta)
        prior_beta_mean_val = prior_beta_shape / prior_beta_rate
        x_range = np.linspace(0, max(data) * 1.2, 1000)
        prior_pred_pdf = stats.gamma.pdf(x_range, alpha, scale=1 / prior_beta_mean_val)

        # Posterior predictive (using posterior mean of beta)
        posterior_pred_pdf = stats.gamma.pdf(x_range, alpha, scale=1 / posterior_beta_mean)

        ax2.plot(x_range, prior_pred_pdf, "b--", label="Prior Predictive", linewidth=2)
        ax2.plot(x_range, posterior_pred_pdf, "r-", label="Posterior Predictive", linewidth=2)
        ax2.set_xlabel("Data")
        ax2.set_ylabel("Density")
        ax2.set_title("Data and Predictive Distributions")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return posterior_shape, posterior_rate, posterior_beta_mean


# %%
# Generate some example data
np.random.seed(42)
true_alpha = 2.0
true_beta = 1.5
n_samples = 500
data = np.random.gamma(true_alpha, scale=1 / true_beta, size=n_samples)

# Specify known alpha and prior on mean
alpha_known = 2.0  # Assume we know the true alpha
prior_mean_shape = 3.0  # Shape parameter of prior on mean
prior_mean_rate = 3.0  # Rate parameter of prior on mean

print(f"True parameters: α = {true_alpha}, β = {true_beta}")
print(f"Known α = {alpha_known}")
print(f"Prior on mean: Gamma({prior_mean_shape}, {prior_mean_rate})")
print(f"Data: n = {len(data)}, sample mean = {np.mean(data):.3f}")
print()

# Perform Bayesian inference
post_shape, post_rate, post_mean = bayesian_gamma_inference(data, alpha_known, prior_mean_shape, prior_mean_rate)

print(f"Posterior distribution of β: Gamma({post_shape:.3f}, {post_rate:.3f})")
print(f"Posterior mean of β: {post_mean:.3f}")
print(f"True β: {true_beta:.3f}")
# %%
