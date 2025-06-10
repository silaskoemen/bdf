# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def sample_normal(n, mean, std):
    """Sample n points from a normal distribution with specified mean and std."""
    return np.random.normal(mean, std, size=n)


def sample_and_print_distributions(data, prior_alpha, prior_beta, true_std=None, mean=True):
    """
    Plot the data likelihood (normal distribution) and posterior distribution for the variance.
    Also plot posterior predictive likelihood using either the mean or mode of the posterior variance.

    Args:
        data: The observed data
        prior_alpha: Shape parameter for the inverse gamma prior
        prior_beta: Scale parameter for the inverse gamma prior
        true_std: True standard deviation (if known)
        mean: Whether to use the mean (True) or mode (False) of the posterior as variance estimate
    """
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data)

    # Calculate posterior parameters for inverse gamma
    post_alpha = prior_alpha + n / 2
    post_beta = prior_beta + (n * sample_var) / 2

    # Calculate posterior estimate for variance (mean or mode)
    if mean:
        # Mean of inverse gamma: beta/(alpha-1) for alpha > 1
        posterior_var_estimate = post_beta / (post_alpha - 1) if post_alpha > 1 else sample_var
        estimate_type = "mean"
    else:
        # Mode of inverse gamma: beta/(alpha+1)
        posterior_var_estimate = post_beta / (post_alpha + 1)
        estimate_type = "mode"

    posterior_std = np.sqrt(posterior_var_estimate)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Data and likelihood
    x = np.linspace(min(data) - 3 * np.std(data), max(data) + 3 * np.std(data), 1000)
    likelihood = stats.norm.pdf(x, sample_mean, np.sqrt(sample_var))
    posterior_likelihood = stats.norm.pdf(x, sample_mean, posterior_std)

    ax1.hist(data, bins=20, density=True, alpha=0.5, label="Data histogram")
    ax1.plot(x, likelihood, "r-", label=f"Sample likelihood (μ={sample_mean:.2f}, σ²={sample_var:.2f})")
    ax1.plot(
        x,
        posterior_likelihood,
        "b-",
        label=f"Posterior likelihood (μ={sample_mean:.2f}, σ²={posterior_var_estimate:.2f})",
    )
    if true_std is not None:
        true_likelihood = stats.norm.pdf(x, sample_mean, true_std)
        ax1.plot(x, true_likelihood, "g--", label=f"True distribution (σ={true_std:.2f})")
    ax1.set_title("Data and Normal Likelihood")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Plot 2: Posterior distribution for variance
    variance_range = np.linspace(0.1, 3 * sample_var, 1000)
    posterior_density = stats.invgamma.pdf(variance_range, post_alpha, scale=post_beta)
    prior_density = stats.invgamma.pdf(variance_range, prior_alpha, scale=prior_beta)

    ax2.plot(variance_range, posterior_density, "b-", label=f"Posterior (α={post_alpha:.2f}, β={post_beta:.2f})")
    ax2.plot(variance_range, prior_density, "k--", label=f"Prior (α={prior_alpha}, β={prior_beta})")
    if true_std is not None:
        ax2.axvline(true_std**2, color="g", linestyle="--", label=f"True variance ({true_std**2:.2f})")
    ax2.axvline(
        posterior_var_estimate,
        color="b",
        linestyle=":",
        label=f"Posterior {estimate_type} ({posterior_var_estimate:.2f})",
    )
    ax2.set_title("Prior and Posterior for Variance")
    ax2.set_xlabel("Variance")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# %%
mean = 0.5  # Mean of the normal distribution
std_dev = 0.1  # Standard deviation of the normal distribution
n_samples = 40  # Number of samples
prior_alpha = 2  # Shape parameter for inverse gamma prior
prior_beta = 1e-7  # Scale parameter for inverse gamma prior

# Generate samples
data = sample_normal(n_samples, mean, std_dev)

# Plot results
sample_and_print_distributions(data, prior_alpha, prior_beta, true_std=std_dev, mean=False)

# Print summary
print(f"Sample mean: {np.mean(data):.4f}")
print(f"Sample variance: {np.var(data):.4f}")
print(f"Prior mean for variance: {prior_beta/(prior_alpha-1) if prior_alpha > 1 else 'Undefined'}")

# Posterior parameters
post_alpha = prior_alpha + n_samples / 2
post_beta = prior_beta + (n_samples * np.var(data)) / 2
post_mean = post_beta / (post_alpha - 1) if post_alpha > 1 else "Undefined"

print(f"Posterior mean for variance: {post_mean}")
# %%
