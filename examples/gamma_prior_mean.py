# %%
# --- Gamma Distribution with Adaptive Shape and Gamma Prior on Rate ---
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

# Set random seed for reproducibility
np.random.seed(42)

# %%
# =============================================================================
# STEP 1: Generate data from a known Gamma distribution
# =============================================================================
# True parameters of the data-generating process
TRUE_ALPHA = 10.0  # Shape parameter
TRUE_BETA = 2.0  # Rate parameter (1/scale)

# True mean and variance
true_mean = TRUE_ALPHA / TRUE_BETA
true_variance = TRUE_ALPHA / (TRUE_BETA**2)

print("=" * 60)
print("TRUE DATA-GENERATING DISTRIBUTION")
print("=" * 60)
print(f"True α (shape): {TRUE_ALPHA}")
print(f"True β (rate):  {TRUE_BETA}")
print(f"True mean:      {true_mean:.3f}")
print(f"True variance:  {true_variance:.3f}")
print()

# Generate data
n_samples = 1000
data = gamma.rvs(a=TRUE_ALPHA, scale=1 / TRUE_BETA, size=n_samples)

# Sample statistics
sample_mean = np.mean(data)
sample_var = np.var(data, ddof=1)

print("OBSERVED SAMPLE STATISTICS")
print("=" * 60)
print(f"Sample size:     {n_samples}")
print(f"Sample mean:     {sample_mean:.3f}")
print(f"Sample variance: {sample_var:.3f}")
print()

# %%
# =============================================================================
# STEP 2: Estimate α from data using Method of Moments
# =============================================================================
# α_est = (mean²) / variance
alpha_estimated = (sample_mean**2) / sample_var

print("ESTIMATED SHAPE PARAMETER (Method of Moments)")
print("=" * 60)
print(f"Estimated α: {alpha_estimated:.3f}")
print(f"True α:      {TRUE_ALPHA:.3f}")
print(f"Difference:  {abs(alpha_estimated - TRUE_ALPHA):.3f}")
print()

# %%
# =============================================================================
# STEP 3: Specify prior on β (rate parameter)
# =============================================================================
# Prior: β ~ Gamma(a0, b0)
# You can experiment with different priors:

# Weak prior (high uncertainty)
a0_beta = 2.0
b0_beta = 1.0

# Alternative: Informative prior centered near the truth
# a0_beta = 4.0
# b0_beta = 2.0  # Prior mean = 4/2 = 2.0

prior_mean_beta = a0_beta / b0_beta
prior_var_beta = a0_beta / (b0_beta**2)

print("PRIOR ON β (RATE PARAMETER)")
print("=" * 60)
print(f"Prior: β ~ Gamma(a0={a0_beta}, b0={b0_beta})")
print(f"Prior mean of β: {prior_mean_beta:.3f}")
print(f"Prior std of β:  {np.sqrt(prior_var_beta):.3f}")
print()

# %%
# =============================================================================
# STEP 4: Calculate posterior using conjugate update
# =============================================================================
# Posterior: β | data ~ Gamma(a0 + n*α, b0 + Σx)
data_sum = np.sum(data)
a_posterior = a0_beta + n_samples * alpha_estimated
b_posterior = b0_beta + data_sum

posterior_mean_beta = a_posterior / b_posterior
posterior_var_beta = a_posterior / (b_posterior**2)

print("POSTERIOR OF β (RATE PARAMETER)")
print("=" * 60)
print(f"Posterior: β | data ~ Gamma(a={a_posterior:.1f}, b={b_posterior:.1f})")
print(f"Posterior mean of β: {posterior_mean_beta:.3f}")
print(f"Posterior std of β:  {np.sqrt(posterior_var_beta):.3f}")
print(f"True β:              {TRUE_BETA:.3f}")
print()

# %%
# =============================================================================
# STEP 5: Visualize the data with fitted distributions
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Histogram of data with true distribution ---
ax = axes[0, 0]
ax.hist(data, bins=50, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Observed Data")

x_range = np.linspace(0, max(data), 1000)
true_pdf = gamma.pdf(x_range, a=TRUE_ALPHA, scale=1 / TRUE_BETA)
ax.plot(x_range, true_pdf, "r-", linewidth=2, label=f"True: Γ(α={TRUE_ALPHA}, β={TRUE_BETA})")

ax.set_xlabel("Value", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Data vs True Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# --- Plot 2: Histogram with posterior predictive distribution ---
ax = axes[0, 1]
ax.hist(data, bins=50, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Observed Data")

# Posterior predictive using posterior mean of β
posterior_pdf = gamma.pdf(x_range, a=alpha_estimated, scale=1 / posterior_mean_beta)
ax.plot(
    x_range,
    posterior_pdf,
    "g-",
    linewidth=2,
    label=f"Posterior: Γ(α={alpha_estimated:.2f}, β={posterior_mean_beta:.2f})",
)

# Also plot the true distribution for comparison
ax.plot(x_range, true_pdf, "r--", linewidth=1.5, alpha=0.7, label=f"True: Γ(α={TRUE_ALPHA}, β={TRUE_BETA})")

ax.set_xlabel("Value", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Data vs Posterior Predictive Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# --- Plot 3: Prior and Posterior distribution of β ---
ax = axes[1, 0]

beta_range = np.linspace(0, 5, 1000)
prior_pdf_beta = gamma.pdf(beta_range, a=a0_beta, scale=1 / b0_beta)
posterior_pdf_beta = gamma.pdf(beta_range, a=a_posterior, scale=1 / b_posterior)

ax.plot(beta_range, prior_pdf_beta, "b-", linewidth=2, label=f"Prior: Γ({a0_beta}, {b0_beta})")
ax.plot(beta_range, posterior_pdf_beta, "g-", linewidth=2, label=f"Posterior: Γ({a_posterior:.1f}, {b_posterior:.1f})")
ax.axvline(TRUE_BETA, color="red", linestyle="--", linewidth=2, label=f"True β = {TRUE_BETA}")
ax.axvline(
    prior_mean_beta, color="blue", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Prior mean = {prior_mean_beta:.2f}"
)
ax.axvline(
    posterior_mean_beta,
    color="green",
    linestyle=":",
    linewidth=1.5,
    alpha=0.7,
    label=f"Posterior mean = {posterior_mean_beta:.2f}",
)

ax.set_xlabel("β (Rate Parameter)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Prior vs Posterior Distribution of β", fontsize=14, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- Plot 4: Posterior predictive samples ---
ax = axes[1, 1]

# Draw samples from posterior predictive distribution
n_pred_samples = 1000
# 1. Sample β from its posterior
beta_samples = gamma.rvs(a=a_posterior, scale=1 / b_posterior, size=n_pred_samples)
# 2. For each β, sample from Gamma(α, β)
predictive_samples = gamma.rvs(a=alpha_estimated, scale=1 / beta_samples)

ax.hist(data, bins=50, density=True, alpha=0.4, color="skyblue", edgecolor="black", label="Observed Data")
ax.hist(
    predictive_samples,
    bins=50,
    density=True,
    alpha=0.4,
    color="lightgreen",
    edgecolor="black",
    label="Posterior Predictive Samples",
)

ax.set_xlabel("Value", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Observed Data vs Posterior Predictive Samples", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# STEP 6: Summary statistics
# =============================================================================
print("SUMMARY")
print("=" * 60)
print("Distribution Means:")
print(f"  True mean:                {true_mean:.3f}")
print(f"  Sample mean:              {sample_mean:.3f}")
print(f"  Posterior predictive mean: {alpha_estimated / posterior_mean_beta:.3f}")
print()
print("Distribution Variances:")
print(f"  True variance:                {true_variance:.3f}")
print(f"  Sample variance:              {sample_var:.3f}")
print(f"  Posterior predictive variance: {alpha_estimated / (posterior_mean_beta**2):.3f}")
print()
print("Rate Parameter β:")
print(f"  True β:           {TRUE_BETA:.3f}")
print(f"  Prior mean β:     {prior_mean_beta:.3f}")
print(f"  Posterior mean β: {posterior_mean_beta:.3f}")
print(
    f"  Posterior covers true: {a_posterior/b_posterior - 2*np.sqrt(posterior_var_beta) < TRUE_BETA < a_posterior/b_posterior + 2*np.sqrt(posterior_var_beta)}"
)

# %%
