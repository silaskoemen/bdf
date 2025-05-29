# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np
from scipy import stats

# Define parameters for 5 gamma distributions that better approximate a normal distribution
# Using smaller shape (more spread) with smaller scale (shifted left) and
# larger shape (less spread) with larger scale (shifted right) to balance skewness
shapes = [2.5, 5, 9, 12, 15]  # Varying shapes from small (skewed) to large (more symmetric)
scales = [5.5, 3, 2, 1.5, 1.2]  # Inversely proportional scales to balance the means
n_samples = 2000  # samples per distribution

np.random.seed(42)  # For reproducibility
# Generate samples from each gamma distribution
samples = []
for shape, scale in zip(shapes, scales):
    samples.append(np.random.gamma(shape, scale, n_samples))

# Plot histogram of all samples combined
plt.figure(figsize=(10, 6))
plt.hist(np.concatenate(samples), bins=50, alpha=0.5, color="gray", label="Combined")

# Plot histogram for each gamma distribution
colors = ["blue", "green", "red", "purple", "orange"]
for i, (s, shape, scale) in enumerate(zip(samples, shapes, scales)):
    plt.hist(s, bins=50, alpha=0.5, color=colors[i], label=f"Gamma(k={shape}, θ={scale})")

# Plot normal distribution that approximates the combined data
combined = np.concatenate(samples)
mu, std = combined.mean(), combined.std()
x = np.linspace(min(combined), max(combined), 100)
plt.plot(
    x,
    stats.norm.pdf(x, mu, std) * len(combined) * (max(combined) - min(combined)) / 50,
    "k--",
    linewidth=2,
    label=f"Normal(μ={mu:.2f}, σ={std:.2f})",
)

plt.title("Mixture of 5 Gamma Distributions Approximating a Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/figures/locally_gamma_normal.png", dpi=300)
plt.show()
# %%
# Test normality of the combined samples

# Jarque-Bera test for normality
jb_stat, jb_p = stats.jarque_bera(combined)
print(f"Jarque-Bera test:")
print(f"Statistic: {jb_stat:.4f}")
print(f"p-value: {jb_p:.4g}")
print(f"Conclusion: {'Data appears normal' if jb_p > 0.05 else 'Data does not appear normal'}\n")  # type: ignore | views jb_p as "_T_co@tuple"

# Kolmogorov-Smirnov test for normality
ks_stat, ks_p = stats.kstest(combined, stats.norm.cdf, args=(mu, std))
print(f"Kolmogorov-Smirnov test:")
print(f"Statistic: {ks_stat:.4f}")
print(f"p-value: {ks_p:.4g}")
print(f"Conclusion: {'Data appears normal' if ks_p > 0.05 else 'Data does not appear normal'}")

# %%
