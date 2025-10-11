# %%
# Investigate how scipy parameterizations work for the Poisson distribution with Gamma prior on rate lambda
from scipy.stats import gamma, nbinom, poisson

LAMBDA = 0.5  # rate, 1/mean for poi params
ALPHA, BETA = 100, 20  # alpha and beta of gamma distribution (corresponds to shape and rate), mean is alpha/beta
N_SAMPLES = 1000

data = poisson(1 / LAMBDA).rvs(N_SAMPLES)

# %%
ALPHA_POST = ALPHA + data.sum()
BETA_POST = BETA + N_SAMPLES

print(
    f"True mean: {1/LAMBDA}, data mean: {data.mean()}, prior mean: {ALPHA/BETA}, posterior mean: {ALPHA_POST/BETA_POST}"
)


# %%
import matplotlib.pyplot as plt
import numpy as np

# Plot samples, gamma dist (prior on mean), horiz line for data mean, posterior dist
x_vals = np.linspace(data.min() - 1, data.max() + 1, 1000)
plt.hist(data, density=True)
plt.axvline(1 / LAMBDA, label="True mean", linestyle="--", color="crimson")
plt.axvline(ALPHA_POST / BETA_POST, label="Posterior mean", linestyle="--", color="dodgerblue")
plt.axvline(ALPHA / BETA, label="Prior mean", linestyle="--", color="black")
plt.plot(x_vals, gamma(a=ALPHA, scale=1 / BETA).pdf(x_vals), label="prior dist on Poi rate")
plt.plot(x_vals, gamma(a=ALPHA_POST, scale=1 / BETA_POST).pdf(x_vals), label="posterior dist on Poi rate")
plt.grid(alpha=0.2)
plt.legend()
plt.show()
# %%
MEAN, VARIANCE = 2, 1 / 40
ALPHA, BETA = MEAN**2 / VARIANCE, MEAN / VARIANCE


ALPHA_POST = ALPHA + data.sum()
BETA_POST = BETA + N_SAMPLES

print(
    f"True mean: {1/LAMBDA}, data mean: {data.mean()}, prior mean: {ALPHA/BETA}, posterior mean: {ALPHA_POST/BETA_POST}"
)

# %%
# --- Posterior predictive NB ---
nb_data = nbinom(ALPHA_POST, BETA_POST / (1 + BETA_POST)).rvs(N_SAMPLES)
bins = range(0, data.max() + 1)
plt.hist(data, density=True, label="true", bins=bins)
plt.hist(nb_data, density=True, alpha=0.4, label="PP", bins=bins)
plt.grid(alpha=0.2)
plt.legend()
plt.show()
# %%
