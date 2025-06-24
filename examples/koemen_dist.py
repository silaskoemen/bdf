# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def recentered_skew_normal(x, mu, sigma, alpha):
    """
    Recentered skew-normal PDF with fixed mean mu, scale sigma, skewness alpha.
    """
    delta = alpha / np.sqrt(1 + alpha**2)
    shift = sigma * delta * np.sqrt(2 / np.pi)
    z = (x - mu + shift) / sigma
    return 2 / sigma * norm.pdf(z) * norm.cdf(alpha * z)


# Parameters
mu = 1
sigma = 2
alphas = [-5, 0, 5]
colors = ["blue", "black", "red"]
labels = [r"$\alpha = -5$", r"$\alpha = 0$", r"$\alpha = 5$"]

# X-axis range
x = np.linspace(-3, 5, 500)

# Plot
plt.figure(figsize=(8, 5))

for alpha, color, label in zip(alphas, colors, labels):
    y = recentered_skew_normal(x, mu, sigma, alpha)
    print(f"Alpha: {alpha}, Mean: {mu}, Sigma: {sigma}, sample mean: {np.sum(y*x)/np.sum(y)}")
    plt.plot(x, y, color=color, label=label, linewidth=2)

# Annotations
plt.title("Recentered Skew-Normal Distributions (mean = 1)", fontsize=14)
plt.xlabel("x")
plt.ylabel("Density")
plt.axvline(mu, color="gray", linestyle="--", label=r"Mean ($\mu = 1$)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# %%
########## Mean-preserving skewed density with Hermite polynomial basis ##########
import numpy as np
from scipy.stats import norm, skewnorm

mu = 1.0
sigma = 2.0


def hermite3(t):
    return (t**3 - 3 * t) / np.sqrt(6)


def f(x, mu, sigma, alpha):
    t = (x - mu) / sigma
    base = norm.pdf(t)
    val = base * (1 + alpha * hermite3(t)) / sigma
    return np.clip(val, 0, None)  # clip negative values to zero to ensure positivity


x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

alphas = [-0.25, -0.1, 0, 0.1, 0.2]  # keep alpha small to ensure positivity

plt.figure(figsize=(10, 6))
for alpha in alphas:
    y = f(x, mu, sigma, alpha)
    plt.plot(x, y, label=f"alpha={alpha:.2f}")
# Plot skewnormal on top for comparison
plt.plot(x, skewnorm(5, mu, sigma).pdf(x), "k--", label="SkewNormal PDF", linewidth=1.5)
plt.title("Mean-preserving skewed density with Hermite polynomial basis")
plt.xlabel("x")
plt.ylabel("f(x; mu, alpha)")
plt.legend()
plt.grid()
plt.show()

# Check mean numerically:
for alpha in alphas:
    y = f(x, mu, sigma, alpha)
    mean = np.trapezoid(x * y, x)
    integral = np.trapezoid(y, x)
    print(f"alpha={alpha:.3f}, numerical mean={mean:.5f}, integral={integral:.5f}")

# %%
# Sample from a skewnormal and then fit the mean-preserving skewed density
from scipy.stats import skewnorm


def sample_skewnormal(alpha, xi, omega, n_samples=1000):
    """
    Sample from a skew-normal distribution and return the samples.
    """
    return skewnorm.rvs(a=alpha, loc=xi, scale=omega, size=n_samples)


# Generate samples
n_samples = 1000
sn_alpha, sn_xi, sn_omega = 2, 3, 5
samples = sample_skewnormal(sn_alpha, sn_xi, sn_omega, n_samples)


# Fit the mean-preserving skewed density
def fit_skewed_density(samples):
    """
    Fit the mean-preserving skewed density to the samples.
    """
    # Calculate the MLE of mu, alpha and sigma given the data
    # Define joint optimizer over parameters
    from scipy.optimize import minimize

    def objective(params):
        mu, alpha, sigma = params
        if sigma <= 0:
            return np.inf  # penalize non-positive sigma
        density = f(samples, mu, sigma, alpha)
        return -np.sum(np.log(density + 1e-10))  # add small value to avoid log(0)

    # Initial guess for mu, alpha, sigma
    initial_guess = [np.mean(samples), 0, np.std(samples)]
    # Minimize the negative log-likelihood
    result = minimize(objective, initial_guess, bounds=[(None, None), (-10, 10), (1e-5, None)])
    if result.success:
        mu_fit, alpha_fit, sigma_fit = result.x
        return mu_fit, alpha_fit, sigma_fit
    else:
        raise ValueError("Optimization failed to converge")


mu_fit, alpha_fit, sigma_fit = fit_skewed_density(samples)
hermite_3_t = hermite3((samples - samples.mean()) / samples.std())
mu_mle, alpha_mle, sigma_mle = np.mean(samples), np.sum(hermite_3_t) / np.sum(hermite_3_t**2), np.std(samples)
print(f"Fitted parameters: mu={mu_fit:.3f}, alpha={alpha_fit:.3f}, sigma={sigma_fit:.3f}")
print(f"MLE parameters: mu={mu_mle:.3f}, alpha={alpha_mle:.3f}, sigma={sigma_mle:.3f}")
print(
    f"True data mean: {np.mean(samples):.3f}, std: {np.std(samples):.3f}, skew: {np.mean(((samples - np.mean(samples)) / np.std(samples))**3):.3f}"
)
# Plot the fitted density
x_fit = np.linspace(sn_xi - 4 * sn_omega, sn_xi + 4 * sn_omega, 1000)
y_fit = f(x_fit, mu_fit, sigma_fit, alpha_fit)
y_mle = f(x_fit, mu_mle, sigma_mle, alpha_mle)
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.5, label="Sampled Data", color="lightgray")
plt.plot(x_fit, skewnorm(sn_alpha, sn_xi, sn_omega).pdf(x_fit), "k--", label="SkewNormal PDF", linewidth=1.5)
plt.plot(x_fit, y_fit, label="Fitted Mean-Preserving Skewed Density", color="blue", linewidth=2)
plt.plot(x_fit, y_mle, label="MLE Mean-Preserving Skewed Density", color="orange", linestyle="--", linewidth=2)
plt.title("Fitted Mean-Preserving Skewed Density")
plt.xlabel("x")
plt.ylabel("Density")
plt.axvline(mu_fit, color="red", linestyle="--", label=f"Mean (mu) = {mu_fit:.3f}")
plt.legend()
plt.grid()
plt.show()

"""

########## Useful for possibly bimodal distributions ##########
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 0.0
sigma = 1.0
beta = 2.0

def T_alpha(x, alpha):
    return x + alpha * np.tanh(beta * (x - mu))

def dT_alpha(x, alpha):
    return 1 + alpha * beta / np.cosh(beta * (x - mu))**2

def f(x, mu, sigma, alpha):
    Tx = T_alpha(x, alpha)
    density = norm.pdf((Tx - mu)/sigma) / sigma
    return density * np.abs(dT_alpha(x, alpha))

# Plot
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
alphas = [-0.9/beta, -0.5/beta, 0, 0.5/beta, .9/beta]

plt.figure(figsize=(10,6))
for alpha in alphas:
    y = f(x, mu, sigma, alpha)
    plt.plot(x, y, label=f'alpha={alpha:.2f}')
plt.title('Skewed pdf with mean fixed by construction')
plt.legend()
plt.show()

# Check means numerically (should be ~mu)
for alpha in alphas:
    y = f(x, mu, sigma, alpha)
    mean = np.trapz(x * y, x)
    print(f"alpha={alpha:.3f}, mean={mean:.5f}")

"""
# %%
