# %%
# import numpy as np
import warnings

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.special import gamma
from scipy.stats import weibull_min
from sklearn.datasets import make_regression

warnings.filterwarnings("ignore")

# %% Simulate regression data
np.random.seed(42)
X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)

# Transform y to positive values (Weibull requires positive data)
y = np.abs(y - y.min()) + 1

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")


# %% Method of Moments estimation
def weibull_mom(data):
    """Estimate Weibull parameters using Method of Moments"""
    x_bar = np.mean(data)
    s2 = np.var(data, ddof=1)
    cv2 = s2 / (x_bar**2)

    # Solve for k using the CV equation
    def cv_equation(k):
        numerator = gamma(1 + 2 / k) - gamma(1 + 1 / k) ** 2
        denominator = gamma(1 + 1 / k) ** 2
        return numerator / denominator - cv2

    k_hat = fsolve(cv_equation, 1.5)[0]
    lambda_hat = x_bar / gamma(1 + 1 / k_hat)

    return k_hat, lambda_hat


# %% MLE estimation
def weibull_mle(data):
    """Estimate Weibull parameters using Maximum Likelihood"""
    n = len(data)

    # Solve for k using the MLE equation
    def mle_equation(k):
        numerator = np.sum(data**k * np.log(data))
        denominator = np.sum(data**k)
        return numerator / denominator - 1 / k - np.mean(np.log(data))

    k_hat = fsolve(mle_equation, 1.5)[0]
    lambda_hat = (np.mean(data**k_hat)) ** (1 / k_hat)

    return k_hat, lambda_hat


# %% Negative log-likelihood function
def weibull_nll(data, k, lambda_):
    """Calculate negative log-likelihood for Weibull distribution"""
    if k <= 0 or lambda_ <= 0:
        return np.inf
    return -np.sum(weibull_min.logpdf(data, c=k, scale=lambda_))


# %% Find best split based on Weibull likelihood
def find_best_split(X, y, feature_idx=0):
    """Find best split point that minimizes total NLL"""
    sorted_indices = np.argsort(X[:, feature_idx])
    sorted_x = X[sorted_indices, feature_idx]
    sorted_y = y[sorted_indices]

    best_nll = np.inf
    best_split = None
    best_params = None

    # Try splits at different quantiles
    for pct in np.linspace(0.2, 0.8, 20):
        split_idx = int(len(sorted_y) * pct)

        left_y = sorted_y[:split_idx]
        right_y = sorted_y[split_idx:]

        if len(left_y) < 10 or len(right_y) < 10:
            continue

        # Estimate parameters for both sides using MLE
        try:
            k_left, lambda_left = weibull_mle(left_y)
            k_right, lambda_right = weibull_mle(right_y)

            nll_left = weibull_nll(left_y, k_left, lambda_left)
            nll_right = weibull_nll(right_y, k_right, lambda_right)
            total_nll = nll_left + nll_right

            if total_nll < best_nll:
                best_nll = total_nll
                best_split = sorted_x[split_idx]
                best_params = {"left": (k_left, lambda_left), "right": (k_right, lambda_right), "split_idx": split_idx}
        except:
            continue

    return best_split, best_params, sorted_indices


# %% Execute split
feature_to_split = 0
split_value, params, sorted_indices = find_best_split(X, y, feature_to_split)

print(f"\nBest split on feature {feature_to_split}: {split_value:.2f}")
print(f"Left side - k: {params['left'][0]:.3f}, λ: {params['left'][1]:.3f}")
print(f"Right side - k: {params['right'][0]:.3f}, λ: {params['right'][1]:.3f}")

# %% Prepare data for plotting
sorted_x = X[sorted_indices, feature_to_split]
sorted_y = y[sorted_indices]
split_idx = params["split_idx"]

left_y = sorted_y[:split_idx]
right_y = sorted_y[split_idx:]

k_left, lambda_left = params["left"]
k_right, lambda_right = params["right"]

# %% Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Scatter plot with split
ax = axes[0, 0]
ax.scatter(sorted_x[:split_idx], left_y, alpha=0.5, label="Left side", c="blue")
ax.scatter(sorted_x[split_idx:], right_y, alpha=0.5, label="Right side", c="red")
ax.axvline(split_value, color="black", linestyle="--", linewidth=2, label=f"Split at {split_value:.2f}")
ax.set_xlabel(f"Feature {feature_to_split}")
ax.set_ylabel("y")
ax.set_title("Data Split")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Left side histogram with fitted Weibull
ax = axes[0, 1]
ax.hist(left_y, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")
x_range = np.linspace(left_y.min(), left_y.max(), 200)
ax.plot(
    x_range,
    weibull_min.pdf(x_range, c=k_left, scale=lambda_left),
    "b-",
    linewidth=2,
    label=f"Weibull(k={k_left:.2f}, λ={lambda_left:.2f})",
)
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title("Left Side: Data vs Fitted Weibull")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Right side histogram with fitted Weibull
ax = axes[1, 0]
ax.hist(right_y, bins=30, density=True, alpha=0.6, color="red", edgecolor="black")
x_range = np.linspace(right_y.min(), right_y.max(), 200)
ax.plot(
    x_range,
    weibull_min.pdf(x_range, c=k_right, scale=lambda_right),
    "r-",
    linewidth=2,
    label=f"Weibull(k={k_right:.2f}, λ={lambda_right:.2f})",
)
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title("Right Side: Data vs Fitted Weibull")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Q-Q plots for both sides
ax = axes[1, 1]
# Left side Q-Q
theoretical_left = weibull_min.ppf(np.linspace(0.01, 0.99, len(left_y)), c=k_left, scale=lambda_left)
ax.scatter(np.sort(theoretical_left), np.sort(left_y), alpha=0.5, label="Left side", c="blue")
# Right side Q-Q
theoretical_right = weibull_min.ppf(np.linspace(0.01, 0.99, len(right_y)), c=k_right, scale=lambda_right)
ax.scatter(np.sort(theoretical_right), np.sort(right_y), alpha=0.5, label="Right side", c="red")
# Reference line
all_theoretical = np.concatenate([theoretical_left, theoretical_right])
ax.plot(
    [all_theoretical.min(), all_theoretical.max()],
    [all_theoretical.min(), all_theoretical.max()],
    "k--",
    linewidth=1,
    label="Perfect fit",
)
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title("Q-Q Plot: Weibull Fit Quality")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Compare MoM vs MLE for left side
k_mom_left, lambda_mom_left = weibull_mom(left_y)
k_mle_left, lambda_mle_left = weibull_mle(left_y)

print(f"\nLeft side comparison:")
print(f"MoM: k={k_mom_left:.3f}, λ={lambda_mom_left:.3f}")
print(f"MLE: k={k_mle_left:.3f}, λ={lambda_mle_left:.3f}")
print(f"MoM NLL: {weibull_nll(left_y, k_mom_left, lambda_mom_left):.2f}")
print(f"MLE NLL: {weibull_nll(left_y, k_mle_left, lambda_mle_left):.2f}")


# %%
# MSE-based splitting (traditional regression tree approach)
def find_best_split_mse(X, y, feature_idx=0):
    """Find best split point that minimizes MSE"""
    sorted_indices = np.argsort(X[:, feature_idx])
    sorted_x = X[sorted_indices, feature_idx]
    sorted_y = y[sorted_indices]

    best_mse = np.inf
    best_split = None
    best_split_idx = None

    # Try splits at different quantiles
    for pct in np.linspace(0.2, 0.8, 20):
        split_idx = int(len(sorted_y) * pct)

        left_y = sorted_y[:split_idx]
        right_y = sorted_y[split_idx:]

        if len(left_y) < 10 or len(right_y) < 10:
            continue

        # Calculate MSE for both sides
        mse_left = np.var(left_y) * len(left_y)
        mse_right = np.var(right_y) * len(right_y)
        total_mse = mse_left + mse_right

        if total_mse < best_mse:
            best_mse = total_mse
            best_split = sorted_x[split_idx]
            best_split_idx = split_idx

    return best_split, best_split_idx, sorted_indices


# Execute MSE split
split_value_mse, split_idx_mse, sorted_indices_mse = find_best_split_mse(X, y, feature_to_split)

sorted_x_mse = X[sorted_indices_mse, feature_to_split]
sorted_y_mse = y[sorted_indices_mse]

left_y_mse = sorted_y_mse[:split_idx_mse]
right_y_mse = sorted_y_mse[split_idx_mse:]

# Fit Weibull to MSE-based splits for comparison
k_left_mse, lambda_left_mse = weibull_mle(left_y_mse)
k_right_mse, lambda_right_mse = weibull_mle(right_y_mse)

print(f"\nMSE-based split on feature {feature_to_split}: {split_value_mse:.2f}")
print(f"Left side - k: {k_left_mse:.3f}, λ: {lambda_left_mse:.3f}")
print(f"Right side - k: {k_right_mse:.3f}, λ: {lambda_right_mse:.3f}")

# %% Comparison plot: Weibull split vs MSE split
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Left side histogram - Weibull split
ax = axes[0, 0]
ax.hist(left_y, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")
x_range = np.linspace(left_y.min(), left_y.max(), 200)
ax.plot(
    x_range,
    weibull_min.pdf(x_range, c=k_left, scale=lambda_left),
    "b-",
    linewidth=2,
    label=f"Weibull(k={k_left:.2f}, λ={lambda_left:.2f})",
)
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title(f"Weibull Split - Left Side (n={len(left_y)})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Right side histogram - Weibull split
ax = axes[0, 1]
ax.hist(right_y, bins=30, density=True, alpha=0.6, color="red", edgecolor="black")
x_range = np.linspace(right_y.min(), right_y.max(), 200)
ax.plot(
    x_range,
    weibull_min.pdf(x_range, c=k_right, scale=lambda_right),
    "r-",
    linewidth=2,
    label=f"Weibull(k={k_right:.2f}, λ={lambda_right:.2f})",
)
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title(f"Weibull Split - Right Side (n={len(right_y)})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Left side histogram - MSE split
ax = axes[1, 0]
ax.hist(left_y_mse, bins=30, density=True, alpha=0.6, color="blue", edgecolor="black")
x_range = np.linspace(left_y_mse.min(), left_y_mse.max(), 200)
ax.plot(
    x_range,
    weibull_min.pdf(x_range, c=k_left_mse, scale=lambda_left_mse),
    "b-",
    linewidth=2,
    label=f"Weibull(k={k_left_mse:.2f}, λ={lambda_left_mse:.2f})",
)
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title(f"MSE Split - Left Side (n={len(left_y_mse)})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Right side histogram - MSE split
ax = axes[1, 1]
ax.hist(right_y_mse, bins=30, density=True, alpha=0.6, color="red", edgecolor="black")
x_range = np.linspace(right_y_mse.min(), right_y_mse.max(), 200)
ax.plot(
    x_range,
    weibull_min.pdf(x_range, c=k_right_mse, scale=lambda_right_mse),
    "r-",
    linewidth=2,
    label=f"Weibull(k={k_right_mse:.2f}, λ={lambda_right_mse:.2f})",
)
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.set_title(f"MSE Split - Right Side (n={len(right_y_mse)})")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Comparison: Weibull-based Split (top) vs MSE-based Split (bottom)", fontsize=14, y=1.00)
plt.tight_layout()
plt.show()

# Print comparison metrics
print(f"\n=== Split Comparison ===")
print(f"Weibull split value: {split_value:.2f}")
print(f"MSE split value: {split_value_mse:.2f}")
print(
    f"\nWeibull split NLL: {weibull_nll(left_y, k_left, lambda_left) + weibull_nll(right_y, k_right, lambda_right):.2f}"
)
print(
    f"MSE split NLL: {weibull_nll(left_y_mse, k_left_mse, lambda_left_mse) + weibull_nll(right_y_mse, k_right_mse, lambda_right_mse):.2f}"
)
# %%
