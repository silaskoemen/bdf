# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KernelDensity
from tqdm import tqdm


def friedman_function(x):
    """
    The Friedman function.
    y(x) = 10 * sin(pi * x1 * x2) + 20 * (x3 - 0.5)^2 + 10 * x4 + 5 * x5 + e
    where x_i are drawn from U[0, 1] and e is N(0, 1).
    """
    return 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 10 * x[:, 3] + 5 * x[:, 4]


def generate_data(n_samples=2000, n_features=10, noise_std=1.0):
    """Generates data based on the Friedman function."""
    X = np.random.rand(n_samples, n_features)
    y = friedman_function(X) + np.random.normal(0, noise_std, n_samples)
    return X, y.reshape(-1, 1)


def generate_skew_normal_data(n_samples=2000):
    """
    Generates data where the parameters of a skew-normal distribution
    (skew, location, scale) are functions of 4 features, plus noise.

    The features X are drawn from U[0, 1]. The target y is generated from
    a skew-normal distribution whose parameters depend on the features for
    that sample. This creates a complex, heteroscedastic relationship.
    """
    n_features = 4
    X = np.random.rand(n_samples, n_features)

    # Define skew, location, and scale as functions of features
    # These functions are chosen to create non-trivial relationships
    alpha = 7.5 * np.sin(np.pi * X[:, 0]) + 5 * (X[:, 1] - 0.5)  # Skew parameter
    mu = 15 * (X[:, 2] - 0.5)  # Location parameter
    sigma = 1 + 3 * X[:, 3]  # Scale parameter (must be > 0)

    # Add noise to each parameter
    alpha += np.random.normal(0, 0.5, n_samples)
    mu += np.random.normal(0, 0.5, n_samples)
    sigma += np.random.normal(0, 0.2, n_samples)
    sigma = np.maximum(0.1, sigma)  # Ensure scale is positive

    # Generate y values from the skew-normal distribution for each sample
    y = np.asarray(skewnorm.rvs(a=alpha, loc=mu, scale=sigma, size=n_samples))

    return X, y.reshape(-1, 1)


def generate_multimodal_data(n_samples=2000):
    """
    Generates data where the target y is drawn from a mixture of distributions.
    The mixture component is determined by complex interactions between features,
    and noise is added to the decision boundaries to create overlapping regimes.
    """
    n_features = 4
    X = np.random.rand(n_samples, n_features)
    y = np.zeros(n_samples)

    # Define different modes (distributions)
    modes = {
        "mode1": {"type": "normal", "params": {"loc": -5, "scale": 1.0}},
        "mode2": {"type": "normal", "params": {"loc": 5, "scale": 1.5}},
        "mode3": {"type": "skewnorm", "params": {"a": 5, "loc": 0, "scale": 2}},
        "mode4": {"type": "bimodal", "params": {"loc1": -8, "scale1": 1, "loc2": 8, "scale2": 1, "mix": 0.5}},
    }

    # Define complex, interacting rules for assigning modes based on features
    # Add noise to create soft, probabilistic boundaries between regimes
    noise_scale = 0.1

    # Rule 1: Non-linear interaction between X0 and X1
    score1 = X[:, 0] * X[:, 1] - 0.2 + np.random.normal(0, noise_scale, n_samples)

    # Rule 2: Interaction between X2 and X3
    score2 = X[:, 2] - X[:, 3] + np.random.normal(0, noise_scale, n_samples)

    # Assign samples to modes based on which rule is met first
    mask1 = score1 < 0
    mask2 = ~mask1 & (score2 > 0.5)
    mask3 = ~mask1 & ~mask2 & (X[:, 0] > 0.7)
    mask4 = ~mask1 & ~mask2 & ~mask3  # The "else" case

    masks = [mask1, mask2, mask3, mask4]
    mode_keys = ["mode1", "mode2", "mode3", "mode4"]

    for mask, key in zip(masks, mode_keys):
        n_masked = np.sum(mask)
        if n_masked == 0:
            continue

        mode = modes[key]
        if mode["type"] == "normal":
            y[mask] = np.random.normal(size=n_masked, **mode["params"])
        elif mode["type"] == "skewnorm":
            y[mask] = skewnorm.rvs(size=n_masked, **mode["params"])
        elif mode["type"] == "bimodal":
            p = mode["params"]
            # Create a mixture of two normal distributions
            comp1 = np.random.normal(p["loc1"], p["scale1"], n_masked)
            comp2 = np.random.normal(p["loc2"], p["scale2"], n_masked)
            is_comp1 = np.random.rand(n_masked) < p["mix"]  # type: ignore
            y[mask] = np.where(is_comp1, comp1, comp2)

    return X, y.reshape(-1, 1)


def get_kde_log_likelihood(y, bandwidth, cv=5):
    """
    Calculates the cross-validated log-likelihood of a dataset for a given bandwidth.
    This uses k-fold cross-validation to provide a more regularized estimate of
    the likelihood, which helps prevent overfitting to the specific data in a node.
    """

    # Cannot perform cross-validation with fewer samples than folds.
    # Also, KDE requires at least 2 points to be meaningful.
    if len(y) < cv or len(y) < 2:
        # Fallback to in-sample for very small nodes, though this is less ideal.
        # A node with < 2 samples has no variance, so KDE is not well-defined.
        if len(y) < 2:
            return -np.inf
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(y)
        return kde.score(y)

    kf = KFold(n_splits=cv)
    log_likelihoods = []

    for train_index, test_index in kf.split(y):
        y_train, y_test = y[train_index], y[test_index]

        # Ensure the training split is not too small to fit a KDE
        if len(y_train) < 2:
            continue

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(y_train)
        log_likelihoods.append(kde.score(y_test))

    if not log_likelihoods:
        return -np.inf  # Should not happen if len(y) >= cv

    return np.mean(log_likelihoods)


def find_best_split(X, y, bandwidth):
    """
    Finds the best split for a node in a decision tree.
    The criterion is maximizing the sum of log-likelihoods of the children nodes.
    """
    best_gain = -np.inf
    best_feature_idx = -1
    best_threshold = -1
    n_samples, n_features = X.shape

    # Calculate parent log-likelihood to compute gain
    parent_ll = get_kde_log_likelihood(y, bandwidth)

    for feature_idx in tqdm(range(n_features)):
        # Use unique values as potential split points
        # To avoid testing every single value, we can test on quantiles
        # This makes the process faster and less prone to overfitting on specific values.
        num_thresholds = 30
        # Generate quantiles, excluding 0 and 1 to avoid empty splits at the extremes
        quantiles = np.linspace(0, 1, num_thresholds + 2)[1:-1]
        thresholds = np.quantile(X[:, feature_idx], q=quantiles)
        # Ensure thresholds are unique, as some quantiles might be identical for sparse data
        thresholds = np.unique(thresholds)

        for threshold in thresholds:
            left_indices = X[:, feature_idx] <= threshold
            right_indices = ~left_indices

            # Ensure children are not empty
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            y_left, y_right = y[left_indices], y[right_indices]

            ll_left = get_kde_log_likelihood(y_left, bandwidth)
            ll_right = get_kde_log_likelihood(y_right, bandwidth)

            # Weighted sum of children log-likelihoods
            current_gain = (len(y_left) * ll_left + len(y_right) * ll_right) / n_samples - parent_ll

            if current_gain > best_gain:
                best_gain = current_gain
                best_feature_idx = feature_idx
                best_threshold = threshold

    return best_feature_idx, best_threshold


# %%
bandwidth = 2  # User-specified bandwidth for KDE
n_samples = 1000  # Number of samples to generate

print(f"Using user-specified bandwidth: {bandwidth}")

# 1. Generate data
# X, y = generate_data(n_samples=n_samples)
# X, y = generate_skew_normal_data(n_samples=n_samples)
X, y = generate_multimodal_data(n_samples=n_samples)

# 2. Create a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training on {len(y_train)} samples, testing on {len(y_test)} samples.")

# 3. Find splits using only the training data
print("\n--- Finding Splits on Training Data ---")

# Level 1 split
feature1, threshold1 = find_best_split(X_train, y_train, bandwidth)
print(f"Split 1: Feature {feature1} <= {threshold1:.2f}")

# Level 2 splits
train_left1_mask = X_train[:, feature1] <= threshold1
X_train_left1, y_train_left1 = X_train[train_left1_mask], y_train[train_left1_mask]
X_train_right1, y_train_right1 = X_train[~train_left1_mask], y_train[~train_left1_mask]

feature2a, threshold2a = find_best_split(X_train_left1, y_train_left1, bandwidth)
print(f"  Leaf 1.1 Split: Feature {feature2a} <= {threshold2a:.2f}")
feature2b, threshold2b = find_best_split(X_train_right1, y_train_right1, bandwidth)
print(f"  Leaf 1.2 Split: Feature {feature2b} <= {threshold2b:.2f}")

# Level 3 splits
train_left2_mask = X_train_left1[:, feature2a] <= threshold2a
X_train_leaf1, y_train_leaf1 = X_train_left1[train_left2_mask], y_train_left1[train_left2_mask]
X_train_leaf2, y_train_leaf2 = X_train_left1[~train_left2_mask], y_train_left1[~train_left2_mask]

train_right2_mask = X_train_right1[:, feature2b] <= threshold2b
X_train_leaf3, y_train_leaf3 = X_train_right1[train_right2_mask], y_train_right1[train_right2_mask]
X_train_leaf4, y_train_leaf4 = X_train_right1[~train_right2_mask], y_train_right1[~train_right2_mask]

feature3a, threshold3a = find_best_split(X_train_leaf1, y_train_leaf1, bandwidth)
print(f"    Leaf 2.1 Split: Feature {feature3a} <= {threshold3a:.2f}")
feature3b, threshold3b = find_best_split(X_train_leaf2, y_train_leaf2, bandwidth)
print(f"    Leaf 2.2 Split: Feature {feature3b} <= {threshold3b:.2f}")
feature3c, threshold3c = find_best_split(X_train_leaf3, y_train_leaf3, bandwidth)
print(f"    Leaf 2.3 Split: Feature {feature3c} <= {threshold3c:.2f}")
feature3d, threshold3d = find_best_split(X_train_leaf4, y_train_leaf4, bandwidth)
print(f"    Leaf 2.4 Split: Feature {feature3d} <= {threshold3d:.2f}")


# 4. Propagate the test data down the tree defined by the splits found above
print("\n--- Propagating Test Data Down the Tree ---")

# Level 1
test_left1_mask = X_test[:, feature1] <= threshold1
X_test_left1, y_test_left1 = X_test[test_left1_mask], y_test[test_left1_mask]
X_test_right1, y_test_right1 = X_test[~test_left1_mask], y_test[~test_left1_mask]

# Level 2
test_left2_mask = X_test_left1[:, feature2a] <= threshold2a
X_test_leaf1, y_test_leaf1 = X_test_left1[test_left2_mask], y_test_left1[test_left2_mask]
X_test_leaf2, y_test_leaf2 = X_test_left1[~test_left2_mask], y_test_left1[~test_left2_mask]

test_right2_mask = X_test_right1[:, feature2b] <= threshold2b
X_test_leaf3, y_test_leaf3 = X_test_right1[test_right2_mask], y_test_right1[test_right2_mask]
X_test_leaf4, y_test_leaf4 = X_test_right1[~test_right2_mask], y_test_right1[~test_right2_mask]

# Level 3 (final leaves)
test_final_leaf1_mask = X_test_leaf1[:, feature3a] <= threshold3a
y_test_final_leaf1 = y_test_leaf1[test_final_leaf1_mask]
y_test_final_leaf2 = y_test_leaf1[~test_final_leaf1_mask]

test_final_leaf2_mask = X_test_leaf2[:, feature3b] <= threshold3b
y_test_final_leaf3 = y_test_leaf2[test_final_leaf2_mask]
y_test_final_leaf4 = y_test_leaf2[~test_final_leaf2_mask]

test_final_leaf3_mask = X_test_leaf3[:, feature3c] <= threshold3c
y_test_final_leaf5 = y_test_leaf3[test_final_leaf3_mask]
y_test_final_leaf6 = y_test_leaf3[~test_final_leaf3_mask]

test_final_leaf4_mask = X_test_leaf4[:, feature3d] <= threshold3d
y_test_final_leaf7 = y_test_leaf4[test_final_leaf4_mask]
y_test_final_leaf8 = y_test_leaf4[~test_final_leaf4_mask]

# Also get the final training data leaves for plotting the KDE
train_final_leaf1_mask = X_train_leaf1[:, feature3a] <= threshold3a
y_train_final_leaf1 = y_train_leaf1[train_final_leaf1_mask]
y_train_final_leaf2 = y_train_leaf1[~train_final_leaf1_mask]

train_final_leaf2_mask = X_train_leaf2[:, feature3b] <= threshold3b
y_train_final_leaf3 = y_train_leaf2[train_final_leaf2_mask]
y_train_final_leaf4 = y_train_leaf2[~train_final_leaf2_mask]

train_final_leaf3_mask = X_train_leaf3[:, feature3c] <= threshold3c
y_train_final_leaf5 = y_train_leaf3[train_final_leaf3_mask]
y_train_final_leaf6 = y_train_leaf3[~train_final_leaf3_mask]

train_final_leaf4_mask = X_train_leaf4[:, feature3d] <= threshold3d
y_train_final_leaf7 = y_train_leaf4[train_final_leaf4_mask]
y_train_final_leaf8 = y_train_leaf4[~train_final_leaf4_mask]


# 5. Plot the KDE from the training data against the histogram of the test data for each leaf
leaves_y_train = [
    y_train_final_leaf1,
    y_train_final_leaf2,
    y_train_final_leaf3,
    y_train_final_leaf4,
    y_train_final_leaf5,
    y_train_final_leaf6,
    y_train_final_leaf7,
    y_train_final_leaf8,
]
leaves_y_test = [
    y_test_final_leaf1,
    y_test_final_leaf2,
    y_test_final_leaf3,
    y_test_final_leaf4,
    y_test_final_leaf5,
    y_test_final_leaf6,
    y_test_final_leaf7,
    y_test_final_leaf8,
]

fig, axes = plt.subplots(4, 2, figsize=(14, 20), sharex=True, sharey=True)
fig.suptitle(f"Train KDE vs. Test Histogram in Terminal Leaves (Bandwidth: {bandwidth})", fontsize=16)

plot_range = np.linspace(np.min(y) - 2, np.max(y) + 2, 1000).reshape(-1, 1)

for i, (ax, y_train_leaf, y_test_leaf) in enumerate(zip(axes.flatten(), leaves_y_train, leaves_y_test)):
    # Plot KDE from training data
    if len(y_train_leaf) > 1:
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(y_train_leaf)
        log_dens = kde.score_samples(plot_range)
        ax.plot(plot_range, np.exp(log_dens), label=f"Train KDE (n={len(y_train_leaf)})", color="blue")
        ax.fill_between(plot_range.ravel(), np.exp(log_dens), alpha=0.2, color="blue")

    # Plot histogram of test data
    if len(y_test_leaf) > 0:
        ax.hist(
            y_test_leaf, bins=35, density=True, alpha=0.6, label=f"Test Hist (n={len(y_test_leaf)})", color="orange"
        )

    ax.set_title(f"Leaf {i+1}")
    ax.set_xlabel("y value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
plt.show()


# %%
def get_skewnorm_log_likelihood(y):
    """
    Fits a skew-normal distribution to the data and returns the total log-likelihood
    and the fitted parameters.
    """
    # skewnorm.fit requires at least 3 points to reliably estimate parameters.
    if len(y) < 3:
        return -np.inf, (0, np.mean(y) if len(y) > 0 else 0, 1)

    try:
        # Suppress warnings that can occur during fitting if data is ill-conditioned
        with np.errstate(all="ignore"):
            # Fit the skew-normal distribution. floc=0 and fscale=1 can sometimes help convergence
            # but we will let it fit all parameters.
            a, loc, scale = skewnorm.fit(y.ravel())

        # Ensure scale is positive, as fit can sometimes return a small negative number
        if scale is None or scale <= 0:
            return -np.inf, (a, loc, scale)

        # Calculate the total log-likelihood of the data under the fitted distribution
        log_likelihood = np.sum(skewnorm.logpdf(y, a=a, loc=loc, scale=scale))
        return log_likelihood, (a, loc, scale)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        # Fitting can fail for various reasons (e.g., data is constant or ill-conditioned)
        return -np.inf, (0, np.mean(y), np.std(y) if np.std(y) > 0 else 1)


def find_best_split_skewnorm(X, y):
    """
    Finds the best split for a node by fitting a skew-normal distribution.
    The criterion is maximizing the sum of log-likelihoods of the children nodes.
    """
    best_gain = -np.inf
    best_feature_idx = -1
    best_threshold = -1
    n_samples, n_features = X.shape

    # Calculate parent log-likelihood to compute information gain
    parent_ll, _ = get_skewnorm_log_likelihood(y)
    if parent_ll == -np.inf:  # Cannot split a node that cannot be fit
        return -1, -1

    for feature_idx in tqdm(range(n_features), desc="Finding Skew-Normal Splits"):
        # Use unique quantiles as potential split points for efficiency
        num_thresholds = 30
        quantiles = np.linspace(0, 1, num_thresholds + 2)[1:-1]
        thresholds = np.unique(np.quantile(X[:, feature_idx], q=quantiles))

        for threshold in thresholds:
            left_indices = X[:, feature_idx] <= threshold
            right_indices = ~left_indices

            # Ensure children are not empty
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            y_left, y_right = y[left_indices], y[right_indices]

            ll_left, _ = get_skewnorm_log_likelihood(y_left)
            ll_right, _ = get_skewnorm_log_likelihood(y_right)

            # If either child fails to fit, this split is invalid
            if ll_left == -np.inf or ll_right == -np.inf:
                continue

            # Information gain is the sum of children's likelihoods minus the parent's
            current_gain = (ll_left + ll_right) - parent_ll

            if current_gain > best_gain:
                best_gain = current_gain
                best_feature_idx = feature_idx
                best_threshold = threshold

    return best_feature_idx, best_threshold


print("\n\n--- Finding Splits using Skew-Normal Likelihood ---")

# Find splits on the training data
feature1_sn, threshold1_sn = find_best_split_skewnorm(X_train, y_train)
print(f"Split 1 (SN): Feature {feature1_sn} <= {threshold1_sn:.2f}")

# Propagate training data down the first split
train_left1_mask_sn = X_train[:, feature1_sn] <= threshold1_sn
X_train_left1_sn, y_train_left1_sn = X_train[train_left1_mask_sn], y_train[train_left1_mask_sn]
X_train_right1_sn, y_train_right1_sn = X_train[~train_left1_mask_sn], y_train[~train_left1_mask_sn]

# Find second-level splits
feature2a_sn, threshold2a_sn = find_best_split_skewnorm(X_train_left1_sn, y_train_left1_sn)
print(f"  Leaf 1.1 Split (SN): Feature {feature2a_sn} <= {threshold2a_sn:.2f}")
feature2b_sn, threshold2b_sn = find_best_split_skewnorm(X_train_right1_sn, y_train_right1_sn)
print(f"  Leaf 1.2 Split (SN): Feature {feature2b_sn} <= {threshold2b_sn:.2f}")

# Propagate test data down the full tree
test_left1_mask_sn = X_test[:, feature1_sn] <= threshold1_sn
X_test_left1_sn, y_test_left1_sn = X_test[test_left1_mask_sn], y_test[test_left1_mask_sn]
X_test_right1_sn, y_test_right1_sn = X_test[~test_left1_mask_sn], y_test[~test_left1_mask_sn]

test_left2_mask_sn = X_test_left1_sn[:, feature2a_sn] <= threshold2a_sn
y_test_leaf1_sn = y_test_left1_sn[test_left2_mask_sn]
y_test_leaf2_sn = y_test_left1_sn[~test_left2_mask_sn]

test_right2_mask_sn = X_test_right1_sn[:, feature2b_sn] <= threshold2b_sn
y_test_leaf3_sn = y_test_right1_sn[test_right2_mask_sn]
y_test_leaf4_sn = y_test_right1_sn[~test_right2_mask_sn]

# Get corresponding training data leaves to fit the distributions for plotting
train_left2_mask_sn = X_train_left1_sn[:, feature2a_sn] <= threshold2a_sn
X_train_leaf1_sn, y_train_leaf1_sn = X_train_left1_sn[train_left2_mask_sn], y_train_left1_sn[train_left2_mask_sn]
X_train_leaf2_sn, y_train_leaf2_sn = X_train_left1_sn[~train_left2_mask_sn], y_train_left1_sn[~train_left2_mask_sn]

train_right2_mask_sn = X_train_right1_sn[:, feature2b_sn] <= threshold2b_sn
X_train_leaf3_sn, y_train_leaf3_sn = X_train_right1_sn[train_right2_mask_sn], y_train_right1_sn[train_right2_mask_sn]
X_train_leaf4_sn, y_train_leaf4_sn = X_train_right1_sn[~train_right2_mask_sn], y_train_right1_sn[~train_right2_mask_sn]

# Find third-level splits
feature3a_sn, threshold3a_sn = find_best_split_skewnorm(X_train_leaf1_sn, y_train_leaf1_sn)
print(f"    Leaf 2.1 Split (SN): Feature {feature3a_sn} <= {threshold3a_sn:.2f}")
feature3b_sn, threshold3b_sn = find_best_split_skewnorm(X_train_leaf2_sn, y_train_leaf2_sn)
print(f"    Leaf 2.2 Split (SN): Feature {feature3b_sn} <= {threshold3b_sn:.2f}")
feature3c_sn, threshold3c_sn = find_best_split_skewnorm(X_train_leaf3_sn, y_train_leaf3_sn)
print(f"    Leaf 2.3 Split (SN): Feature {feature3c_sn} <= {threshold3c_sn:.2f}")
feature3d_sn, threshold3d_sn = find_best_split_skewnorm(X_train_leaf4_sn, y_train_leaf4_sn)
print(f"    Leaf 2.4 Split (SN): Feature {feature3d_sn} <= {threshold3d_sn:.2f}")

# Propagate test data to the final 8 leaves
X_test_leaf1_sn, y_test_leaf1_sn = X_test_left1_sn[test_left2_mask_sn], y_test_left1_sn[test_left2_mask_sn]
X_test_leaf2_sn, y_test_leaf2_sn = X_test_left1_sn[~test_left2_mask_sn], y_test_left1_sn[~test_left2_mask_sn]
X_test_leaf3_sn, y_test_leaf3_sn = X_test_right1_sn[test_right2_mask_sn], y_test_right1_sn[test_right2_mask_sn]
X_test_leaf4_sn, y_test_leaf4_sn = X_test_right1_sn[~test_right2_mask_sn], y_test_right1_sn[~test_right2_mask_sn]

test_final_leaf1_mask_sn = X_test_leaf1_sn[:, feature3a_sn] <= threshold3a_sn
y_test_final_leaf1_sn = y_test_leaf1_sn[test_final_leaf1_mask_sn]
y_test_final_leaf2_sn = y_test_leaf1_sn[~test_final_leaf1_mask_sn]

test_final_leaf2_mask_sn = X_test_leaf2_sn[:, feature3b_sn] <= threshold3b_sn
y_test_final_leaf3_sn = y_test_leaf2_sn[test_final_leaf2_mask_sn]
y_test_final_leaf4_sn = y_test_leaf2_sn[~test_final_leaf2_mask_sn]

test_final_leaf3_mask_sn = X_test_leaf3_sn[:, feature3c_sn] <= threshold3c_sn
y_test_final_leaf5_sn = y_test_leaf3_sn[test_final_leaf3_mask_sn]
y_test_final_leaf6_sn = y_test_leaf3_sn[~test_final_leaf3_mask_sn]

test_final_leaf4_mask_sn = X_test_leaf4_sn[:, feature3d_sn] <= threshold3d_sn
y_test_final_leaf7_sn = y_test_leaf4_sn[test_final_leaf4_mask_sn]
y_test_final_leaf8_sn = y_test_leaf4_sn[~test_final_leaf4_mask_sn]

# Get corresponding training data for the final 8 leaves
train_final_leaf1_mask_sn = X_train_leaf1_sn[:, feature3a_sn] <= threshold3a_sn
y_train_final_leaf1_sn = y_train_leaf1_sn[train_final_leaf1_mask_sn]
y_train_final_leaf2_sn = y_train_leaf1_sn[~train_final_leaf1_mask_sn]

train_final_leaf2_mask_sn = X_train_leaf2_sn[:, feature3b_sn] <= threshold3b_sn
y_train_final_leaf3_sn = y_train_leaf2_sn[train_final_leaf2_mask_sn]
y_train_final_leaf4_sn = y_train_leaf2_sn[~train_final_leaf2_mask_sn]

train_final_leaf3_mask_sn = X_train_leaf3_sn[:, feature3c_sn] <= threshold3c_sn
y_train_final_leaf5_sn = y_train_leaf3_sn[train_final_leaf3_mask_sn]
y_train_final_leaf6_sn = y_train_leaf3_sn[~train_final_leaf3_mask_sn]

train_final_leaf4_mask_sn = X_train_leaf4_sn[:, feature3d_sn] <= threshold3d_sn
y_train_final_leaf7_sn = y_train_leaf4_sn[train_final_leaf4_mask_sn]
y_train_final_leaf8_sn = y_train_leaf4_sn[~train_final_leaf4_mask_sn]

# Plotting
leaves_y_train_sn = [
    y_train_final_leaf1_sn,
    y_train_final_leaf2_sn,
    y_train_final_leaf3_sn,
    y_train_final_leaf4_sn,
    y_train_final_leaf5_sn,
    y_train_final_leaf6_sn,
    y_train_final_leaf7_sn,
    y_train_final_leaf8_sn,
]
leaves_y_test_sn = [
    y_test_final_leaf1_sn,
    y_test_final_leaf2_sn,
    y_test_final_leaf3_sn,
    y_test_final_leaf4_sn,
    y_test_final_leaf5_sn,
    y_test_final_leaf6_sn,
    y_test_final_leaf7_sn,
    y_test_final_leaf8_sn,
]

fig_sn, axes_sn = plt.subplots(4, 2, figsize=(14, 20), sharex=True, sharey=True)
fig_sn.suptitle("Train Skew-Normal Fit vs. Test Histogram in Terminal Leaves", fontsize=16)

plot_range = np.linspace(np.min(y) - 2, np.max(y) + 2, 1000).reshape(-1, 1)

for i, (ax, y_train_leaf, y_test_leaf) in enumerate(zip(axes_sn.flatten(), leaves_y_train_sn, leaves_y_test_sn)):
    # Fit skew-normal on training data and plot its PDF
    if len(y_train_leaf) > 2:
        _, params = get_skewnorm_log_likelihood(y_train_leaf)
        a, loc, scale = params
        if scale is not None and scale > 0:
            pdf = skewnorm.pdf(plot_range, a=a, loc=loc, scale=scale)
            ax.plot(plot_range, pdf, label=f"Train Fit (n={len(y_train_leaf)})", color="blue")
            ax.fill_between(plot_range.ravel(), pdf.ravel(), alpha=0.2, color="blue")

    # Plot histogram of test data
    if len(y_test_leaf) > 0:
        ax.hist(
            y_test_leaf, bins=35, density=True, alpha=0.6, label=f"Test Hist (n={len(y_test_leaf)})", color="orange"
        )

    ax.set_title(f"Leaf {i+1}")
    ax.set_xlabel("y value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
plt.show()
# %%
