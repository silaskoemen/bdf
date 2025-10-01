# %% Imports
import numpy as np

# Data params
n_samples = 600
n_features = 1  # 1 or 2 supported in toy generator below
random_state = 42

# Toy data mix parameters (for mixture of Gaussians)
mix_weights = np.array([0.6, 0.4])  # must sum to 1
# For 1D:
means_1d = np.array([0.0, 3.0])
stds_1d = np.array([0.5, 0.8])
# For 2D:
means_2d = np.array([[0.0, 0.0], [3.0, 3.0]])
stds_2d = np.array([0.6, 0.8])  # isotropic per component

# KDE params
kernel = "gaussian"  # currently only 'gaussian' is implemented
bandwidth = "scott"  # > 0, isotropic
min_bandwidth = 1e-6  # clamp to avoid too small values
# Cross-validation: 'loo' for leave-one-out, 1 for in-sample, or int >= 2 for K-fold
cv = 2  # 'loo', 1, or int (e.g. 2, 5, 10)
kfold_shuffle = True


# %% Toy data generation
def make_toy_mog(n, d, weights, means_1d, stds_1d, means_2d, stds_2d, seed=None):
    rng = np.random.default_rng(seed)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    k = len(weights)
    comp_idx = rng.choice(k, size=n, p=weights)
    if d == 1:
        means = means_1d
        stds = stds_1d
        x = rng.normal(loc=means[comp_idx], scale=stds[comp_idx], size=n)
        return x.reshape(-1, 1)
    elif d == 2:
        means = means_2d
        stds = stds_2d
        x = np.empty((n, d))
        for i in range(n):
            mu = means[comp_idx[i]]
            s = stds[comp_idx[i]]
            x[i] = rng.normal(loc=mu, scale=s, size=d)
        return x
    else:
        raise ValueError("This toy generator supports n_features in {1, 2}.")


X = make_toy_mog(
    n=n_samples,
    d=n_features,
    weights=mix_weights,
    means_1d=means_1d,
    stds_1d=stds_1d,
    means_2d=means_2d,
    stds_2d=stds_2d,
    seed=random_state,
)


# %% Utilities
def _check_bandwidth(h, min_h=None):
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("bandwidth must be a positive finite float.")
    if min_h is None:
        min_h = float(min_bandwidth)
    min_h = float(min_h)
    if not np.isfinite(min_h) or min_h <= 0:
        raise ValueError("min_bandwidth must be a positive finite float.")
    if h < min_h:
        h = min_h
    return h


def _pairwise_sq_dists(A, B):
    # Returns squared Euclidean distances between rows of A and B
    # A: (n_a, d), B: (n_b, d)
    # Efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    A2 = np.sum(A * A, axis=1, keepdims=True)  # (n_a, 1)
    B2 = np.sum(B * B, axis=1, keepdims=True).T  # (1, n_b)
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    # Clamp tiny negatives to zero due to numerical errors
    np.maximum(D2, 0.0, out=D2)
    return D2


def _logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _compute_rule_of_thumb_bandwidth(X, method):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 2:
        raise ValueError("At least two samples are required to estimate the bandwidth.")
    method = method.lower()
    if method == "scott":
        scale = float(np.sqrt(np.mean(np.var(X, axis=0, ddof=1))))
        return scale * n ** (-1.0 / (d + 4))
    if method == "silverman":
        if d == 1:
            X_1d = X[:, 0]
            std = float(np.std(X_1d, ddof=1))
            iqr = float(np.subtract(*np.percentile(X_1d, [75, 25])))
            scale = min(std, iqr / 1.349) if iqr > 0 else std
            if scale <= 0:
                scale = std
            return 0.9 * scale * n ** (-1.0 / 5)
        scale = float(np.sqrt(np.mean(np.var(X, axis=0, ddof=1))))
        factor = (4.0 / (d + 2)) ** (1.0 / (d + 4))
        return factor * scale * n ** (-1.0 / (d + 4))
    raise ValueError("bandwidth method must be 'scott' or 'silverman'.")


def _resolve_bandwidth(bandwidth, X, min_h=None):
    if isinstance(bandwidth, str):
        h = _compute_rule_of_thumb_bandwidth(X, bandwidth)
    else:
        h = bandwidth
    return _check_bandwidth(h, min_h=min_h)


# %% KDE log-density (Gaussian kernel, isotropic bandwidth)
def gaussian_kde_logpdf(X_eval, X_train, bandwidth):
    """
    Computes log p(X_eval) under KDE fitted on X_train with Gaussian kernel and isotropic bandwidth.
    """
    X_eval = np.asarray(X_eval, dtype=float)
    X_train = np.asarray(X_train, dtype=float)

    # Early exit for empty evaluation set
    if X_eval.shape[0] == 0:
        return np.empty((0,), dtype=float)

    h = _resolve_bandwidth(bandwidth, X_train)
    n_train, d = X_train.shape
    if n_train < 1:
        raise ValueError("Training set must contain at least one sample.")
    D2 = _pairwise_sq_dists(X_eval, X_train)  # (n_eval, n_train)
    log_c = -0.5 * d * np.log(2.0 * np.pi) - d * np.log(h)
    log_weights = log_c - 0.5 * D2 / (h * h)
    log_sum = _logsumexp(log_weights, axis=1)  # (n_eval,)
    return log_sum - np.log(n_train)


# %% In-sample NLL (includes self-contribution)
def kde_insample_nll(X, bandwidth, kernel="gaussian"):
    if kernel != "gaussian":
        raise NotImplementedError("Only Gaussian kernel is implemented.")
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 1:
        raise ValueError("At least one sample is required.")
    h = _resolve_bandwidth(bandwidth, X)
    D2 = _pairwise_sq_dists(X, X)  # (n, n)
    log_c = -0.5 * d * np.log(2.0 * np.pi) - d * np.log(h)
    log_weights = log_c - 0.5 * D2 / (h * h)
    # include diagonal (self) terms
    log_density = _logsumexp(log_weights, axis=1) - np.log(n)
    if np.any(~np.isfinite(log_density)):
        raise FloatingPointError("Non-finite log densities encountered. Try a larger bandwidth.")
    nll_total = -np.sum(log_density)
    nll_mean = nll_total / n
    return nll_total, nll_mean


# %% Leave-One-Out NLL
def kde_loo_nll(X, bandwidth, kernel="gaussian"):
    if kernel != "gaussian":
        raise NotImplementedError("Only Gaussian kernel is implemented.")
    h = _resolve_bandwidth(bandwidth, X)
    n, d = X.shape
    if n < 2:
        raise ValueError("LOO requires at least 2 samples.")
    D2 = _pairwise_sq_dists(X, X)  # (n, n)
    log_c = -0.5 * d * np.log(2.0 * np.pi) - d * np.log(h)
    log_weights = log_c - 0.5 * D2 / (h * h)
    # Exclude self-contribution by setting diagonal to -inf
    np.fill_diagonal(log_weights, -np.inf)
    # log of averaged sum over n-1 terms
    log_density = _logsumexp(log_weights, axis=1) - np.log(n - 1)
    if np.any(~np.isfinite(log_density)):
        raise FloatingPointError("Non-finite log densities encountered. Try a larger bandwidth.")
    nll_total = -np.sum(log_density)
    nll_mean = nll_total / n
    return nll_total, nll_mean


# %% K-Fold CV NLL
def _kfold_indices(n, k, shuffle=True, seed=None):
    if k < 2:
        raise ValueError("k must be >= 2 for K-Fold.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = idx[start:stop]
        train_idx = np.concatenate((idx[:start], idx[stop:]))
        folds.append((train_idx, test_idx))
        current = stop
    return folds


def kde_kfold_nll(X, bandwidth, k=2, kernel="gaussian", shuffle=True, seed=None):
    if kernel != "gaussian":
        raise NotImplementedError("Only Gaussian kernel is implemented.")
    X = np.asarray(X, dtype=float)

    n = X.shape[0]
    if k < 2:
        raise ValueError("k must be >= 2 for K-Fold.")
    if k > n:
        raise ValueError("k cannot exceed number of samples n (k <= n).")
    # Exact equality with LOO when k == n
    if k == n:
        return kde_loo_nll(X, bandwidth, kernel=kernel)

    folds = _kfold_indices(n, k, shuffle=shuffle, seed=seed)
    nll_total = 0.0
    n_eval_total = 0
    for train_idx, test_idx in folds:
        if test_idx.size == 0:
            continue  # defensive; shouldn't happen with k <= n
        X_tr = X[train_idx]
        X_te = X[test_idx]
        logpdf = gaussian_kde_logpdf(X_te, X_tr, bandwidth)
        if np.any(~np.isfinite(logpdf)):
            raise FloatingPointError("Non-finite log densities encountered. Try a larger bandwidth.")
        nll_total += -np.sum(logpdf)
        n_eval_total += X_te.shape[0]
    nll_mean = nll_total / n_eval_total
    return nll_total, nll_mean


# %% Unified scorer
def kde_cv_nll(X, bandwidth, cv: str | int = "loo", kernel="gaussian", shuffle=True, seed=None):
    if cv == "loo":
        return kde_loo_nll(X, bandwidth, kernel=kernel)
    if isinstance(cv, int):
        if cv == 1:
            return kde_insample_nll(X, bandwidth, kernel=kernel)
        if cv < 2:
            raise ValueError("cv must be 'loo', 1, or an int >= 2.")
        return kde_kfold_nll(X, bandwidth, k=cv, kernel=kernel, shuffle=shuffle, seed=seed)
    raise ValueError("cv must be 'loo', 1, or an integer >= 2.")


# %% Run scoring
nll_total, nll_mean = kde_cv_nll(
    X,
    bandwidth=bandwidth,
    cv=cv,
    kernel=kernel,
    shuffle=kfold_shuffle,
    seed=random_state,
)

bandwidth_value = _resolve_bandwidth(bandwidth, X)
print(f"Settings: kernel={kernel}, bandwidth_spec={bandwidth}, resolved_bandwidth={bandwidth_value:.6f}, cv={cv}")
print(f"Total NLL: {nll_total:.6f}")
print(f"Mean NLL per sample: {nll_mean:.6f}")
# %%
nll_total_loo, nll_mean_loo = kde_loo_nll(X, bandwidth)
nll_total_k, nll_mean_k = kde_kfold_nll(X, bandwidth, k=X.shape[0], shuffle=True, seed=random_state)
print("LOO vs K-fold(k=n) Δtotal:", abs(nll_total_loo - nll_total_k))
print("LOO vs K-fold(k=n) Δmean :", abs(nll_mean_loo - nll_mean_k))
# %%
