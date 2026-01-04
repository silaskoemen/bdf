import math
from typing import Literal

import numpy as np
import pytest

from bdf.distributions.kde import KDE, BayesianKDE, BayesianKDEParams, KDEParams

# -----------------------------
# Pure-python reference helpers
# -----------------------------

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _kernel_matrix(y: np.ndarray, h: float, kernel: str, *, compact_support: bool) -> np.ndarray:
    """
    Return dense K where K[i,j] = K(y_i, y_j), with diagonal = 0.
    Matches Rust split-scoring kernel matrix semantics:
    - diagonal excluded
    - Gaussian optionally truncated at 4*h (if compact_support True)
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    assert n >= 2
    assert h > 0

    K = np.zeros((n, n), dtype=float)

    if kernel == "gaussian":
        cutoff2 = (4.0 * h) ** 2 if compact_support else float("inf")
        norm = INV_SQRT_2PI / h
        for i in range(n):
            for j in range(i + 1, n):
                d = y[i] - y[j]
                d2 = d * d
                if d2 > cutoff2:
                    continue
                kij = norm * math.exp(-0.5 * d2 / (h * h))
                K[i, j] = kij
                K[j, i] = kij
    elif kernel == "epanechnikov":
        inv_h = 1.0 / h
        inv_h2 = inv_h * inv_h
        norm = 0.75 * inv_h
        for i in range(n):
            for j in range(i + 1, n):
                d = y[i] - y[j]
                u2 = (d * d) * inv_h2
                if u2 <= 1.0:
                    one_minus = 1.0 - u2
                    kij = norm * one_minus if one_minus > 0.0 else 0.0
                else:
                    kij = 0.0
                K[i, j] = kij
                K[j, i] = kij
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return K


def _loo_nll_from_row_sums(row_sums: np.ndarray) -> float:
    """
    If row_sums[i] = sum_{j != i} K[i,j], then
    NLL = n*ln(n-1) - sum_i ln(row_sums[i]).
    Matches Rust's kde_full_nll_from_row_sums.
    """
    row_sums = np.asarray(row_sums, dtype=float).ravel()
    n = row_sums.size
    if n < 2:
        return float("inf")
    if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        return float("inf")
    return float(n * math.log(n - 1) - np.log(row_sums).sum())


def _subset_nll_from_row_sums(indices: np.ndarray, row_sums_for_subset: np.ndarray) -> float:
    """
    indices: integer indices of the subset
    row_sums_for_subset is length n_total, but only values at indices are used.
    Matches Rust's kde_nll_from_row_sums (for the Left side).
    """
    m = int(indices.size)
    if m < 2:
        return float("inf")
    vals = row_sums_for_subset[indices]
    if np.any(~np.isfinite(vals)) or np.any(vals <= 0.0):
        return float("inf")
    return float(m * math.log(m - 1) - np.log(vals).sum())


def _subset_nll_right(indices: np.ndarray, sum_to_left: np.ndarray, rs_total: np.ndarray) -> float:
    """
    Right side formula: s_i = rs_total[i] - sum_to_left[i]
    Matches Rust's kde_nll_from_row_sums_right.
    """
    m = int(indices.size)
    if m < 2:
        return float("inf")
    vals = rs_total[indices] - sum_to_left[indices]
    if np.any(~np.isfinite(vals)) or np.any(vals <= 0.0):
        return float("inf")
    return float(m * math.log(m - 1) - np.log(vals).sum())


def _rust_like_best_split_kde(
    X: np.ndarray,
    y: np.ndarray,
    *,
    eta: float,
    min_samples_leaf: int,
    min_child_weight: float,
    reg_gamma: float,
    kernel: str,
    h: float,
    compact_support: bool,
) -> tuple[int | None, float | None, float]:
    """
    Reference implementation for Rust's FAST KDE path in splitter.rs:
    - Parent bandwidth
    - Dense kernel matrix
    - Candidate splits at (i+1)%stride==0 along sorted feature
    - Penalty: reg_gamma*(ln(num_features_tried)+ln(num_thresholds_tried))
    Returns (best_feature, best_threshold, best_gain).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    assert y.size == n
    if n < 2:
        return None, None, 0.0

    stride = int(max(n * eta, 1.0))
    num_features_tried = d

    K = _kernel_matrix(y, h, kernel, compact_support=compact_support)
    rs_total = K.sum(axis=1)
    current_score = _loo_nll_from_row_sums(rs_total)

    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for feat in range(d):
        col = X[:, feat]
        order = np.argsort(col, kind="mergesort")  # stable-ish, deterministic
        sum_to_left = np.zeros(n, dtype=float)

        local_best_gain = 0.0
        local_best_threshold = None
        num_thresholds_tried = 0

        # emulate Rust: walk i=0..n-2, update sum_to_left using the row of the point moved to left
        for i in range(n - 1):
            idx = int(order[i])
            sum_to_left += K[idx, :]

            if (i + 1) % stride != 0:
                continue

            feat_val = float(col[idx])
            next_feat_val = float(col[int(order[i + 1])])
            if feat_val >= next_feat_val:
                continue

            left_n = i + 1
            right_n = n - left_n
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue
            if float(left_n) < float(min_child_weight) or float(right_n) < float(min_child_weight):
                continue

            num_thresholds_tried += 1

            left_indices = order[:left_n]
            right_indices = order[left_n:]

            left_score = _subset_nll_from_row_sums(left_indices, sum_to_left)
            right_score = _subset_nll_right(right_indices, sum_to_left, rs_total)

            if not np.isfinite(left_score) or not np.isfinite(right_score):
                continue

            gain = current_score - (left_score + right_score)
            if gain > local_best_gain:
                local_best_gain = gain
                local_best_threshold = 0.5 * (feat_val + next_feat_val)

        if reg_gamma > 0.0 and num_thresholds_tried > 0 and local_best_threshold is not None:
            local_best_gain -= reg_gamma * (math.log(float(num_features_tried)) + math.log(float(num_thresholds_tried)))

        if local_best_threshold is not None and local_best_gain > best_gain:
            best_gain = float(local_best_gain)
            best_feature = int(feat)
            best_threshold = float(local_best_threshold)

    if best_feature is None:
        return None, None, 0.0
    return best_feature, best_threshold, best_gain


# -----------------------------
# Numerical correctness tests
# -----------------------------


def test_kde_validate_targets_rejects_invalid():
    dist = KDE(KDEParams(bandwidth=0.5, kernel="gaussian"))
    with pytest.raises(ValueError):
        dist.validate_targets(np.array([1.0]))  # needs >=2

    with pytest.raises(ValueError):
        dist.validate_targets(np.array([1.0, np.nan]))

    with pytest.raises(ValueError):
        dist.validate_targets(np.array([1.0, np.inf]))


def test_kde_bandwidth_fixed_and_min_bandwidth():
    dist = KDE(KDEParams(bandwidth=1e-10, min_bandwidth=1e-6, kernel="gaussian"))
    y = np.array([0.0, 1.0, 2.0])
    params = dist.calc_posterior_params(y)
    assert params["bandwidth"] == pytest.approx(1e-6)


def test_kde_bandwidth_scott_matches_formula():
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    dist = KDE(KDEParams(bandwidth="scott", min_bandwidth=1e-10, kernel="gaussian"))
    h = dist.calc_posterior_params(y)["bandwidth"]
    expected = np.std(y, ddof=1) * (y.size ** (-0.2))
    assert h == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_kde_bandwidth_silverman_matches_formula():
    y = np.array([0.0, 1.0, 2.0, 100.0], dtype=float)
    dist = KDE(KDEParams(bandwidth="silverman", min_bandwidth=1e-10, kernel="gaussian"))
    h = dist.calc_posterior_params(y)["bandwidth"]

    std = float(np.std(y, ddof=1))
    iqr = float(np.subtract(*np.percentile(y, [75, 25])))
    scale = min(std, iqr / 1.349) if iqr > 0 else std
    if scale <= 0:
        scale = std
    expected = max(1e-10, 0.9 * scale * (y.size ** (-0.2)))

    assert h == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_kde_gaussian_log_kernel_values():
    dist = KDE(KDEParams(bandwidth=2.0, kernel="gaussian"))
    y = np.array([0.0, 2.0])
    logK = dist._gaussian_log_kernel(y, y, 2.0)

    # K(x,y)= (1/sqrt(2pi)/h)*exp(-0.5*((x-y)/h)^2)
    # logK = -0.5*diff^2 - 0.5*ln(2pi) - ln(h) with diff=(x-y)/h
    expected_00 = -0.5 * 0.0 - 0.5 * math.log(2 * math.pi) - math.log(2.0)
    expected_01 = -0.5 * ((0.0 - 2.0) / 2.0) ** 2 - 0.5 * math.log(2 * math.pi) - math.log(2.0)

    assert logK[0, 0] == pytest.approx(expected_00, rel=0, abs=1e-14)
    assert logK[0, 1] == pytest.approx(expected_01, rel=0, abs=1e-14)


def test_kde_epanechnikov_log_kernel_support_and_values():
    dist = KDE(KDEParams(bandwidth=1.0, kernel="epanechnikov"))
    eval_points = np.array([0.0, 2.0])
    ref = np.array([0.0])

    logK = dist._epanechnikov_log_kernel(eval_points, ref, 1.0)
    # At u=0: log(0.75) + log(1-0) - log(h)
    assert logK[0, 0] == pytest.approx(math.log(0.75) - math.log(1.0), abs=1e-14)
    # At u=2: outside support => -inf
    assert np.isneginf(logK[1, 0])


def test_kde_loo_cv_matches_naive_leave_one_out():
    y = np.array([-2.0, -1.0, 1.0, 2.0], dtype=float)
    dist = KDE(KDEParams(bandwidth=0.7, kernel="gaussian", score_method="nll", score_correction="loo_cv"))
    ll_fast = dist._loo_cv_log_likelihood(y)

    ll_naive = np.empty_like(ll_fast)
    for i in range(y.size):
        train = np.delete(y, i)
        params = dist.calc_posterior_params(train)
        ll_naive[i] = dist._plugin_log_likelihood(np.array([y[i]]), params)[0]

    assert np.allclose(ll_fast, ll_naive, rtol=0, atol=1e-10)


def test_kde_translation_invariance_fixed_bandwidth():
    y = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
    dist = KDE(KDEParams(bandwidth=0.5, kernel="gaussian", score_method="nll", score_correction=None))
    nll1 = dist.nll(y)
    nll2 = dist.nll(y + 123.456)
    assert nll1 == pytest.approx(nll2, rel=0, abs=1e-10)


def test_bayesian_kde_posterior_bandwidth_formula():
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    params = BayesianKDEParams(bandwidth=0.5, kernel="gaussian", prior_h=2.0, m_h=3.0, min_bandwidth=1e-10)
    dist = BayesianKDE(params)

    post = dist.calc_posterior_params(y)
    data_h = float(post["data_bandwidth"])
    n = float(post["n"])
    expected = (params.m_h * params.prior_h + n * data_h) / (params.m_h + n)
    assert float(post["bandwidth"]) == pytest.approx(expected, rel=0, abs=1e-14)


def test_kde_fft_param_validation_rejects_epanechnikov():
    with pytest.raises(ValueError):
        KDEParams(kde_backend="fft", kernel="epanechnikov")


# -----------------------------------
# Rust <-> Python equivalence tests
# -----------------------------------


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov"])
def test_rust_python_split_equivalence_pairwise_parent(kernel: Literal["gaussian", "epanechnikov"]):
    bdf_rs = pytest.importorskip("bdf_rs")

    rng = np.random.default_rng(0)
    n = 60

    # Mixture targets -> meaningful KDE split gains
    y = np.concatenate([rng.normal(-2.0, 0.3, n // 2), rng.normal(2.0, 0.3, n // 2)]).astype(float)

    # Feature 0 correlates with y; feature 1 is noise (best should be feature 0)
    X0 = y + rng.normal(0.0, 0.1, n)
    X1 = rng.normal(0.0, 1.0, n)
    X = np.column_stack([X0, X1]).astype(float, copy=False)

    eta = 0.1
    reg_gamma = 0.2
    min_samples_leaf = 5
    min_child_weight = 5.0

    # Fixed bandwidth avoids any cross-language std/iqr subtlety
    h = 0.8

    dist = KDE(
        KDEParams(
            bandwidth=h,
            kernel=kernel,
            min_bandwidth=1e-10,
            kde_backend="pairwise",
            bandwidth_policy="parent",
            use_compact_support=False,
        )
    )
    spec = dist.to_rust_spec()

    feat_r, thr_r, gain_r, left_r, right_r, _, _ = bdf_rs.find_best_split(  # type: ignore[attr-defined]
        X, y, min_samples_leaf, float(min_child_weight), spec, eta, float(reg_gamma), None, "map"
    )

    feat_p, thr_p, gain_p = _rust_like_best_split_kde(
        X,
        y,
        eta=eta,
        min_samples_leaf=min_samples_leaf,
        min_child_weight=min_child_weight,
        reg_gamma=reg_gamma,
        kernel=kernel,
        h=h,
        compact_support=False,
    )

    assert feat_r == feat_p
    assert thr_r == pytest.approx(thr_p, rel=0, abs=1e-10)
    assert gain_r == pytest.approx(gain_p, rel=0, abs=1e-9)

    assert left_r is not None and right_r is not None
    assert int(np.sum(left_r)) + int(np.sum(right_r)) == n
    assert np.all(~(left_r & right_r))


def test_rust_compact_support_close_to_exact_gaussian():
    bdf_rs = pytest.importorskip("bdf_rs")

    rng = np.random.default_rng(1)
    n = 120
    y = rng.normal(0.0, 2.0, n).astype(float)
    X = np.column_stack([y + rng.normal(0, 0.2, n), rng.normal(0, 1, n)]).astype(float, copy=False)

    eta = 0.05
    h = 0.6
    min_samples_leaf = 10
    min_child_weight = 10.0
    reg_gamma = 0.0

    dist_exact = KDE(
        KDEParams(
            bandwidth=h,
            kernel="gaussian",
            kde_backend="pairwise",
            bandwidth_policy="parent",
            use_compact_support=False,
        )
    )
    dist_compact = KDE(
        KDEParams(
            bandwidth=h,
            kernel="gaussian",
            kde_backend="pairwise",
            bandwidth_policy="parent",
            use_compact_support=True,
        )
    )

    spec_exact = dist_exact.to_rust_spec()
    spec_compact = dist_compact.to_rust_spec()

    feat_e, thr_e, gain_e, _, _, _, _ = bdf_rs.find_best_split(  # type: ignore[attr-defined]
        X, y, min_samples_leaf, min_child_weight, spec_exact, eta, reg_gamma, None, "map"
    )
    feat_c, thr_c, gain_c, _, _, _, _ = bdf_rs.find_best_split(  # type: ignore[attr-defined]
        X, y, min_samples_leaf, min_child_weight, spec_compact, eta, reg_gamma, None, "map"
    )

    assert feat_e == feat_c
    assert thr_c == pytest.approx(thr_e, abs=0.25)
    assert gain_c == pytest.approx(gain_e, abs=0.25)


def test_rust_fft_close_to_pairwise_gaussian():
    bdf_rs = pytest.importorskip("bdf_rs")

    rng = np.random.default_rng(2)
    n = 300
    y = np.concatenate([rng.normal(-1.5, 0.5, n // 2), rng.normal(1.5, 0.5, n // 2)]).astype(float)
    X = np.column_stack([y + rng.normal(0, 0.3, n), rng.normal(0, 1, n)]).astype(float, copy=False)

    eta = 0.05
    h = 0.7
    min_samples_leaf = 20
    min_child_weight = 20.0
    reg_gamma = 0.0

    dist_pairwise = KDE(
        KDEParams(
            bandwidth=h,
            kernel="gaussian",
            kde_backend="pairwise",
            bandwidth_policy="parent",
            use_compact_support=False,
        )
    )
    dist_fft = KDE(
        KDEParams(
            bandwidth=h,
            kernel="gaussian",
            kde_backend="fft",
            bandwidth_policy="parent",
            fft_grid_points=128,
        )
    )

    spec_pairwise = dist_pairwise.to_rust_spec()
    spec_fft = dist_fft.to_rust_spec()

    feat_p, thr_p, gain_p, _, _, _, _ = bdf_rs.find_best_split(  # type: ignore[attr-defined]
        X, y, min_samples_leaf, min_child_weight, spec_pairwise, eta, reg_gamma, None, "map"
    )
    feat_f, thr_f, gain_f, _, _, _, _ = bdf_rs.find_best_split(  # type: ignore[attr-defined]
        X, y, min_samples_leaf, min_child_weight, spec_fft, eta, reg_gamma, None, "map"
    )

    # Approximation checks: same feature, threshold & gain reasonably close.
    assert feat_f == feat_p
    assert thr_f == pytest.approx(thr_p, abs=0.35)
    assert gain_f == pytest.approx(gain_p, rel=1.0)


def test_rust_switch_backend_matches_fft_when_forced():
    bdf_rs = pytest.importorskip("bdf_rs")

    rng = np.random.default_rng(3)
    n = 200
    y = rng.normal(0, 1, n).astype(float)
    X = np.column_stack([y + rng.normal(0, 0.2, n), rng.normal(0, 1, n)]).astype(float, copy=False)

    eta = 0.05
    h = 0.8

    dist_fft = KDE(
        KDEParams(
            bandwidth=h,
            kernel="gaussian",
            kde_backend="fft",
            bandwidth_policy="parent",
            fft_grid_points=1024,
        )
    )
    dist_switch = KDE(
        KDEParams(
            bandwidth=h,
            kernel="gaussian",
            kde_backend="switch",
            kde_backend_switch_size=1,  # n*n will exceed -> force FFT
            bandwidth_policy="parent",
            fft_grid_points=1024,
        )
    )

    spec_fft = dist_fft.to_rust_spec()
    spec_switch = dist_switch.to_rust_spec()

    out_fft = bdf_rs.find_best_split(X, y, 10, 10.0, spec_fft, eta, 0.0, None, "map")  # type: ignore[attr-defined]
    out_sw = bdf_rs.find_best_split(X, y, 10, 10.0, spec_switch, eta, 0.0, None, "map")  # type: ignore[attr-defined]

    assert out_sw[0] == out_fft[0]
    assert out_sw[1] == pytest.approx(out_fft[1], abs=1e-10)
    assert out_sw[2] == pytest.approx(out_fft[2], abs=1e-9)
