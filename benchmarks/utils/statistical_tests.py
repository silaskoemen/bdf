"""Statistical significance tests for algorithm comparison.

Implements:
- Friedman test with Iman-Davenport correction
- Nemenyi post-hoc test with critical difference
- Wilcoxon signed-rank test for pairwise comparisons
- Win/Tie/Loss analysis
- Effect size measures (Vargha-Delaney A12)
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FriedmanResult:
    """Result of Friedman test."""

    statistic: float
    p_value: float
    iman_davenport_statistic: float
    iman_davenport_p_value: float
    avg_ranks: dict[str, float]
    n_datasets: int
    n_algorithms: int
    reject_null: bool  # at alpha=0.05


@dataclass
class NemenyiResult:
    """Result of Nemenyi post-hoc test."""

    critical_difference: float
    alpha: float
    significant_pairs: list[tuple[str, str]]
    non_significant_pairs: list[tuple[str, str]]
    avg_ranks: dict[str, float]


@dataclass
class WilcoxonResult:
    """Result of Wilcoxon signed-rank test."""

    statistic: float
    p_value: float
    effect_size_r: float  # r = Z / sqrt(N)
    a12: float  # Vargha-Delaney effect size
    reject_null: bool  # at alpha=0.05
    n_samples: int
    adjusted_p_value: float | None = None  # Holm-adjusted p-value, set by pairwise_wilcoxon_tests


@dataclass
class WinTieLossResult:
    """Win/Tie/Loss counts."""

    wins: int
    ties: int
    losses: int
    n_total: int
    sign_test_p_value: float


def compute_ranks(
    metric_matrix: np.ndarray,
    lower_is_better: bool = True,
) -> np.ndarray:
    """Compute ranks per dataset (row).

    Args:
        metric_matrix: Shape (n_datasets, n_algorithms). NaN allowed.
        lower_is_better: If True, lower values get lower (better) ranks.

    Returns:
        Rank matrix of same shape. Average ranks used for ties.
    """
    n_datasets, n_algorithms = metric_matrix.shape
    ranks = np.zeros_like(metric_matrix)

    for i in range(n_datasets):
        row = metric_matrix[i, :]
        valid_mask = ~np.isnan(row)

        if not valid_mask.any():
            ranks[i, :] = np.nan
            continue

        # Rank valid entries
        if lower_is_better:
            row_ranks = stats.rankdata(row[valid_mask], method="average")
        else:
            # Higher is better: negate to rank
            row_ranks = stats.rankdata(-row[valid_mask], method="average")

        ranks[i, valid_mask] = row_ranks
        ranks[i, ~valid_mask] = np.nan

    return ranks


def friedman_test(
    metric_matrix: np.ndarray,
    algorithm_names: list[str],
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> FriedmanResult:
    """Perform Friedman test with Iman-Davenport correction.

    Tests H0: All algorithms perform equally.

    Args:
        metric_matrix: Shape (n_datasets, n_algorithms).
        algorithm_names: Names for each algorithm (column).
        lower_is_better: If True, lower metric values are better.
        alpha: Significance level.

    Returns:
        FriedmanResult with test statistics and average ranks.
    """
    # Remove rows (datasets) with any NaN
    valid_rows = ~np.isnan(metric_matrix).any(axis=1)
    clean_matrix = metric_matrix[valid_rows, :]

    n_datasets, n_algorithms = clean_matrix.shape

    if n_datasets < 2:
        raise ValueError("Need at least 2 datasets for Friedman test")
    if n_algorithms < 2:
        raise ValueError("Need at least 2 algorithms for Friedman test")

    # Compute ranks
    ranks = compute_ranks(clean_matrix, lower_is_better)
    avg_ranks = np.nanmean(ranks, axis=0)

    # Friedman chi-squared statistic
    k = n_algorithms
    N = n_datasets
    sum_squared_ranks = np.sum(avg_ranks**2)

    chi2_f = (12 * N / (k * (k + 1))) * (sum_squared_ranks - (k * (k + 1) ** 2) / 4)

    # p-value from chi-squared distribution
    p_chi2 = 1 - stats.chi2.cdf(chi2_f, k - 1)

    # Iman-Davenport correction (more powerful)
    if chi2_f == 0:
        F_f = 0
        p_f = 1.0
    else:
        F_f = ((N - 1) * chi2_f) / (N * (k - 1) - chi2_f)
        # F distribution with (k-1, (k-1)(N-1)) degrees of freedom
        p_f = 1 - stats.f.cdf(F_f, k - 1, (k - 1) * (N - 1))

    avg_ranks_dict = {name: rank for name, rank in zip(algorithm_names, avg_ranks)}

    return FriedmanResult(
        statistic=chi2_f,
        p_value=p_chi2,
        iman_davenport_statistic=F_f,
        iman_davenport_p_value=p_f,
        avg_ranks=avg_ranks_dict,
        n_datasets=N,
        n_algorithms=k,
        reject_null=p_f < alpha,
    )


def nemenyi_critical_difference(
    n_algorithms: int,
    n_datasets: int,
    alpha: float = 0.05,
) -> float:
    """Compute Nemenyi critical difference.

    Args:
        n_algorithms: Number of algorithms (k).
        n_datasets: Number of datasets (N).
        alpha: Significance level.

    Returns:
        Critical difference value. Two algorithms are significantly different
        if |rank_i - rank_j| > CD.
    """
    # Studentized range critical values for Nemenyi test
    # q_alpha values for alpha=0.05, different k values
    # Source: Demsar (2006) Table
    q_alpha_005 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
    }

    q_alpha_010 = {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
        11: 2.978,
        12: 3.030,
        13: 3.077,
        14: 3.120,
        15: 3.159,
    }

    if alpha == 0.05:
        q_table = q_alpha_005
    elif alpha == 0.10:
        q_table = q_alpha_010
    else:
        warnings.warn(f"Using alpha=0.05 q-values for alpha={alpha}")
        q_table = q_alpha_005

    if n_algorithms not in q_table:
        # Approximate using largest available
        if n_algorithms > max(q_table.keys()):
            q_alpha = q_table[max(q_table.keys())]
        else:
            q_alpha = q_table[min(q_table.keys())]
        warnings.warn(f"q_alpha approximated for k={n_algorithms}")
    else:
        q_alpha = q_table[n_algorithms]

    cd = q_alpha * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_datasets))
    return cd


def nemenyi_test(
    metric_matrix: np.ndarray,
    algorithm_names: list[str],
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> NemenyiResult:
    """Perform Nemenyi post-hoc test.

    Should be used after Friedman test rejects null hypothesis.

    Args:
        metric_matrix: Shape (n_datasets, n_algorithms).
        algorithm_names: Names for each algorithm.
        lower_is_better: If True, lower values are better.
        alpha: Significance level.

    Returns:
        NemenyiResult with critical difference and significant pairs.
    """
    # Remove rows with NaN
    valid_rows = ~np.isnan(metric_matrix).any(axis=1)
    clean_matrix = metric_matrix[valid_rows, :]

    n_datasets, n_algorithms = clean_matrix.shape

    # Compute ranks and averages
    ranks = compute_ranks(clean_matrix, lower_is_better)
    avg_ranks = np.nanmean(ranks, axis=0)

    # Compute critical difference
    cd = nemenyi_critical_difference(n_algorithms, n_datasets, alpha)

    # Find significant and non-significant pairs
    significant_pairs = []
    non_significant_pairs = []

    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            pair = (algorithm_names[i], algorithm_names[j])
            if rank_diff > cd:
                significant_pairs.append(pair)
            else:
                non_significant_pairs.append(pair)

    avg_ranks_dict = {name: rank for name, rank in zip(algorithm_names, avg_ranks)}

    return NemenyiResult(
        critical_difference=cd,
        alpha=alpha,
        significant_pairs=significant_pairs,
        non_significant_pairs=non_significant_pairs,
        avg_ranks=avg_ranks_dict,
    )


def wilcoxon_signed_rank_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> WilcoxonResult:
    """Perform Wilcoxon signed-rank test for pairwise comparison.

    Tests H0: The two algorithms perform equally.

    Args:
        x: Metric values for algorithm A (one per dataset).
        y: Metric values for algorithm B (one per dataset).
        alpha: Significance level.

    Returns:
        WilcoxonResult with test statistics and effect sizes.
    """
    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]

    n = len(x_clean)

    if n < 5:
        warnings.warn(f"Wilcoxon test unreliable with n={n} < 5 samples")

    if n == 0:
        return WilcoxonResult(
            statistic=np.nan,
            p_value=1.0,
            effect_size_r=0.0,
            a12=0.5,
            reject_null=False,
            n_samples=0,
        )

    # Perform test
    try:
        stat, p_value = stats.wilcoxon(x_clean, y_clean, alternative="two-sided")
    except ValueError:
        # All differences are zero
        return WilcoxonResult(
            statistic=0.0,
            p_value=1.0,
            effect_size_r=0.0,
            a12=0.5,
            reject_null=False,
            n_samples=n,
        )

    # Effect size r = Z / sqrt(N)
    # Approximate Z from p-value
    z = stats.norm.ppf(1 - p_value / 2)
    effect_size_r = z / np.sqrt(n)

    # Vargha-Delaney A12 effect size
    a12 = vargha_delaney_a12(x_clean, y_clean)

    return WilcoxonResult(
        statistic=stat,
        p_value=p_value,
        effect_size_r=effect_size_r,
        a12=a12,
        reject_null=p_value < alpha,
        n_samples=n,
    )


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Vargha-Delaney A12 effect size.

    A12 measures the probability that a randomly chosen value from x
    is greater than a randomly chosen value from y.

    Interpretation:
        A12 = 0.5: No difference
        A12 > 0.5: x tends to be larger
        A12 < 0.5: y tends to be larger
        |A12 - 0.5| > 0.21: Large effect

    Args:
        x: Values for first group.
        y: Values for second group.

    Returns:
        A12 statistic in [0, 1].
    """
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0.5

    # Count wins and ties
    wins = 0
    ties = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                wins += 1
            elif xi == yj:
                ties += 1

    a12 = (wins + 0.5 * ties) / (m * n)
    return a12


def win_tie_loss(
    x: np.ndarray,
    y: np.ndarray,
    lower_is_better: bool = True,
    tie_threshold: float = 0.01,
    use_relative: bool = True,
) -> WinTieLossResult:
    """Compute Win/Tie/Loss counts.

    Args:
        x: Metric values for algorithm A (control).
        y: Metric values for algorithm B (challenger).
        lower_is_better: If True, A wins when x < y.
        tie_threshold: Threshold for declaring a tie.
        use_relative: If True, threshold is relative difference.
            If False, threshold is absolute difference.

    Returns:
        WinTieLossResult with counts and sign test p-value.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]

    n = len(x_clean)

    if n == 0:
        return WinTieLossResult(
            wins=0,
            ties=0,
            losses=0,
            n_total=0,
            sign_test_p_value=1.0,
        )

    wins = 0
    ties = 0
    losses = 0

    for xi, yi in zip(x_clean, y_clean):
        if use_relative:
            # Relative difference
            denom = max(abs(xi), abs(yi), 1e-10)
            diff = abs(xi - yi) / denom
        else:
            diff = abs(xi - yi)

        if diff < tie_threshold:
            ties += 1
        elif lower_is_better:
            if xi < yi:
                wins += 1
            else:
                losses += 1
        else:
            if xi > yi:
                wins += 1
            else:
                losses += 1

    # Sign test: under H0, P(win) = P(loss) = 0.5 (ignoring ties)
    n_decisive = wins + losses
    if n_decisive > 0:
        # Two-sided binomial test
        result = stats.binomtest(min(wins, losses), n_decisive, 0.5, alternative="two-sided")
        p_value = result.pvalue
    else:
        p_value = 1.0

    return WinTieLossResult(
        wins=wins,
        ties=ties,
        losses=losses,
        n_total=n,
        sign_test_p_value=p_value,
    )


def pairwise_wilcoxon_tests(
    metric_matrix: np.ndarray,
    algorithm_names: list[str],
    control_idx: int = 0,
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> dict[str, WilcoxonResult]:
    """Perform Wilcoxon tests comparing control to all others with Holm correction.

    Applies Holm's step-down procedure to control the family-wise error rate
    across all pairwise comparisons. This is more powerful than Nemenyi (which
    corrects for all k(k-1)/2 pairs) since only k-1 comparisons are made.

    Args:
        metric_matrix: Shape (n_datasets, n_algorithms).
        algorithm_names: Names for each algorithm.
        control_idx: Index of control algorithm (default: 0, first one).
        lower_is_better: If True, lower metric values are better.
        alpha: Significance level.

    Returns:
        Dict mapping challenger name -> WilcoxonResult with adjusted_p_value
        and reject_null set according to Holm correction.
    """
    control_values = metric_matrix[:, control_idx]

    results = {}
    for j, name in enumerate(algorithm_names):
        if j == control_idx:
            continue

        challenger_values = metric_matrix[:, j]

        if lower_is_better:
            result = wilcoxon_signed_rank_test(control_values, challenger_values, alpha)
        else:
            result = wilcoxon_signed_rank_test(challenger_values, control_values, alpha)

        results[name] = result

    # Apply Holm correction across all pairwise tests
    if results:
        names = list(results.keys())
        raw_p_values = [results[name].p_value for name in names]
        adjusted = holm_adjusted_p_values(raw_p_values)

        for name, adj_p in zip(names, adjusted):
            results[name].adjusted_p_value = adj_p
            results[name].reject_null = adj_p < alpha

    return results


def pairwise_win_tie_loss(
    metric_matrix: np.ndarray,
    algorithm_names: list[str],
    control_idx: int = 0,
    lower_is_better: bool = True,
    tie_threshold: float = 0.01,
) -> dict[str, WinTieLossResult]:
    """Compute Win/Tie/Loss for control vs all others.

    Args:
        metric_matrix: Shape (n_datasets, n_algorithms).
        algorithm_names: Names for each algorithm.
        control_idx: Index of control algorithm.
        lower_is_better: If True, lower metric values are better.
        tie_threshold: Relative threshold for declaring ties.

    Returns:
        Dict mapping challenger name -> WinTieLossResult.
    """
    control_values = metric_matrix[:, control_idx]

    results = {}
    for j, name in enumerate(algorithm_names):
        if j == control_idx:
            continue

        challenger_values = metric_matrix[:, j]
        result = win_tie_loss(control_values, challenger_values, lower_is_better, tie_threshold)
        results[name] = result

    return results


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm's step-down procedure for multiple testing correction.

    Args:
        p_values: List of p-values from pairwise tests.
        alpha: Family-wise error rate.

    Returns:
        List of booleans indicating rejection (True = significant).
    """
    adjusted = holm_adjusted_p_values(p_values)
    return [p <= alpha for p in adjusted]


def holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    """Compute Holm-adjusted p-values for multiple testing correction.

    Uses Holm's step-down procedure. Adjusted p-values maintain the ordering
    of the original p-values while controlling the family-wise error rate.

    Args:
        p_values: List of raw p-values from pairwise tests.

    Returns:
        List of adjusted p-values (same order as input). Compare directly
        against alpha to determine significance.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and keep track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * n
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm adjustment: p * (n - rank)
        adj_p = min(p * (n - rank), 1.0)
        # Enforce monotonicity: adjusted p-values must be non-decreasing in sorted order
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max

    return adjusted
