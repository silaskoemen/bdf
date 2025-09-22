# %%
import numpy as np
import seaborn as sns
from scipy.stats import skewnorm


def simulate_posterior_returns(
    n_days: int = 365,
    n_posterior_distributions: int = 10,
    n_samples_per_distribution: int = 100,
    mean_scale: float = 0.025,
    mean_scale_perturbation: float = 0.005,
    loc_perturbation: float = 0.005,
    loc_noise: float = 0.005,
    skew_min: float = -5,
    skew_max: float = 5,
    skew_scale_perturbation: float = 0.005,
):
    """
    Simulates the posterior distribution of daily log returns for a stock.

    For each day, a true mean is sampled. Then, a set of posterior
    distributions (skew-normal) are generated around this mean, from which
    return samples are drawn.

    Args:
        n_days (int): The number of days to simulate (e.g., 365 for a year).
        n_posterior_distributions (int): The number of posterior distributions
                                         to generate each day.
        n_samples_per_distribution (int): The number of samples to draw from
                                          each posterior distribution.
        user_scale (float): The scale (standard deviation) parameter for the
                            skew-normal distributions.

    Returns:
        tuple[list[float], list[np.ndarray]]: A tuple containing:
            - A list of the true daily mean log returns.
            - A list of arrays, where each array holds the combined posterior
              samples for that day.
    """
    daily_means = []
    posterior_samples = []

    # Parameters for the daily mean distribution (Normal)
    mean_loc = 0.0

    for _ in range(n_days):
        # 1. Sample the true mean for the day from a Normal distribution
        daily_scale = mean_scale + np.random.uniform(-mean_scale_perturbation, mean_scale_perturbation)
        daily_scale = max(daily_scale, 1e-5)  # Ensure scale is positive
        true_daily_mean = np.random.normal(loc=mean_loc, scale=mean_scale)
        daily_means.append(true_daily_mean)
        measured_daily_mean = true_daily_mean + np.random.normal(0, loc_noise)

        day_samples = []
        # 2. Generate posterior distributions and sample from them
        for _ in range(n_posterior_distributions):
            # a. Location varies around the true mean
            loc = measured_daily_mean + np.random.uniform(-loc_perturbation, loc_perturbation)

            # b. Skewness is sampled from a Uniform distribution
            skew = np.random.uniform(skew_min, skew_max)
            skew_scale = daily_scale + np.random.uniform(-skew_scale_perturbation, skew_scale_perturbation)
            skew_scale = max(skew_scale, 1e-5)  # Ensure scale is positive
            # c. Draw samples from the resulting skew-normal distribution
            samples = skewnorm.rvs(a=skew, loc=loc, scale=skew_scale, size=n_samples_per_distribution)
            day_samples.extend(samples)

        posterior_samples.append(np.array(day_samples))

    return daily_means, posterior_samples


def print_verify_simulation(means, posteriors, mean_scale, n_days):
    print(f"Simulation finished for {n_days} days.")
    print(f"Scale parameter used: {mean_scale}")
    print("-" * 30)

    # Verify the dimensions of the output
    print(f"Number of daily means generated: {len(means)}")
    print(f"Number of daily posterior sample sets: {len(posteriors)}")
    if len(posteriors) > 0:
        print(f"Samples generated for Day 1: {len(posteriors[0])}")

    # Print statistics for the first day as an example
    if n_days > 0:
        print("\n--- Example: Day 1 Statistics ---")
        print(f"True Mean: {means[0]:.4f}")
        print(f"Posterior Sample Mean: {np.mean(posteriors[0]):.4f}")
        print(f"Posterior Sample Std Dev: {np.std(posteriors[0]):.4f}")
        print(f"Posterior Sample Min: {np.min(posteriors[0]):.4f}")
        print(f"Posterior Sample Max: {np.max(posteriors[0]):.4f}")

    # Optional: Plotting the distribution for the first day
    try:
        import matplotlib.pyplot as plt

        if n_days > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(posteriors[0], kde=True, bins=50)
            plt.axvline(means[0], color="r", linestyle="--", label=f"True Mean: {means[0]:.4f}")
            plt.title("Posterior Distribution of Log Returns for Day 1")
            plt.xlabel("Log Return")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    except ImportError:
        print("\nMatplotlib and Seaborn not found. Skipping plot.")


# %%
# --- Configuration ---
N_DAYS = 1000
MEAN_SCALE: float = 0.01
MEAN_SCALE_PERTURBATION: float = 0.005
LOC_PERTURBATION: float = 0.005
LOC_NOISE: float = 0.01
SKEW_MIN: float = -5
SKEW_MAX: float = 5
SKEW_SCALE_PERTURBATION: float = 0.005

# --- Simulation ---
means, posteriors = simulate_posterior_returns(
    n_days=N_DAYS,
    mean_scale=MEAN_SCALE,
    mean_scale_perturbation=MEAN_SCALE_PERTURBATION,
    loc_perturbation=LOC_PERTURBATION,
    loc_noise=LOC_NOISE,
    skew_min=SKEW_MIN,
    skew_max=SKEW_MAX,
    skew_scale_perturbation=SKEW_SCALE_PERTURBATION,
)

# --- Output & Verification ---
print_verify_simulation(means=means, posteriors=posteriors, mean_scale=MEAN_SCALE, n_days=N_DAYS)
# %%
# Investigate which decision metric leads to the best trading performance
# Options:
# - Mean of posterior samples
# - Median of posterior samples
# - Sharpe ratio of posterior samples
# - Sortino ratio of posterior samples
# - Mean divided by probability of negative returns


def trade_mean(posterior_samples, threshold=0.0) -> bool:
    """
    Trades based on the mean of posterior samples.

    Args:
        posterior_samples (list[np.ndarray]): List of arrays containing posterior samples.
        threshold (float): Threshold for making a trade decision.

    Returns:
        int: Number of trades made.
    """
    return np.mean(posterior_samples) > threshold


def trade_median(posterior_samples, threshold=0.0) -> bool:
    """
    Trades based on the median of posterior samples.

    Args:
        posterior_samples (list[np.ndarray]): List of arrays containing posterior samples.
        threshold (float): Threshold for making a trade decision.

    Returns:
        int: Number of trades made.
    """
    return np.median(posterior_samples) > threshold


def trade_sharpe(posterior_samples, risk_free_rate=0.0, threshold=0.0) -> bool:
    """
    Trades based on the Sharpe ratio of posterior samples.

    Args:
        posterior_samples (list[np.ndarray]): List of arrays containing posterior samples.
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
        threshold (float): Threshold for making a trade decision.

    Returns:
        int: Number of trades made.
    """
    mean_return = np.mean(posterior_samples)
    std_dev = np.std(posterior_samples)
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev > 0 else 0
    return sharpe_ratio > threshold


def trade_sortino(posterior_samples, risk_free_rate=0.0, threshold=0.0) -> bool:
    """
    Trades based on the Sortino ratio of posterior samples.

    Args:
        posterior_samples (list[np.ndarray]): List of arrays containing posterior samples.
        risk_free_rate (float): Risk-free rate for Sortino ratio calculation.
        threshold (float): Threshold for making a trade decision.

    Returns:
        int: Number of trades made.
    """
    mean_return = np.mean(posterior_samples)
    downside_returns = np.array([x for x in posterior_samples if x < risk_free_rate])
    if len(downside_returns) < 2:
        downside_std_dev = 1e-5
    else:
        downside_std_dev = np.std(downside_returns)
    sortino_ratio = (mean_return - risk_free_rate) / downside_std_dev
    return sortino_ratio > threshold


def trade_mean_prob_negative(posterior_samples, threshold=0.0) -> bool:
    """
    Trades based on the mean of posterior samples divided by the probability of negative returns.

    Args:
        posterior_samples (list[np.ndarray]): List of arrays containing posterior samples.
        threshold (float): Threshold for making a trade decision.

    Returns:
        int: Number of trades made.
    """
    mean_return = np.mean(posterior_samples)
    prob_negative = np.mean(np.array(posterior_samples) < 0)
    return (mean_return / (prob_negative + 1e-8)) > threshold


def trade_mean_prob_positive(posterior_samples, threshold=0.0) -> bool:
    """
    Trades based on the mean of posterior samples divided by the probability of positive returns.

    Args:
        posterior_samples (list[np.ndarray]): List of arrays containing posterior samples.
        threshold (float): Threshold for making a trade decision.

    Returns:
        int: Number of trades made.
    """
    mean_return = np.mean(posterior_samples)
    prob_positive = np.mean(np.array(posterior_samples) > 0)
    return (mean_return * (prob_positive + 1e-8)) > threshold


def evaluate_trading_strategy(strategy_func, posteriors, returns, plot=False, **kwargs):
    """
    Evaluates a trading strategy based on posterior samples.

    Args:
        strategy_func (function): The trading strategy function to evaluate.
        posteriors (list[np.ndarray]): List of arrays containing posterior samples.
        returns (list[float]): List of actual returns for each day.
        threshold (float): Threshold for making a trade decision.

    Returns:
        int: Number of trades made.
    """
    log_returns = []
    trading_days = 0
    for day_samples, rtn in zip(posteriors, returns):
        if strategy_func(day_samples, **kwargs):
            log_returns.append(rtn)
            trading_days += 1
        else:
            log_returns.append(0.0)
    # Plot cumulative returns, meaning np.exp -1 of cum sum of returns
    if plot:
        cumsum_log_returns = np.cumsum(np.array(log_returns))
        # Convert to real returns
        cumsum_returns = np.exp(cumsum_log_returns) - 1
        # Plot cumulative returns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(100 * cumsum_returns, label=f"{strategy_func.__name__} Cumulative Returns (%)")
        plt.title(f"Cumulative Returns using {strategy_func.__name__}")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        print(f"{trading_days} trades out of {len(posteriors)} days.")
    return np.exp(np.sum(log_returns)) - 1  # Convert log returns to actual returns


# %%
# Example usage of the trading strategies
for raw_strategy in [trade_mean, trade_median]:
    print("\n", "#" * 20, f"Evaluating strategy: {raw_strategy.__name__}", "#" * 20)
    for threshold in [0.0, 0.005, 0.01, 0.015, 0.02]:
        if threshold == 0.0:
            total_return = evaluate_trading_strategy(raw_strategy, posteriors, means, threshold=threshold, plot=True)
        total_return = evaluate_trading_strategy(raw_strategy, posteriors, means, threshold=threshold)
        print(f"Total return using {raw_strategy.__name__} with {threshold =}: {100*total_return:.2f}%")

for agg_strategy in [trade_sharpe, trade_sortino, trade_mean_prob_negative, trade_mean_prob_positive]:
    print("\n", "#" * 20, f"Evaluating strategy: {agg_strategy.__name__}", "#" * 20)
    for threshold in np.arange(0.0, 0.5, 0.01):
        if threshold == 0.0:
            total_return = evaluate_trading_strategy(agg_strategy, posteriors, means, threshold=threshold, plot=True)
        else:
            total_return = evaluate_trading_strategy(agg_strategy, posteriors, means, threshold=threshold)
        print(f"Total return using {agg_strategy.__name__} with {threshold = :.3f}: {100*total_return:.2f}%")
# %%
