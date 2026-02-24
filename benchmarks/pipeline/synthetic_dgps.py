"""Synthetic Data Generating Processes (DGPs) for rigorous benchmarking.

This module provides JMLR-quality synthetic datasets with known ground truth for evaluating
distributional regression methods. Each DGP includes:
- Feature generation (X)
- Target generation with known conditional distribution p(y|x)
- Ground truth functions: mean(x), variance(x), quantiles(x)
- Metadata for reproducibility and analysis

All DGPs are designed to test specific capabilities of distributional methods:
- Heteroscedasticity: Variance changes with covariates
- Discontinuities: Step functions and change points
- Multimodality: Mixture distributions
- Heavy tails: Non-Gaussian error distributions
- Sparsity: Uneven sampling in feature space
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
from scipy.stats import expon, norm, poisson, skewnorm
from scipy.stats import t as student_t


class DGPType(Enum):
    """Categories of data generating processes."""

    HETEROSCEDASTIC = "heteroscedastic"
    STEP_FUNCTION = "step_function"
    BIMODAL_MIXTURE = "bimodal_mixture"
    HEAVY_TAILED = "heavy_tailed"
    SPARSE_SAMPLING = "sparse_sampling"
    GAUSSIAN_HETEROSCEDASTIC = "gaussian_heteroscedastic"
    POISSON_COUNT = "poisson_count"
    EXPONENTIAL_WAITING_TIME = "exponential_waiting_time"
    MULTIMODAL_MIXTURE_FIXED = "multimodal_mixture_fixed"
    SKEWED_HETEROSCEDASTIC = "skewed_heteroscedastic"


@dataclass
class GroundTruthFunctions:
    """Ground truth functions for a DGP.

    These functions allow computing metrics against the true conditional distribution.
    """

    mean_fn: Callable[[np.ndarray], np.ndarray]  # E[Y|X=x]
    variance_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None  # Var[Y|X=x]
    quantile_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None  # Q_τ[Y|X=x]
    density_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None  # p(y|X=x)
    sample_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None  # Sample from p(Y|X=x)


@dataclass
class SyntheticDataset:
    """Container for synthetic dataset with ground truth."""

    X: np.ndarray  # Features (n_samples, n_features)
    y: np.ndarray  # Targets (n_samples,)
    dgp_type: DGPType  # Type of DGP
    name: str  # Human-readable name
    ground_truth: GroundTruthFunctions  # Ground truth functions
    seed: int  # Random seed for reproducibility
    n_features: int  # Total number of features
    n_informative: int  # Number of informative features
    noise_level: float  # Relative noise level (for applicable DGPs)
    metadata: dict  # Additional metadata


# =============================================================================
# DGP 1: Heteroscedastic Sinusoidal
# =============================================================================


def generate_heteroscedastic_sinusoidal(
    n_samples: int = 5000,
    seed: int = 42,
    noise_scale: float = 0.1,
) -> SyntheticDataset:
    """Generate data from y = sin(x) + N(0, σ²(x)) where variance changes with x.

    The variance function σ²(x) increases at x = π:
    - For x < π: σ(x) = noise_scale * sqrt(|x|)
    - For x ≥ π: σ(x) = noise_scale * sqrt(4|x|)

    This tests whether models can:
    1. Capture nonlinear mean function
    2. Adapt uncertainty estimates to local variance
    3. Detect variance change points

    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        noise_scale: Scaling factor for heteroscedastic noise

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 2 * np.pi, size=(n_samples, 1))

    # Ground truth mean and variance functions
    def mean_fn(x):
        return np.sin(x)

    def variance_fn(x):
        var = np.where(x < np.pi, noise_scale**2 * np.abs(x), noise_scale**2 * 4 * np.abs(x))
        return var

    def std_fn(x):
        return np.sqrt(variance_fn(x))

    def quantile_fn(x, tau):
        """Quantile function for normal distribution."""
        return mean_fn(x) + std_fn(x) * norm.ppf(tau)

    def density_fn(y, x):
        """Conditional density p(y|x)."""
        return norm.pdf(y, loc=mean_fn(x), scale=std_fn(x))

    def sample_fn(x, n_samples_per_x):
        """Sample from conditional distribution."""
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            samples[i] = norm.rvs(loc=mean_fn(x[i]), scale=std_fn(x[i]), size=n_samples_per_x, random_state=seed + i)
        return samples

    # Generate y with heteroscedastic noise
    y_mean = mean_fn(X)
    y_std = std_fn(X)
    y = y_mean.squeeze() + rng.randn(n_samples) * y_std.squeeze()

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.HETEROSCEDASTIC,
        name="heteroscedastic_sinusoidal",
        ground_truth=ground_truth,
        seed=seed,
        n_features=1,
        n_informative=1,
        noise_level=noise_scale,
        metadata={
            "variance_change_point": np.pi,
            "variance_ratio": 4.0,  # σ²(x≥π) / σ²(x<π)
            "mean_function": "sin(x)",
        },
    )


# =============================================================================
# DGP 2: Step Function with Constant Variance
# =============================================================================


def generate_step_function(
    n_samples: int = 5000,
    seed: int = 42,
    noise_std: float = 0.1,
    n_steps: int = 5,
) -> SyntheticDataset:
    """Generate piecewise constant function with discontinuities.

    The mean function is:
    y = step_height * floor(k * x / (2π)) + N(0, σ²)

    where k is the number of steps and step_height varies per step.

    This tests whether models can:
    1. Detect discontinuities
    2. Maintain appropriate uncertainty near change points
    3. Avoid over-smoothing

    Args:
        n_samples: Number of samples
        seed: Random seed
        noise_std: Standard deviation of Gaussian noise
        n_steps: Number of discrete steps in the function

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 2 * np.pi, size=(n_samples, 1))

    # Define step heights (deterministic for reproducibility)
    step_heights = np.array([0.0, 1.0, -0.5, 1.5, 0.5])[:n_steps]
    boundaries = np.linspace(0, 2 * np.pi, n_steps + 1)

    def mean_fn(x):
        """Piecewise constant mean function."""
        x_flat = x.flatten()
        result = np.zeros_like(x_flat)
        for i in range(n_steps):
            mask = (x_flat >= boundaries[i]) & (x_flat < boundaries[i + 1])
            result[mask] = step_heights[i]
        # Handle right endpoint
        result[x_flat >= boundaries[-1]] = step_heights[-1]
        return result.reshape(x.shape)

    def variance_fn(x):
        """Constant variance."""
        return np.full_like(x, noise_std**2)

    def quantile_fn(x, tau):
        """Quantile function for normal distribution."""
        return mean_fn(x) + noise_std * norm.ppf(tau)

    def density_fn(y, x):
        """Conditional density p(y|x)."""
        return norm.pdf(y, loc=mean_fn(x), scale=noise_std)

    def sample_fn(x, n_samples_per_x):
        """Sample from conditional distribution."""
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            samples[i] = norm.rvs(loc=mean_fn(x[i]), scale=noise_std, size=n_samples_per_x, random_state=seed + i)
        return samples

    # Generate y
    y_mean = mean_fn(X)
    y = y_mean.squeeze() + rng.randn(n_samples) * noise_std

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.STEP_FUNCTION,
        name=f"step_function_k{n_steps}",
        ground_truth=ground_truth,
        seed=seed,
        n_features=1,
        n_informative=1,
        noise_level=noise_std,
        metadata={
            "n_steps": n_steps,
            "step_heights": step_heights.tolist(),
            "boundaries": boundaries.tolist(),
        },
    )


# =============================================================================
# DGP 3: Bimodal Mixture with Input-Dependent Mixing
# =============================================================================


def generate_bimodal_mixture(
    n_samples: int = 5000,
    seed: int = 42,
    noise_std: float = 0.1,
    separation: float = 2.0,
) -> SyntheticDataset:
    """Generate mixture of two linear models with input-dependent mixing probability.

    The conditional distribution is:
    p(y|x) = π(x) N(y; x + c₁, σ²) + (1-π(x)) N(y; x + c₂, σ²)

    where π(x) = 0.1 + 0.8x for x ∈ [0, 1], and c₁ = 0, c₂ = separation.

    This tests whether models can:
    1. Capture multimodal distributions
    2. Adapt mixture weights with covariates
    3. Provide uncertainty that reflects multiple modes

    Args:
        n_samples: Number of samples
        seed: Random seed
        noise_std: Standard deviation within each component
        separation: Distance between the two modes

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, 1))

    def mixing_prob(x):
        """Probability of first component: π(x) = 0.1 + 0.8x."""
        return 0.1 + 0.8 * x

    def mean_fn(x):
        """Expected value: E[Y|X=x] = π(x)(x + 0) + (1-π(x))(x + separation)."""
        pi_x = mixing_prob(x)
        return pi_x * x + (1 - pi_x) * (x + separation)

    def variance_fn(x):
        """Variance: Var[Y|X=x] = π(x)σ² + (1-π(x))σ² + π(x)(1-π(x))separation²."""
        pi_x = mixing_prob(x)
        # Variance within components
        within_var = noise_std**2
        # Variance between components
        between_var = pi_x * (1 - pi_x) * separation**2
        return within_var + between_var

    def quantile_fn(x, tau):
        """Approximate quantile by inverting mixture CDF numerically."""
        # For simplicity, we'll use a grid-based approximation
        # More sophisticated: use numerical root-finding
        y_grid = np.linspace(x.min() - separation, x.max() + 2 * separation, 1000)
        cdf_vals = np.zeros_like(y_grid)

        pi_x = mixing_prob(x)
        for i, y_val in enumerate(y_grid):
            cdf_vals[i] = pi_x * norm.cdf(y_val, loc=x, scale=noise_std) + (1 - pi_x) * norm.cdf(
                y_val, loc=x + separation, scale=noise_std
            )

        # Find quantile
        idx = np.searchsorted(cdf_vals, tau)
        return y_grid[min(idx, len(y_grid) - 1)]

    def density_fn(y, x):
        """Mixture density."""
        pi_x = mixing_prob(x)
        component1 = norm.pdf(y, loc=x, scale=noise_std)
        component2 = norm.pdf(y, loc=x + separation, scale=noise_std)
        return pi_x * component1 + (1 - pi_x) * component2

    def sample_fn(x, n_samples_per_x):
        """Sample from mixture distribution."""
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            pi_x_i = mixing_prob(x[i])
            # Sample component indicators
            component = rng.binomial(1, pi_x_i, size=n_samples_per_x)
            # Sample from respective components
            samples[i] = np.where(
                component,
                norm.rvs(loc=x[i, 0], scale=noise_std, size=n_samples_per_x, random_state=seed + i),
                norm.rvs(loc=x[i, 0] + separation, scale=noise_std, size=n_samples_per_x, random_state=seed + i + 1),
            )
        return samples

    # Generate y from mixture
    y = np.zeros(n_samples)
    for i in range(n_samples):
        pi_x = mixing_prob(X[i])
        if rng.rand() < pi_x:
            # First component
            y[i] = X[i, 0] + rng.randn() * noise_std
        else:
            # Second component
            y[i] = X[i, 0] + separation + rng.randn() * noise_std

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.BIMODAL_MIXTURE,
        name="bimodal_mixture",
        ground_truth=ground_truth,
        seed=seed,
        n_features=1,
        n_informative=1,
        noise_level=noise_std,
        metadata={
            "separation": separation,
            "mixing_fn": "π(x) = 0.1 + 0.8x",
            "component1_mean": "x",
            "component2_mean": f"x + {separation}",
        },
    )


# =============================================================================
# DGP 4: Heavy-Tailed Noise
# =============================================================================


def generate_heavy_tailed(
    n_samples: int = 5000,
    seed: int = 42,
    df: float = 3.0,
    scale: float = 0.5,
) -> SyntheticDataset:
    """Generate data with Student-t distributed noise (heavy tails).

    The model is:
    y = x₁² + x₂ + t_df * scale

    where t_df is Student-t with df degrees of freedom.

    This tests whether models can:
    1. Handle outliers robustly
    2. Provide wider uncertainty estimates for heavy-tailed data
    3. Capture tail behavior accurately

    Args:
        n_samples: Number of samples
        seed: Random seed
        df: Degrees of freedom for Student-t (lower = heavier tails)
        scale: Scale parameter for noise

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, size=(n_samples, 2))

    def mean_fn(x):
        """E[Y|X=x] = x₁² + x₂."""
        return x[:, 0] ** 2 + x[:, 1]

    def variance_fn(x):
        """Var[Y|X=x] = scale² * df/(df-2) for df > 2."""
        if df > 2:
            return np.full(x.shape[0], scale**2 * df / (df - 2))
        else:
            return np.full(x.shape[0], np.inf)  # Variance undefined for df ≤ 2

    def quantile_fn(x, tau):
        """Quantile function for location-scale t distribution."""
        return mean_fn(x) + scale * student_t.ppf(tau, df)

    def density_fn(y, x):
        """Conditional density p(y|x)."""
        return student_t.pdf((y - mean_fn(x)) / scale, df) / scale

    def sample_fn(x, n_samples_per_x):
        """Sample from conditional distribution."""
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            samples[i] = mean_fn(x[i : i + 1]) + scale * student_t.rvs(df, size=n_samples_per_x, random_state=seed + i)
        return samples

    # Generate y with Student-t noise
    y_mean = mean_fn(X)
    noise = student_t.rvs(df, size=n_samples, scale=scale, random_state=seed)
    y = y_mean + noise

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.HEAVY_TAILED,
        name=f"heavy_tailed_df{df}",
        ground_truth=ground_truth,
        seed=seed,
        n_features=2,
        n_informative=2,
        noise_level=scale,
        metadata={
            "df": df,
            "scale": scale,
            "mean_function": "x₁² + x₂",
            "noise_distribution": f"Student-t(df={df})",
        },
    )


# =============================================================================
# DGP 5: Sparse Sampling (Heterogeneous Density in X)
# =============================================================================


def generate_sparse_sampling(
    n_samples: int = 1000,
    seed: int = 42,
    noise_std: float = 0.1,
    sparsity_type: str = "linear_decay",
) -> SyntheticDataset:
    """Generate data with non-uniform sampling density in feature space.

    The underlying function is y = sin(x) + N(0, σ²), but x is sampled
    non-uniformly. This creates regions with sparse data where uncertainty
    should be higher.

    Sparsity types:
    - "linear_decay": p(x) ∝ 1 - x/(2π), dense at x=0, sparse at x=2π
    - "gaps": p(x) has explicit gaps/holes

    This tests whether models:
    1. Increase uncertainty in sparse regions
    2. Avoid overfitting in dense regions
    3. Handle unbalanced data distribution

    Args:
        n_samples: Number of samples (actual may be less due to rejection)
        seed: Random seed
        noise_std: Standard deviation of Gaussian noise
        sparsity_type: Type of sparsity pattern

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)

    if sparsity_type == "linear_decay":
        # Rejection sampling with acceptance probability 1 - x/(2π)
        x_vals = []
        while len(x_vals) < n_samples:
            trial_x = rng.uniform(0, 2 * np.pi)
            accept_prob = 1 - trial_x / (2 * np.pi)
            if accept_prob > rng.uniform():
                x_vals.append(trial_x)
        X = np.array(x_vals).reshape(-1, 1)

    elif sparsity_type == "gaps":
        # Sample from [0, π/2] ∪ [3π/2, 2π], leaving gap at [π/2, 3π/2]
        n_left = n_samples // 2
        n_right = n_samples - n_left
        x_left = rng.uniform(0, np.pi / 2, size=n_left)
        x_right = rng.uniform(3 * np.pi / 2, 2 * np.pi, size=n_right)
        X = np.concatenate([x_left, x_right]).reshape(-1, 1)

    else:
        raise ValueError(f"Unknown sparsity_type: {sparsity_type}")

    n_actual = X.shape[0]

    def mean_fn(x):
        """E[Y|X=x] = sin(x)."""
        return np.sin(x)

    def variance_fn(x):
        """Homoscedastic variance."""
        return np.full_like(x, noise_std**2)

    def quantile_fn(x, tau):
        """Quantile function."""
        return mean_fn(x) + noise_std * norm.ppf(tau)

    def density_fn(y, x):
        """Conditional density."""
        return norm.pdf(y, loc=mean_fn(x), scale=noise_std)

    def sample_fn(x, n_samples_per_x):
        """Sample from conditional distribution."""
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            samples[i] = norm.rvs(loc=mean_fn(x[i]), scale=noise_std, size=n_samples_per_x, random_state=seed + i)
        return samples

    # Generate y
    y_mean = mean_fn(X)
    y = y_mean.squeeze() + rng.randn(n_actual) * noise_std

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.SPARSE_SAMPLING,
        name=f"sparse_{sparsity_type}",
        ground_truth=ground_truth,
        seed=seed,
        n_features=1,
        n_informative=1,
        noise_level=noise_std,
        metadata={
            "sparsity_type": sparsity_type,
            "mean_function": "sin(x)",
            "sampling_density": "non-uniform (see sparsity_type)",
        },
    )


# =============================================================================
# DGP 6: Gaussian Heteroscedastic (home for NormalMuNormal)
# =============================================================================


def generate_gaussian_heteroscedastic(
    n_samples: int = 5000,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate data from a heteroscedastic Gaussian model.

    The model is:
        y = 2*x1 + sin(3*x2) + N(0, sigma(x)^2)
        sigma(x) = 0.3 + 0.2*|x1|

    This is the "home" DGP for NormalMuNormal in the misspecification study.

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, size=(n_samples, 2))

    def mean_fn(x):
        """E[Y|X=x] = 2*x1 + sin(3*x2)."""
        return 2 * x[:, 0] + np.sin(3 * x[:, 1])

    def std_fn(x):
        """sigma(x) = 0.3 + 0.2*|x1|."""
        return 0.3 + 0.2 * np.abs(x[:, 0])

    def variance_fn(x):
        return std_fn(x) ** 2

    def quantile_fn(x, tau):
        return mean_fn(x) + std_fn(x) * norm.ppf(tau)

    def density_fn(y, x):
        return norm.pdf(y, loc=mean_fn(x), scale=std_fn(x))

    def sample_fn(x, n_samples_per_x):
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            samples[i] = norm.rvs(
                loc=mean_fn(x[i : i + 1]),
                scale=std_fn(x[i : i + 1]),
                size=n_samples_per_x,
                random_state=seed + i,
            )
        return samples

    y_mean = mean_fn(X)
    y_std = std_fn(X)
    y = y_mean + rng.randn(n_samples) * y_std

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.GAUSSIAN_HETEROSCEDASTIC,
        name="gaussian_heteroscedastic",
        ground_truth=ground_truth,
        seed=seed,
        n_features=2,
        n_informative=2,
        noise_level=0.3,
        metadata={
            "mean_function": "2*x1 + sin(3*x2)",
            "std_function": "0.3 + 0.2*|x1|",
            "noise_distribution": "Normal",
        },
    )


# =============================================================================
# DGP 7: Poisson Count (home for GammaMVLambdaPoisson)
# =============================================================================


def generate_poisson_count(
    n_samples: int = 5000,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate count data from a Poisson model with log-linear rate.

    The model is:
        log(lambda(x)) = 0.5 + 0.8*x1 + 0.3*x2
        y | x ~ Poisson(lambda(x))

    Lambda ranges from ~1.6 (x near 0) to ~45 (x near 3).
    This is the "home" DGP for GammaMVLambdaPoisson in the misspecification study.

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 3, size=(n_samples, 2))

    def rate_fn(x):
        """lambda(x) = exp(0.5 + 0.8*x1 + 0.3*x2)."""
        return np.exp(0.5 + 0.8 * x[:, 0] + 0.3 * x[:, 1])

    def mean_fn(x):
        return rate_fn(x)

    def variance_fn(x):
        return rate_fn(x)  # Poisson: variance = mean

    def quantile_fn(x, tau):
        return poisson.ppf(tau, mu=rate_fn(x))

    def density_fn(y, x):
        return poisson.pmf(np.round(y).astype(int), mu=rate_fn(x))

    def sample_fn(x, n_samples_per_x):
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            lam = rate_fn(x[i : i + 1])[0]
            samples[i] = poisson.rvs(mu=lam, size=n_samples_per_x, random_state=seed + i)
        return samples

    lam = rate_fn(X)
    y = rng.poisson(lam).astype(float)

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.POISSON_COUNT,
        name="poisson_count",
        ground_truth=ground_truth,
        seed=seed,
        n_features=2,
        n_informative=2,
        noise_level=0.0,
        metadata={
            "rate_function": "exp(0.5 + 0.8*x1 + 0.3*x2)",
            "noise_distribution": "Poisson",
            "lambda_range": "~1.6 to ~45",
        },
    )


# =============================================================================
# DGP 8: Exponential Waiting Time (home for GammaMVLambdaExponential)
# =============================================================================


def generate_exponential_waiting_time(
    n_samples: int = 5000,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate waiting-time data from an Exponential model with linear rate.

    The model is:
        rate(x) = 0.5 + 0.3*x1 + 0.2*x2
        y | x ~ Exponential(rate=rate(x))     [mean = 1/rate(x)]

    X in [0.5, 3]^2, so rate ranges from ~0.75 to ~1.9, mean from ~0.5 to ~1.3.
    All y values are strictly positive.
    This is the "home" DGP for GammaMVLambdaExponential in the misspecification study.

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.5, 3, size=(n_samples, 2))

    def rate_fn(x):
        """rate(x) = 0.5 + 0.3*x1 + 0.2*x2."""
        return 0.5 + 0.3 * x[:, 0] + 0.2 * x[:, 1]

    def mean_fn(x):
        return 1.0 / rate_fn(x)

    def variance_fn(x):
        return 1.0 / rate_fn(x) ** 2

    def quantile_fn(x, tau):
        return expon.ppf(tau, scale=1.0 / rate_fn(x))

    def density_fn(y, x):
        return expon.pdf(y, scale=1.0 / rate_fn(x))

    def sample_fn(x, n_samples_per_x):
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            scale = 1.0 / rate_fn(x[i : i + 1])[0]
            samples[i] = expon.rvs(scale=scale, size=n_samples_per_x, random_state=seed + i)
        return samples

    rates = rate_fn(X)
    y = rng.exponential(scale=1.0 / rates)

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.EXPONENTIAL_WAITING_TIME,
        name="exponential_waiting_time",
        ground_truth=ground_truth,
        seed=seed,
        n_features=2,
        n_informative=2,
        noise_level=0.0,
        metadata={
            "rate_function": "0.5 + 0.3*x1 + 0.2*x2",
            "noise_distribution": "Exponential",
            "rate_range": "~0.75 to ~1.9",
        },
    )


# =============================================================================
# DGP 9: Multimodal Mixture with Fixed Weights (home for KDE)
# =============================================================================


def generate_multimodal_mixture_fixed(
    n_samples: int = 5000,
    seed: int = 42,
    sigma: float = 0.2,
) -> SyntheticDataset:
    """Generate bimodal mixture data with equal weights and oscillating separation.

    The model is:
        y | x ~ 0.5 * N(2x, sigma^2) + 0.5 * N(2x + 2*sin(2*pi*x), sigma^2)

    The second mode oscillates relative to the first, creating varying separation.
    No parametric BDF distribution can capture this — KDE is the natural fit.
    This is the "home" DGP for KDE in the misspecification study.

    Args:
        n_samples: Number of samples
        seed: Random seed
        sigma: Standard deviation within each component

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, 1))

    def mu1_fn(x):
        """First mode center: 2x."""
        return 2 * x.flatten()

    def mu2_fn(x):
        """Second mode center: 2x + 2*sin(2*pi*x)."""
        return 2 * x.flatten() + 2 * np.sin(2 * np.pi * x.flatten())

    def mean_fn(x):
        """E[Y|X=x] = 0.5*(mu1 + mu2)."""
        return 0.5 * (mu1_fn(x) + mu2_fn(x))

    def variance_fn(x):
        """Var[Y|X=x] = sigma^2 + 0.25*(mu1 - mu2)^2."""
        diff = mu1_fn(x) - mu2_fn(x)
        return np.full_like(diff, sigma**2) + 0.25 * diff**2

    def quantile_fn(x, tau):
        """Numerically invert mixture CDF."""
        m1 = mu1_fn(x)
        m2 = mu2_fn(x)
        # Use bisection on a fine grid
        y_low = np.minimum(m1, m2) - 5 * sigma
        y_high = np.maximum(m1, m2) + 5 * sigma
        y_grid = np.linspace(y_low, y_high, 2000).T  # (n_x, 2000)
        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            cdf_vals = 0.5 * norm.cdf(y_grid[i], loc=m1[i], scale=sigma) + 0.5 * norm.cdf(
                y_grid[i], loc=m2[i], scale=sigma
            )
            idx = np.searchsorted(cdf_vals, tau)
            idx = min(idx, len(y_grid[i]) - 1)
            result[i] = y_grid[i, idx]
        return result

    def density_fn(y, x):
        """Mixture density."""
        m1 = mu1_fn(x)
        m2 = mu2_fn(x)
        return 0.5 * norm.pdf(y, loc=m1, scale=sigma) + 0.5 * norm.pdf(y, loc=m2, scale=sigma)

    def sample_fn(x, n_samples_per_x):
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            m1 = mu1_fn(x[i : i + 1])[0]
            m2 = mu2_fn(x[i : i + 1])[0]
            component = rng.binomial(1, 0.5, size=n_samples_per_x)
            samples[i] = np.where(
                component,
                norm.rvs(loc=m1, scale=sigma, size=n_samples_per_x, random_state=seed + i),
                norm.rvs(loc=m2, scale=sigma, size=n_samples_per_x, random_state=seed + i + n_samples),
            )
        return samples

    # Generate y from mixture
    y = np.zeros(n_samples)
    for i in range(n_samples):
        m1 = mu1_fn(X[i : i + 1])[0]
        m2 = mu2_fn(X[i : i + 1])[0]
        if rng.rand() < 0.5:
            y[i] = m1 + rng.randn() * sigma
        else:
            y[i] = m2 + rng.randn() * sigma

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.MULTIMODAL_MIXTURE_FIXED,
        name="multimodal_mixture_fixed",
        ground_truth=ground_truth,
        seed=seed,
        n_features=1,
        n_informative=1,
        noise_level=sigma,
        metadata={
            "mixing_weight": 0.5,
            "mu1_function": "2x",
            "mu2_function": "2x + 2*sin(2*pi*x)",
            "component_sigma": sigma,
            "noise_distribution": "Gaussian mixture",
        },
    )


# =============================================================================
# DGP 10: Skewed Heteroscedastic (home for SkewNormal)
# =============================================================================


def generate_skewed_heteroscedastic(
    n_samples: int = 5000,
    seed: int = 42,
    alpha: float = 5.0,
) -> SyntheticDataset:
    """Generate data from a skew-normal model with input-dependent location and scale.

    The model is:
        y | x ~ SkewNormal(alpha, xi(x), omega(x))
        xi(x) = 1.5*x1 + 0.5*sin(2*x2)
        omega(x) = 0.3 + 0.15*|x1|

    The skewness parameter alpha is constant across the feature space, producing
    right-skewed residuals at every point. This is the "home" DGP for SkewNormal
    in the misspecification study.

    Args:
        n_samples: Number of samples
        seed: Random seed
        alpha: Skewness parameter (positive = right-skewed)

    Returns:
        SyntheticDataset with ground truth functions
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, size=(n_samples, 2))

    delta = alpha / np.sqrt(1 + alpha**2)

    def xi_fn(x):
        """Location parameter xi(x)."""
        return 1.5 * x[:, 0] + 0.5 * np.sin(2 * x[:, 1])

    def omega_fn(x):
        """Scale parameter omega(x)."""
        return 0.3 + 0.15 * np.abs(x[:, 0])

    def mean_fn(x):
        """E[Y|X=x] = xi + omega * delta * sqrt(2/pi)."""
        return xi_fn(x) + omega_fn(x) * delta * np.sqrt(2 / np.pi)

    def variance_fn(x):
        """Var[Y|X=x] = omega^2 * (1 - 2*delta^2/pi)."""
        return omega_fn(x) ** 2 * (1 - 2 * delta**2 / np.pi)

    def quantile_fn(x, tau):
        """Quantile function for skew-normal."""
        return skewnorm.ppf(tau, a=alpha, loc=xi_fn(x), scale=omega_fn(x))

    def density_fn(y, x):
        """Conditional density p(y|x)."""
        return skewnorm.pdf(y, a=alpha, loc=xi_fn(x), scale=omega_fn(x))

    def sample_fn(x, n_samples_per_x):
        """Sample from conditional distribution."""
        n_x = x.shape[0]
        samples = np.zeros((n_x, n_samples_per_x))
        for i in range(n_x):
            samples[i] = skewnorm.rvs(
                a=alpha,
                loc=xi_fn(x[i : i + 1])[0],
                scale=omega_fn(x[i : i + 1])[0],
                size=n_samples_per_x,
                random_state=seed + i,
            )
        return samples

    # Generate y
    xi = xi_fn(X)
    omega = omega_fn(X)
    y = skewnorm.rvs(a=alpha, loc=xi, scale=omega, random_state=seed)

    ground_truth = GroundTruthFunctions(
        mean_fn=mean_fn,
        variance_fn=variance_fn,
        quantile_fn=quantile_fn,
        density_fn=density_fn,
        sample_fn=sample_fn,
    )

    return SyntheticDataset(
        X=X,
        y=y,
        dgp_type=DGPType.SKEWED_HETEROSCEDASTIC,
        name="skewed_heteroscedastic",
        ground_truth=ground_truth,
        seed=seed,
        n_features=2,
        n_informative=2,
        noise_level=0.3,
        metadata={
            "alpha": alpha,
            "xi_function": "1.5*x1 + 0.5*sin(2*x2)",
            "omega_function": "0.3 + 0.15*|x1|",
            "noise_distribution": f"SkewNormal(alpha={alpha})",
        },
    )


# =============================================================================
# DGP Registry
# =============================================================================

DGP_REGISTRY = {
    "heteroscedastic_sinusoidal": generate_heteroscedastic_sinusoidal,
    "step_function": generate_step_function,
    "bimodal_mixture": generate_bimodal_mixture,
    "heavy_tailed": generate_heavy_tailed,
    "sparse_sampling": generate_sparse_sampling,
    "gaussian_heteroscedastic": generate_gaussian_heteroscedastic,
    "poisson_count": generate_poisson_count,
    "exponential_waiting_time": generate_exponential_waiting_time,
    "multimodal_mixture_fixed": generate_multimodal_mixture_fixed,
    "skewed_heteroscedastic": generate_skewed_heteroscedastic,
}


def get_dgp(name: str, **kwargs) -> SyntheticDataset:
    """Get a synthetic dataset by name.

    Args:
        name: Name of the DGP (see DGP_REGISTRY keys)
        **kwargs: Arguments to pass to the DGP generator

    Returns:
        SyntheticDataset

    Raises:
        ValueError: If DGP name not found
    """
    if name not in DGP_REGISTRY:
        raise ValueError(f"Unknown DGP: {name}. Available: {list(DGP_REGISTRY.keys())}")
    return DGP_REGISTRY[name](**kwargs)
