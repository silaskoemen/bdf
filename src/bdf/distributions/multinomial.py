"""Multinomial distribution implementations for BDF.

Provides conjugate Multinomial-Dirichlet models:
- DirichletAlphaMultinomial: Dirichlet(α) prior on category probabilities (concentration parameterization)
- DirichletMeanMultinomial: Dirichlet prior via mean probabilities + strength (more interpretable)

Both support:
- Closed-form Bayesian evidence (NLE)
- Dirichlet-Multinomial posterior predictive
- Efficient inference (conjugate updates)

Note: Data format is categorical (integer labels 0, 1, ..., K-1), not one-hot.
"""

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator
from scipy.special import gammaln
from scipy.stats import dirichlet

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class DirichletAlphaMultinomialParams(BDFDistributionParams):
    """Parameters for Multinomial with Dirichlet(α) prior on probabilities p.

    Prior: p ~ Dirichlet(α) where α = [α₁, ..., αₖ]
    Likelihood: n | p ~ Multinomial(N, p) where n = [n₁, ..., nₖ]
    Posterior: p | n ~ Dirichlet(α + n)
    Posterior predictive: n_new | data ~ Dirichlet-Multinomial(α_post, N)

    Notes:
    - α (alpha): Concentration parameters (all αᵢ > 0)
    - E[pᵢ] = αᵢ / Σαⱼ
    - Larger Σα → more concentrated around E[p]
    - α = [1, 1, ..., 1] → uniform prior (Jeffreys prior is α = 0.5 each)
    """

    # Prior hyperparameters
    alpha: list[float] = Field(
        default_factory=lambda: [1.0], description="Concentration parameters α for each category (length K)"
    )

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True, description="Use Dirichlet-Multinomial posterior predictive (marginalizes p uncertainty)."
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v):
        """Validate concentration parameters."""
        if len(v) == 0:
            raise ValueError("alpha must have at least one element")
        if not all(a > 0 for a in v):
            raise ValueError("All concentration parameters alpha must be positive")
        return v


class DirichletMeanMultinomialParams(BDFDistributionParams):
    """Parameters for Multinomial with Dirichlet prior specified via mean probabilities + strength.

    Prior: p ~ Dirichlet(α) where α = m · μ
    - μ (mean_probs): Prior mean probabilities E[p] = μ (must sum to 1)
    - m (strength): Total concentration = Σαᵢ (like pseudo-sample-size)

    This parameterization is more interpretable:
    - mean_probs: Your prior belief about category distribution
    - strength: How confident you are (higher → less variance in posterior)

    Example:
        mean_probs=[0.5, 0.3, 0.2], strength=10.0
        → α = [5.0, 3.0, 2.0]
        → Prior expects 50% cat 0, 30% cat 1, 20% cat 2
        → Equivalent to 10 prior pseudo-observations
    """

    # Prior hyperparameters (mean-strength parameterization)
    mean_probs: list[float] = Field(
        default_factory=lambda: [1.0], description="Prior mean probabilities μ for each category (must sum to 1)"
    )
    strength: float = Field(default=1.0, gt=0, description="Total concentration m = Σαᵢ (pseudo-sample-size)")

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nle")
    use_posterior_predictive: bool = Field(default=True, description="Use Dirichlet-Multinomial posterior predictive.")

    @field_validator("mean_probs")
    @classmethod
    def validate_mean_probs(cls, v):
        """Validate mean probabilities."""
        if len(v) == 0:
            raise ValueError("mean_probs must have at least one element")
        if not all(p > 0 for p in v):
            raise ValueError("All mean_probs must be positive")
        prob_sum = sum(v)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(f"mean_probs must sum to 1.0, got {prob_sum}")
        return v


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class DirichletAlphaMultinomial(BDFDistribution):
    """Multinomial-Dirichlet conjugate model with concentration (α) parameterization.

    Supports:
    - Closed-form Bayesian evidence (NLE)
    - Dirichlet-Multinomial posterior predictive (integrates out p uncertainty)
    - Efficient conjugate updates

    Data format: Integer labels in {0, 1, ..., K-1}
    Example: np.array([0, 1, 0, 2, 1]) → 5 observations, 3 categories
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = DirichletAlphaMultinomialParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: DirichletAlphaMultinomialParams):
        super().__init__(params)
        self.alpha = np.array(params.alpha)
        self.n_categories = len(self.alpha)

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Dirichlet posterior parameters for p.

        Returns dict with:
        - posterior_alpha: List of posterior concentrations [α₁ + n₁, ..., αₖ + nₖ]
        - posterior_probs: Posterior mean E[p | data] = α_post / Σα_post
        """
        # Count occurrences of each category
        counts = np.bincount(data.astype(int), minlength=self.n_categories)

        # Handle case where data has more categories than prior
        if len(counts) > self.n_categories:
            raise ValueError(
                f"Data contains {len(counts)} categories but prior only has {self.n_categories}. "
                f"Adjust alpha to match."
            )

        # Conjugate update: Dirichlet(α) + Multinomial counts → Dirichlet(α + n)
        alpha_post = self.alpha + counts[: self.n_categories]

        # Posterior mean probabilities
        probs_post = alpha_post / np.sum(alpha_post)

        return {
            "posterior_alpha": alpha_post.tolist(),
            "posterior_probs": probs_post.tolist(),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Categorical likelihood: log p(y | p_post).

        Uses posterior mean probabilities as point estimate.
        For each data point yᵢ, returns log p(yᵢ = k) = log(p_post[k])
        """
        probs_post = np.array(params["posterior_probs"])

        # Categorical log-likelihood for each observation
        # data[i] is the category index, so log p(y[i]) = log(probs[y[i]])
        log_probs = np.log(probs_post + 1e-10)  # Add epsilon for numerical stability
        return log_probs[data.astype(int)]

    def _num_parameters(self) -> int:
        """Number of free parameters: K - 1 (probabilities sum to 1)."""
        return self.n_categories - 1

    def _sample_posterior_params(
        self, params: dict[str, float], size: int | tuple[int, int], random_state: int
    ) -> np.ndarray:
        """Sample from posterior predictive (Dirichlet-Multinomial).

        If use_posterior_predictive=True:
            Sample p ~ Dirichlet(α_post), then y ~ Categorical(p)
        Else:
            Sample y ~ Categorical(p_post) where p_post = E[p | data]
        """
        if self.params.use_posterior_predictive:
            # Posterior predictive: integrate out p
            # For single draws, equivalent to: p ~ Dir(α_post), y ~ Cat(p)
            alpha_post = np.array(params["posterior_alpha"])

            rng = np.random.default_rng(random_state)

            # Sample from Dirichlet-Multinomial (equivalent to sampling p then y)
            samples = np.empty(size, dtype=int)
            n_samples = size if isinstance(size, int) else np.prod(size)
            for i in range(n_samples):
                # Sample probabilities from posterior Dirichlet
                p_sample = rng.dirichlet(alpha_post)
                # Sample category from Categorical(p_sample)
                samples[i] = rng.choice(self.n_categories, p=p_sample)

            return samples.reshape(size)
        else:
            # Plug-in: Categorical(p_post)
            probs_post = np.array(params["posterior_probs"])
            rng = np.random.default_rng(random_state)
            return rng.choice(self.n_categories, size=size, p=probs_post)

    def validate_targets(self, data: np.ndarray):
        """Validate data for Multinomial distribution."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Multinomial data must be non-negative integers (category labels)")
        if not np.all(data == np.floor(data)):
            raise ValueError("Multinomial data must be integers (category labels)")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

        max_category = int(np.max(data))
        if max_category >= self.n_categories:
            raise ValueError(
                f"Data contains category {max_category} but only {self.n_categories} categories in prior. "
                f"Increase alpha length or check data encoding."
            )

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean category (mode of posterior probabilities).

        Returns the category index with highest posterior probability.
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        probs_post = np.array(params["posterior_probs"])
        return float(np.argmax(probs_post))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance (entropy of categorical distribution).

        Returns the entropy H[p] = -Σ pᵢ log pᵢ as a measure of uncertainty.
        Higher entropy → more uniform distribution → higher variance.
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        probs_post = np.array(params["posterior_probs"])
        # Entropy (use as variance proxy for categorical)
        entropy = -float(np.sum(probs_post * np.log(probs_post + 1e-10)))
        return entropy

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Multinomial-Dirichlet conjugate.

        p(n | prior) = ∫ p(n | p) p(p) dp

        Closed form for Dirichlet-Multinomial:
        log p(n) = log B(α + n) - log B(α) + log N!/(n₁! n₂! ... nₖ!)
        where B(α) = Π Γ(αᵢ) / Γ(Σαᵢ)

        Simplifies to:
        log p(n) = [Σ log Γ(αᵢ + nᵢ) - Σ log Γ(αᵢ)] - [log Γ(Σαᵢ + N) - log Γ(Σαᵢ)]
        """
        counts = np.bincount(data.astype(int), minlength=self.n_categories)
        len(data)

        alpha_post = self.alpha + counts[: self.n_categories]

        # Log evidence using Beta function identity
        log_ev = 0.0

        # Numerator: log B(α + n) = Σ log Γ(αᵢ + nᵢ) - log Γ(Σ(αᵢ + nᵢ))
        log_ev += np.sum(gammaln(alpha_post)) - gammaln(np.sum(alpha_post))

        # Denominator: log B(α) = Σ log Γ(αᵢ) - log Γ(Σαᵢ)
        log_ev -= np.sum(gammaln(self.alpha)) - gammaln(np.sum(self.alpha))

        # Note: Multinomial coefficient log(N!/(n₁!...nₖ!)) cancels in marginal
        # (already accounted for in Dirichlet-Multinomial formula)

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Dirichlet-Multinomial (Pólya distribution).

        For a single new observation y_new:
        p(y_new = k | data) = (αₖ + nₖ) / (Σαᵢ + N)

        This is the predictive distribution integrating out p ~ Dirichlet(α_post).
        """
        alpha_post = np.array(params["posterior_alpha"])
        alpha_sum = np.sum(alpha_post)

        # Predictive probabilities for each category
        predictive_probs = alpha_post / alpha_sum

        # Log-likelihood for each data point
        log_probs = np.log(predictive_probs + 1e-10)
        return log_probs[data.astype(int)]

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample p from prior Dirichlet(α).

        Returns probability vectors p (not categorical samples).
        To get prior predictive samples, draw p ~ Dir(α) then y ~ Cat(p).
        """
        rng = np.random.default_rng(random_state)
        # Returns (size, n_categories) array of probability vectors
        return dirichlet.rvs(alpha=self.alpha, size=size, random_state=rng)


class DirichletMeanMultinomial(BDFDistribution):
    """Multinomial-Dirichlet conjugate model with mean-strength parameterization.

    Same as DirichletAlphaMultinomial, but prior specified via:
    - mean_probs = E[p] (prior mean probabilities)
    - strength = Σαᵢ (total concentration, like pseudo-sample-size)

    Internally converts to α = strength · mean_probs.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = DirichletMeanMultinomialParams

    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: DirichletMeanMultinomialParams):
        super().__init__(params)
        self.mean_probs = np.array(params.mean_probs)
        self.strength = params.strength

        # Convert mean-strength to concentration
        self.alpha = self.strength * self.mean_probs
        self.n_categories = len(self.alpha)

    # ========================================================================
    # REQUIRED METHODS (identical to DirichletAlphaMultinomial after conversion)
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Dirichlet posterior parameters."""
        counts = np.bincount(data.astype(int), minlength=self.n_categories)

        if len(counts) > self.n_categories:
            raise ValueError(f"Data contains {len(counts)} categories but prior only has {self.n_categories}.")

        alpha_post = self.alpha + counts[: self.n_categories]
        probs_post = alpha_post / np.sum(alpha_post)

        return {
            "posterior_alpha": alpha_post.tolist(),
            "posterior_probs": probs_post.tolist(),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Categorical likelihood."""
        probs_post = np.array(params["posterior_probs"])
        log_probs = np.log(probs_post + 1e-10)
        return log_probs[data.astype(int)]

    def _num_parameters(self) -> int:
        return self.n_categories - 1

    def _sample_posterior_params(
        self, params: dict[str, float], size: int | tuple[int, ...], random_state: int
    ) -> np.ndarray:
        """Sample from posterior predictive or plug-in."""
        if self.params.use_posterior_predictive:
            alpha_post = np.array(params["posterior_alpha"])
            rng = np.random.default_rng(random_state)

            samples = np.empty(size, dtype=int)
            n_samples = size if isinstance(size, int) else np.prod(size)
            for i in range(n_samples):
                p_sample = rng.dirichlet(alpha_post)
                samples[i] = rng.choice(self.n_categories, p=p_sample)

            return samples.reshape(size)
        else:
            probs_post = np.array(params["posterior_probs"])
            rng = np.random.default_rng(random_state)
            return rng.choice(self.n_categories, size=size, p=probs_post)

    def validate_targets(self, data: np.ndarray):
        """Validate data."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Multinomial data must be non-negative integers")
        if not np.all(data == np.floor(data)):
            raise ValueError("Multinomial data must be integers")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

        max_category = int(np.max(data))
        if max_category >= self.n_categories:
            raise ValueError(f"Data contains category {max_category} but only {self.n_categories} categories in prior.")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean category."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        probs_post = np.array(params["posterior_probs"])
        return float(np.argmax(probs_post))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance (entropy)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        probs_post = np.array(params["posterior_probs"])
        entropy = -float(np.sum(probs_post * np.log(probs_post + 1e-10)))
        return entropy

    # ========================================================================
    # OPTIONAL METHODS
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence."""
        counts = np.bincount(data.astype(int), minlength=self.n_categories)
        alpha_post = self.alpha + counts[: self.n_categories]

        log_ev = np.sum(gammaln(alpha_post)) - gammaln(np.sum(alpha_post))
        log_ev -= np.sum(gammaln(self.alpha)) - gammaln(np.sum(self.alpha))

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: Dirichlet-Multinomial."""
        alpha_post = np.array(params["posterior_alpha"])
        predictive_probs = alpha_post / np.sum(alpha_post)
        log_probs = np.log(predictive_probs + 1e-10)
        return log_probs[data.astype(int)]

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample p from prior Dirichlet(α)."""
        rng = np.random.default_rng(random_state)
        return dirichlet.rvs(alpha=self.alpha, size=size, random_state=rng)
