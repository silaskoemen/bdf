"""Negative Binomial distribution implementations for BDF.

Provides:
- FrequentistNegativeBinomial: MLE estimation (no prior)
- NormalMeanNegativeBinomial: Normal prior on mean, MoM parameter mapping
- GammaMSLambdaNegBin: Gamma prior on mean λ with strength parameterization (conjugate)

Note: Negative Binomial requires non-negative integer count data (y ∈ {0, 1, 2, ...}).

Parameterization: NB(r, p) where:
- r > 0: number of successes (can be non-integer for overdispersion)
- p ∈ (0, 1): success probability
- Mean: μ = r(1-p)/p
- Variance: σ² = r(1-p)/p²
"""

import warnings
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import nbinom

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class FrequentistNegativeBinomialParams(BDFDistributionParams):
    """Parameters for Frequentist Negative Binomial (MLE).

    No priors. Estimates r (dispersion) and p (success probability) via MLE using
    Method of Moments for initial values.
    """

    # Scoring defaults
    score_method: Literal["nll"] = Field(default="nll", description="Frequentist model only supports NLL (plug-in).")
    use_posterior_predictive: bool = Field(
        default=False, description="Posterior predictive not supported for frequentist model."
    )
    estimation_method: Literal["mom", "mle"] = Field(
        default="mom",
        description="Use closed-form Method of Moments ('mom') or fast Newton MLE ('mle').",
    )


class NormalMeanNegativeBinomialParams(BDFDistributionParams):
    """Parameters for Negative Binomial with Normal prior on the mean.

    Methodology:
    1. Place Normal(μ₀, σ₀²) prior on the population mean.
    2. Update posterior mean μ_post using Bayesian update (CLT-based).
    3. Estimate NegBinom parameters (r, p) using Method of Moments based on
       μ_post and sample variance.
    """

    prior_mean: float = Field(default=1.0, gt=0, description="Prior mean for the population mean (must be > 0).")
    prior_std: float = Field(default=1.0, gt=0, description="Prior standard deviation for the population mean.")

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nll", description="NLE not supported for this hybrid model.")
    use_posterior_predictive: bool = Field(default=False, description="Posterior predictive not currently supported.")


class GammaMSLambdaNegBinParams(BDFDistributionParams):
    """Parameters for Negative Binomial with Gamma prior on mean λ (Mean-Strength parameterization).

    Prior: λ ~ Gamma(α₀, β₀) where α₀ = strength·mean, β₀ = strength
    Likelihood: y | λ, φ ~ NegativeBinomial(λ, φ)
    Posterior: λ | y ~ Gamma(strength·mean + Σy, strength + n)

    This provides exact linear shrinkage:
        E[λ|y] = (strength·mean + n·ȳ) / (strength + n)

    Parameters
    ----------
    mean_lambda : float, default=1.0
        Prior mean for rate parameter λ (grand mean, shrinkage target).
    strength_lambda : float, default=1.0
        Prior strength (pseudo-observations). Higher values = stronger prior.
        Interpretation: "strength_lambda pseudo-observations with mean mean_lambda"
    phi : float or None, default=None
        Dispersion parameter φ (also called r).
        - If None: estimated via Method of Moments (φ = μ²/(σ²-μ))
        - If provided: treated as fixed hyperparameter
        Larger φ → less overdispersion (approaches Poisson as φ→∞).
        NB variance: Var[Y] = λ + λ²/φ
    score_method : {'nle', 'nll'}, default='nle'
        Scoring method. NLE uses closed-form Bayesian evidence.
    use_posterior_predictive : bool, default=True
        Use posterior predictive (integrates out λ uncertainty) or plug-in.

    Notes
    -----
    The dispersion parameter φ controls overdispersion:
    - Larger φ → less overdispersion (approaches Poisson as φ→∞)
    - Smaller φ → more overdispersion
    - φ is estimated per-leaf if not provided (allows adaptive overdispersion)
    """

    # Prior hyperparameters (mean-strength parameterization)
    mean_lambda: float = Field(default=1.0, gt=0, description="Prior mean E[λ] (grand mean, shrinkage target)")
    strength_lambda: float = Field(
        default=1.0, gt=0, description="Prior strength (pseudo-observations). Higher = stronger prior."
    )

    # Dispersion parameter (None => estimate via MoM)
    phi: float | None = Field(
        default=None,
        gt=0,
        description="Dispersion φ. If None, estimated via Method of Moments; otherwise treated as fixed.",
    )

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True,
        description="Use posterior predictive (marginalizes λ uncertainty) or plug-in Negative Binomial.",
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class FrequentistNegativeBinomial(BDFDistribution):
    r"""Negative Binomial with frequentist estimation (no prior).

    **Usage:** ``dist="FrequentistNegativeBinomial"``

    Parameters :math:`r` (dispersion) and :math:`p` (success probability)
    are estimated via method of moments.

    .. math:: y \sim \text{NegBin}(r, p)

    Parameters
    ----------
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    GammaMSLambdaNegBin : Bayesian version with Gamma prior on the mean.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = FrequentistNegativeBinomialParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        self.validate_targets(data)
        if data.size < 2:
            raise ValueError("Negative Binomial requires at least two observations.")
        if self.params.estimation_method == "mom":
            return self._estimate_params_mom(data)
        return self._estimate_params_mle(data)

    def _ensure_overdispersion(self, mean: float, var: float) -> float:
        """Clamp variance so NB assumptions hold."""
        return float(np.maximum(var, mean + 1e-6))

    def _estimate_params_mle(self, data: np.ndarray) -> dict[str, float]:
        mean = float(np.mean(data))
        var = self._ensure_overdispersion(mean, float(np.var(data, ddof=1)))
        r = (mean**2) / (var - mean)  # MoM init
        for _ in range(5):
            grad = np.sum(digamma(data + r) - digamma(r)) + len(data) * np.log(r / (r + mean))
            hess = np.sum(polygamma(1, data + r) - polygamma(1, r)) + len(data) * (1 / r - 1 / (r + mean))
            delta = grad / (hess + 1e-12)
            r = np.maximum(r - delta, 1e-6)
            if abs(delta) < 1e-6:
                break
        p = r / (r + mean)
        return {"r": float(r), "p": float(np.clip(p, 1e-6, 1 - 1e-6))}

    def _estimate_params_mom(self, data: np.ndarray) -> dict[str, float]:
        mean = float(np.mean(data))
        var = self._ensure_overdispersion(mean, float(np.var(data, ddof=1)))
        p = mean / var
        r = (mean**2) / (var - mean)
        return {"r": float(np.maximum(r, 1e-6)), "p": float(np.clip(p, 1e-6, 1 - 1e-6))}

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Negative Binomial log-likelihood."""
        r = params["r"]
        p = params["p"]
        # scipy.stats.nbinom uses (n, p) parameterization where n=r
        return nbinom.logpmf(data, n=r, p=p)

    def _num_parameters(self) -> int:
        """r and p (2 parameters)."""
        return 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from fitted Negative Binomial."""
        r = params["r"]
        p = params["p"]
        return np.array(nbinom.rvs(n=r, p=p, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Negative Binomial."""
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Negative Binomial requires non-negative count data (y >= 0)")
        if not np.all(data == np.floor(data)):
            raise ValueError("Negative Binomial requires integer count data")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Mean of Negative Binomial: μ = r(1-p)/p."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / p

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Variance of Negative Binomial: σ² = r(1-p)/p²."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / (p**2)


class NormalMeanNegativeBinomial(BDFDistribution):
    r"""Negative Binomial with Normal prior on the mean.

    **Usage:** ``dist="NormalMeanNegativeBinomial"``

    **Model:**

    *   **Prior:** :math:`\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)`
    *   **Likelihood:** :math:`y \mid r, p \sim \text{NegBin}(r, p)`
    *   **Posterior:** Precision-weighted update on :math:`\mu`, then MoM mapping
        :math:`(\mu_\text{post}, s^2) \to (r, p)`

    Parameters
    ----------
    prior_mean : float, default=1.0
        Prior mean for the population mean (must be > 0).
    prior_std : float, default=1.0
        Prior standard deviation for the population mean.
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    FrequentistNegativeBinomial : MLE estimation without prior.
    GammaMSLambdaNegBin : Fully conjugate Gamma prior on rate.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMeanNegativeBinomialParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: NormalMeanNegativeBinomialParams):
        super().__init__(params)
        self.prior_mean = self.params.prior_mean
        self.prior_std = self.params.prior_std

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate posterior mean, then solve for NegBinom params via MoM."""
        n = float(len(data))

        if n < 2:
            # Insufficient data: return prior-informed defaults
            return {"r": 1.0, "p": 0.5, "posterior_mean_mu": self.prior_mean}

        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # Negative Binomial requires variance > mean
        if sample_var <= sample_mean:
            warnings.warn(
                f"Negative Binomial requires variance > mean. "
                f"Got mean={sample_mean:.3f}, var={sample_var:.3f}. "
                f"Adding variance inflation.",
                UserWarning,
            )
            sample_var = max(sample_mean * 1.1, sample_var + 0.1)

        # 1. Bayesian update on mean (Normal-Normal conjugate, CLT approximation)
        # Precision-weighted average
        prior_prec = 1.0 / (self.prior_std**2)
        # Use sample variance as approximate likelihood variance (CLT)
        data_prec = n / sample_var
        post_prec = prior_prec + data_prec

        post_mu = (prior_prec * self.prior_mean + data_prec * sample_mean) / post_prec

        # 2. Method of Moments mapping (μ_post, sample_var) → (r, p)
        # Using posterior mean but sample variance (not updated by prior)
        p_est = post_mu / sample_var
        r_est = (post_mu**2) / (sample_var - post_mu)

        # Safety clamps
        p_est = np.clip(p_est, 1e-6, 1 - 1e-6)
        r_est = max(r_est, 1e-4)

        return {
            "r": float(r_est),
            "p": float(p_est),
            "posterior_mean_mu": float(post_mu),  # For debugging/inspection
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Negative Binomial log-likelihood."""
        r = params["r"]
        p = params["p"]
        return nbinom.logpmf(data, n=r, p=p)

    def _num_parameters(self) -> int:
        """r and p (2 parameters)."""
        return 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from fitted Negative Binomial."""
        r = params["r"]
        p = params["p"]
        return np.array(nbinom.rvs(n=r, p=p, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Negative Binomial."""
        if data.ndim != 1:
            raise ValueError("Data must be 1D")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Negative Binomial requires non-negative count data (y >= 0)")
        if not np.all(data == np.floor(data)):
            raise ValueError("Negative Binomial requires integer count data")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Mean of Negative Binomial: μ = r(1-p)/p."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / p

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Variance of Negative Binomial: σ² = r(1-p)/p²."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)
        r = params["r"]
        p = params["p"]
        return r * (1 - p) / (p**2)


class GammaMSLambdaNegBin(BDFDistribution[GammaMSLambdaNegBinParams]):
    """Negative Binomial with Gamma prior on mean λ (Mean-Strength parameterization).

    **Usage:** ``dist="GammaMSLambdaNegBin"``

    **Model Specification:**

    *   **Prior:** :math:`\\lambda \\sim \\text{Gamma}(\\alpha_0, \\beta_0)` where
        :math:`\\alpha_0 = m \\cdot \\lambda_0`, :math:`\\beta_0 = m`
    *   **Likelihood:** :math:`y \\mid \\lambda, \\phi \\sim \\text{NegBin}(\\lambda, \\phi)`
    *   **Posterior:** :math:`\\lambda \\mid y \\sim \\text{Gamma}(\\alpha_0 + \\sum y_i, \\beta_0 + n)`

    **Key Properties:**

    *   **Exact Linear Shrinkage:** :math:`E[\\lambda \\mid y] = \\frac{m \\lambda_0 + n \\bar{y}}{m + n}`
    *   **Closed-form Evidence:** Supports NLE scoring via Bayesian marginal likelihood
    *   **Posterior Predictive:** Integrates out :math:`\\lambda` uncertainty for robust predictions

    Parameters
    ----------
    mean_lambda : float, default=1.0
        Prior mean for :math:`\\lambda` (:math:`\\lambda_0`). Grand mean, shrinkage target.
    strength_lambda : float, default=1.0
        Prior strength (pseudo-observations, :math:`m`). Higher = stronger regularization.
    phi : float or None, default=None
        Dispersion parameter :math:`\\phi`. If None, estimated via MoM per leaf.
        If provided, treated as fixed hyperparameter across all leaves.
    score_method : {'nle', 'nll'}, default='nle'
        Scoring method. 'nle' uses exact Bayesian evidence.
    use_posterior_predictive : bool, default=True
        If True, uses posterior predictive (marginalizes :math:`\\lambda` uncertainty).

    Examples
    --------
    >>> from bdf import BDFRegressor
    >>> model = BDFRegressor(
    ...     dist="GammaMSLambdaNegBin",
    ...     params={
    ...         "mean_lambda": "auto",      # Use global mean
    ...         "strength_lambda": 1.0,     # 1 pseudo-observation
    ...         "phi": None,                # Estimate per leaf
    ...         "score_method": "nle",
    ...         "use_posterior_predictive": True,
    ...     }
    ... )
    >>> model.fit(X, y)  # doctest: +SKIP

    Notes
    -----
    This is a fully conjugate Bayesian model. The Gamma prior on :math:`\\lambda` combined
    with the Negative Binomial likelihood gives:

    - **Exact conjugate updates** (no approximations)
    - **Interpretable hyperparameters**: strength_lambda = number of pseudo-observations
    - **Linear shrinkage**: :math:`E[\\lambda \\mid y] = \\frac{m \\lambda_0 + n \\bar{y}}{m + n}`

    The dispersion parameter :math:`\\phi` (also called :math:`r`) is estimated via Method of
    Moments when not provided, allowing adaptive overdispersion per leaf.

    See Also
    --------
    :class:`.FrequentistNegativeBinomial` : MLE estimation without prior
    :class:`.NormalMeanNegativeBinomial` : Hybrid approach with Normal prior on mean
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = GammaMSLambdaNegBinParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: dict[str, Any] | GammaMSLambdaNegBinParams):
        super().__init__(params)
        self.mean_lambda = self.params.mean_lambda
        self.strength_lambda = self.params.strength_lambda
        self.phi = self.params.phi

        # Derived shape-rate parameters for Gamma prior
        # λ ~ Gamma(α, β) where α = m·λ₀, β = m
        self.alpha_lambda = self.strength_lambda * self.mean_lambda
        self.beta_lambda = self.strength_lambda

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Gamma posterior parameters for λ and estimate dispersion φ.

        Returns dict with:
        - posterior_alpha: Posterior shape α_post = m·λ₀ + Σy
        - posterior_beta: Posterior rate β_post = m + n
        - posterior_lambda: Posterior mean E[λ|y] = α_post/β_post
        - phi: Dispersion parameter (estimated or fixed)
        """
        n = data.shape[0]
        sum_counts = np.sum(data)

        # Conjugate update for λ: Gamma(α, β) → Gamma(α + Σy, β + n)
        alpha_post = self.alpha_lambda + sum_counts
        beta_post = self.beta_lambda + n
        lambda_post = alpha_post / beta_post

        # Use fixed φ or estimate via MoM
        if self.phi is not None:
            phi = self.phi
        else:
            phi = self._estimate_phi_mom(data)

        return {
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
            "posterior_lambda": float(lambda_post),
            "phi": float(phi),
        }

    def _estimate_phi_mom(self, data: np.ndarray) -> float:
        """Estimate dispersion parameter φ using Method of Moments.

        φ = μ²/(σ²-μ) where μ = sample mean, σ² = sample variance
        """
        n = len(data)
        if n < 2:
            # Insufficient data: use default
            return 1.0

        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # Ensure overdispersion (NB requires var > mean)
        if sample_var <= sample_mean:
            # No overdispersion: φ = μ² gives Var = μ + 1 (nearly Poisson).
            # Floor at 1e4 so small means don't produce tiny φ.
            return float(max(sample_mean**2, 1e-4))

        # MoM estimator: φ = μ²/(σ²-μ)
        phi_est = (sample_mean**2) / (sample_var - sample_mean)

        # Ensure positive and reasonable
        return float(np.maximum(phi_est, 1e-6))

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Negative Binomial log-likelihood with posterior mean λ.

        Uses NB(y | λ_post, φ) where λ_post = E[λ|y].
        """
        lambda_post = params["posterior_lambda"]
        phi = params["phi"]

        # NB parameterization: r=φ, p=φ/(φ+λ)
        # scipy.stats.nbinom uses (n, p) where n=r
        r = phi
        p = phi / (phi + lambda_post)

        return nbinom.logpmf(data, n=r, p=p)

    def _num_parameters(self) -> int:
        """Number of parameters estimated from data.

        - If phi=None: 2 parameters (λ and φ both estimated)
        - If phi is fixed: 1 parameter (only λ estimated)

        This affects BIC/AIC corrections which penalize model complexity.
        """
        return 2 if self.phi is None else 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive or plug-in Negative Binomial.

        If use_posterior_predictive=True:
            Sample from compound Gamma-NB (marginalizes λ uncertainty)
        Else:
            Sample from plug-in NB(λ_post, φ)
        """
        if self.params.use_posterior_predictive:
            # Posterior predictive: integrate out λ
            # This is a Beta-Negative-Binomial distribution
            # For simplicity, we use Monte Carlo:
            # 1. Sample λ ~ Gamma(α_post, β_post)
            # 2. Sample y ~ NB(λ, φ)

            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]
            phi = params["phi"]

            rng = np.random.default_rng(random_state)

            # Sample λ from posterior Gamma
            # scipy.stats.gamma uses (a=shape, scale=1/rate)
            lambda_samples = rng.gamma(shape=alpha_post, scale=1.0 / beta_post, size=size)

            # Sample y from NB(λ, φ) for each λ
            samples = np.empty(size)
            for i, lam in enumerate(lambda_samples):
                p = phi / (phi + lam)
                samples[i] = nbinom.rvs(n=phi, p=p, random_state=rng)

            return samples
        else:
            # Plug-in: NB(λ_post, φ)
            lambda_post = params["posterior_lambda"]
            phi = params["phi"]

            r = phi
            p = phi / (phi + lambda_post)

            return np.array(nbinom.rvs(n=r, p=p, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Negative Binomial distribution."""
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got shape {data.shape}")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(data >= 0):
            raise ValueError("Negative Binomial requires non-negative count data (y >= 0)")
        if not np.all(data == np.floor(data)):
            raise ValueError("Negative Binomial requires integer count data")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean of the distribution.

        For Negative Binomial with mean λ: E[Y] = λ
        For posterior predictive: E[Y] = E[λ|y] (same as plug-in for NB)
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        # Both plug-in and posterior predictive have mean = λ_post
        return params["posterior_lambda"]

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance of the distribution.

        For plug-in NB(λ, φ): Var[Y] = λ + λ²/φ
        For posterior predictive: Var[Y] = E[λ] + E[λ²]/φ (accounts for λ uncertainty)
        """
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        phi = params["phi"]

        if self.params.use_posterior_predictive:
            # Posterior predictive variance includes λ uncertainty
            # Var[Y] = E[Var[Y|λ]] + Var[E[Y|λ]]
            # where E[Y|λ] = λ, Var[Y|λ] = λ + λ²/φ

            alpha_post = params["posterior_alpha"]
            beta_post = params["posterior_beta"]

            # E[λ] and Var[λ] from Gamma posterior
            mean_lambda = alpha_post / beta_post
            var_lambda = alpha_post / (beta_post**2)

            # E[Var[Y|λ]] = E[λ + λ²/φ] = E[λ] + E[λ²]/φ
            # E[λ²] = Var[λ] + E[λ]²
            mean_lambda_sq = var_lambda + mean_lambda**2

            ev_vy = mean_lambda + mean_lambda_sq / phi

            # Var[E[Y|λ]] = Var[λ]
            ve_y = var_lambda

            return ev_vy + ve_y
        else:
            # Plug-in NB variance: λ + λ²/φ
            lambda_post = params["posterior_lambda"]
            return lambda_post + (lambda_post**2) / phi

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Gamma-NegativeBinomial conjugate.

        Computes p(y | prior) = ∫ p(y | λ, φ) p(λ) dλ

        The dispersion φ is estimated from data (not integrated out).
        The log evidence integrates over λ only.
        """
        n = data.shape[0]
        if n == 0:
            return 0.0

        sum_counts = np.sum(data)

        # Estimate or use fixed φ
        if self.phi is not None:
            phi = self.phi
        else:
            phi = self._estimate_phi_mom(data)

        # Posterior parameters
        alpha_post = self.alpha_lambda + sum_counts
        beta_post = self.beta_lambda + n

        # Log evidence for Gamma-Poisson (core conjugate part):
        # log p(y) = log Γ(α_post) - log Γ(α₀) + α₀·log(β₀) - α_post·log(β_post)
        log_ev = gammaln(alpha_post) - gammaln(self.alpha_lambda)
        log_ev += self.alpha_lambda * np.log(self.beta_lambda)
        log_ev -= alpha_post * np.log(beta_post)

        # NB-specific normalization terms (depend on φ):
        # For each y_i: log[Γ(y_i + φ) / (Γ(φ) · y_i!)]
        # Plus constant terms involving φ
        for y_i in data:
            log_ev += gammaln(y_i + phi) - gammaln(phi) - gammaln(y_i + 1)

        # Additional φ-dependent constant (from NB-Gamma convolution)
        log_ev += n * phi * np.log(phi)

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: integrates out λ uncertainty.

        This is the Beta-Negative-Binomial distribution (also called Gamma-Negative-Binomial).

        The closed form involves:
        p(y_new | y) = ∫ NB(y_new | λ, φ) Gamma(λ | α_post, β_post) dλ
        """
        alpha_post = params["posterior_alpha"]
        beta_post = params["posterior_beta"]
        phi = params["phi"]

        # Beta-NB is a compound distribution
        # For computational efficiency, we use the equivalent formulation:
        # Y ~ BetaNegBin(r, α, β) where r=φ, α=α_post, β=β_post

        # The PMF involves:
        # p(y) ∝ Γ(y+r) Γ(α+y) Γ(β+r) / [Γ(r) Γ(y+1) Γ(α) Γ(β) Γ(α+β+y+r)]

        # Simplified computation:
        log_probs = np.empty_like(data, dtype=float)

        for i, y in enumerate(data):
            # Beta-Negative-Binomial PMF (log scale)
            log_p = (
                gammaln(y + phi)
                + gammaln(alpha_post + y)
                + gammaln(beta_post + phi)
                - gammaln(phi)
                - gammaln(y + 1)
                - gammaln(alpha_post)
                - gammaln(beta_post)
                - gammaln(alpha_post + beta_post + y + phi)
            )
            # Add normalizing constant
            log_p += gammaln(alpha_post + beta_post)

            log_probs[i] = log_p

        return log_probs

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample λ from prior Gamma(α₀, β₀).

        Returns rate parameters λ (not NB samples).
        To get prior predictive samples, draw λ ~ Gamma then y ~ NB(λ, φ).
        """
        rng = np.random.default_rng(random_state)
        # scipy.stats.gamma: a=shape, scale=1/rate
        return np.array(rng.gamma(shape=self.alpha_lambda, scale=1.0 / self.beta_lambda, size=size))

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        For GammaMSLambdaNegBin:
        - 'mean_lambda': Use sample mean (grand mean, natural shrinkage target)
        """
        if key == "mean_lambda":
            return float(np.mean(data))
        else:
            raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")


"""
NOTE: Use gamma prior on precision to avoid instability for tiny leaves?
Principled? Good idea? Use standalone NormalMean and NormalMeanGammaVariance?
def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        self.validate_targets(data)
        n = float(len(data))
        if n < 2:
            raise ValueError("Need at least two observations for NB.")
        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1))

        # Normal-Normal update on μ
        prior_prec_mu = 1.0 / (self.prior_std**2)
        data_prec_mu = n / np.maximum(sample_var, 1e-6)
        post_prec_mu = prior_prec_mu + data_prec_mu
        post_mu = (prior_prec_mu * self.prior_mean + data_prec_mu * sample_mean) / post_prec_mu

        # Gamma prior on precision τ = 1/σ²
        alpha_post = self.params.prior_var_shape + 0.5 * (n - 1)
        beta_post = self.params.prior_var_rate + 0.5 * (n - 1) * sample_var
        post_var = 1.0 / (alpha_post / beta_post)  # E[σ² | data]

        post_var = float(np.maximum(post_var, post_mu + 1e-6))
        p_hat = post_mu / post_var
        r_hat = (post_mu**2) / (post_var - post_mu)

        return {
            "r": float(np.maximum(r_hat, 1e-6)),
            "p": float(np.clip(p_hat, 1e-6, 1 - 1e-6)),
            "posterior_mean_mu": float(post_mu),
            "posterior_mean_var": float(post_var),
        }
"""
