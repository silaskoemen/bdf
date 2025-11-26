from typing import ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.stats import norm
from scipy.stats import t as student_t

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class NormalMuNormalParams(BDFDistributionParams):
    r"""Parameters for Normal-Normal conjugate model (known variance).

    **Model Specification:**

    *   **Prior:** :math:`\mu \sim \mathcal{N}(\mu_0, \sigma_\mu^2)`
    *   **Likelihood:** :math:`y \mid \mu \sim \mathcal{N}(\mu, \sigma^2)` (where :math:`\sigma` is estimated from data)
    *   **Posterior:** :math:`\mu \mid y \sim \mathcal{N}(\mu_n, \sigma_n^2)`

    Parameters
    ----------
    mu_mu : float
        Prior mean for :math:`\mu`.
    sigma_mu : float
        Prior standard deviation for :math:`\mu`.

    See Also
    --------
    :class:`.NormalMuNormal` : The implementation class using these parameters.
    """

    # Prior hyperparameters
    mu_mu: float = Field(default=0.0, description="Prior mean for μ")
    sigma_mu: float = Field(default=1.0, gt=0, description="Prior std for μ")

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True, description="Use Student's t posterior predictive (marginalizes μ uncertainty)."
    )


class NormGammaNormalParams(BDFDistributionParams):
    """Parameters for Normal-Gamma conjugate model (unknown mean and variance).

    Prior: μ | σ² ~ N(μ₀, σ²/n₀), σ² ~ InvGamma(ν₀/2, ν₀φ₀/2)
    Posterior: μ | σ², y ~ N(μₙ, σ²/nₙ), σ² | y ~ InvGamma(νₙ/2, νₙφₙ/2)
    Posterior predictive: y_new | y ~ StudentT(νₙ, μₙ, φₙ/nₙ(1 + 1/nₙ))
    """

    # Prior hyperparameters
    mu_zero: float = Field(default=0.0, description="Prior mean μ₀")
    prior_n: float = Field(default=1.0, gt=0, description="Prior precision parameter n₀")
    prior_nu: float = Field(default=3.0, gt=0, description="Prior degrees of freedom ν₀")
    prior_phi: float = Field(default=1.0, gt=0, description="Prior scale parameter φ₀")

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nle")
    use_posterior_predictive: bool = Field(default=True, description="Use Student's t posterior predictive.")


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class NormalMuNormal(BDFDistribution):
    r"""Normal-Normal conjugate model (μ unknown, σ² estimated from data).

    **String Alias:** ``'normal_normal'``

    **Model Specification:**

    *   **Prior:** :math:`\mu \sim \mathcal{N}(\mu_0, \sigma_\mu^2)`
    *   **Likelihood:** :math:`y \mid \mu \sim \mathcal{N}(\mu, \sigma^2)`
    *   **Posterior:** :math:`\mu \mid y \sim \mathcal{N}(\mu_n, \sigma_n^2)`

    Parameters
    ----------
    mu_mu : float, default=0.0
        Prior mean for :math:`\mu` (:math:`\mu_0`).

    sigma_mu : float, default=1.0
        Prior standard deviation for :math:`\mu` (:math:`\sigma_\mu`).

    score_method : {'nle', 'nll'}, default='nle'
        Scoring method.
        *   ``'nle'``: Uses exact Bayesian evidence (Negative Log Evidence).
        *   ``'nll'``: Uses plug-in Negative Log Likelihood.

        .. note:: This overrides the base default of 'nll' because this is a conjugate model.

    use_posterior_predictive : bool, default=True
        If True, uses the Student's t posterior predictive distribution for inference.
        If False, uses the plug-in Normal distribution with MAP estimates.

    score_correction : {'aic', 'bic', 'loo_cv', 'kfold_cv'} or None, default=None
        Correction term for NLL scoring. Ignored if ``score_method='nle'``.

    score_cv_folds : int, default=3
        Number of folds if ``score_correction='kfold_cv'``.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMuNormalParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = True
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: NormalMuNormalParams):
        super().__init__(params)
        self.mu_mu = params.mu_mu
        self.sigma_mu = params.sigma_mu

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Normal-Normal posterior parameters.

        Returns dict with:
        - posterior_mu: Posterior mean of μ
        - posterior_sigma: Posterior std of μ (for PP) or sample std (for plug-in)
        - sample_std: Always store sample std for likelihood evaluation
        """
        n = data.shape[0]
        if n == 0:
            raise ValueError("Data must contain at least one observation to compute posterior parameters.")
        elif n == 1:
            # With one data point, sample std is undefined; use small value to avoid division by zero
            sample_mean = data[0]
            sample_std = 1e-10
        else:
            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)

        # Avoid division by zero
        sample_var = max(sample_std**2, 1e-10)

        # Posterior mean (weighted average of prior and data)
        precision_prior = 1 / (self.sigma_mu**2)
        precision_data = n / sample_var

        posterior_mu = (precision_data * sample_mean + precision_prior * self.mu_mu) / (
            precision_data + precision_prior
        )

        # Posterior std of μ (parameter uncertainty)
        posterior_sigma_mu = np.sqrt(1 / (precision_data + precision_prior))

        return {
            "posterior_mu": float(posterior_mu),
            "posterior_sigma": float(posterior_sigma_mu),  # Used for PP
            "sample_std": float(sample_std),  # Used for plug-in
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in: N(x | μ_posterior, σ_sample)."""
        mu = params["posterior_mu"]
        sigma = params["sample_std"]
        return norm.logpdf(data, loc=mu, scale=sigma)

    def _num_parameters(self) -> int:
        """Only μ is estimated (σ known from data)."""
        return 1

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from posterior predictive N(μ_post, σ_μ² + σ_data²)."""
        mu = params["posterior_mu"]
        sigma_mu = params["posterior_sigma"]
        sample_std = params["sample_std"]

        # Posterior predictive variance = parameter uncertainty + data noise
        pred_std = np.sqrt(sigma_mu**2 + sample_std**2)

        return np.array(norm.rvs(loc=mu, scale=pred_std, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data for Normal distribution."""
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains infinite values")

        std = np.std(data)
        if not (np.isfinite(std) and std >= 0.0):
            raise ValueError(f"Standard deviation must be finite and non-negative, got {std}")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean of μ."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        return params["posterior_mu"]

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance of μ."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        return params["posterior_sigma"] ** 2

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Normal-Normal conjugate.

        p(y | prior) = ∫ p(y | μ) p(μ) dμ

        Marginal distribution of sample mean: N(μ₀, σ²/n + σ_μ²)
        Plus term for deviations from mean.
        """
        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)

        # Marginal variance of sample mean
        marginal_var = (sample_std**2 / n) + self.sigma_mu**2

        # Log evidence for sample mean
        log_ev = -0.5 * np.log(2 * np.pi * marginal_var)
        log_ev -= 0.5 * (sample_mean - self.mu_mu) ** 2 / marginal_var

        # Log evidence for deviations from mean (independent of prior)
        if n > 1:
            log_ev -= 0.5 * (n - 1) * (1 + np.log(2 * np.pi * sample_std**2))

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: N(x | μ_post, σ_μ² + σ_data²).

        Integrates out uncertainty in μ.
        """
        mu = params["posterior_mu"]
        sigma_mu = params["posterior_sigma"]
        sample_std = params["sample_std"]

        # Posterior predictive variance
        pred_var = sigma_mu**2 + sample_std**2

        return norm.logpdf(data, loc=mu, scale=np.sqrt(pred_var))

    def _loo_cv_log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Efficient analytical LOO for Normal (override default).

        For plug-in: analytical LOO means and stds
        For PP: recompute posterior for each fold
        """
        n = data.shape[0]
        full_mean = np.mean(data)
        full_std = np.std(data, ddof=1)

        # Analytical LOO means
        loo_means = (n * full_mean - data) / (n - 1)

        if self.params.use_posterior_predictive:
            # Full Bayesian: recompute posterior for each LOO fold
            loo_ll = np.empty(n)
            for i in range(n):
                train_data = np.delete(data, i)
                loo_params = self.calc_posterior_params(train_data)
                loo_ll[i] = self._posterior_predictive_log_likelihood(data[i : i + 1], loo_params)[0]
            return loo_ll
        else:
            # Plug-in: analytical LOO std
            if n > 2:
                loo_vars = ((n - 1) * full_std**2 - (data - full_mean) ** 2) / (n - 2)
                loo_stds = np.sqrt(np.maximum(loo_vars, 1e-10))
            else:
                loo_stds = np.full(n, full_std)

            return norm.logpdf(data, loc=loo_means, scale=loo_stds)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample μ from prior N(μ₀, σ_μ²)."""
        rng = np.random.default_rng(random_state)
        return rng.normal(self.mu_mu, self.sigma_mu, size=size)


class NormGammaNormal(BDFDistribution):
    """Normal-Gamma conjugate model (μ and σ² both unknown).

    Supports:
    - Closed-form Bayesian evidence
    - Student's t posterior predictive
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormGammaNormalParams

    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: NormGammaNormalParams):
        super().__init__(params)
        self.mu_zero = params.mu_zero
        self.prior_n = params.prior_n
        self.prior_nu = params.prior_nu
        self.prior_phi = params.prior_phi

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Normal-Gamma posterior parameters."""
        n = data.shape[0]
        if n == 0:
            raise ValueError("Data must contain at least one observation to compute posterior parameters.")
        elif n == 1:
            # With one data point, sample std is undefined; use small value to avoid division by zero
            sample_mean = data[0]
            sample_var = 1e-10
        else:
            sample_mean = np.mean(data)
            sample_var = np.var(data, ddof=1)

        # Posterior hyperparameters
        post_n = self.prior_n + n
        post_nu = self.prior_nu + n
        post_phi = (
            self.prior_nu * self.prior_phi
            + (n - 1) * sample_var
            + (n * self.prior_n / post_n) * (sample_mean - self.mu_zero) ** 2
        )

        # Posterior mean
        posterior_mu = (self.prior_n * self.mu_zero + n * sample_mean) / post_n

        # Posterior mode of σ² (if ν > 2), else use mean
        if post_nu > 2:
            posterior_sigma = np.sqrt(post_phi / (post_nu - 2))
        else:
            posterior_sigma = np.sqrt(post_phi / post_nu)

        return {
            "posterior_mu": float(posterior_mu),
            "posterior_sigma": float(posterior_sigma),
            "post_n": float(post_n),
            "post_nu": float(post_nu),
            "post_phi": float(post_phi),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Normal likelihood with posterior mode."""
        mu = params["posterior_mu"]
        sigma = params["posterior_sigma"]
        return norm.logpdf(data, loc=mu, scale=sigma)

    def _num_parameters(self) -> int:
        """μ and σ² both estimated."""
        return 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from Student's t posterior predictive."""
        mu = params["posterior_mu"]
        post_n = params["post_n"]
        post_nu = params["post_nu"]
        post_phi = params["post_phi"]

        df = post_nu
        scale = np.sqrt(post_phi / post_nu * (1 + 1 / post_n))

        return np.array(student_t.rvs(df=df, loc=mu, scale=scale, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        """Validate data."""
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains infinite values")

        std = np.std(data)
        if not (np.isfinite(std) and std >= 0.0):
            raise ValueError(f"Standard deviation must be finite and non-negative, got {std}")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        return params["posterior_mu"]

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        return params["posterior_sigma"] ** 2

    # ========================================================================
    # OPTIONAL METHODS
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Normal-Gamma conjugate."""
        from scipy.special import gammaln

        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        post_n = self.prior_n + n
        post_nu = self.prior_nu + n
        post_phi = (
            self.prior_nu * self.prior_phi
            + (n - 1) * sample_var
            + (n * self.prior_n / post_n) * (sample_mean - self.mu_zero) ** 2
        )

        log_ev = -0.5 * n * np.log(2 * np.pi)
        log_ev += 0.5 * np.log(self.prior_n / post_n)
        log_ev += gammaln(post_nu / 2) - gammaln(self.prior_nu / 2)
        log_ev += (self.prior_nu / 2) * np.log(self.prior_phi)
        log_ev -= (post_nu / 2) * np.log(post_phi)

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Student's t posterior predictive."""
        mu = params["posterior_mu"]
        post_n = params["post_n"]
        post_nu = params["post_nu"]
        post_phi = params["post_phi"]

        df = post_nu
        scale = np.sqrt(post_phi / post_nu * (1 + 1 / post_n))

        return student_t.logpdf(data, df=df, loc=mu, scale=scale)

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Cannot easily sample from Normal-Gamma prior (hierarchical)."""
        raise NotImplementedError("NormGammaNormal prior sampling not implemented (requires hierarchical sampling).")
