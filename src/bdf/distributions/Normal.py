from typing import Any, ClassVar, Literal

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
    sigma_mu_auto_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale factor for automatic sigma_mu if 'auto' is used. Resolved upon `fit`, discarded from final params.",
        exclude=True,
    )

    # Scoring defaults for conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nle", description="Conjugate model defaults to nle (Bayesian evidence)."
    )
    use_posterior_predictive: bool = Field(
        default=True, description="Use Student's t posterior predictive (marginalizes μ uncertainty)."
    )


class NormalMuInvGammaSigmaNormalParams(BDFDistributionParams):
    """Parameters for Normal-Inverse-Gamma conjugate model (unknown mean and variance).

    Prior: μ | σ² ~ N(μ₀, σ²/n₀), σ² ~ InvGamma(ν₀/2, ν₀φ₀/2)
    Posterior: μ | σ², y ~ N(μₙ, σ²/nₙ), σ² | y ~ InvGamma(νₙ/2, νₙφₙ/2)
    Posterior predictive: y_new | y ~ StudentT(νₙ, μₙ, φₙ/nₙ(1 + 1/nₙ))
    """

    # Prior hyperparameters
    mu_mu: float = Field(default=0.0, description="Prior mean μ₀ for μ")
    n_mu: float = Field(default=1.0, gt=0, description="Prior precision scale n₀ for μ")
    nu_sigma: float = Field(default=3.0, gt=0, description="Prior degrees of freedom ν₀ for σ²")
    phi_sigma: float = Field(default=1.0, gt=0, description="Prior scale parameter φ₀ for σ²")

    # Scoring defaults
    score_method: Literal["nle", "nll"] = Field(default="nle")
    use_posterior_predictive: bool = Field(default=True, description="Use Student's t posterior predictive.")


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class NormalMuNormal(BDFDistribution[NormalMuNormalParams]):
    r"""Normal-Normal conjugate model (unknown mean, variance estimated from data).

    **Usage:** ``dist="NormalMuNormal"``

    **Model:**

    *   **Prior:** :math:`\mu \sim \mathcal{N}(\mu_0, \sigma_\mu^2)`
    *   **Likelihood:** :math:`y \mid \mu \sim \mathcal{N}(\mu, \sigma^2)`
    *   **Posterior:** :math:`\mu \mid y \sim \mathcal{N}(\mu_n, \sigma_n^2)`

    Parameters
    ----------
    mu_mu : float or "auto", default=0.0
        Prior mean for :math:`\mu`. If ``"auto"``, set to the sample mean of
        ``y`` at fit time.
    sigma_mu : float or "auto", default=1.0
        Prior standard deviation for :math:`\mu`. If ``"auto"``, set to
        ``std(y) * sigma_mu_auto_scale`` at fit time.
    sigma_mu_auto_scale : float, default=1.0
        Multiplier applied when ``sigma_mu="auto"``. Ignored otherwise.

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    NormalMuInvGammaSigmaNormal : Full conjugate model with unknown mean and variance.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMuNormalParams

    # Capabilities
    _supports_nle = True
    _has_fast_loo_cv = True
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: dict[str, Any] | NormalMuNormalParams):
        super().__init__(params)
        self.mu_mu = self.params.mu_mu
        self.sigma_mu = self.params.sigma_mu

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate Normal-Normal posterior parameters.

        Returns dict with:
        - posterior_mu: Posterior mean of μ
        - posterior_sigma_mu: Posterior std of μ (for PP) or sample std (for plug-in)
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
            "posterior_sigma_mu": float(posterior_sigma_mu),  # Used for PP
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
        sigma_mu = params["posterior_sigma_mu"]
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

        return params["posterior_sigma_mu"] ** 2

    # ========================================================================
    # OPTIONAL METHODS (OVERRIDE FOR EFFICIENCY)
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Empirical Bayes evidence for Normal with prior on μ, plug-in σ.

        This is a hybrid approach:
        - Prior: μ ~ N(μ₀, σ_μ²)
        - Likelihood: yᵢ | μ ~ N(μ, σ²) with σ² estimated from data

        The log evidence decomposes as:
            log p(y) = log p(ȳ | μ₀, σ_μ, σ̂) + log p(residuals | σ̂)

        where σ̂ is the sample standard deviation.

        Edge case: When σ̂ = 0 (constant data), all residuals are zero.
        The residual term vanishes (log p(0|0,σ→0) → 0 in the limit).
        """
        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)

        # Marginal variance of sample mean under prior
        marginal_var = (sample_std**2 / n) + self.sigma_mu**2

        # Log evidence for sample mean
        log_ev = -0.5 * np.log(2 * np.pi * marginal_var)
        log_ev -= 0.5 * (sample_mean - self.mu_mu) ** 2 / marginal_var

        # Log evidence for deviations from mean (independent of prior)
        # When sample_std=0 (constant data), residuals are all zero → term vanishes
        if n > 1 and sample_std > 0:
            log_ev -= 0.5 * (n - 1) * (1 + np.log(2 * np.pi * sample_std**2))

        return float(log_ev)

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive: N(x | μ_post, σ_μ² + σ_data²).

        Integrates out uncertainty in μ.
        """
        mu = params["posterior_mu"]
        sigma_mu = params["posterior_sigma_mu"]
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

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        For NormalMuNormal:
        - 'mu_mu': Use sample mean
        - 'sigma_mu': Use sample std and 'sigma_mu_auto_scale' if defined
        """
        if key == "mu_mu":
            return float(np.mean(data))
        if key == "sigma_mu":
            assert params is not None, "'params' must be provided to resolve 'sigma_mu' automatically."
            assert (
                "sigma_mu_auto_scale" in params
            ), "'sigma_mu_auto_scale' must be defined in params to resolve 'sigma_mu' automatically."
            sample_std = np.std(data, ddof=1)
            scale = params["sigma_mu_auto_scale"]
            return float(sample_std * scale)
        else:
            raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")


class NormalMuInvGammaSigmaNormal(BDFDistribution[NormalMuInvGammaSigmaNormalParams]):
    r"""Normal-Inverse-Gamma conjugate model (unknown mean and variance).

    **Usage:** ``dist="NormalMuInvGammaSigmaNormal"``

    **Model:**

    *   **Prior:** :math:`\mu \mid \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2/n_0)`,
        :math:`\sigma^2 \sim \text{InvGamma}(\nu_0/2, \nu_0 \phi_0/2)`
    *   **Likelihood:** :math:`y \mid \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma^2)`
    *   **Posterior Predictive:** :math:`y_\text{new} \mid y \sim t_{\nu_n}(\mu_n, \phi_n(1+1/n_n))`

    Parameters
    ----------
    mu_mu : float, default=0.0
        Prior mean :math:`\mu_0` for :math:`\mu`.
    n_mu : float, default=1.0
        Prior precision scale :math:`n_0`. Acts as pseudo-sample-size
        weighting the prior mean.
    nu_sigma : float, default=3.0
        Prior degrees of freedom :math:`\nu_0` for :math:`\sigma^2`.
    phi_sigma : float, default=1.0
        Prior scale parameter :math:`\phi_0` for :math:`\sigma^2`.

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    NormalMuNormal : Simpler model with known variance.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMuInvGammaSigmaNormalParams

    _supports_nle = True
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = True

    def __init__(self, params: dict[str, Any] | NormalMuInvGammaSigmaNormalParams):
        super().__init__(params)
        self.mu_mu = self.params.mu_mu
        self.n_mu = self.params.n_mu
        self.nu_sigma = self.params.nu_sigma
        self.phi_sigma = self.params.phi_sigma

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
        post_n = self.n_mu + n
        post_nu = self.nu_sigma + n
        post_phi = (
            self.nu_sigma * self.phi_sigma
            + (n - 1) * sample_var
            + (n * self.n_mu / post_n) * (sample_mean - self.mu_mu) ** 2
        )

        # Posterior mean
        posterior_mu = (self.n_mu * self.mu_mu + n * sample_mean) / post_n

        # Posterior mode of σ² (if ν > 2), else use mean
        if post_nu > 2:
            posterior_sigma_mu = np.sqrt(post_phi / (post_nu - 2))
        else:
            posterior_sigma_mu = np.sqrt(post_phi / post_nu)

        return {
            "posterior_mu": float(posterior_mu),
            "posterior_sigma_mu": float(posterior_sigma_mu),
            "post_n": float(post_n),
            "post_nu": float(post_nu),
            "post_phi": float(post_phi),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in Normal likelihood with posterior mode."""
        mu = params["posterior_mu"]
        sigma = params["posterior_sigma_mu"]
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

        return params["posterior_sigma_mu"] ** 2

    # ========================================================================
    # OPTIONAL METHODS
    # ========================================================================

    def log_evidence(self, data: np.ndarray) -> float:
        """Exact Bayesian evidence for Normal-Inverse-Gamma conjugate.

        Uses Murphy MLAPP eq 4.127 parameterization:
        log p(D) = -n/2·log(2π) + 1/2·log(κ₀/κₙ) + logΓ(αₙ) - logΓ(α₀)
                   + α₀·log(β₀) - αₙ·log(βₙ)

        where α = ν/2, β = νφ/2 (InvGamma rate parameterization).
        """
        from scipy.special import gammaln

        n = data.shape[0]
        sample_mean = np.mean(data)
        ssd = np.sum((data - sample_mean) ** 2)  # Sum of squared deviations

        # Posterior parameters
        kappa_n = self.n_mu + n
        nu_n = self.nu_sigma + n
        interaction = (self.n_mu * n / kappa_n) * (sample_mean - self.mu_mu) ** 2

        # Murphy's parameterization: α = ν/2, β = νφ/2
        alpha_0 = self.nu_sigma / 2
        alpha_n = nu_n / 2
        beta_0 = self.nu_sigma * self.phi_sigma / 2
        beta_n = beta_0 + ssd / 2 + interaction / 2

        log_ev = -0.5 * n * np.log(2 * np.pi)
        log_ev += 0.5 * np.log(self.n_mu / kappa_n)
        log_ev += gammaln(alpha_n) - gammaln(alpha_0)
        log_ev += alpha_0 * np.log(beta_0)
        log_ev -= alpha_n * np.log(beta_n)

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
