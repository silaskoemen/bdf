import warnings
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.stats import skewnorm

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class NormalMeanPseudoAlphaSkewNormalParams(BDFDistributionParams):
    r"""Parameters for Skew-Normal with Normal prior on mean and pseudo-prior on alpha.

    **Model Specification:**

    *   **Prior on mean:** :math:`\mu \sim \mathcal{N}(\mu_\mu, \sigma_\mu^2)`
    *   **Pseudo-prior on skewness:** :math:`\alpha \sim` shrinkage toward :math:`\alpha_0` with strength :math:`m`
    *   **Likelihood:** :math:`y \mid \alpha, \xi, \omega \sim \text{SkewNormal}(\alpha, \xi, \omega)`

    Parameters
    ----------
    mu_mu : float
        Prior mean for the data mean (used in Normal-like posterior update for location).
    sigma_mu : float
        Prior standard deviation for the data mean.
    sigma_mu_auto_scale : float
        Scale factor for automatic sigma_mu if 'auto' is used. Multiplied by sample std.
    prior_alpha : float
        Prior mean for skewness parameter alpha (shrinkage target).
    m_alpha : float
        Prior strength (pseudo sample size) for alpha shrinkage.
    """

    # Prior hyperparameters
    mu_mu: float = Field(default=0.0, description="Prior mean for data mean μ")
    sigma_mu: float = Field(default=1.0, gt=0, description="Prior std for data mean μ")
    sigma_mu_auto_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale factor for automatic sigma_mu if 'auto' is used. Resolved upon fit, discarded from final params.",
        exclude=True,
    )
    prior_alpha: float = Field(default=0.0, description="Prior mean (shrinkage target) for skewness α")
    m_alpha: float = Field(default=10.0, gt=0, description="Prior strength for α shrinkage")

    # Scoring defaults for non-conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nll",
        description="Non-conjugate model uses NLL scoring.",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="Modular inference does not support posterior predictive.",
    )


class NormalMeanNormalGammaSkewNormalParams(BDFDistributionParams):
    r"""Parameters for Skew-Normal with Normal prior on mean and Normal prior on skewness gamma.

    **Model Specification:**

    *   **Prior on mean:** :math:`\mu \sim \mathcal{N}(\mu_\mu, \sigma_\mu^2)`
    *   **Prior on skewness:** :math:`\gamma \sim \mathcal{N}(\mu_\gamma, \sigma_\gamma^2)`
    *   **Likelihood:** :math:`y \mid \alpha, \xi, \omega \sim \text{SkewNormal}(\alpha, \xi, \omega)`

    Uses moment-matching to convert between gamma (sample skewness) and alpha parameterization.

    Parameters
    ----------
    mu_mu : float
        Prior mean for the data mean.
    sigma_mu : float
        Prior standard deviation for the data mean.
    sigma_mu_auto_scale : float
        Scale factor for automatic sigma_mu if 'auto' is used. Multiplied by sample std.
    mu_gamma : float
        Prior mean for skewness parameter gamma.
    sigma_gamma : float
        Prior standard deviation for skewness parameter gamma (skewness is scale-free, no auto-scale needed).
    """

    # Prior hyperparameters
    mu_mu: float = Field(default=0.0, description="Prior mean for data mean μ")
    sigma_mu: float = Field(default=1.0, gt=0, description="Prior std for data mean μ")
    sigma_mu_auto_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale factor for automatic sigma_mu if 'auto' is used. Resolved upon fit, discarded from final params.",
        exclude=True,
    )
    mu_gamma: float = Field(default=0.0, description="Prior mean for skewness γ")
    sigma_gamma: float = Field(default=0.5, gt=0, description="Prior std for skewness γ")

    # Scoring defaults for non-conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nll",
        description="Non-conjugate model uses NLL scoring.",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="Modular inference does not support posterior predictive.",
    )


class NormalXiNormalAlphaSkewNormalMAPParams(BDFDistributionParams):
    r"""Parameters for Skew-Normal with Normal priors on xi and alpha using MAP estimation.

    **Model Specification:**

    *   **Prior on location:** :math:`\xi \sim \mathcal{N}(\mu_\xi, \sigma_\xi^2)`
    *   **Prior on skewness:** :math:`\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)`
    *   **Likelihood:** :math:`y \mid \alpha, \xi, \omega \sim \text{SkewNormal}(\alpha, \xi, \omega)`

    Uses numerical optimization (Nelder-Mead) to find MAP estimates.

    Parameters
    ----------
    mu_xi : float
        Prior mean for location parameter xi.
    sigma_xi : float
        Prior standard deviation for location parameter xi.
    sigma_xi_auto_scale : float
        Scale factor for automatic sigma_xi if 'auto' is used. Multiplied by sample std.
    mu_alpha : float
        Prior mean for skewness parameter alpha.
    sigma_alpha : float
        Prior standard deviation for skewness parameter alpha (alpha is scale-free, no auto-scale needed).
    """

    # Prior hyperparameters
    mu_xi: float = Field(default=0.0, description="Prior mean for location ξ")
    sigma_xi: float = Field(default=1.0, gt=0, description="Prior std for location ξ")
    sigma_xi_auto_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale factor for automatic sigma_xi if 'auto' is used. Resolved upon fit, discarded from final params.",
        exclude=True,
    )
    mu_alpha: float = Field(default=0.0, description="Prior mean for skewness α")
    sigma_alpha: float = Field(default=5.0, gt=0, description="Prior std for skewness α")

    # Scoring defaults for non-conjugate model
    score_method: Literal["nle", "nll"] = Field(
        default="nll",
        description="Non-conjugate model uses NLL scoring.",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="MAP inference does not support posterior predictive.",
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class NormalMeanPseudoAlphaSkewNormal(BDFDistribution[NormalMeanPseudoAlphaSkewNormalParams]):
    r"""Skew-Normal with Normal prior on mean and pseudo-prior on skewness.

    **Usage:** ``dist="NormalMeanPseudoAlphaSkewNormal"``

    **Model:**

    *   **Prior on mean:** :math:`\mu \sim \mathcal{N}(\mu_\mu, \sigma_\mu^2)`
    *   **Pseudo-prior on skewness:** :math:`\alpha` shrunk toward :math:`\alpha_0`
        with strength :math:`m`
    *   **Likelihood:** :math:`y \mid \alpha, \xi, \omega \sim \text{SkewNormal}(\alpha, \xi, \omega)`

    Parameters
    ----------
    mu_mu : float or "auto", default=0.0
        Prior mean for the data mean. If ``"auto"``, set to the sample mean at
        fit time.
    sigma_mu : float or "auto", default=1.0
        Prior standard deviation for the data mean. If ``"auto"``, set to the
        sample standard deviation times ``sigma_mu_auto_scale`` at fit time.
    sigma_mu_auto_scale : float, default=1.0
        Multiplier applied when ``sigma_mu="auto"``. Ignored otherwise.
    prior_alpha : float, default=0.0
        Shrinkage target for skewness :math:`\alpha`.
    m_alpha : float, default=10.0
        Shrinkage strength (pseudo sample size) for :math:`\alpha`.
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    NormalMeanNormalGammaSkewNormal : Alternative with Normal prior on skewness gamma.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMeanPseudoAlphaSkewNormalParams

    # Capabilities
    _supports_nle = False  # Not conjugate
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: dict[str, Any] | NormalMeanPseudoAlphaSkewNormalParams):
        super().__init__(params)
        self.mu_mu = self.params.mu_mu
        self.sigma_mu = self.params.sigma_mu
        self.prior_alpha = self.params.prior_alpha
        self.m_alpha = self.params.m_alpha

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate posterior parameters using modular moment-based updates.

        Returns dict with:
        - posterior_alpha: Posterior skewness parameter
        - posterior_xi: Posterior location parameter
        - posterior_omega: Posterior scale parameter
        """
        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # Clip sample skewness to valid range
        sample_skewness = np.clip(
            np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-5)) ** 3),
            -0.99,
            0.99,
        )

        # Step 1: Posterior mean for location (Normal prior update)
        precision_prior = 1 / self.sigma_mu**2
        precision_data = n / (sample_var + 1e-5)
        posterior_mean = (precision_prior * self.mu_mu + precision_data * sample_mean) / (
            precision_prior + precision_data
        )

        # Step 2: Convert sample skewness to alpha via moment-matching
        delta = np.sign(sample_skewness) * np.sqrt(
            np.pi
            / 2
            * np.abs(sample_skewness) ** (2 / 3)
            / (np.abs(sample_skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        alpha_mle = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 3)

        # Apply shrinkage prior
        posterior_alpha = (n / (n + self.m_alpha)) * alpha_mle + (self.m_alpha / (n + self.m_alpha)) * self.prior_alpha

        # Step 3: Estimate omega from sample variance adjusted for skewness
        posterior_omega = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)

        # Handle numerical issues
        if posterior_omega <= 0 or not np.isfinite(posterior_omega):
            warnings.warn("Posterior omega invalid, using small positive value.", UserWarning)
            posterior_omega = 1e-5

        # Step 4: Recover xi from posterior mean
        posterior_xi = posterior_mean - posterior_omega * posterior_alpha / np.sqrt(1 + posterior_alpha**2) * np.sqrt(
            2 / np.pi
        )

        return {
            "posterior_alpha": float(posterior_alpha),
            "posterior_xi": float(posterior_xi),
            "posterior_omega": float(posterior_omega),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in log-likelihood using posterior point estimates."""
        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]
        return skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega)

    def _num_parameters(self) -> int:
        """Skew-normal has 3 parameters: α, ξ, ω."""
        return 3

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from the fitted skew-normal distribution."""
        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]
        return np.asarray(skewnorm.rvs(a=alpha, loc=xi, scale=omega, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray) -> np.ndarray:
        """Validate data for skew-normal distribution."""
        arr = np.asarray(data, dtype=float)
        if np.any(np.isnan(arr)):
            raise ValueError("Data contains NaN values")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Data contains infinite values")
        return arr

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean: E[X] = ξ + ω·δ·√(2/π) where δ = α/√(1+α²)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]

        delta = alpha / np.sqrt(1 + alpha**2)
        return float(xi + omega * delta * np.sqrt(2 / np.pi))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance: Var[X] = ω²(1 - 2δ²/π) where δ = α/√(1+α²)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        omega = params["posterior_omega"]

        delta = alpha / np.sqrt(1 + alpha**2)
        return float(omega**2 * (1 - 2 * delta**2 / np.pi))

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        For NormalMeanPseudoAlphaSkewNormal:
        - 'mu_mu': Use sample mean
        - 'sigma_mu': Use sample std * sigma_mu_auto_scale
        """
        if key == "mu_mu":
            return float(np.mean(data))
        elif key == "sigma_mu":
            if params is None:
                raise ValueError("'params' must be provided to resolve 'sigma_mu' automatically.")
            if "sigma_mu_auto_scale" not in params:
                raise ValueError("'sigma_mu_auto_scale' must be defined in params to resolve 'sigma_mu' automatically.")
            sample_std = np.std(data, ddof=1)
            scale = params["sigma_mu_auto_scale"]
            return float(sample_std * scale)
        else:
            raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")


class NormalMeanNormalGammaSkewNormal(BDFDistribution[NormalMeanNormalGammaSkewNormalParams]):
    r"""Skew-Normal with Normal priors on mean and skewness gamma.

    **Usage:** ``dist="NormalMeanNormalGammaSkewNormal"``

    **Model:**

    *   **Prior on mean:** :math:`\mu \sim \mathcal{N}(\mu_\mu, \sigma_\mu^2)`
    *   **Prior on skewness:** :math:`\gamma \sim \mathcal{N}(\mu_\gamma, \sigma_\gamma^2)`
    *   **Likelihood:** :math:`y \mid \alpha, \xi, \omega \sim \text{SkewNormal}(\alpha, \xi, \omega)`

    Parameters
    ----------
    mu_mu : float or "auto", default=0.0
        Prior mean for the data mean. If ``"auto"``, set to the sample mean at
        fit time.
    sigma_mu : float or "auto", default=1.0
        Prior standard deviation for the data mean. If ``"auto"``, set to the
        sample standard deviation times ``sigma_mu_auto_scale`` at fit time.
    sigma_mu_auto_scale : float, default=1.0
        Multiplier applied when ``sigma_mu="auto"``. Ignored otherwise.
    mu_gamma : float, default=0.0
        Prior mean for skewness :math:`\gamma`.
    sigma_gamma : float, default=0.5
        Prior standard deviation for skewness :math:`\gamma`.
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    NormalMeanPseudoAlphaSkewNormal : Alternative with pseudo-prior on alpha.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMeanNormalGammaSkewNormalParams

    # Capabilities
    _supports_nle = False  # Not conjugate
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: dict[str, Any] | NormalMeanNormalGammaSkewNormalParams):
        super().__init__(params)
        self.mu_mu = self.params.mu_mu
        self.sigma_mu = self.params.sigma_mu
        self.mu_gamma = self.params.mu_gamma
        self.sigma_gamma = self.params.sigma_gamma

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate posterior parameters using modular moment-based updates.

        Returns dict with:
        - posterior_alpha: Posterior skewness parameter
        - posterior_xi: Posterior location parameter
        - posterior_omega: Posterior scale parameter
        """
        n = data.shape[0]
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        sample_skewness = np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-7)) ** 3)

        # Variance of skewness estimator
        var_skewness = (6 * n * (n - 1)) / ((n - 2) * (n + 1) * (n + 3))

        # Step 1: Posterior skewness gamma (Normal prior)
        precision_prior = 1 / self.sigma_gamma**2
        precision_data = n / (var_skewness + 1e-7)
        posterior_skewness = (precision_prior * self.mu_gamma + precision_data * sample_skewness) / (
            precision_prior + precision_data
        )
        posterior_skewness = np.clip(posterior_skewness, -0.995, 0.995)

        # Step 2: Posterior mean for location (Normal prior update)
        precision_prior_mu = 1 / self.sigma_mu**2
        precision_data_mu = n / (sample_var + 1e-7)
        posterior_mean = (precision_prior_mu * self.mu_mu + precision_data_mu * sample_mean) / (
            precision_prior_mu + precision_data_mu
        )

        # Step 3: Convert skewness to alpha via moment-matching
        delta = np.sign(posterior_skewness) * np.sqrt(
            np.pi
            / 2
            * np.abs(posterior_skewness) ** (2 / 3)
            / (np.abs(posterior_skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        posterior_alpha = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 3)

        # Step 4: Estimate omega from sample variance adjusted for skewness
        posterior_omega = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)

        # Handle numerical issues
        if np.isnan(posterior_omega) or np.isinf(posterior_omega):
            warnings.warn("Posterior omega invalid, setting to large value.", UserWarning)
            posterior_omega = 1e7
        elif posterior_omega <= 0:
            warnings.warn("Posterior omega non-positive, using small positive value.", UserWarning)
            posterior_omega = 1e-7

        # Step 5: Recover xi from posterior mean
        posterior_xi = posterior_mean - posterior_omega * posterior_alpha / np.sqrt(1 + posterior_alpha**2) * np.sqrt(
            2 / np.pi
        )

        return {
            "posterior_alpha": float(posterior_alpha),
            "posterior_xi": float(posterior_xi),
            "posterior_omega": float(posterior_omega),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in log-likelihood using posterior point estimates."""
        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]
        return skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega)

    def _num_parameters(self) -> int:
        """Skew-normal has 3 parameters: α, ξ, ω."""
        return 3

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from the fitted skew-normal distribution."""
        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]
        return np.asarray(skewnorm.rvs(a=alpha, loc=xi, scale=omega, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray) -> np.ndarray:
        """Validate data for skew-normal distribution."""
        arr = np.asarray(data, dtype=float)
        if np.any(np.isnan(arr)):
            raise ValueError("Data contains NaN values")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Data contains infinite values")
        return arr

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean: E[X] = ξ + ω·δ·√(2/π) where δ = α/√(1+α²)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]

        delta = alpha / np.sqrt(1 + alpha**2)
        return float(xi + omega * delta * np.sqrt(2 / np.pi))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance: Var[X] = ω²(1 - 2δ²/π) where δ = α/√(1+α²)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        omega = params["posterior_omega"]

        delta = alpha / np.sqrt(1 + alpha**2)
        return float(omega**2 * (1 - 2 * delta**2 / np.pi))

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        For NormalMeanNormalGammaSkewNormal:
        - 'mu_mu': Use sample mean
        - 'sigma_mu': Use sample std * sigma_mu_auto_scale
        """
        if key == "mu_mu":
            return float(np.mean(data))
        elif key == "sigma_mu":
            if params is None:
                raise ValueError("'params' must be provided to resolve 'sigma_mu' automatically.")
            if "sigma_mu_auto_scale" not in params:
                raise ValueError("'sigma_mu_auto_scale' must be defined in params to resolve 'sigma_mu' automatically.")
            sample_std = np.std(data, ddof=1)
            scale = params["sigma_mu_auto_scale"]
            return float(sample_std * scale)
        else:
            raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")


class NormalXiNormalAlphaSkewNormalMAP(BDFDistribution[NormalXiNormalAlphaSkewNormalMAPParams]):
    r"""Skew-Normal with Normal priors on location and skewness via MAP estimation.

    **Usage:** ``dist="NormalXiNormalAlphaSkewNormalMAP"``

    **Model:**

    *   **Prior on location:** :math:`\xi \sim \mathcal{N}(\mu_\xi, \sigma_\xi^2)`
    *   **Prior on skewness:** :math:`\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)`
    *   **Likelihood:** :math:`y \mid \alpha, \xi, \omega \sim \text{SkewNormal}(\alpha, \xi, \omega)`

    Uses Nelder-Mead optimization to find MAP estimates of :math:`(\xi, \alpha)`.

    Parameters
    ----------
    mu_xi : float or "auto", default=0.0
        Prior mean for location :math:`\xi`. If ``"auto"``, set to the sample
        mean at fit time.
    sigma_xi : float or "auto", default=1.0
        Prior standard deviation for location :math:`\xi`. If ``"auto"``, set to
        the sample standard deviation times ``sigma_xi_auto_scale`` at fit time.
    sigma_xi_auto_scale : float, default=1.0
        Multiplier applied when ``sigma_xi="auto"``. Ignored otherwise.
    mu_alpha : float, default=0.0
        Prior mean for skewness :math:`\alpha`.
    sigma_alpha : float, default=5.0
        Prior standard deviation for skewness :math:`\alpha`.
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    NormalMeanPseudoAlphaSkewNormal : Faster modular inference alternative.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalXiNormalAlphaSkewNormalMAPParams

    # Capabilities
    _supports_nle = False  # Not conjugate
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: dict[str, Any] | NormalXiNormalAlphaSkewNormalMAPParams):
        super().__init__(params)
        self.mu_xi = self.params.mu_xi
        self.sigma_xi = self.params.sigma_xi
        self.mu_alpha = self.params.mu_alpha
        self.sigma_alpha = self.params.sigma_alpha

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Calculate MAP posterior parameters via numerical optimization.

        Returns dict with:
        - posterior_alpha: MAP skewness parameter
        - posterior_xi: MAP location parameter
        - posterior_omega: Scale parameter (from moment-matching, fixed during optimization)
        """
        from scipy.optimize import minimize

        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # Initial estimates from moment matching
        sample_skewness = np.clip(
            np.mean(((data - sample_mean) / np.sqrt(sample_var + 1e-7)) ** 3),
            -0.995,
            0.995,
        )
        delta = np.sign(sample_skewness) * np.sqrt(
            np.pi
            / 2
            * np.abs(sample_skewness) ** (2 / 3)
            / (np.abs(sample_skewness) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3))
        )
        alpha_init = np.sign(delta) * (np.abs(delta) / np.sqrt(1 - delta**2)) ** (1 / 3)
        omega_init = np.sqrt(sample_var) / np.sqrt(1 - 2 * delta**2 / np.pi)

        # Handle invalid omega
        if not np.isfinite(omega_init) or omega_init <= 0:
            omega_init = np.sqrt(sample_var)

        xi_init = sample_mean - omega_init * alpha_init / np.sqrt(1 + alpha_init**2) * np.sqrt(2 / np.pi)

        # Define negative log posterior
        def neg_log_posterior(theta: np.ndarray) -> float:
            xi, alpha = theta

            # Prior terms (Gaussian priors)
            prior_xi = -0.5 * ((xi - self.mu_xi) / self.sigma_xi) ** 2
            prior_alpha = -0.5 * ((alpha - self.mu_alpha) / self.sigma_alpha) ** 2

            # Likelihood term
            try:
                likelihood = np.sum(skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega_init))
                if not np.isfinite(likelihood):
                    return 1e10
            except Exception:
                return 1e10

            return -(prior_xi + prior_alpha + likelihood)

        # Optimize with limited iterations for speed
        x0 = np.array([xi_init, alpha_init])
        result = minimize(
            neg_log_posterior,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 10, "xatol": 1e-4, "fatol": 1e-4},
        )

        posterior_xi, posterior_alpha = result.x
        posterior_omega = omega_init

        return {
            "posterior_alpha": float(posterior_alpha),
            "posterior_xi": float(posterior_xi),
            "posterior_omega": float(posterior_omega),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in log-likelihood using MAP estimates."""
        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]
        return skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega)

    def _num_parameters(self) -> int:
        """Skew-normal has 3 parameters: α, ξ, ω."""
        return 3

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        """Sample from the fitted skew-normal distribution."""
        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]
        return np.asarray(skewnorm.rvs(a=alpha, loc=xi, scale=omega, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray) -> np.ndarray:
        """Validate data for skew-normal distribution."""
        arr = np.asarray(data, dtype=float)
        if np.any(np.isnan(arr)):
            raise ValueError("Data contains NaN values")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Data contains infinite values")
        return arr

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get posterior mean: E[X] = ξ + ω·δ·√(2/π) where δ = α/√(1+α²)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        xi = params["posterior_xi"]
        omega = params["posterior_omega"]

        delta = alpha / np.sqrt(1 + alpha**2)
        return float(xi + omega * delta * np.sqrt(2 / np.pi))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get posterior variance: Var[X] = ω²(1 - 2δ²/π) where δ = α/√(1+α²)."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        alpha = params["posterior_alpha"]
        omega = params["posterior_omega"]

        delta = alpha / np.sqrt(1 + alpha**2)
        return float(omega**2 * (1 - 2 * delta**2 / np.pi))

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        For NormalXiNormalAlphaSkewNormalMAP:
        - 'mu_xi': Use sample mean
        - 'sigma_xi': Use sample std * sigma_xi_auto_scale
        """
        if key == "mu_xi":
            return float(np.mean(data))
        elif key == "sigma_xi":
            if params is None:
                raise ValueError("'params' must be provided to resolve 'sigma_xi' automatically.")
            if "sigma_xi_auto_scale" not in params:
                raise ValueError("'sigma_xi_auto_scale' must be defined in params to resolve 'sigma_xi' automatically.")
            sample_std = np.std(data, ddof=1)
            scale = params["sigma_xi_auto_scale"]
            return float(sample_std * scale)
        else:
            raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")
