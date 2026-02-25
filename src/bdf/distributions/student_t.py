import warnings
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field
from scipy.stats import t as student_t

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams

# ============================================================================
# PARAMS
# ============================================================================


class FrequentistStudentTParams(BDFDistributionParams):
    """Frequentist Student-t distribution (robust, non-conjugate).

    y ~ t_ν(μ, σ)

    - df=None: estimate ν jointly with (μ, σ)
    - df=float: treat ν as fixed hyperparameter
    """

    # Degrees of freedom (None => estimated per node)
    df: float | None = Field(
        default=None,
        gt=2.0,
        description="Degrees of freedom ν. If None, ν is estimated; otherwise treated as fixed (ν > 2 for finite variance).",
    )

    estimation_method: Literal["em", "mle", "mom"] = Field(
        default="mom",
        description=(
            "Parameter estimation method for (μ, σ[, ν]): "
            "'em' uses a fast IRLS/EM-like scheme; 'mle' uses generic numerical optimization, 'mom' uses method of moments (4th moment to obtain nu)."
        ),
    )

    score_method: Literal["nll"] = Field(
        default="nll",
        description="Student-t is non-conjugate; only 'nll' (negative log-likelihood) is supported.",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="Posterior predictive not supported (no conjugate pair); always use plug-in.",
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATION
# ============================================================================


class FrequentistStudentT(BDFDistribution[FrequentistStudentTParams]):
    r"""Student-t distribution with frequentist estimation.

    **Usage:** ``dist="FrequentistStudentT"``

    Robust alternative to Normal for heavy-tailed data. Parameters are
    estimated from data via method of moments with no prior regularization.

    .. math:: y \sim t_\nu(\mu, \sigma)

    Parameters
    ----------
    df : float or None, default=None
        Degrees of freedom :math:`\nu`. If None, estimated jointly with
        :math:`\mu` and :math:`\sigma`. If provided, treated as fixed
        (must be > 2 for finite variance).
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    NormalMeanStudentT : Student-t with Normal prior on the mean.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = FrequentistStudentTParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: dict[str, Any] | FrequentistStudentTParams):
        super().__init__(params)
        self.df = self.params.df
        self.estimation_method = self.params.estimation_method

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        if data.size == 0:
            raise ValueError("Data must contain at least one observation.")

        try:
            if self.estimation_method == "em":
                mu, sigma, df = self._fit_em(data)
            elif self.estimation_method == "mle":
                mu, sigma, df = self._fit_mle(data)
            elif self.estimation_method == "mom":
                mu, sigma, df = self._fit_mom(data)
            else:
                raise ValueError(f"Unknown estimation_method: {self.estimation_method}")
        except Exception:
            # Fallback to Normal MLE if Student-t fit fails
            warnings.warn("Student-t parameter estimation failed; falling back to Normal MLE.", RuntimeWarning)
            mu = float(np.mean(data))
            sigma = float(np.std(data, ddof=1)) if data.size > 1 else 1.0
            df = float(self.df) if self.df is not None else 5.0

        # Safety guards (preserve user df when fixed)
        sigma = max(sigma, 1e-8)
        if self.df is not None:
            df = float(self.df)
        else:
            df = max(df, 2.05)

        return {"mu": float(mu), "sigma": float(sigma), "df": float(df)}

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        mu = params["mu"]
        sigma = params["sigma"]
        df = params["df"]
        return student_t.logpdf(data, df=df, loc=mu, scale=sigma)

    def _num_parameters(self) -> int:
        # μ, σ always estimated; ν only if df is None
        return 3 if self.df is None else 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        mu = params["mu"]
        sigma = params["sigma"]
        df = params["df"]
        return np.array(student_t.rvs(df=df, loc=mu, scale=sigma, size=size, random_state=random_state))

    def validate_targets(self, data: np.ndarray):
        if data.ndim != 1:
            raise ValueError("Data must be 1D.")
        if data.size == 0:
            raise ValueError("Data cannot be empty.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values.")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'.")
            params = self.calc_posterior_params(data)
        return float(params["mu"])

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'.")
            params = self.calc_posterior_params(data)
        df = params["df"]
        sigma = params["sigma"]
        if df <= 2:
            return float(np.inf)
        return float(sigma**2 * df / (df - 2))

    # ========================================================================
    # INTERNAL FIT METHODS
    # ========================================================================

    def _fit_em(self, data: np.ndarray) -> tuple[float, float, float]:
        y = data.astype(float)
        n = y.size

        mu = float(np.mean(y))
        sigma2 = float(np.var(y, ddof=1)) if n > 1 else 1.0
        sigma2 = max(sigma2, 1e-16)

        df = float(self.df) if self.df is not None else 5.0

        # EM loop: ONLY update (μ, σ) if df is fixed
        for _ in range(10):
            r2 = ((y - mu) ** 2) / sigma2
            w = (df + 1.0) / (df + r2)

            w_sum = np.sum(w)
            if w_sum <= 0:
                break

            mu = float(np.sum(w * y) / w_sum)
            sigma2 = float(np.sum(w * (y - mu) ** 2) / n)
            sigma2 = max(sigma2, 1e-16)

        # If df is free, update ONCE after EM convergence (not inside loop)
        if self.df is None:
            df = self._update_df_newton(df, y, mu, sigma2)

        return mu, np.sqrt(sigma2), df

    def _update_df_newton(self, df_init: float, y: np.ndarray, mu: float, sigma2: float) -> float:
        """Single Newton iteration for df after EM converges."""
        from scipy.optimize import brentq

        r2 = ((y - mu) ** 2) / sigma2
        w = (df_init + 1.0) / (df_init + r2)

        mean_log_w = np.mean(np.log(np.maximum(w, 1e-12)))
        mean_w = np.mean(w)

        # Solve: ψ((ν+1)/2) - log((ν+1)/2) - ψ(ν/2) + log(ν/2) + mean_log_w - mean_w = 0
        # Use root finding instead of Newton (more stable for small nodes)
        from scipy.special import digamma

        def objective(nu):
            if nu <= 1.0:
                return 1e9
            lhs = digamma((nu + 1) / 2) - np.log((nu + 1) / 2)
            rhs = digamma(nu / 2) - np.log(nu / 2) - mean_log_w + mean_w
            return lhs - rhs

        try:
            df_new = brentq(objective, 2.05, 100.0)
            return float(df_new)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            warnings.warn(f"DF update via root finding failed: {e}; keeping previous df.", RuntimeWarning)
            return df_init  # Fallback if solver fails

    def _fit_mle(self, data: np.ndarray) -> tuple[float, float, float]:
        from scipy.optimize import minimize

        y = data.astype(float)
        mu0 = float(np.mean(y))
        sigma0 = max(float(np.std(y, ddof=1)), 1e-8) if y.size > 1 else 1.0

        if self.df is None:
            from scipy.special import expit

            # Parameterize df via sigmoid: df = 2.05 + 97.95 * sigmoid(x)
            # This enforces df ∈ [2.05, 100] smoothly
            df0 = 5.0
            logit_df0 = np.log((df0 - 2.05) / (100 - df0))

            x0 = np.array([mu0, np.log(sigma0), logit_df0])

            def optim_nll(theta):
                mu, log_sigma, logit_df = theta
                sigma = np.exp(log_sigma)
                df = 2.05 + 97.95 * expit(logit_df)  # Maps R → [2.05, 100]

                ll = student_t.logpdf(y, df=df, loc=mu, scale=sigma)
                if not np.all(np.isfinite(ll)):
                    return 1e12
                return -np.sum(ll)

            res = minimize(optim_nll, x0=x0, method="L-BFGS-B")
            mu_hat, log_sigma_hat, logit_df_hat = res.x
            sigma_hat = np.exp(log_sigma_hat)
            df_hat = 2.05 + 97.95 * expit(logit_df_hat)
        else:
            # Optimize (mu, log_sigma) only, df fixed
            x0 = np.array([mu0, np.log(sigma0)])
            fixed_df = float(self.df)

            def nll(theta: np.ndarray) -> float:
                mu, log_sigma = theta
                sigma = np.exp(log_sigma)
                return float(-np.sum(student_t.logpdf(y, df=fixed_df, loc=mu, scale=sigma)))

            res = minimize(nll, x0=x0, method="L-BFGS-B")
            mu_hat, log_sigma_hat = res.x
            sigma_hat = float(np.exp(log_sigma_hat))
            df_hat = fixed_df

        return float(mu_hat), sigma_hat, df_hat

    def _fit_mom(self, data: np.ndarray) -> tuple[float, float, float]:
        """Fit using method of moments: μ = mean, σ = s * ((v-2)/ν), ν from 4th moment."""
        y = data.astype(float)
        n = y.size

        mu = float(np.mean(y))
        sigma2 = float(np.var(y, ddof=1)) if n > 1 else 1.0
        sigma2 = max(sigma2, 1e-10)

        if self.df is None:
            # Method of moments for df using excess kurtosis
            m4 = np.mean((y - mu) ** 4)
            excess_kurtosis = m4 / (sigma2**2) - 3
            if excess_kurtosis <= 0:
                df = 100.0
            else:
                df = max(2.05, min(100.0, 6 / excess_kurtosis + 4))
        else:
            df = self.df

        return mu, np.sqrt(sigma2 * (df - 2) / df), float(df)


# ============================================================================
# NORMAL MEAN PRIOR STUDENT-T (CLT moment prior)
# ============================================================================


class NormalMeanStudentTParams(BDFDistributionParams):
    """Student-t with Normal prior on the mean (CLT moment prior).

    y ~ t_ν(μ, σ)

    Prior: μ ~ N(mu_mean, sigma_mean²)
    Posterior mean via Normal-Normal conjugate update (CLT approximation).
    ν estimated via Method of Moments (excess kurtosis) or fixed.
    σ recovered from variance identity: Var(Y) = σ²ν/(ν-2).
    """

    mu_mean: float = Field(default=0.0, description="Prior mean for μ.")
    sigma_mean: float = Field(default=1.0, gt=0, description="Prior std for μ.")
    sigma_mean_auto_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale factor for automatic sigma_mean if 'auto' is used. Resolved upon `fit`, discarded from final params.",
        exclude=True,
    )
    df: float | None = Field(
        default=None,
        gt=2.0,
        description="Degrees of freedom ν. If None, ν is estimated via MoM; otherwise treated as fixed (ν > 2 for finite variance).",
    )
    score_method: Literal["nll"] = Field(
        default="nll",
        description="Student-t is non-conjugate; only 'nll' (negative log-likelihood) is supported.",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="Posterior predictive not supported (no conjugate pair); always use plug-in.",
    )


class NormalMeanStudentT(BDFDistribution[NormalMeanStudentTParams]):
    r"""Student-t distribution with Normal prior on the mean.

    **Usage:** ``dist="NormalMeanStudentT"``

    Combines heavy-tail flexibility of Student-t with Bayesian regularization
    on the location parameter via a Normal prior.

    **Model:**

    *   **Prior:** :math:`\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)`
    *   **Likelihood:** :math:`y \sim t_\nu(\mu, \sigma)`
    *   **Inference:** Posterior mean via Normal-Normal update (CLT approximation).
        :math:`\nu` estimated via method of moments or fixed.

    Parameters
    ----------
    mu_mean : float or "auto", default=0.0
        Prior mean for :math:`\mu`. If ``"auto"``, set to the sample mean of
        ``y`` at fit time.
    sigma_mean : float or "auto", default=1.0
        Prior standard deviation for :math:`\mu`. If ``"auto"``, set to
        ``std(y) * sigma_mean_auto_scale`` at fit time.
    sigma_mean_auto_scale : float, default=1.0
        Multiplier applied when ``sigma_mean="auto"``. Ignored otherwise.
    df : float or None, default=None
        Degrees of freedom :math:`\nu`. If None, estimated via method of
        moments. If provided, treated as fixed (must be > 2).
    score_method : {"nll"}
        Only NLL scoring is supported (non-conjugate model).

    See Also
    --------
    BDFDistributionParams : Common scoring and inference parameters shared by all distributions.
    FrequentistStudentT : Student-t without prior regularization.
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = NormalMeanStudentTParams

    _supports_nle = False
    _has_fast_loo_cv = False
    _has_fast_kfold_cv = False
    _supports_posterior_predictive = False

    def __init__(self, params: dict[str, Any] | NormalMeanStudentTParams):
        super().__init__(params)
        self.mu_mean = float(self.params.mu_mean)
        self.sigma_mean = float(self.params.sigma_mean)
        self.df = self.params.df

    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        if key == "mu_mean":
            return float(np.mean(data))
        if key == "sigma_mean":
            if params is None:
                raise ValueError("'params' must be provided to resolve 'sigma_mean' automatically.")
            if "sigma_mean_auto_scale" not in params:
                raise ValueError(
                    "'sigma_mean_auto_scale' must be defined in params to resolve 'sigma_mean' automatically."
                )
            sample_std = np.std(data, ddof=1)
            scale = params["sigma_mean_auto_scale"]
            return float(sample_std * scale)
        raise ValueError(f"Unknown parameter '{key}' for auto resolution in {cls.__name__}")

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float]:
        """Bayesian update on mean, then MoM for df and sigma."""
        n = float(len(data))

        if n < 2:
            df = float(self.df) if self.df is not None else 5.0
            return {"mu": self.mu_mean, "sigma": 1.0, "df": df, "posterior_mean_mu": self.mu_mean}

        sample_mean = float(np.mean(data))
        sample_var = float(np.var(data, ddof=1))
        sample_var = max(sample_var, 1e-10)

        # 1. Bayesian update on mean (Normal-Normal conjugate, CLT approximation)
        prior_prec = 1.0 / (self.sigma_mean**2)
        data_prec = n / sample_var
        post_prec = prior_prec + data_prec
        post_mu = (prior_prec * self.mu_mean + data_prec * sample_mean) / post_prec

        # 2. Estimate df from excess kurtosis (MoM)
        if self.df is not None:
            df = float(self.df)
        else:
            m4 = float(np.mean((data - sample_mean) ** 4))
            excess_kurtosis = m4 / (sample_var**2) - 3
            if excess_kurtosis <= 0:
                df = 100.0
            else:
                df = max(2.05, min(100.0, 6.0 / excess_kurtosis + 4.0))

        # 3. Recover sigma from variance identity: Var(Y) = σ²ν/(ν-2)
        sigma = float(np.sqrt(sample_var * (df - 2) / df))
        sigma = max(sigma, 1e-8)

        return {
            "mu": float(post_mu),
            "sigma": sigma,
            "df": df,
            "posterior_mean_mu": float(post_mu),
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        return student_t.logpdf(data, df=params["df"], loc=params["mu"], scale=params["sigma"])

    def _num_parameters(self) -> int:
        return 3 if self.df is None else 2

    def _sample_posterior_params(self, params: dict[str, float], size: int, random_state: int) -> np.ndarray:
        return np.array(
            student_t.rvs(
                df=params["df"], loc=params["mu"], scale=params["sigma"], size=size, random_state=random_state
            )
        )

    def validate_targets(self, data: np.ndarray):
        if data.ndim != 1:
            raise ValueError("Data must be 1D.")
        if data.size == 0:
            raise ValueError("Data cannot be empty.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values.")

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'.")
            params = self.calc_posterior_params(data)
        return float(params["mu"])

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'.")
            params = self.calc_posterior_params(data)
        df = params["df"]
        sigma = params["sigma"]
        if df <= 2:
            return float(np.inf)
        return float(sigma**2 * df / (df - 2))
