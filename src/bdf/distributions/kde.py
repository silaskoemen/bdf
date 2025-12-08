import warnings
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator
from scipy.fft import irfft, rfft

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams

# ============================================================================
# PARAMS CLASSES
# ============================================================================


class KDEParams(BDFDistributionParams):
    """Parameters for KDE distribution.

    **Model Specification:**
    - Non-parametric density estimation using kernel smoothing
    - Bandwidth selection via rule-of-thumb or fixed value

    Parameters
    ----------
    bandwidth : float | str
        Bandwidth for KDE. Can be positive float, 'scott', or 'silverman'.
    kernel : str
        Kernel type: 'gaussian' or 'epanechnikov'.
    min_bandwidth : float
        Minimum allowable bandwidth to prevent numerical issues.
    """

    # KDE hyperparameters
    bandwidth: str | float = Field(
        default="scott",
        description="Bandwidth: positive float, 'scott', or 'silverman'.",
    )
    kernel: Literal["gaussian", "epanechnikov"] = Field(
        default="gaussian",
        description="Kernel type for KDE.",
    )
    min_bandwidth: float = Field(
        default=1e-6,
        gt=1e-10,
        description="Minimum allowable bandwidth.",
    )

    # FFT configuration - DEFERRED FOR NOW
    use_fft: bool = Field(
        default=False,
        description="Use FFT-based KDE for likelihood calculations.",
    )
    # fft_grid_points: int = Field(
    #     default=512,
    #     ge=64,
    #     description="Number of grid points for FFT-based KDE.",
    # )
    # fft_grid_edges: np.ndarray | None = Field(
    #     default=None,
    #     description="Precomputed grid edges for FFT (set by regressor).",
    # )
    # fft_kernel_rfft: np.ndarray | None = Field(
    #     default=None,
    #     description="Precomputed kernel FFT at reference bandwidth=1 (set by regressor).",
    # )
    # fft_grid_delta: float | None = Field(
    #     default=None,
    #     description="Grid spacing for FFT (set by regressor).",
    # )

    # Scoring defaults for non-parametric model
    score_method: Literal["nle", "nll"] = Field(
        default="nll",
        description="KDE uses NLL scoring (no closed-form evidence).",
    )
    score_correction: Literal["aic", "bic", "loo_cv", "kfold_cv"] | None = Field(
        default="loo_cv",
        description="Default to LOO-CV for KDE (natural cross-validation).",
    )
    use_posterior_predictive: bool = Field(
        default=False,
        description="KDE doesn't have posterior predictive in traditional sense.",
    )

    model_config = {"extra": "forbid"}

    # @model_validator(mode="after")
    # def validate_fft_config(self) -> "KDEParams":
    #     if self.use_fft and self.kernel != "gaussian":
    #         raise ValueError("FFT-based KDE only supports Gaussian kernel.")
    #     return self
    @field_validator("use_fft")
    def warn_fft_deferred(cls, v: bool) -> bool:
        if v:
            warnings.warn("FFT-based KDE is currently deferred and not implemented.")
        return False


class BayesianKDEParams(KDEParams):
    """Parameters for Bayesian KDE with prior on bandwidth.

    Extends KDE with pseudo-Bayesian bandwidth estimation:
    posterior_h = (m_h * prior_h + n * data_h) / (m_h + n)
    """

    prior_h: float = Field(
        default=1.0,
        gt=0,
        description="Prior bandwidth for pseudo-Bayesian estimation.",
    )
    m_h: float = Field(
        default=1.0,
        gt=0,
        description="Prior weight for bandwidth (pseudo sample size).",
    )


# ============================================================================
# DISTRIBUTION IMPLEMENTATIONS
# ============================================================================


class KDE(BDFDistribution):
    """Kernel Density Estimation distribution.

    **String Alias:** ``'kde'``

    Non-parametric density estimation using kernel smoothing.
    Supports both direct evaluation and FFT-based computation.

    Parameters
    ----------
    bandwidth : float or {'scott', 'silverman'}, default='scott'
        Bandwidth selection method or fixed value.
    kernel : {'gaussian', 'epanechnikov'}, default='gaussian'
        Kernel function.
    use_fft : bool, default=False
        Use FFT for fast density evaluation (requires precomputed grid).
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = KDEParams

    # Capabilities
    _supports_nle = False  # No closed-form evidence
    _has_fast_loo_cv = True  # O(n²) but vectorized
    _has_fast_kfold_cv = True
    _supports_posterior_predictive = False

    def __init__(self, params: KDEParams):
        super().__init__(params)

    # ========================================================================
    # REQUIRED METHODS
    # ========================================================================

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float | np.ndarray]:
        """Calculate KDE parameters from data.

        Returns dict with:
        - data: The kernel centers (copy of input data)
        - bandwidth: Computed or specified bandwidth
        - n: Number of data points
        """
        data = np.asarray(data, dtype=float).ravel()
        n = data.size

        if n < 2:
            raise ValueError("KDE requires at least 2 data points.")

        bandwidth = self._compute_bandwidth(data)

        return {
            "data": data.copy(),
            "bandwidth": float(bandwidth),
            "n": n,
        }

    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Compute log-likelihood using KDE density estimate."""
        eval_points = np.asarray(data, dtype=float).ravel()
        ref_data = params["data"]
        h = params["bandwidth"]

        if self.params.use_fft and self._fft_ready():
            return self._fft_log_likelihood(eval_points, ref_data, h)
        else:
            return self._direct_log_likelihood(eval_points, ref_data, h)

    def _num_parameters(self) -> int:
        """Effective number of parameters for KDE.

        For KDE, this is approximately n/h (bandwidth controls complexity).
        Return 1 as conservative estimate (just bandwidth).
        """
        return 1

    def _sample_posterior_params(
        self, params: dict[str, float | np.ndarray], size: int, random_state: int
    ) -> np.ndarray:
        """Sample from KDE distribution."""
        data = np.asarray(params["data"], dtype=float).ravel()
        h = params["bandwidth"]

        if data.size == 0:
            raise ValueError("No reference data for KDE sampling.")

        rng = np.random.default_rng(random_state)

        # Sample kernel centers
        centers = rng.choice(data, size=size, replace=True)

        # Add kernel noise
        if self.params.kernel == "gaussian":
            noise = rng.normal(0, h, size=size)
        elif self.params.kernel == "epanechnikov":
            noise = h * self._sample_epanechnikov(size, rng)
        else:
            raise ValueError(f"Unknown kernel: {self.params.kernel}")

        return centers + noise

    def validate_targets(self, data: np.ndarray) -> np.ndarray:
        """Validate data for KDE."""
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 0:
            raise ValueError("Data must contain at least one observation.")
        if not np.isfinite(arr).all():
            raise ValueError("Data must be finite real numbers.")
        if arr.shape[0] < 2:
            raise ValueError("KDE requires at least 2 observations.")
        return arr

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Mean of KDE is sample mean."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        return float(np.mean(params["data"]))

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Variance of KDE = sample variance + kernel variance."""
        if params is None:
            if data is None:
                raise ValueError("Provide either 'data' or 'params'")
            params = self.calc_posterior_params(data)

        sample_var = float(np.var(params["data"], ddof=1))
        h = params["bandwidth"]

        if self.params.kernel == "gaussian":
            kernel_var = h**2
        elif self.params.kernel == "epanechnikov":
            kernel_var = 0.2 * h**2  # Second moment of Epanechnikov
        else:
            kernel_var = h**2

        return sample_var + kernel_var

    # ========================================================================
    # BANDWIDTH COMPUTATION
    # ========================================================================

    def _compute_bandwidth(self, data: np.ndarray) -> float:
        """Compute bandwidth from data or params."""
        bw = self.params.bandwidth

        if isinstance(bw, str):
            if bw == "scott":
                return self._bandwidth_scott(data)
            elif bw == "silverman":
                return self._bandwidth_silverman(data)
            else:
                raise ValueError(f"Unknown bandwidth method: {bw}")
        elif isinstance(bw, (int, float)):
            if bw <= 0:
                raise ValueError("Bandwidth must be positive.")
            return max(float(bw), self.params.min_bandwidth)
        else:
            raise TypeError(f"Invalid bandwidth type: {type(bw)}")

    def _bandwidth_scott(self, data: np.ndarray) -> float:
        """Scott's rule: h = σ * n^(-1/5)"""
        n = data.size
        std = float(np.std(data, ddof=1))
        return max(self.params.min_bandwidth, std * n ** (-0.2))

    def _bandwidth_silverman(self, data: np.ndarray) -> float:
        """Silverman's rule: h = 0.9 * min(σ, IQR/1.349) * n^(-1/5)"""
        n = data.size
        std = float(np.std(data, ddof=1))
        iqr = float(np.subtract(*np.percentile(data, [75, 25])))

        scale = min(std, iqr / 1.349) if iqr > 0 else std
        if scale <= 0:
            scale = std

        return max(self.params.min_bandwidth, 0.9 * scale * n ** (-0.2))

    # ========================================================================
    # DIRECT LIKELIHOOD COMPUTATION
    # ========================================================================

    def _direct_log_likelihood(self, eval_points: np.ndarray, ref_data: np.ndarray, h: float) -> np.ndarray:
        """Direct O(n*m) kernel evaluation."""
        n_ref = ref_data.size

        if n_ref == 0:
            return np.full(eval_points.size, -np.inf)

        # Compute kernel contributions: shape (n_eval, n_ref)
        if self.params.kernel == "gaussian":
            log_kernel = self._gaussian_log_kernel(eval_points, ref_data, h)
        elif self.params.kernel == "epanechnikov":
            log_kernel = self._epanechnikov_log_kernel(eval_points, ref_data, h)
        else:
            raise ValueError(f"Unknown kernel: {self.params.kernel}")

        # Log-sum-exp over reference points, then subtract log(n)
        return np.logaddexp.reduce(log_kernel, axis=1) - np.log(n_ref)

    def _gaussian_log_kernel(self, eval_points: np.ndarray, ref_data: np.ndarray, h: float) -> np.ndarray:
        """Gaussian kernel log-contributions (n_eval × n_ref)."""
        diffs = (eval_points[:, None] - ref_data[None, :]) / h
        log_k = -0.5 * diffs**2
        log_k -= 0.5 * np.log(2 * np.pi)
        log_k -= np.log(h)
        return log_k

    def _epanechnikov_log_kernel(self, eval_points: np.ndarray, ref_data: np.ndarray, h: float) -> np.ndarray:
        """Epanechnikov kernel log-contributions (n_eval × n_ref)."""
        u = (eval_points[:, None] - ref_data[None, :]) / h
        log_k = np.full_like(u, -np.inf, dtype=float)
        mask = np.abs(u) <= 1.0
        log_k[mask] = np.log(0.75) + np.log(1 - u[mask] ** 2) - np.log(h)
        return log_k

    # ========================================================================
    # FFT-BASED LIKELIHOOD (FAST)
    # ========================================================================

    def _fft_ready(self) -> bool:
        """Check if FFT precomputation is available."""
        return (
            self.params.fft_grid_edges is not None
            and self.params.fft_kernel_rfft is not None
            and self.params.fft_grid_delta is not None
        )

    def _fft_log_likelihood(self, eval_points: np.ndarray, ref_data: np.ndarray, h: float) -> np.ndarray:
        """FFT-based KDE evaluation.

        Steps:
        1. Bin reference data into grid
        2. FFT of bin counts
        3. Multiply by scaled kernel FFT
        4. IFFT to get density on grid
        5. Interpolate to evaluation points
        """
        edges = self.params.fft_grid_edges
        kernel_rfft_ref = self.params.fft_kernel_rfft
        delta = self.params.fft_grid_delta
        n_grid = self.params.fft_grid_points

        n_ref = ref_data.size

        # Step 1: Bin counts
        counts, _ = np.histogram(ref_data, bins=edges)
        counts = counts.astype(float)

        # Step 2: FFT of counts
        counts_rfft = rfft(counts)

        # Step 3: Scale kernel FFT by bandwidth
        # For Gaussian: K(x/h)/h, so in frequency domain: h * K_hat(h*ω)
        # With reference bandwidth=1, we need to rescale frequencies
        freq = np.fft.rfftfreq(n_grid, d=delta)
        kernel_rfft_scaled = kernel_rfft_ref * np.exp(-2 * np.pi**2 * freq**2 * (h**2 - 1))

        # Step 4: Multiply and IFFT
        density_grid = irfft(counts_rfft * kernel_rfft_scaled, n=n_grid)
        density_grid = density_grid / n_ref  # Normalize by number of points
        density_grid = np.maximum(density_grid, 1e-300)  # Avoid log(0)

        # Step 5: Interpolate to evaluation points
        grid_centers = 0.5 * (edges[:-1] + edges[1:])
        log_density = np.interp(eval_points, grid_centers, np.log(density_grid))

        # Handle points outside grid
        outside = (eval_points < edges[0]) | (eval_points > edges[-1])
        log_density[outside] = -np.inf

        return log_density

    # ========================================================================
    # EFFICIENT LOO-CV
    # ========================================================================

    def _loo_cv_log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Efficient vectorized LOO-CV for KDE."""
        data = np.asarray(data, dtype=float).ravel()
        n = data.size

        if n < 2:
            return np.full(n, -np.inf)

        params = self.calc_posterior_params(data)
        h: float = params["bandwidth"]  # type: ignore

        # Compute all pairwise kernel contributions (use correct kernel!)
        if self.params.kernel == "gaussian":
            log_kernel = self._gaussian_log_kernel(data, data, h)
        elif self.params.kernel == "epanechnikov":
            log_kernel = self._epanechnikov_log_kernel(data, data, h)
        else:
            raise ValueError(f"Unknown kernel: {self.params.kernel}")

        # Exclude self-contribution by setting diagonal to -inf
        np.fill_diagonal(log_kernel, -np.inf)

        # LOO log-likelihood: log-sum-exp over other points, minus log(n-1)
        return np.logaddexp.reduce(log_kernel, axis=1) - np.log(n - 1)

    def _kfold_log_likelihood(self, data: np.ndarray, n_folds: int, shuffle: bool, seed: int | None) -> np.ndarray:
        """K-fold CV for KDE."""
        data = np.asarray(data, dtype=float).ravel()
        n = data.size

        if n < 2:
            return np.full(n, -np.inf)

        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
        fold_sizes[: n % n_folds] += 1

        cv_ll = np.empty(n, dtype=float)
        start = 0

        for fold_size in fold_sizes:
            stop = start + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate((indices[:start], indices[stop:]))
            start = stop

            if train_idx.size == 0:
                cv_ll[val_idx] = -np.inf
                continue

            train_params = self.calc_posterior_params(data[train_idx])
            cv_ll[val_idx] = self._plugin_log_likelihood(data[val_idx], train_params)

        return cv_ll

    # ========================================================================
    # SAMPLING UTILITIES
    # ========================================================================

    @staticmethod
    def _sample_epanechnikov(size: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from Epanechnikov kernel using rejection sampling."""
        samples = np.empty(size, dtype=float)
        filled = 0
        while filled < size:
            remaining = size - filled
            candidates = rng.uniform(-1, 1, remaining)
            accept = rng.uniform(0, 1, remaining) <= (1 - candidates**2)
            n_accept = accept.sum()
            if n_accept:
                samples[filled : filled + n_accept] = candidates[accept]
                filled += n_accept
        return samples


class BayesianKDE(KDE):
    """Bayesian KDE with prior on bandwidth.

    **String Alias:** ``'bayesian_kde'``

    Posterior bandwidth is weighted average of prior and data-based estimate:
    h_posterior = (m_h * h_prior + n * h_data) / (m_h + n)
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = BayesianKDEParams

    def __init__(self, params: BayesianKDEParams):
        super().__init__(params)

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, float | np.ndarray]:
        """Calculate posterior bandwidth using pseudo-Bayesian weighting."""
        data = np.asarray(data, dtype=float).ravel()
        n = data.size

        if n < 2:
            raise ValueError("KDE requires at least 2 data points.")

        data_h = self._compute_bandwidth(data)
        prior_h = self.params.prior_h
        m_h = self.params.m_h

        posterior_h = (m_h * prior_h + n * data_h) / (m_h + n)
        posterior_h = max(posterior_h, self.params.min_bandwidth)

        return {
            "data": data.copy(),
            "bandwidth": float(posterior_h),
            "data_bandwidth": float(data_h),
            "n": n,
        }
