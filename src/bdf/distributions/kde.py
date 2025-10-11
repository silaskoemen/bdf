import warnings

import numpy as np
from pydantic import Field

from bdf.utils.constants import RANDOM_SEED

from .bdf_distribution import BDFDistribution, BDFDistributionParams


class KDEParams(BDFDistributionParams):
    """Parameters for the KDE distribution.

    Attributes
    ----------
    bandwidth : float | str
        Bandwidth for the kernel density estimation. Can be a positive float or 'scott' or 'silverman' for rule-of-thumb methods.
    """

    bandwidth: str | float | int = Field(
        default="scott",
        description="Bandwidth for the kernel density estimation. Can be a positive float or 'scott' or 'silverman' for rule-of-thumb methods.",
    )
    kernel: str = Field(
        default="gaussian",
        description="Kernel type for the kernel density estimation. Currently, only 'gaussian' and 'epanechnikov' are supported.",
    )
    cv: str | int = Field(
        default=2,  # 2-fold cv by default; balancing bias-variance trade-off and computational cost
        description="Cross-validation strategy for bandwidth selection: 'loo' for leave-one-out, 1 for in-sample, or an integer >= 2 for K-fold CV.",
    )
    min_bandwidth: float | int = Field(
        default=1e-6,
        gt=1e-10,
        description="Minimum allowable bandwidth to prevent numerical issues.",
    )
    recompute_bandwidth_splits: bool = Field(
        default=False,
        description="Whether to recompute bandwidth for each split in cross-validation.",
    )
    shuffle_splits: bool = Field(
        default=True,
        description="Whether to shuffle data before creating folds in K-fold cross-validation.",
    )

    class Config:
        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "bandwidth" not in data:
            missing_fields["bandwidth"] = self.__class__.model_fields["bandwidth"].default
        if "kernel" not in data:
            missing_fields["kernel"] = self.__class__.model_fields["kernel"].default
        if "cv" not in data:
            missing_fields["cv"] = self.__class__.model_fields["cv"].default
        if "min_bandwidth" not in data:
            missing_fields["min_bandwidth"] = self.__class__.model_fields["min_bandwidth"].default
        if "recompute_bandwidth_splits" not in data:
            missing_fields["recompute_bandwidth_splits"] = self.__class__.model_fields[
                "recompute_bandwidth_splits"
            ].default
        if "shuffle_splits" not in data:
            missing_fields["shuffle_splits"] = self.__class__.model_fields["shuffle_splits"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class KDEBase(BDFDistribution):
    """Base implementation for all Kernel Density Estimation (KDE) distributions.
    Calculation of (posterior) bandwidth has to be implemented by subclasses, all
    (log-)likelihood and sampling methods are implemented here.
    """

    def __init__(self, prior_params: BDFDistributionParams, params: KDEParams):
        super().__init__(prior_params=prior_params, params=params)

    def nll(self, data: np.ndarray) -> float:
        """Compute the negative log-likelihood of the data given the KDE."""
        return -np.sum(self.log_likelihood(data))

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> dict | tuple:
        raise NotImplementedError("Subclasses must implement this method.")

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Likelihood computation not implemented currently, use log_likelihood instead.")

    def _calc_data_bandwidth(self, data: np.ndarray) -> float:
        """Calculate bandwidth from data using specified method or value."""
        bandwidth = self.params.bandwidth  # type: ignore[attr-defined] | doesn't see pydantic field
        if isinstance(bandwidth, str):
            match bandwidth:
                case "scott":
                    return self._compute_bandwidth_scott(data)
                case "silverman":
                    return self._compute_bandwidth_silverman(data)
                case _:
                    raise ValueError("bandwidth string must be 'scott' or 'silverman'.")
        elif isinstance(bandwidth, (float, int)):
            if bandwidth <= 0:
                raise ValueError("bandwidth must be a positive float.")
            return float(bandwidth)
        else:
            raise TypeError("bandwidth must be a positive float or 'scott' or 'silverman'.")

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data given the KDE."""
        _, h = self.calc_posterior_params(data, return_dict=False)
        match self.params.cv:  # type: ignore[arg-defined]
            case "loo":
                return self._compute_loo_loglik(data, h)
            case 1:
                return self._compute_insample_loglik(data, h)
            case int() if self.params.cv >= 2:  # type: ignore[arg-defined]
                return self._compute_kfold_loglik(data, h, self.params.cv, False, None)  # type: ignore[arg-defined]
            case _:
                raise ValueError("cv must be 'loo', 1 (insample), or an integer >= 2.")

    def _compute_loo_loglik(self, data: np.ndarray, h: float) -> np.ndarray:
        """Compute leave-one-out log-likelihood, calling _compute_kernel_loglik to route to the appropriate kernel.

        Args
        ----
        `data` : np.ndarray
            Data points for which to compute the log-likelihood.
        `h` : float
            Bandwidth for the kernel density estimation.

        Returns
        -------
        np.ndarray
            Log-likelihood values for each data point.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]

        n = data.shape[0]
        if n < 2:
            return np.full(n, -np.inf, dtype=float)

        # Evaluate each point against all samples (including self)
        all_loglik = self._compute_kernel_loglik(data, data, h)

        # all_loglik is shape (n, n); mask diagonal to drop self-contribution
        np.fill_diagonal(all_loglik, -np.inf)

        # Aggregate per row (log-mean over n-1 refs)
        log_counts = np.log(n - 1)
        row_logsum = np.logaddexp.reduce(all_loglik, axis=1)
        return row_logsum - log_counts

    def _compute_kfold_loglik(
        self,
        data: np.ndarray,
        h: float,
        k: int,
        seed: int = RANDOM_SEED,
    ) -> np.ndarray:
        """K-fold log-likelihood using posterior bandwidth `h` (or per-fold updates)."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]

        n = data.shape[0]
        if n < 2:
            return np.full(n, -np.inf, dtype=float)

        if k <= 1:
            raise ValueError("k-fold cross-validation requires k >= 2.")
        if k > n:
            k = n  # fallback to LOO

        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        if self.params.shuffle_splits:  # type: ignore[attr-defined]
            rng.shuffle(indices)

        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[: n % k] += 1

        loglik = np.empty(n, dtype=float)
        start = 0

        for fold_size in fold_sizes:
            stop = start + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate((indices[:start], indices[stop:]))
            start = stop

            eval_data = data[val_idx]
            ref_data = data[train_idx]

            if ref_data.size == 0:
                loglik[val_idx] = -np.inf
                continue

            if self.params.recompute_bandwidth_splits:  # type: ignore[attr-defined]
                fold_params = self.calc_posterior_params(ref_data, return_dict=False)
                h_fold = float(fold_params[-1])  # expects (..., posterior_h)
            else:
                h_fold = h

            fold_log_weights = self._compute_kernel_loglik(eval_data, ref_data, h_fold)
            log_norm = np.log(ref_data.shape[0])
            loglik[val_idx] = np.logaddexp.reduce(fold_log_weights, axis=1) - log_norm

        return loglik

    def _compute_insample_loglik(self, data: np.ndarray, h: float) -> np.ndarray:
        """Compute in-sample log-likelihood, calling _compute_kernel_loglik to route to the appropriate kernel.

        Args
        ----
        `data` : np.ndarray
            Data points for which to compute the log-likelihood.
        `h` : float
            Bandwidth for the kernel density estimation.

        Returns
        -------
        np.ndarray
            Log-likelihood values for each data point.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]

        n = data.shape[0]
        if n < 1:
            return np.full(n, -np.inf, dtype=float)

        all_loglik = self._compute_kernel_loglik(data, data, h)

        # all_loglik is shape (n, n); aggregate per row (log-mean over n refs)
        log_counts = np.log(n)
        row_logsum = np.logaddexp.reduce(all_loglik, axis=1)
        return row_logsum - log_counts

    def _compute_kernel_loglik(self, ref_data: np.ndarray, eval_data: np.ndarray, h: float) -> np.ndarray:
        """Compute log-likelihood using scipy's KDE implementation."""
        if self.params.kernel == "gaussian":  # type: ignore
            return self._compute_gaussian_loglik(ref_data, eval_data, h)
        elif self.params.kernel == "epanechnikov":  # type: ignore[arg-defined]
            return self._compute_epanechnikov_loglik(ref_data, eval_data, h)
        else:
            raise ValueError("Unsupported kernel type. Use 'gaussian' or 'epanechnikov'.")

    def _compute_gaussian_loglik(self, ref_data: np.ndarray, eval_data: np.ndarray, h: float) -> np.ndarray:
        """Compute log-likelihood using Gaussian kernel."""
        ref = np.asarray(ref_data, dtype=float)
        eva = np.asarray(eval_data, dtype=float)
        if ref.ndim == 1:
            ref = ref[:, None]
        if eva.ndim == 1:
            eva = eva[:, None]

        n_ref, dim = ref.shape
        if n_ref == 0:
            return np.full(eva.shape[0], -np.inf, dtype=float)

        diffs = (eva[:, None, :] - ref[None, :, :]) / h
        quad = np.sum(diffs**2, axis=2)  # shape (n_eval, n_ref)
        log_kernel = -0.5 * quad  # Gaussian exponent
        log_norm = dim * np.log(h) + 0.5 * dim * np.log(2 * np.pi) + np.log(n_ref)
        return np.logaddexp.reduce(log_kernel, axis=1) - log_norm

    def _compute_epanechnikov_loglik(self, ref_data: np.ndarray, eval_data: np.ndarray, h: float) -> np.ndarray:
        """Compute log-likelihood using the Epanechnikov kernel (currently 1D only)."""
        ref = np.asarray(ref_data, dtype=float)
        eva = np.asarray(eval_data, dtype=float)

        if ref.ndim == 2 and ref.shape[1] != 1 or eva.ndim == 2 and eva.shape[1] != 1:
            raise NotImplementedError("Epanechnikov kernel currently supports only 1D data.")

        ref = ref.reshape(-1, 1)
        eva = eva.reshape(-1, 1)

        n_ref = ref.shape[0]
        if n_ref == 0:
            return np.full(eva.shape[0], -np.inf, dtype=float)
        if h <= 0:
            raise ValueError("Bandwidth must be strictly positive for Epanechnikov kernel.")

        u = (eva - ref.T) / h  # shape (n_eval, n_ref)
        mask = np.abs(u) <= 1.0

        # Kernel contributions: K(u) = 0.75 * (1 - u^2) for |u| <= 1, else 0
        log_kernel = np.full_like(u, -np.inf, dtype=float)
        safe_vals = 1.0 - u[mask] ** 2
        safe_vals = np.clip(safe_vals, 0.0, None)
        log_kernel[mask] = np.log(0.75) + np.log(safe_vals, out=safe_vals)

        log_norm = np.log(n_ref) + np.log(h)
        return np.logaddexp.reduce(log_kernel, axis=1) - log_norm

    def _compute_bandwidth_silverman(self, X: np.ndarray) -> float:
        """X is currently only supported for 1D, else X_1d = X[:, 0]"""
        std = float(np.std(X, ddof=1))
        iqr = float(np.subtract(*np.percentile(X, [75, 25])))
        scale = min(std, iqr / 1.349) if iqr > 0 else std
        if scale <= 0:
            scale = std
        return max(self.params.min_bandwidth, 0.9 * scale * X.shape[0] ** (-1.0 / 5))  # type: ignore

    def _compute_bandwidth_scott(self, X: np.ndarray) -> float:
        """Compute bandwidth using Scott's or Silverman's rule of thumb."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if n < 2:
            raise ValueError("At least two samples are required to estimate bandwidth.")

        scale = float(np.sqrt(np.mean(np.var(X, axis=0, ddof=1))))
        return max(self.params.min_bandwidth, scale * n ** (-1.0 / (d + 4)))  # type: ignore

    def _pairwise_sq_dists(self, A, B):
        """Compute squared Euclidean distances between rows of A and B."""
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True).T
        D2 = A2 + B2 - 2.0 * (A @ B.T)
        np.maximum(D2, 0.0, out=D2)
        return D2


class PseudoHKDEParams(BDFDistributionParams):
    """Parameters for the PseudoHKDE distribution.

    Attributes
    ----------
    prior_h : float
        Prior bandwidth for the kernel density estimation.
    m_h : float
        Weighting factor for the prior bandwidth.
    """

    prior_h: float = Field(default=1.0, description="Prior bandwidth for the kernel density estimation.")
    m_h: float = Field(default=1.0, description="Weighting factor for the prior bandwidth.")

    class Config:
        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "prior_h" not in data:
            missing_fields["prior_h"] = self.__class__.model_fields["prior_h"].default
        if "m_h" not in data:
            missing_fields["m_h"] = self.__class__.model_fields["m_h"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class PseudoHKDE(KDEBase):
    """
    Kernel Density Estimation distribution.

    Uses Gaussian kernel with isotropic bandwidth for non-parametric density estimation.
    """

    def __init__(self, prior_params: dict | PseudoHKDEParams, params: dict | KDEParams | None = None):
        if isinstance(prior_params, dict):
            prior_params = PseudoHKDEParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, PseudoHKDEParams
        ), "prior_params must be an instance of NormalNormalParams after possible conversion from dict."
        if params is None:  # keep consistent with other distributions, but needed here
            params = KDEParams.model_validate({})  # type: ignore
        else:
            params = KDEParams.model_validate(params)  # type: ignore
        super().__init__(prior_params=prior_params, params=params)

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> dict | tuple:
        """Calculate the posterior bandwidth using a pseudo-Bayesian approach.

        The posterior bandwidth is a weighted average of the prior bandwidth and the
        bandwidth estimated from the data using Silverman's rule of thumb.

        Args:
            data (np.ndarray): The input data for bandwidth estimation.
            return_dict (bool): If True, return the parameters as a dictionary.

        Returns:
            dict or tuple: The posterior parameters as a dictionary or tuple.
        """
        n = len(data)
        if n < 2:
            raise ValueError("At least two data points are required to estimate bandwidth.")
        data_h = self._calc_data_bandwidth(data)
        prior_h = self.prior_params.prior_h  # type: ignore[attr-defined]
        m_h = self.prior_params.m_h  # type: ignore[attr-defined]
        posterior_h = (m_h * prior_h + n * data_h) / (m_h + n)
        if return_dict:
            return {"data": data, "posterior_h": posterior_h}
        else:
            return data, posterior_h


# NOTE/TODO: Investigate whether this even makes sense, would need to reimplement (log-)likelihood calculations
# to include penalty term; also, how to choose lambda_h?
class PenalizedHKDEParams(BDFDistributionParams):
    """Parameters for the PenalizedHKDE distribution.

    Attributes
    ----------
    lambda_h : float
        Penalty parameter for bandwidth regularization.
    """

    lambda_h: float = Field(default=1.0, description="Penalty parameter of `h` for `nll` calculation.")

    class Config:
        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "lambda_h" not in data:
            missing_fields["lambda_h"] = self.__class__.model_fields["lambda_h"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class PenalizedHKDE(KDEBase):
    """
    Kernel Density Estimation distribution with penalized bandwidth.

    Uses Gaussian kernel with isotropic bandwidth for non-parametric density estimation.
    The bandwidth is estimated by minimizing the penalized negative log-likelihood.
    """

    def __init__(self, prior_params: dict | PenalizedHKDEParams, params: dict | KDEParams | None = None):
        if isinstance(prior_params, dict):
            prior_params = PenalizedHKDEParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, PenalizedHKDEParams
        ), "prior_params must be an instance of PenalizedHKDEParams after possible conversion from dict."
        if params is None:  # keep consistent with other distributions, but needed here
            params = KDEParams.model_validate({})  # type: ignore
        else:
            params = KDEParams.model_validate(params)  # type: ignore
        super().__init__(prior_params=prior_params, params=params)

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> dict | tuple:
        """Calculate the posterior bandwidth by minimizing the penalized negative log-likelihood.

        The posterior bandwidth is found by minimizing the sum of the negative log-likelihood
        of the data and a penalty term proportional to the bandwidth.

        Args:
            data (np.ndarray): The input data for bandwidth estimation.
            return_dict (bool): If True, return the parameters as a dictionary.
        Returns:
            dict or tuple: The posterior parameters as a dictionary or tuple.
        """
        n = len(data)
        if n < 2:
            raise ValueError("At least two data points are required to estimate bandwidth.")
        data_h = self._calc_data_bandwidth(data)
        lambda_h = self.prior_params.lambda_h  # type: ignore[attr-defined]
        posterior_h = max(data_h, lambda_h**0.5)  # simple heuristic to avoid too small bandwidths
        if return_dict:
            return {"data": data, "posterior_h": posterior_h}
        else:
            return data, posterior_h
