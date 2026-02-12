import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast, get_type_hints

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bdf.utils.constants import RANDOM_SEED


class BDFDistributionParams(BaseModel, ABC):
    """Common parameters shared by all BDF distributions.

    Every distribution inherits these fields and may **override** their defaults
    or restrict their allowed values. For example, non-conjugate distributions
    lock ``score_method`` to ``"nll"`` and set ``use_posterior_predictive=False``.
    When a distribution overrides a default, it is documented in that
    distribution's own docstring; parameters not mentioned there use the
    defaults listed below.

    Parameters
    ----------
    score_method : {"nle", "nll"}, default="nle"
        Scoring method for split evaluation.

        * ``"nle"`` — Bayesian marginal likelihood (negative log evidence).
          Only available for conjugate distributions.
        * ``"nll"`` — Plug-in negative log-likelihood.
    score_correction : {"aic", "bic", "loo_cv", "kfold_cv"} or None, default=None
        Complexity correction applied when ``score_method="nll"``.
        Ignored when using ``"nle"``.
    score_cv_folds : int, default=3
        Number of folds for k-fold cross-validation (only used when
        ``score_correction="kfold_cv"``).
    score_cv_shuffle : bool, default=True
        Shuffle data before creating CV splits.
    score_cv_seed : int, default=1234
        Random seed for CV fold assignment.
    use_posterior_predictive : bool, default=True
        If ``True``, use the posterior predictive distribution (integrates out
        parameter uncertainty) for scoring and inference. If ``False``, use
        plug-in point estimates. Not all distributions support this; those
        that don't override this to ``False``.
    """

    # Scoring configuration (common to all distributions)
    score_method: Literal["nle", "nll"] = Field(
        default="nle",
        description="Scoring for splits: 'nle' (Bayesian marginal likelihood, negative log evidence) or 'nll' (plug-in negative log likelihood).",
    )
    score_correction: Literal["aic", "bic", "loo_cv", "kfold_cv"] | None = Field(
        default=None,
        description="Correction for NLL: None, 'aic', 'bic', 'loo_cv', or 'kfold_cv' (only used for `nll`)",
    )
    score_cv_folds: int = Field(default=3, ge=2, description="K-fold CV folds (only used for kfold_cv).")
    score_cv_shuffle: bool = Field(default=True, description="Shuffle data before CV splits (only for kfold_cv).")
    score_cv_seed: int = Field(default=1234, description="Random seed for CV (only for kfold_cv).")

    use_posterior_predictive: bool = Field(
        default=True, description="Use posterior predictive (True) or plug-in MAP (False) for NLL/inference."
    )

    tree_prior_mode: Literal["linear", "defer", "bernoulli"] = Field(
        default="linear",
        description="Tree prior mode: 'linear' (positive penalty terms), 'defer' (log params, includes current stop prob), ' \
            'bernoulli' (log params, includes probability of making children leaves).",
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_by_name=True)


P = TypeVar("P", bound=BDFDistributionParams)


class BDFDistribution(ABC, Generic[P]):
    """Abstract base class for all BDF distributions.

    Subclasses MUST implement all abstract methods:
    - calc_posterior_params
    - _plugin_log_likelihood
    - _num_parameters
    - _sample_posterior_params
    - validate_targets
    - get_posterior_mean
    - get_posterior_variance

    Subclasses MAY override optional methods for efficiency/capabilities:
    - nle
    - log_evidence
    - _posterior_predictive_log_likelihood
    - _loo_cv_log_likelihood
    - _kfold_log_likelihood
    - sample_prior
    """

    params_cls: ClassVar[type[BDFDistributionParams]] = BDFDistributionParams
    _registry: ClassVar[dict[str, type["BDFDistribution"]]] = {}
    params: P

    # Capability flags (override in subclasses)
    _supports_nle: bool = False
    _has_fast_loo_cv: bool = False
    _has_fast_kfold_cv: bool = False
    _supports_posterior_predictive: bool = False

    def __init__(self, params: dict[str, Any] | P):
        self.params = cast(P, self.params_cls.model_validate(params))
        self._validate_scoring_support()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            # Auto-detect params_cls from type hints
            if not hasattr(cls, "params_cls"):
                hints = get_type_hints(cls.__init__)
                params_type = hints.get("params")
                if params_type and issubclass(params_type, BDFDistributionParams):
                    cls.params_cls = params_type
                else:
                    raise TypeError(f"{cls.__name__} must set `params_cls`")
            BDFDistribution._registry[cls.__name__] = cls

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.params == other.params

    def __hash__(self) -> int:
        params_tuple = tuple(sorted(self.params.model_dump().items()))
        return hash((self.__class__.__name__, params_tuple))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"

    def __str__(self):
        return f"{self.__class__.__name__}(params={self.params})"

    # ============================================================================
    # VALIDATION & SERIALIZATION
    # ============================================================================
    @classmethod
    def resolve_auto_params(cls, key: str, data: np.ndarray, params: dict[str, Any] | None = None) -> Any:
        """Resolve 'auto' parameters based on data.

        Override in subclasses to implement distribution-specific auto-parameter logic.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement auto-parameter resolution for '{key}'")

    def _validate_scoring_support(self):
        """Validate that requested scoring is supported by this distribution."""
        if self.params.score_method == "nle" and not self._supports_nle:
            raise ValueError(
                f"{self.__class__.__name__} does not support score_method='nle' "
                f"(not a conjugate model). Use score_method='nll' instead."
            )

        if self.params.score_correction == "loo_cv" and not self._has_fast_loo_cv:
            warnings.warn(
                f"{self.__class__.__name__} does not implement efficient LOOCV; "
                f"falling back to naive n-fold refitting (slow).",
                UserWarning,
            )

        if self.params.score_correction == "kfold_cv" and not self._has_fast_kfold_cv:
            warnings.warn(
                f"{self.__class__.__name__} does not implement efficient k-fold CV; "
                f"falling back to naive refitting (slow).",
                UserWarning,
            )

        if self.params.use_posterior_predictive and not self._supports_posterior_predictive:
            warnings.warn(
                f"{self.__class__.__name__} does not support posterior predictive; "
                f"using plug-in estimates instead.",
                UserWarning,
            )
            self.params = self.params.model_copy(update={"use_posterior_predictive": False})

    def to_rust_spec(self) -> dict[str, Any]:
        """Create fully-validated Rust spec with all scoring metadata."""
        spec = {
            # Distribution identity
            "dist_type": self.__class__.__name__,
            # ALL parameters (hyperparameters + scoring config) - already validated
            **self.params.model_dump(),
            # Add computed metadata
            "num_parameters": self._num_parameters(),
            # Fallback
            "_python_object": self,
        }
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "BDFDistribution":
        dist_name = spec.get("dist_type")
        if dist_name is None:
            raise ValueError("Missing 'dist_type' in distribution spec.")
        dist_cls = cls._registry.get(dist_name)
        if dist_cls is None:
            raise ValueError(f"Unknown distribution class '{dist_name}'.")
        params_data = spec.get("params", {})
        params_model = dist_cls.params_cls.model_validate(params_data)
        return dist_cls(params=params_model)

    # ============================================================================
    # REQUIRED ABSTRACT METHODS (every distribution MUST implement)
    # ============================================================================

    @abstractmethod
    def calc_posterior_params(self, data: np.ndarray) -> dict[str, Any]:
        """Calculate posterior parameters from data (REQUIRED)."""
        pass

    @abstractmethod
    def _plugin_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Plug-in log-likelihood using MAP/MLE (REQUIRED).

        This is the CORE method every distribution must implement.
        Evaluates log p(data | params) where params are point estimates.
        """
        pass

    @abstractmethod
    def _num_parameters(self) -> int:
        """Number of parameters for AIC/BIC (REQUIRED)."""
        pass

    @abstractmethod
    def _sample_posterior_params(self, params: dict[str, Any], size: int, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution using provided parameters (REQUIRED).

        Returns a 1D array of `size` i.i.d. samples from the posterior.
        """
        pass

    @abstractmethod
    def validate_targets(self, data: np.ndarray):
        """Validate data (REQUIRED)."""
        pass

    @abstractmethod
    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior mean of the distribution."""
        pass

    @abstractmethod
    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get the posterior standard deviation of the distribution."""
        pass

    # ============================================================================
    # SCORING METHODS
    # ============================================================================

    def score(self, data: np.ndarray) -> float:
        """Unified scoring for splits (lower is better). CONCRETE."""
        if self.params.score_method == "nle":
            return self.nle(data)

        nll = self.nll(data)

        match self.params.score_correction:
            case None:
                score = nll
            case "aic":
                score = nll + self._num_parameters()
            case "bic":
                score = nll + 0.5 * self._num_parameters() * np.log(data.shape[0])
            case "loo_cv":
                score = -np.sum(self._loo_cv_log_likelihood(data))
            case "kfold_cv":
                score = -np.sum(
                    self._kfold_log_likelihood(
                        data, self.params.score_cv_folds, self.params.score_cv_shuffle, self.params.score_cv_seed
                    )
                )
            case _:
                raise ValueError(f"Unknown correction: {self.params.score_correction}")

        return float(score)

    def log_likelihood(self, data: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Log-likelihood of each data point (CONCRETE: don't override).

        Automatically routes to posterior predictive or plug-in based on config.
        Override only if you need custom routing logic.
        """
        if params is None:
            params = self.calc_posterior_params(data)

        # Route based on use_posterior_predictive flag
        if self.params.use_posterior_predictive:
            if not self._supports_posterior_predictive:
                # This should never happen due to _validate_scoring_support,
                # but guard anyway for safety
                warnings.warn(f"{self.__class__.__name__} falling back to plug-in (PP not supported)", UserWarning)
                return self._plugin_log_likelihood(data, params)
            return self._posterior_predictive_log_likelihood(data, params)
        else:
            return self._plugin_log_likelihood(data, params)

    def likelihood(self, data: np.ndarray, params: dict | None = None) -> np.ndarray:
        """Likelihood of each data point (CONCRETE: uses log_likelihood).

        Override only for numerical stability (e.g., avoiding exp overflow).
        """
        return np.exp(self.log_likelihood(data, params))

    def nll(self, data: np.ndarray, params: dict | None = None) -> float:
        """Negative log-likelihood (CONCRETE)."""
        if params is None:
            params = self.calc_posterior_params(data)
        return float(-np.sum(self.log_likelihood(data, params=params)))

    def sample_posterior(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
        size: int = 1,
        random_state: int = RANDOM_SEED,
    ) -> np.ndarray:
        """Sample from posterior (CONCRETE).

        Returns a 1D array of `size` i.i.d. samples from the posterior.
        """
        if params is None and data is None:
            raise ValueError("Provide either 'data' or 'params'")
        if params is None:
            params = self.calc_posterior_params(data)  # type: ignore
        return self._sample_posterior_params(params, size, random_state)

    # ============================================================================
    # OPTIONAL METHODS (override for capabilities/efficiency)
    # ============================================================================

    def nle(self, data: np.ndarray) -> float:
        """Negative log evidence (OPTIONAL: conjugate only)."""
        if not self._supports_nle:
            raise NotImplementedError(f"{self.__class__.__name__} does not implement nle (not a conjugate model).")
        return float(-self.log_evidence(data))

    def log_evidence(self, data: np.ndarray) -> float:
        """Log evidence (OPTIONAL: conjugate only)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement log_evidence")

    def _posterior_predictive_log_likelihood(self, data: np.ndarray, params: dict) -> np.ndarray:
        """Posterior predictive log-likelihood (OPTIONAL).

        Only implement if _supports_posterior_predictive=True.

        For conjugate models:
        - Normal-Normal → Student's t
        - Gamma-Exponential → Lomax
        - etc.

        Raises NotImplementedError if called on non-conjugate model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement posterior predictive. "
            f"Set _supports_posterior_predictive=True and implement this method, "
            f"or set use_posterior_predictive=False in params."
        )

    def _loo_cv_log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """LOO CV (DEFAULT: slow naive implementation)."""
        n = data.shape[0]
        loo_ll = np.empty(n)
        for i in range(n):
            train = np.delete(data, i)
            test = data[i : i + 1]
            params = self.calc_posterior_params(train)
            loo_ll[i] = self.log_likelihood(test, params=params)[0]
        return loo_ll

    def _kfold_log_likelihood(self, data: np.ndarray, n_folds: int, shuffle: bool, seed: int | None) -> np.ndarray:
        """K-fold CV (DEFAULT: sklearn implementation)."""
        from sklearn.model_selection import KFold

        cv_ll = np.empty(data.shape[0])
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        for train_idx, test_idx in kf.split(data):
            params = self.calc_posterior_params(data[train_idx])
            cv_ll[test_idx] = self.log_likelihood(data[test_idx], params=params)
        return cv_ll

    def sample_prior(self, size: int, random_state: int = RANDOM_SEED) -> np.ndarray:
        """Sample from prior (OPTIONAL: only for Bayesian distributions).

        Frequentist distributions don't have priors and should not implement this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sample_prior " f"(frequentist distribution has no prior)."
        )
