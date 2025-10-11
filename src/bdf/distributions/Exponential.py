"""Exponential data with Gamma prior on lambda, posterior is of lambda is Gamma

Gamma uses parameterization of alpha and beta (as in https://en.wikipedia.org/wiki/Conjugate_prior),
corresponding to shape and rate (inverse scale) (alpha and lambda in https://en.wikipedia.org/wiki/Gamma_distribution)

Implemented versions:
- GammaABLambdaExponential: Gamma prior on mean lambda with parameters alpha and beta
- GammaMVLambdaExponential: Gamma prior on mean lambda with parameters mean and variance
- GammaABLambdaExponentialPP: Gamma prior on mean lambda with parameters alpha and beta, posterior predictive Lomax distribution
- GammaMVLambdaExponentialPP: Gamma prior on mean lambda with parameters mean and variance, posterior predictive Lomax distribution
- PseudoLambdaExponential: Pseudo prior on mean lambda with mean and strength/number m
- NormalMeanExponential: Normal prior on CLT mean (normal distribution) to obtain normal posterior, use MAP for Exponential

Would also be possible to use normal prior on lambda directly (in Exponential likelihood) and use numerical optimization
to find the MAP posterior estimate, but due to conjugate options available and the performance penalty of numerical optimization,
this is currently not implemented.
"""
import warnings

import numpy as np
from pydantic import Field
from scipy.stats import expon, lomax

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED


class ExponentialBase(BDFDistribution):
    """Base class for all Exponential distribution implementations (where posterior sampling distribution remains exponential).

    All implementations use parameter lambda, so sampling,
    (log)likelihoods and return functions are all identical.
    """

    # This is just a placeholder - child classes will have their own init
    def __init__(self, prior_params, params=None):
        super().__init__(prior_params, params)

    # Child classes MUST implement this method
    def calc_posterior_params(self, data, return_dict=False):
        raise NotImplementedError("Subclasses must implement calc_posterior_params")

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the log-likelihood for.

        Returns
        -------
        np.ndarray
            A numpy array containing the log-likelihood values for each data point.
        """
        posterior_lambda = self.calc_posterior_params(data)
        return expon.logpdf(data, scale=1 / posterior_lambda)

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the likelihood for.

        Returns
        -------
        np.ndarray
            A numpy array containing the likelihood values for each data point.
        """
        posterior_lambda = self.calc_posterior_params(data)
        return expon.pdf(data, scale=1 / posterior_lambda)

    def nll(self, data: np.ndarray) -> float:
        """Compute the negative log-likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the negative log-likelihood for.

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        return -np.sum(self.log_likelihood(data))

    def sample_prior(self, size: int, random_state: int) -> np.ndarray:
        """Sample from the prior distribution.

        Prior is always defined over parameter lambda, which is the mean of the Exponential distribution.
        To sample from this, we need to sample from the prior distribution of lambda.

        Args
        ----
        `size` : int
            The number of samples to draw from the prior distribution.

        Returns
        -------
        np.ndarray
            A numpy array containing samples drawn from the prior distribution.
        """
        raise NotImplementedError(
            "Subclasses must implement sample_prior method for ExponentialBase distribution, due to",
            "different prior distributions for each subclass.",
        )

    def sample_posterior(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
        size: int = 1,
        random_state: int = RANDOM_SEED,
    ) -> np.ndarray:
        """Sample from the posterior distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to use for sampling, if available.
        `params` : dict[str, float] | None, optional
            Additional parameters for sampling, if available.
        `size` : int, optional
            The number of samples to draw from the posterior distribution, default is 1.

        Returns
        -------
        np.ndarray
            A numpy array containing samples drawn from the posterior distribution.
        """
        if params is not None:
            return self._sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self._sample_posterior_data(data, size=size, random_state=random_state)
        else:
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def _sample_posterior_params(self, params: dict[str, float], *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution using provided parameters.

        Args
        ----
        `params` : dict[str, float]
            The posterior parameters to sample from, must include 'alpha', 'xi', and 'omega'.
        `size` : int, optional
            The number of samples to generate, default is 1.

        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution.
        """
        posterior_lambda = params.get("posterior_lambda")
        if posterior_lambda is None:
            raise ValueError("params must contain 'posterior_lambda' key")
        assert posterior_lambda > 0, "'posterior_lambda' parameter must be positive"
        return expon.rvs(scale=1 / posterior_lambda, size=size, random_state=random_state)  # type: ignore

    def _sample_posterior_data(self, data: np.ndarray, *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution using the data.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        `size` : int, optional
            The number of samples to generate, default is 1.

        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution based on the data.
        """
        posterior_lambda = self.calc_posterior_params(data)
        return expon.rvs(scale=1 / posterior_lambda, size=size, random_state=random_state)  # type: ignore

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior mean of the distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to calculate the posterior mean from, if available.
        `params` : dict[str, float] | None, optional
            The posterior parameters to use for calculating the mean, if available.

        Returns
        -------
        float
            The posterior mean of the distribution.
        """
        if params is not None:
            return 1 / params.get("lambda", params.get("posterior_lambda"))  # type: ignore
        elif data is not None:
            posterior_lambda = self.calc_posterior_params(data)
            return 1 / posterior_lambda  # type: ignore
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior mean.")

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get the posterior variance of the distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to calculate the posterior variance from, if available.
        `params` : dict[str, float] | None, optional
            The posterior parameters to use for calculating the variance, if available.

        Returns
        -------
        float
            The posterior variance of the distribution.
        """
        if params is not None:
            return 1 / params.get("posterior_lambda") ** 2  # type: ignore
        elif data is not None:
            posterior_lambda = self.calc_posterior_params(data)
            return 1 / posterior_lambda**2  # type: ignore
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior variance.")

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters of the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.

        Returns
        -------
        dict
            A dictionary containing the posterior parameter 'posterior_lambda'.
        """
        return self.calc_posterior_params(data, return_dict=True)

    def validate_targets(self, data: np.ndarray):
        assert np.all(data > 0), "Targets have to be positive for exponential distribution."
        assert all(np.isfinite(data)), "Targets must be finite for exponential distribution."
        std = np.std(data)
        assert (
            np.isfinite(std) and std is not None and std >= 0.0
        ), f"Standard deviation has to be finite, not None and >=0, got {std}"


class GammaABLambdaExponentialParams(BDFDistributionParams):
    """Parameters for the Gamma prior on mean lambda with parameters alpha and theta ('AB' notation)."""

    alpha_lambda: float = Field(gt=0.0, description="Shape parameter of the Gamma prior for parameter lambda")
    beta_lambda: float = Field(gt=0.0, description="Scale parameter of the Gamma prior for parameter lambda")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "alpha_lambda" not in data:
            missing_fields["alpha_lambda"] = self.__class__.model_fields["alpha_lambda"].default
        if "beta_lambda" not in data:
            missing_fields["beta_lambda"] = self.__class__.model_fields["beta_lambda"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class GammaABLambdaExponential(ExponentialBase):
    """Exponential distribution with a Gamma prior on mean lambda.

    This class implements the Exponential distribution with a Gamma prior on the mean lambda.
    The posterior is also a Gamma distribution.
    """

    def __init__(self, prior_params: GammaABLambdaExponentialParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = GammaABLambdaExponentialParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, GammaABLambdaExponentialParams
        ), "prior_params must be an instance of GammaABLambdaExponentialParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.alpha_lambda = prior_params.alpha_lambda
        self.beta_lambda = prior_params.beta_lambda

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)
        alpha_post = self.alpha_lambda + n
        beta_post = self.beta_lambda + sum_data

        if return_dict:
            return {"posterior_lambda": alpha_post / beta_post}
        else:
            return alpha_post / beta_post


class GammaMVLambdaExponentialParams(BDFDistributionParams):
    """Parameters for the Gamma prior on mean lambda with parameters mean and variance."""

    mean_lambda: float = Field(gt=0.0, description="Mean of the Gamma prior for parameter lambda.")
    var_lambda: float = Field(gt=0.0, description="Variance of the Gamma prior for parameter lambda.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "mean_lambda" not in data:
            missing_fields["mean_lambda"] = self.__class__.model_fields["mean_lambda"].default
        if "var_lambda" not in data:
            missing_fields["var_lambda"] = self.__class__.model_fields["var_lambda"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class GammaMVLambdaExponential(ExponentialBase):
    """Exponential distribution with a Gamma prior on mean lambda.

    This class implements the Exponential distribution with a Gamma prior on the mean lambda.
    The posterior of lambda is also a Gamma distribution, while the sampling distribution
    is again an Exponential distribution at the posterior mean of lambda.
    """

    def __init__(self, prior_params: dict | GammaMVLambdaExponentialParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = GammaMVLambdaExponentialParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, GammaMVLambdaExponentialParams
        ), "prior_params must be an instance of GammaMVLambdaExponentialParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mean_lambda = prior_params.mean_lambda
        self.var_lambda = prior_params.var_lambda

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)

        # Relate mean and variance to alpha and beta of the Gamma distribution
        prior_alpha = self.mean_lambda**2 / self.var_lambda
        prior_beta = self.mean_lambda / self.var_lambda

        alpha_post = prior_alpha + n
        beta_post = prior_beta + sum_data

        if return_dict:
            return {"posterior_lambda": alpha_post / beta_post}
        else:
            return alpha_post / beta_post


class PseudoLambdaExponentialParams(BDFDistributionParams):
    """Parameters for the Pseudo prior on mean lambda with mean and strength/number m."""

    mean_lambda: float = Field(gt=0.0, description="Mean of the Pseudo prior for parameter lambda.")
    m_lambda: float = Field(gt=0.0, description="Strength or number of samples in the Pseudo prior.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "mean_lambda" not in data:
            missing_fields["mean_lambda"] = self.__class__.model_fields["mean_lambda"].default
        if "m_lambda" not in data:
            missing_fields["m_lambda"] = self.__class__.model_fields["m_lambda"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class PseudoLambdaExponential(ExponentialBase):
    """Exponential distribution with a Pseudo prior on mean lambda.

    This class implements the Exponential distribution with a Pseudo prior on the mean lambda.
    The posterior is also an Exponential distribution, and the sampling distribution is again an Exponential distribution.
    """

    def __init__(self, prior_params: PseudoLambdaExponentialParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = PseudoLambdaExponentialParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, PseudoLambdaExponentialParams
        ), "prior_params must be an instance of PseudoLambdaExponentialParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mean_lambda = prior_params.mean_lambda
        self.m_lambda = prior_params.m_lambda

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)

        posterior_lambda = (self.mean_lambda * self.m_lambda + sum_data) / (self.m_lambda + n)

        if return_dict:
            return {"posterior_lambda": posterior_lambda}
        else:
            return posterior_lambda


class NormalMeanExponentialParams(BDFDistributionParams):
    """Parameters for the Normal prior on mean lambda with parameters mu and sigma."""

    mu_mean: float = Field(gt=0.0, description="Mean of the Normal prior for mean.")
    sigma_mean: float = Field(gt=0.0, description="Standard deviation of the Normal prior for mean.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "mu_mean" not in data:
            missing_fields["mu_mean"] = self.__class__.model_fields["mu_mean"].default
        if "sigma_mean" not in data:
            missing_fields["sigma_mean"] = self.__class__.model_fields["sigma_mean"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class NormalMeanExponential(ExponentialBase):
    """Exponential distribution with a Normal prior on mean lambda.

    This class implements the Exponential distribution with a Normal prior on the mean lambda.
    The posterior is also an Exponential distribution, and the sampling distribution is again an Exponential distribution.
    """

    def __init__(self, prior_params: NormalMeanExponentialParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = NormalMeanExponentialParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalMeanExponentialParams
        ), "prior_params must be an instance of NormalMeanExponentialParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mu_mean = prior_params.mu_mean
        self.sigma_mean = prior_params.sigma_mean

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict:
        """Calculate the posterior parameters based on the data. Uses the Normal distribution
        of the sample mean under the CLT, calculates a standard Normal-Normal posterior.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        `return_dict` : bool, optional
            If True, returns a dictionary with the posterior parameter 'posterior_lambda'.
            If False, returns the posterior parameter directly. Default is False.

        Returns
        -------
        float | dict
            The posterior parameter 'posterior_lambda' as a float if `return_dict` is False,
            or as a dictionary if `return_dict` is True.
        """
        n = len(data)
        eps = 1e-10
        sample_mean = data.mean()
        sample_var = np.var(data, ddof=1)
        posterior_mean = (
            (n / (sample_var + eps)) * sample_mean + (1 / (self.sigma_mean**2 + eps)) * self.mu_mean
        ) / ((n / (sample_var + eps)) + (1 / (self.sigma_mean**2 + eps)))

        if return_dict:
            return {"posterior_lambda": 1 / posterior_mean}  # type: ignore
        else:
            return 1 / posterior_mean  # type: ignore


class ExponentialPPBase(BDFDistribution):
    """Base class for Exponential likelihood - Lomax posterior predictive distributions.

    Final distribution is Lomax, meaning parameters returned are alpha and lambda and
    all likelihoods and sampling functions are based on these parameters.
    """

    def calc_posterior_params(self, data, return_dict=False):
        raise NotImplementedError("Subclasses must implement calc_posterior_params")

    def log_likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the log-likelihood for.

        Returns
        -------
        np.ndarray
            A numpy array containing the log-likelihood values for each data point.
        """
        posterior_alpha, posterior_lambda = self.calc_posterior_params(data)
        return lomax.logpdf(data, c=posterior_alpha, scale=posterior_lambda)

    def likelihood(self, data: np.ndarray) -> np.ndarray:
        """Compute the likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the likelihood for.

        Returns
        -------
        np.ndarray
            A numpy array containing the likelihood values for each data point.
        """
        posterior_alpha, posterior_lambda = self.calc_posterior_params(data)
        return expon.pdf(data, c=posterior_alpha, scale=posterior_lambda)

    def nll(self, data: np.ndarray) -> float:
        """Compute the negative log-likelihood of the data given the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to compute the negative log-likelihood for.

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        return -np.sum(self.log_likelihood(data))

    def sample_prior(self, size: int, random_state: int) -> np.ndarray:
        """Sample from the prior distribution.

        Prior is always defined over parameter lambda, which is the mean of the Exponential distribution.
        To sample from this, we need to sample from the prior distribution of lambda.

        Args
        ----
        `size` : int
            The number of samples to draw from the prior distribution.

        Returns
        -------
        np.ndarray
            A numpy array containing samples drawn from the prior distribution.
        """
        raise NotImplementedError(
            "Subclasses must implement sample_prior method for ExponentialBase distribution, due to",
            "different prior distributions for each subclass.",
        )

    def sample_posterior(
        self,
        *,
        data: np.ndarray | None = None,
        params: dict[str, float] | None = None,
        size: int = 1,
        random_state: int = RANDOM_SEED,
    ) -> np.ndarray:
        """Sample from the posterior distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to use for sampling, if available.
        `params` : dict[str, float] | None, optional
            Additional parameters for sampling, if available.
        `size` : int, optional
            The number of samples to draw from the posterior distribution, default is 1.

        Returns
        -------
        np.ndarray
            A numpy array containing samples drawn from the posterior distribution.
        """
        if params is not None:
            return self.sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self.sample_posterior_data(data, size=size, random_state=random_state)
        else:
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )

    def sample_posterior_params(self, params: dict[str, float], *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution using provided parameters.

        Args
        ----
        `params` : dict[str, float]
            The posterior parameters to sample from, must include 'alpha', 'xi', and 'omega'.
        `size` : int, optional
            The number of samples to generate, default is 1.

        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution.
        """
        posterior_alpha = params.get("posterior_alpha")
        posterior_lambda = params.get("posterior_lambda")
        if posterior_lambda is None or posterior_alpha is None:
            raise ValueError("params must contain 'posterior_alpha' and 'posterior_lambda' keys")
        if posterior_alpha <= 0 or posterior_lambda <= 0:
            raise ValueError(f"{posterior_alpha =} and {posterior_lambda =} parameters must be positive")
        return lomax.rvs(c=posterior_alpha, scale=posterior_lambda, size=size, random_state=random_state)  # type: ignore

    def sample_posterior_data(self, data: np.ndarray, *, size: int = 1, random_state: int) -> np.ndarray:
        """Sample from the posterior distribution using the data.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.
        `size` : int, optional
            The number of samples to generate, default is 1.

        Returns
        -------
        np.ndarray
            Samples drawn from the posterior distribution based on the data.
        """
        posterior_alpha, posterior_lambda = self.calc_posterior_params(data)
        return lomax.rvs(c=posterior_alpha, scale=posterior_lambda, size=size, random_state=random_state)  # type: ignore

    def get_posterior_mean(self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None) -> float:
        """Get the posterior mean of the distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to calculate the posterior mean from, if available.
        `params` : dict[str, float] | None, optional
            The posterior parameters to use for calculating the mean, if available.

        Returns
        -------
        float
            The posterior mean of the distribution.
        """
        if params is not None:
            posterior_alpha = params.get("posterior_alpha")  # type: ignore
            posterior_lambda = params.get("posterior_lambda")  # type: ignore
        elif data is not None:
            posterior_alpha, posterior_lambda = self.calc_posterior_params(data)
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior mean.")
        if posterior_alpha <= 1:  # type: ignore
            warnings.warn(
                "Posterior alpha is less than or equal to 1, for which the mean is undefined. Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
        return posterior_lambda / (posterior_alpha - 1)  # type: ignore

    def get_posterior_variance(
        self, *, data: np.ndarray | None = None, params: dict[str, float] | None = None
    ) -> float:
        """Get the posterior variance of the distribution.

        Args
        ----
        `data` : np.ndarray | None, optional
            The data to calculate the posterior variance from, if available.
        `params` : dict[str, float] | None, optional
            The posterior parameters to use for calculating the variance, if available.

        Returns
        -------
        float
            The posterior variance of the distribution.
        """
        if params is not None:
            posterior_alpha = params.get("posterior_alpha")  # type: ignore
            posterior_lambda = params.get("posterior_lambda")  # type: ignore
        elif data is not None:
            posterior_alpha, posterior_lambda = self.calc_posterior_params(data)
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior variance.")
        if posterior_alpha <= 1:  # type: ignore
            warnings.warn(
                "Posterior alpha is less than or equal to 1, for which the variance is undefined. Returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
        elif posterior_alpha <= 2:  # type: ignore
            warnings.warn(
                "Posterior alpha is less than or equal to 2, for which the variance is infinite. Returning inf.",
                UserWarning,
                stacklevel=2,
            )
            return float("inf")
        return posterior_lambda**2 * posterior_alpha / ((posterior_alpha - 1) ** 2 * (posterior_alpha - 2))  # type: ignore

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters of the distribution.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.

        Returns
        -------
        dict
            A dictionary containing the posterior parameter 'posterior_lambda'.
        """
        return self.calc_posterior_params(data, return_dict=True)


class GammaABLambdaExponentialPPParams(BDFDistributionParams):
    """Parameters for the Gamma prior on mean lambda with parameters alpha and beta for posterior predictive Lomax."""

    alpha_lambda: float = Field(gt=0.0, description="Shape parameter of the Gamma prior for parameter lambda.")
    beta_lambda: float = Field(gt=0.0, description="Scale parameter of the Gamma prior for parameter lambda.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "alpha_lambda" not in data:
            missing_fields["alpha_lambda"] = self.__class__.model_fields["alpha_lambda"].default
        if "beta_lambda" not in data:
            missing_fields["beta_lambda"] = self.__class__.model_fields["beta_lambda"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class GammaABLambdaExponentialPP(ExponentialPPBase):
    """Exponential distribution with a Gamma prior on mean lambda for posterior predictive Lomax.

    This class implements the Exponential distribution with a Gamma prior on the mean lambda.
    The posterior of lambda is also a Gamma distribution, and the posterior predictive distribution is a Lomax distribution.
    """

    def __init__(self, prior_params: GammaABLambdaExponentialPPParams):
        super().__init__(prior_params)
        self.alpha_lambda = prior_params.alpha_lambda
        self.beta_lambda = prior_params.beta_lambda

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> tuple[float, float] | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)
        alpha_post = self.alpha_lambda + n
        beta_post = self.beta_lambda + sum_data

        if return_dict:
            return {"posterior_alpha": alpha_post, "posterior_lambda": beta_post}
        else:
            return alpha_post, beta_post


class GammaMVLambdaExponentialPPParams(BDFDistributionParams):
    """Parameters for the Gamma prior on mean lambda with parameters alpha and beta for posterior predictive Lomax."""

    mean_lambda: float = Field(gt=0.0, description="Mean of the Gamma prior for parameter lambda.")
    var_lambda: float = Field(gt=0.0, description="Variance of the Gamma prior for parameter lambda.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "mean_lambda" not in data:
            missing_fields["mean_lambda"] = self.__class__.model_fields["mean_lambda"].default
        if "var_lambda" not in data:
            missing_fields["var_lambda"] = self.__class__.model_fields["var_lambda"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class GammaMVLambdaExponentialPP(ExponentialPPBase):
    """Exponential distribution with a Gamma prior on mean lambda for posterior predictive Lomax.

    This class implements the Exponential distribution with a Gamma prior on the mean lambda.
    The posterior of lambda is also a Gamma distribution, and the posterior predictive distribution is a Lomax distribution.
    """

    def __init__(self, prior_params: GammaMVLambdaExponentialPPParams):
        super().__init__(prior_params)
        self.mean_lambda = prior_params.mean_lambda
        self.var_lambda = prior_params.var_lambda

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> tuple[float, float] | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)

        prior_alpha = self.mean_lambda**2 / self.var_lambda
        prior_beta = self.mean_lambda / self.var_lambda

        alpha_post = prior_alpha + n
        beta_post = prior_beta + sum_data

        if return_dict:
            return {"posterior_alpha": alpha_post, "posterior_lambda": beta_post}
        else:
            return alpha_post, beta_post
