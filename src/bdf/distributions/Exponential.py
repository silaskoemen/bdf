"""Exponential data with Gamma prior, posterior is of lambda is Gamma

Allowed versions:
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
from scipy.stats import expon

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED


class ExponentialBase(BDFDistribution):
    """Base class for all Exponential distribution implementations.

    All implementations use parameter lambda, so sampling,
    (log)likelihoods and return functions are all identical.
    """

    # This is just a placeholder - child classes will have their own init
    def __init__(self, prior_params, params=None):
        super().__init__(prior_params, params)

    # Child classes MUST implement this method
    def calc_posterior_params(self, data, return_dict=True):
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
        posterior_lambda = self.calc_posterior_params(data, return_dict=False)
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
        posterior_lambda = self.calc_posterior_params(data, return_dict=False)
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
        if data is None and params is None:
            raise ValueError(
                "Either 'data' or 'params' must be provided to generate samples from the posterior distribution."
            )
        if params is not None:
            return self.sample_posterior_params(params, size=size, random_state=random_state)
        elif data is not None:
            return self.sample_posterior_data(data, size=size, random_state=random_state)
        else:  # This case should not happen due to the initial check but is required for type safety
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
        lbda = params.get("lambda", params.get("posterior_lambda"))
        if lbda is None:
            raise ValueError("params must contain 'lambda' keys")
        assert lbda > 0, "Lbda parameter must be positive"
        return expon.rvs(scale=1 / lbda, size=size, random_state=random_state)  # type: ignore

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
        posterior_lambda = self.calc_posterior_params(data, return_dict=False)
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
            posterior_lambda = self.calc_posterior_params(data, return_dict=False)
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
            return 1 / params.get("lambda", params.get("posterior_lambda")) ** 2  # type: ignore
        elif data is not None:
            posterior_lambda = self.calc_posterior_params(data, return_dict=False)
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
            A dictionary containing the posterior parameter 'lambda'.
        """
        posterior_lambda = self.calc_posterior_params(data)
        return {"lambda": posterior_lambda}


class GammaABLambdaExponentialParams(BDFDistributionParams):
    """Parameters for the Gamma prior on mean lambda with parameters alpha and theta."""

    alpha: float = Field(gt=0.0, description="Shape parameter of the Gamma prior.")
    beta: float = Field(gt=0.0, description="Scale parameter of the Gamma prior.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "alpha" not in data:
            missing_fields["alpha"] = self.__class__.model_fields["alpha"].default
        if "beta" not in data:
            missing_fields["beta"] = self.__class__.model_fields["beta"].default

        super().__init__(**data)

        # Issue warnings for missing fields
        for field, default_value in missing_fields.items():
            warnings.warn(
                f"No value provided for '{field}', using default: {default_value}",
                UserWarning,
                stacklevel=2,
            )


class GammaATLambdaExponential(ExponentialBase):
    """Exponential distribution with a Gamma prior on mean lambda.

    This class implements the Exponential distribution with a Gamma prior on the mean lambda.
    The posterior is also a Gamma distribution.
    """

    def __init__(self, prior_params: GammaABLambdaExponentialParams):
        super().__init__(prior_params)
        self.prior_alpha = prior_params.alpha
        self.prior_beta = prior_params.beta

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = True) -> float | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)
        alpha_post = self.prior_alpha + n
        beta_post = self.prior_beta + sum_data

        if return_dict:
            return {"lambda": alpha_post / beta_post}
        else:
            return alpha_post / beta_post


class GammaMVLambdaExponentialParams(BDFDistributionParams):
    """Parameters for the Gamma prior on mean lambda with parameters mean and variance."""

    mean: float = Field(gt=0.0, description="Mean of the Gamma prior.")
    variance: float = Field(gt=0.0, description="Variance of the Gamma prior.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "mean" not in data:
            missing_fields["mean"] = self.__class__.model_fields["mean"].default
        if "variance" not in data:
            missing_fields["variance"] = self.__class__.model_fields["variance"].default

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

    def __init__(self, prior_params: GammaMVLambdaExponentialParams):
        super().__init__(prior_params)
        self.prior_mean = prior_params.mean
        self.prior_variance = prior_params.variance

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = True) -> float | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)

        # Relate mean and variance to alpha and beta of the Gamma distribution
        prior_alpha = self.prior_mean**2 / self.prior_variance
        prior_beta = self.prior_mean / self.prior_variance

        alpha_post = prior_alpha + n
        beta_post = prior_beta + sum_data

        if return_dict:
            return {"lambda": alpha_post / beta_post}
        else:
            return alpha_post / beta_post


class ExponentialPPBase(ExponentialBase):
    """Base class for Exponential likelihood - Lomax posterior predictive distributions.

    Final distribution is Lomax, meaning parameters returned are alpha and lambda and
    all likelihoods and sampling functions are based on these parameters.
    """

    pass


class GammaABLambdaExponentialPPParams(BDFDistributionParams):
    """Parameters for the Gamma prior on mean lambda with parameters alpha and beta for posterior predictive Lomax."""

    alpha: float = Field(gt=0.0, description="Shape parameter of the Gamma prior.")
    beta: float = Field(gt=0.0, description="Scale parameter of the Gamma prior.")

    class Config:
        """Pydantic configuration to allow extra fields and use aliases."""

        extra = "forbid"
        validate_by_name = True

    def __init__(self, **data):
        # Check for missing fields before initialization
        missing_fields = {}
        if "alpha" not in data:
            missing_fields["alpha"] = self.__class__.model_fields["alpha"].default
        if "beta" not in data:
            missing_fields["beta"] = self.__class__.model_fields["beta"].default

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
        self.prior_alpha = prior_params.alpha
        self.prior_beta = prior_params.beta

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = True) -> float | dict:
        """Calculate the posterior parameters based on the data."""
        n = len(data)
        sum_data = np.sum(data)
        alpha_post = self.prior_alpha + n
        beta_post = self.prior_beta + sum_data

        if return_dict:
            return {"lambda": alpha_post / beta_post}
        else:
            return alpha_post / beta_post
