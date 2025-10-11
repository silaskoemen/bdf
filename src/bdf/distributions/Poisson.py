import warnings

import numpy as np
from pydantic import Field
from scipy.stats import nbinom, poisson

from bdf.distributions.bdf_distribution import BDFDistribution, BDFDistributionParams
from bdf.utils.constants import RANDOM_SEED


class PoissonBase(BDFDistribution):
    """Base class for all Poisson distribution implementations (where posterior sampling distribution remains Poisson).

    All implementations use parameter lambda (rate), so sampling,
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
        return poisson.logpmf(data, mu=1 / posterior_lambda)

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
        return poisson.pmf(data, mu=1 / posterior_lambda)

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
        return poisson.rvs(mu=1 / posterior_lambda, size=size, random_state=random_state)  # type: ignore

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
        return poisson.rvs(mu=1 / posterior_lambda, size=size, random_state=random_state)  # type: ignore

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
            return 1 / params.get("posterior_lambda")  # type: ignore
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
            return 1 / params.get("posterior_lambda")  # type: ignore
        elif data is not None:
            posterior_lambda = self.calc_posterior_params(data)
            return 1 / posterior_lambda  # type: ignore
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


class GammaABLambdaPoissonParams(BDFDistributionParams):
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


class GammaABLambdaPoisson(BDFDistribution):
    """Gamma-Poisson distribution class for Bayesian Distributional Forests.
    This class models a Poisson distribution with a Gamma prior on the rate parameter.
    """

    def __init__(self, prior_params: GammaABLambdaPoissonParams | dict, params: dict | None = None):
        """Initialize the Gamma-Poisson distribution with prior parameters.

        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'alpha' and 'beta'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        if isinstance(prior_params, dict):
            prior_params = GammaABLambdaPoissonParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, GammaABLambdaPoissonParams
        ), "prior_params must be an instance of GammaABLambdaExponentialParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.alpha_lambda = prior_params.alpha_lambda
        self.beta_lambda = prior_params.beta_lambda

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict:
        """Calculate posterior parameters based on the data.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.

        Returns
        -------
        tuple[float, float]
            A tuple containing the posterior alpha and posterior beta.
        """
        n_events = np.sum(data)
        n = data.shape[0]

        alpha_post = self.alpha_lambda + n_events
        beta_post = self.beta_lambda + n

        if return_dict:
            return {"posterior_lambda": alpha_post / beta_post}
        else:
            return alpha_post / beta_post


class GammaMVLambdaPoissonParams(BDFDistributionParams):
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


class GammaMVLambdaPoisson(BDFDistribution):
    """Gamma-Poisson distribution class for Bayesian Distributional Forests.
    This class models a Poisson distribution with a Gamma prior on the rate parameter.
    """

    def __init__(self, prior_params: GammaMVLambdaPoissonParams | dict, params: dict | None = None):
        """Initialize the Gamma-Poisson distribution with prior parameters.

        Args
        ----
        `prior_params` : dict
            Dictionary containing prior parameters, must include 'alpha' and 'beta'.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        if isinstance(prior_params, dict):
            prior_params = GammaMVLambdaPoissonParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, GammaMVLambdaPoissonParams
        ), "prior_params must be an instance of GammaABLambdaExponentialParams after possible conversion from dict."
        super().__init__(prior_params, params)
        self.mean_lambda = prior_params.mean_lambda
        self.var_lambda = prior_params.var_lambda

    def calc_posterior_params(self, data: np.ndarray, return_dict: bool = False) -> float | dict:
        """Calculate posterior parameters based on the data.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.

        Returns
        -------
        tuple[float, float]
            A tuple containing the posterior alpha and posterior beta.
        """
        n_events = np.sum(data)
        n = data.shape[0]

        alpha_lambda = self.mean_lambda**2 / self.var_lambda
        beta_lambda = self.mean_lambda / self.var_lambda

        alpha_post = alpha_lambda + n_events
        beta_post = beta_lambda + n

        if return_dict:
            return {"posterior_lambda": alpha_post / beta_post}
        else:
            return alpha_post / beta_post


class NormalMeanPoissonParams(BDFDistributionParams):
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


class NormalMeanPoisson(PoissonBase):
    """Exponential distribution with a Normal prior on mean lambda.

    This class implements the Exponential distribution with a Normal prior on the mean lambda.
    The posterior is also an Exponential distribution, and the sampling distribution is again an Exponential distribution.
    """

    def __init__(self, prior_params: NormalMeanPoissonParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = NormalMeanPoissonParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, NormalMeanPoissonParams
        ), "prior_params must be an instance of NormalMeanPoissonParams after possible conversion from dict."
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


class PseudoLambdaPoissonParams(BDFDistributionParams):
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


class PseudoLambdaPoisson(PoissonBase):
    """Exponential distribution with a Pseudo prior on mean lambda.

    This class implements the Exponential distribution with a Pseudo prior on the mean lambda.
    The posterior is also an Exponential distribution, and the sampling distribution is again an Exponential distribution.
    """

    def __init__(self, prior_params: PseudoLambdaPoissonParams, params: dict | None = None):
        if isinstance(prior_params, dict):
            prior_params = PseudoLambdaPoissonParams.model_validate(prior_params)  # type: ignore
        assert isinstance(
            prior_params, PseudoLambdaPoissonParams
        ), "prior_params must be an instance of PseudoLambdaPoissonParams after possible conversion from dict."
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


class PoissonPPBase(BDFDistribution):
    """Base class for all Poisson data implementations that use the NB posterior predictive
    distribution. Because the NB parameters are the same for all sampling and likelihood calculations,
    can be unified once and subclasses by specific parameterizations.
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
        posterior_alpha, posterior_beta = self.calc_posterior_params(data)
        return nbinom.logpmf(data, n=posterior_alpha, p=posterior_beta / (1 + posterior_beta))

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
        posterior_alpha, posterior_beta = self.calc_posterior_params(data)
        return nbinom.pmf(data, n=posterior_alpha, p=posterior_beta / (1 + posterior_beta))

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
        posterior_alpha = params.get("posterior_alpha")
        posterior_beta = params.get("posterior_beta")
        if posterior_alpha is None or posterior_beta is None:
            raise ValueError("params must contain 'posterior_alpha' and 'posterior_beta' keys")
        assert posterior_alpha > 0, "'posterior_alpha' parameter must be positive"
        assert posterior_beta > 0, "'posterior_beta' parameter must be positive"
        return nbinom.rvs(n=posterior_alpha, p=posterior_beta / (1 + posterior_beta), size=size, random_state=random_state)  # type: ignore

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
        posterior_alpha, posterior_beta = self.calc_posterior_params(data)
        return nbinom.rvs(n=posterior_alpha, p=posterior_beta / (1 + posterior_beta), size=size, random_state=random_state)  # type: ignore

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
            posterior_alpha, posterior_beta = params.get("posterior_alpha"), params.get("posterior_beta")  # type: ignore
        elif data is not None:
            posterior_alpha, posterior_beta = self.calc_posterior_params(data)
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior mean.")
        return posterior_alpha * posterior_beta / (1 + posterior_beta)  # type: ignore

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
            posterior_alpha, posterior_beta = params.get("posterior_alpha"), params.get("posterior_beta")  # type: ignore
        elif data is not None:
            posterior_alpha, posterior_beta = self.calc_posterior_params(data)
        else:
            raise ValueError("Either 'data' or 'params' must be provided to calculate the posterior variance.")
        raise NotImplementedError("Variance from alpha and beta of Gamma distribution not implemented yet")

    def get_posterior_params(self, data: np.ndarray) -> dict:
        """Get the posterior parameters of the distribution.

        NOTE: Calculation of params estiates posterior alpha and beta from the Gamma distribution,
        from which n and p of the negative binomial distribution can be calculated.
        At the moment, 'n' and 'p' are also valid keys to return all parameter versions

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.

        Returns
        -------
        dict
            A dictionary containing the posterior parameter 'posterior_alpha'.
        """
        return self.calc_posterior_params(data, return_dict=True)

    def validate_targets(self, data: np.ndarray):
        assert np.all(data > 0), "Targets have to be positive for exponential distribution."
        assert all(np.isfinite(data)), "Targets must be finite for exponential distribution."
        std = np.std(data)
        assert (
            np.isfinite(std) and std is not None and std >= 0.0
        ), f"Standard deviation has to be finite, not None and >=0, got {std}"
