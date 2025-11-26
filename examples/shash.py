"""File to investigate various approximations of the parameters
of the sinh-arcsinh (SAS/SHASH) distribution based on moment estimates
or other relations.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genhyperbolic, norm, skewnorm
from tqdm import tqdm


def c_epsilon_delta(z, epsilon, delta):
    return np.cosh(epsilon + delta * np.arcsinh(z))


def s_epsilon_delta(z, epsilon, delta):
    return np.sinh(epsilon + delta * np.arcsinh(z))


def loglik_shash(params, y) -> float:
    """Log-likelihood function for the sinh-arcsinh (SAS/SHASH) distribution.

    Args
    ----
    y : np.ndarray
        Input data.
    mu : float, optional
        Location parameter (default is 0.0).
    sigma : float, optional
        Scale parameter (default is 1.0).
    epsilon : float, optional
        Skewness parameter.
    delta : float, optional
        Tail heaviness parameter.

    Returns
    -------
    float
        Log-likelihood value.
    """
    mu, sigma, epsilon, delta = params
    if sigma <= 0:
        raise ValueError("Scale parameter sigma must be positive.")

    z = (y - mu) / sigma
    # Calculate the log-likelihood using the skew-normal distribution
    return (
        -np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        + np.log(delta)
        + np.log(c_epsilon_delta(z, epsilon, delta))
        - 0.5 * np.log(1 + z**2)
        - 0.5 * s_epsilon_delta(z, epsilon, delta) ** 2
    )


def lik_shash(
    params: tuple,
    y: np.ndarray,
) -> np.ndarray:
    return np.exp(loglik_shash(params, y))


def nll_shash(
    params: np.ndarray,
    y: np.ndarray,
) -> float:
    """Negative log-likelihood function for the SAS/SHASH distribution.

    Args
    ----
    params : np.ndarray
        Parameters [mu, sigma, epsilon, delta].
    y : np.ndarray
        Input data.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    return -np.mean(loglik_shash(params, y))


def penalized_nll(params, y, lbda_1=0.0, lbda_2=0.5) -> float:
    return nll_shash(params=params, y=y) + lbda_1 * np.abs(params[2]) + lbda_2 * np.abs(params[3] - 1)


def simulate_data(n_mixtures=10, probs=[0.1, 0.5, 0.2, 0.2], samples_per_sim=100, seed=42):
    np.random.seed(seed)
    out_data = np.zeros((n_mixtures * samples_per_sim,))
    probs = np.array(probs) / sum(probs)
    rng = np.random.default_rng(seed)
    for i in tqdm(range(n_mixtures), "Simulating data"):
        dist = rng.choice(["normal", "skewnormal", "genhyp", "uniform"], p=probs)

        match dist:
            case "normal":
                mu = np.random.uniform(-10, 10)
                sigma = np.random.uniform(0.1, 50)
                data = norm.rvs(loc=mu, scale=sigma, size=samples_per_sim)
            case "skewnormal":
                mu = np.random.uniform(-10, 10)
                sigma = np.random.uniform(0.1, 50)
                skew = np.random.uniform(-10, 10)
                data = skewnorm.rvs(a=skew, loc=mu, scale=sigma, size=samples_per_sim)
            case "genhyp":
                p = np.random.uniform(-0.5, 1.5)  # lambda
                a = np.random.uniform(0.1, 10)  # alpha
                b = np.random.uniform(-a + 1e-3, a - 1e-3)  # beta
                loc = np.random.uniform(-5, 5)
                scale = np.random.uniform(0.5, 5)
                data = genhyperbolic.rvs(p, a, b, loc=loc, scale=scale, size=samples_per_sim)
            case "uniform":
                low = np.random.uniform(-10, 0)
                high = np.random.uniform(0, 10)
                data = np.random.uniform(low, high)
            case _:
                raise ValueError(f"Distribution '{dist}' not supported.")

        out_data[(i * samples_per_sim) : ((i + 1) * samples_per_sim)] = data

    return out_data


def calc_posterior_params(data, penalized: bool = False) -> tuple:
    initial_params = np.array([np.mean(data), np.std(data), 0.0, 1])
    fct = penalized_nll if penalized else nll_shash
    result = minimize(
        fct,
        initial_params,
        args=(data,),
        method="L-BFGS-B",
        bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-np.inf, np.inf), (1e-6, np.inf)],
        options={"maxiter": 50},
    )
    return result.x


# %%
# Fit the SAS/SHASH distribution to the data
from scipy.optimize import minimize

data = simulate_data(n_mixtures=10, samples_per_sim=50, probs=[0.4, 2, 0.3, 8])
param_tuple = calc_posterior_params(data, penalized=True)
x_vals = np.arange(data.min(), data.max(), (data.max() - data.min()) / 1000)
plt.hist(data, density=True, color="dodgerblue", bins=35)
liks = lik_shash(param_tuple, x_vals)
plt.plot(x_vals, liks, color="crimson")
plt.show()


# %%
