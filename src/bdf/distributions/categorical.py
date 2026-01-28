"""Dirichlet prior over categories, where certain real number per category is prior.
larger numbers mean stronger influence of priors, as posterior counts absolute number of
events in category"""

from typing import Any

import numpy as np

from bdf.distributions.bdf_distribution import BDFDistribution


class DirichletCategorical(BDFDistribution):
    """Dirichlet-Categorical distribution class for Bayesian Distributional Forests.
    This class models a Categorical distribution with a Dirichlet prior on the category probabilities.
    """

    def __init__(self, params: dict[str, Any]):
        """Initialize the Dirichlet-Categorical distribution with prior parameters.

        Args
        ----
        `params` : dict
            Dictionary containing prior parameters, must include 'alpha' for each category.
        `params` : tuple, optional
            Additional parameters for the distribution, default is None.
        """
        super().__init__(params)
        self.prior_alpha = np.array(params.get("alpha", [1.0] * len(params)))
        if np.any(self.prior_alpha <= 0):
            raise ValueError("Prior parameters 'alpha' must be positive.")
        self.params = tuple(self.prior_alpha)

    def calc_posterior_params(self, data: np.ndarray) -> dict[str, Any]:
        """Calculate posterior parameters based on the data.

        Args
        ----
        `data` : np.ndarray
            The data to calculate the posterior parameters from.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'posterior_alpha' array for each category.
        """
        counts = np.bincount(data, minlength=len(self.prior_alpha))
        posterior_alpha = self.prior_alpha + counts
        return {"posterior_alpha": posterior_alpha}
