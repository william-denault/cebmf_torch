"""Covariate-Enhanced Empirical Bayes Normal Means (cEBNM) solvers."""

# Advanced solvers with covariates
from .cash_solver import cash_posterior_means
from .cov_gb_prior import cgb_posterior_means
from .cov_sharp_gb_prior import sharp_cgb_posterior_means
from .emdn import emdn_posterior_means
from .spiked_emdn import spiked_emdn_posterior_means

__all__ = [
    "cash_posterior_means",
    "cgb_posterior_means",
    "emdn_posterior_means",
    "sharp_cgb_posterior_means",
    "spiked_emdn_posterior_means",
]
