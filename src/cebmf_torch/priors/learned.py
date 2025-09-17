from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

from torch import Tensor

from cebmf_torch.cebnm.cash_solver import cash_posterior_means
from cebmf_torch.cebnm.cov_gb_prior import cgb_posterior_means
from cebmf_torch.cebnm.cov_sharp_gb_prior import sharp_cgb_posterior_means
from cebmf_torch.cebnm.emdn import emdn_posterior_means
from cebmf_torch.cebnm.spiked_emdn import spiked_emdn_posterior_means

from .base import Prior, PriorBuilder


class LearnedPriorType(StrEnum):
    """
    Enum for learned prior types.

    Attributes
    ----------
    CASH : str
        Covariate-assisted spike-and-slab prior.
    CGB : str
        Covariate Generalized-binary prior.
    CGB_SHARP : str
        Covariate sharp Generalized-binary prior.
    EMDN : str
        Empirical Mixture Density Network prior.
    SPIKED_EMDN : str
        Spiked Empirical Mixture Density Network prior.
    """
    CASH = auto()
    CGB = auto()
    CGB_SHARP = auto()
    EMDN = auto()
    SPIKED_EMDN = auto()


builder_functions: dict[LearnedPriorType, Callable] = {
    LearnedPriorType.CASH: cash_posterior_means,
    LearnedPriorType.CGB: cgb_posterior_means,
    LearnedPriorType.CGB_SHARP: sharp_cgb_posterior_means,
    LearnedPriorType.EMDN: emdn_posterior_means,
    LearnedPriorType.SPIKED_EMDN: spiked_emdn_posterior_means,
}


class LearnedBuilder(PriorBuilder):
    """
    Builder for learned priors (e.g., CASH, CGB, EMDN).

    Parameters
    ----------
    type : LearnedPriorType
        The type of learned prior to use.
    """
    def __init__(self, type: LearnedPriorType):
        """
        Initialize the LearnedBuilder.

        Parameters
        ----------
        type : LearnedPriorType
            The type of learned prior to use.
        """
        self.type = type

    @property
    def name(self) -> str:
        """
        Name of the prior type.

        Returns
        -------
        str
            String representation of the prior type.
        """
        return str(self.type)

    def fit(
        self,
        X: Tensor | None,
        betahat: Tensor,
        sebetahat: Tensor,
        model_param: Any | None = None,
    ) -> Prior:
        """
        Fit the learned prior to the data.

        Parameters
        ----------
        X : torch.Tensor or None
            Covariate matrix (may be required for some learned priors).
        betahat : torch.Tensor
            Observed effect size estimates.
        sebetahat : torch.Tensor
            Standard errors of the effect size estimates.
        model_param : Any, optional
            Additional model parameters (default: None).

        Returns
        -------
        Prior
            Fitted prior object with posterior means and related quantities.

        Raises
        ------
        ValueError
            If the prior type is unknown or unsupported.
        """
        obj = builder_functions[self.type](X, betahat, sebetahat, model_param=model_param)

        # A bit annoying that the different types have different ways of handling pi0
        match self.type:
            case LearnedPriorType.CASH | LearnedPriorType.SPIKED_EMDN:
                # optional: could expose from obj.pi_np
                pi0_null = obj.pi_np[:, 0]
            case LearnedPriorType.CGB | LearnedPriorType.CGB_SHARP:
                # π₀(x) from the covariate model
                pi0_null = obj.pi
            case LearnedPriorType.EMDN:
                pi0_null = None
            case _:
                raise ValueError(f"Default pi0 unknown for prior type: {self.type}")

        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=float(obj.loss),
            model_param=model_param,
            pi0_null=pi0_null,
        )
