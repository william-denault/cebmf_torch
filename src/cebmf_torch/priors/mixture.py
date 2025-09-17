from enum import StrEnum, auto
from typing import Any

from torch import Tensor

from cebmf_torch.ebnm.ash import ash

from .base import Prior, PriorBuilder


class MixturePriorType(StrEnum):
    """
    Enum for mixture prior types.

    Attributes
    ----------
    NORM : str
        Normal mixture prior.
    EXP : str
        Exponential mixture prior.
    """

    NORM = auto()
    EXP = auto()


class ASHBuilder(PriorBuilder):
    """
    Builder for adaptive shrinkage (ASH) mixture priors.

    Parameters
    ----------
    type : MixturePriorType
        The type of mixture prior to use.
    """

    def __init__(self, type: MixturePriorType):
        """
        Initialize the ASHBuilder.

        Parameters
        ----------
        type : MixturePriorType
            The type of mixture prior to use.
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
        Fit the ASH mixture prior to the data.

        Parameters
        ----------
        X : torch.Tensor or None
            Optional covariate matrix (not used for ASH priors).
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
        """
        obj = ash(betahat, sebetahat, prior=str(self.type))
        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=-float(obj.log_lik),
            model_param=model_param,
            pi0_null=obj.pi0,
        )
