from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any

import torch
from torch import Tensor

from cebmf_torch.ebnm.ash import ash
from cebmf_torch.utils.mixture import mixture_log_penalty

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


@dataclass
class ASHState:
    """Per-factor component grid and weights for the next coordinate update."""

    prior: MixturePriorType
    mode: float
    scale: Tensor
    pi: Tensor


class ASHBuilder(PriorBuilder):
    """
    Builder for adaptive shrinkage (ASH) mixture priors.

    Parameters
    ----------
    type : MixturePriorType
        The type of mixture prior to use.
    """

    def __init__(self, type: MixturePriorType, **kwargs: Any) -> None:
        """
        Initialize the ASHBuilder.

        Parameters
        ----------
        type : MixturePriorType
            The type of mixture prior to use.
        **kwargs : Any
            Additional keyword arguments specific to the prior type.
        """
        self.type = type
        self.kwargs = kwargs

    def set_kwargs(self, **new_kwargs: Any) -> None:
        """
        Overwrite the keyword arguments for the builder.

        Parameters
        ----------
        **new_kwargs : Any
            New keyword arguments to set, replacing the old ones.
        """
        self.kwargs = new_kwargs

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
        internal_epoch: Any | None = None,
        device: torch.device | None = None,
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
        model_param : ASHState or None, optional
            Previous fit's grid and weights. Reusing both preserves the
            previous prior as a feasible starting point for the next update.
        internal_epoch : Any, optional
            Unused; kept for signature parity with the other builders.
        device : torch.device, optional
            Unused — ASH inherits its device from the input tensors directly.
            Accepted for signature parity with :class:`LearnedBuilder`.

        Returns
        -------
        Prior
            Fitted prior object with posterior means and related quantities.
        """
        del device  # ASH operates on the input tensors' device directly
        fit_kwargs = dict(self.kwargs)
        penalty = self.kwargs.get("penalty", 10.0)
        if (
            isinstance(model_param, ASHState)
            and model_param.prior == self.type
            and model_param.mode == fit_kwargs.get("mode", 0.0)
        ):
            if fit_kwargs.get("scale") is None:
                fit_kwargs["scale"] = model_param.scale
            if torch.equal(torch.as_tensor(fit_kwargs["scale"]).to(model_param.scale), model_param.scale):
                fit_kwargs.setdefault("pi_init", model_param.pi)
        obj = ash(betahat, sebetahat, prior=str(self.type), **fit_kwargs)
        if "pi_init" in fit_kwargs and self.kwargs.get("scale") is None:
            # Keep adaptive grid selection, but do not discard a better prior
            # merely because the new residuals suggest a different grid.
            fresh = ash(betahat, sebetahat, prior=str(self.type), **self.kwargs)

            def criterion(result):
                return result.log_lik + mixture_log_penalty(result.pi, penalty)

            if criterion(fresh) > criterion(obj):
                obj = fresh
        state = ASHState(self.type, obj.mode, obj.scale.detach().clone(), obj.pi.detach().clone())
        # Keep the marginal log-likelihood on-device for cEBMF's KL calculation.
        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=-obj.log_lik,
            model_param=state,
            pi0_null=obj.pi0,
            regularization_loss=-mixture_log_penalty(obj.pi, penalty),
        )
