from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

import torch
from torch import Tensor

from cebmf_torch.ebnm.generalized_binary import ebnm_gb
from cebmf_torch.ebnm.point_exp import ebnm_point_exp
from cebmf_torch.ebnm.point_laplace import ebnm_point_laplace

from .base import Prior, PriorBuilder


class PointPriorType(StrEnum):
    """
    Enum for point prior types.

    Attributes
    ----------
    LAPLACE : str
        Laplace (double exponential) prior.
    EXP : str
        Exponential prior.
    """

    LAPLACE = auto()
    EXP = auto()
    GBINARY = auto()


builder_functions: dict[PointPriorType, Callable] = {
    PointPriorType.LAPLACE: ebnm_point_laplace,
    PointPriorType.EXP: ebnm_point_exp,
    PointPriorType.GBINARY: ebnm_gb,
}


class PointBuilder(PriorBuilder):
    """
    Builder for point priors (Laplace or Exponential).

    Parameters
    ----------
    type : PointPriorType
        The type of point prior to use.
    """

    def __init__(self, type: PointPriorType, **kwargs: Any) -> None:
        """
        Initialize the PointBuilder.

        Parameters
        ----------
        type : PointPriorType
            The type of point prior to use.
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
        Fit the point prior to the data.

        Parameters
        ----------
        X : torch.Tensor or None
            Optional covariate matrix (not used for point priors).
        betahat : torch.Tensor
            Observed effect size estimates.
        sebetahat : torch.Tensor
            Standard errors of the effect size estimates.
        model_param : Any, optional
            Additional model parameters (default: None).
        internal_epoch : Any, optional
            Unused; kept for signature parity with the other builders.
        device : torch.device, optional
            Unused — point priors operate on the input tensors' device.
            Accepted for signature parity with :class:`LearnedBuilder`.

        Returns
        -------
        Prior
            Fitted prior object with posterior means and related quantities.
        """
        del device  # point priors run on the input tensors' device directly
        obj = builder_functions[self.type](betahat, sebetahat, **self.kwargs)
        # `obj.pi_slab` is the slab (non-null) weight by the convention of
        # EBNMPointExp / EBNMLaplaceResult / EBNMGBResult. The Prior.pi0_null
        # field expected by `cebmf._should_prune_factor` (cebmf.py:482) is
        # the *null/spike* weight, so it is `1 - pi_slab`. Both `pi_slab` and
        # `log_lik` are now 0-d tensors on-device — keeping them that way
        # avoids the per-fit-call host sync that the previous `float(...)`
        # casts forced.
        pi_slab_t = obj.pi_slab
        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=-obj.log_lik,
            model_param=model_param,
            pi0_null=1.0 - pi_slab_t,
            pi_slab=pi_slab_t,
        )
