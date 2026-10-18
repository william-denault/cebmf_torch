from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any

import torch
from torch import Tensor

from cebmf_torch.ebnm.generalized_binary import ebnm_gb
from cebmf_torch.ebnm.point_exp import ebnm_point_exp
from cebmf_torch.ebnm.point_laplace import ebnm_point_laplace

from .base import Prior, PriorBuilder

# Slab-weight clamp used when converting pi_slab back to logit(pi_slab)
# for warm-start. Matches the range used inside the L-BFGS closures.
_WARMSTART_EPS = 1e-8
_WARMSTART_LOG_A_FLOOR = 1e-12


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


def _warmstart_par_init_from_result(prior_type: PointPriorType, obj: Any) -> tuple[float, float, float] | None:
    """Extract ``(logit(pi_slab), log(a), mu)`` from an EBNM result for the
    next iteration's warm-start. Returns ``None`` for prior types whose
    underlying solver does not accept this ``par_init`` tuple.

    The L-BFGS solvers in ``ebnm_point_laplace`` / ``ebnm_point_exp`` are
    non-convex; without a warm-start they restart from a fixed init each
    cEBMF iteration and can converge to a different local optimum, breaking
    ELBO monotonicity. Threading the previous solution through ``par_init``
    fixes that.
    """
    if prior_type == PointPriorType.LAPLACE:
        a_val = obj.a
        mu_val = obj.mu
    elif prior_type == PointPriorType.EXP:
        a_val = obj.scale  # named ``scale`` for API compatibility but is the rate
        mu_val = obj.mode
    else:
        # GBINARY uses a different init API (par_init_mu / par_init_pi) and is
        # not handled here. EM there is convex in (mu, pi) for the inner step
        # so the monotonicity story is less acute; can be added separately.
        return None

    pi_slab = obj.pi_slab.detach().clamp(_WARMSTART_EPS, 1.0 - _WARMSTART_EPS)
    logit_pi = (torch.log(pi_slab) - torch.log1p(-pi_slab)).item()
    log_a = torch.log(torch.as_tensor(a_val).clamp_min(_WARMSTART_LOG_A_FLOOR)).item()
    mu_f = float(mu_val)
    return (logit_pi, log_a, mu_f)


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

        # Warm-start: if cEBMF gave us a previous solution for this factor,
        # feed it back as ``par_init`` so L-BFGS doesn't restart from the
        # default each iteration. The previous-iteration ``par_init`` is the
        # only thing we actually need from ``model_param``; user-supplied
        # kwargs take precedence (so explicit ``par_init`` in ``self.kwargs``
        # is not overridden by warm-start).
        fit_kwargs = dict(self.kwargs)
        if (
            isinstance(model_param, dict)
            and "par_init" in model_param
            and "par_init" not in fit_kwargs
            and self.type in (PointPriorType.LAPLACE, PointPriorType.EXP)
        ):
            fit_kwargs["par_init"] = model_param["par_init"]

        obj = builder_functions[self.type](betahat, sebetahat, **fit_kwargs)
        # `obj.pi_slab` is the slab (non-null) weight by the convention of
        # EBNMPointExp / EBNMLaplaceResult / EBNMGBResult. The Prior.pi0_null
        # field expected by `cebmf._should_prune_factor` (cebmf.py:482) is
        # the *null/spike* weight, so it is `1 - pi_slab`. Both `pi_slab` and
        # `log_lik` are now 0-d tensors on-device — keeping them that way
        # avoids the per-fit-call host sync that the previous `float(...)`
        # casts forced.
        pi_slab_t = obj.pi_slab

        # Build the model_param payload for the next iteration's warm-start.
        # For LAPLACE / EXP we encode the converged ``par_init`` tuple; for
        # other priors we pass through whatever the caller had.
        next_par_init = _warmstart_par_init_from_result(self.type, obj)
        if next_par_init is not None:
            next_model_param: Any = {"par_init": next_par_init}
        else:
            next_model_param = model_param

        return Prior(
            post_mean=obj.post_mean,
            post_mean2=obj.post_mean2,
            loss=-obj.log_lik,
            model_param=next_model_param,
            pi0_null=1.0 - pi_slab_t,
            pi_slab=pi_slab_t,
        )
