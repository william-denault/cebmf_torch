# torch_convolved_loglik.py
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto

import torch

from cebmf_torch.utils.distribution_operation import (
    get_data_loglik_exp_torch,
    get_data_loglik_normal_torch,
)
from cebmf_torch.utils.mixture import (
    autoselect_scales_mix_exp,
    autoselect_scales_mix_norm,
    optimize_pi_logL,
)
from cebmf_torch.utils.posterior import (
    PosteriorMean,
    posterior_mean_exp,
    posterior_mean_norm,
)


class PriorType(StrEnum):
    NORM = auto()
    EXP = auto()


@dataclass
class AshConfig:
    mult: float = math.sqrt(2.0)
    penalty: float = 10.
    verbose: bool = True
    threshold_loglikelihood: float = -300.0
    mode: float = 0.0  # only for PriorType.NORM
    batch_size: int | None = 128
    shuffle: bool = False
    seed: int | None = None


def _optimize_mixture_weights(L: torch.Tensor, config: AshConfig) -> torch.Tensor:
    """Optimize mixture weights and return log probabilities."""
    pi0 = optimize_pi_logL(
        L,
        penalty=config.penalty,
        verbose=config.verbose,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        seed=config.seed,
    )
    return torch.log(torch.clamp(pi0, min=1e-32))


def _ash_normal(
    x: torch.Tensor,
    s: torch.Tensor,
    config: AshConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PosteriorMean]:
    scale = autoselect_scales_mix_norm(x, s, mult=config.mult)  # (K,)
    loc = torch.full((scale.shape[0],), config.mode, dtype=x.dtype, device=x.device)

    L = get_data_loglik_normal_torch(x, s, location=loc, scale=scale)  # (J,K)
    log_pi0 = _optimize_mixture_weights(L, config)
    pm_obj = posterior_mean_norm(x, s, log_pi=log_pi0, data_loglik=L, location=loc, scale=scale)
    return scale, log_pi0, L, pm_obj


def _ash_exp(
    x: torch.Tensor,
    s: torch.Tensor,
    config: AshConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PosteriorMean]:
    scale = autoselect_scales_mix_exp(x, s, mult=config.mult)  # (K,) with scale[0]=0 (spike)
    L = get_data_loglik_exp_torch(x, s, scale=scale)  # (J,K)
    log_pi0 = _optimize_mixture_weights(L, config)
    pm_obj = posterior_mean_exp(x, s, log_pi=log_pi0, scale=scale)
    return scale, log_pi0, L, pm_obj


ash_optimisers: dict[
    PriorType,
    Callable[
        [torch.Tensor, torch.Tensor, AshConfig],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, PosteriorMean],
    ],
] = {
    PriorType.NORM: _ash_normal,
    PriorType.EXP: _ash_exp,
}


@dataclass
class ASHResult:
    """Result from ASH (Adaptive SHrinkage) algorithm.

    Attributes:
        post_mean: Posterior means for each observation
        post_mean2: Posterior second moments for each observation
        post_sd: Posterior standard deviations for each observation
        scale: Mixture component scales/standard deviations
        pi0: Null component probability (spike at zero)
        prior: Prior type used ("norm" or "exp")
        log_lik: Total log-likelihood of the fitted model
        mode: Mode parameter used (only relevant for normal prior)
    """

    post_mean: torch.Tensor
    post_mean2: torch.Tensor
    post_sd: torch.Tensor
    scale: torch.Tensor
    pi0: torch.Tensor | float
    prior: str
    log_lik: float = 0.0
    mode: float = 0.0

    @classmethod
    def from_data(cls, x: torch.Tensor, s: torch.Tensor, prior: PriorType, config: AshConfig) -> "ASHResult":
        """Factory method to create ASHResult from data."""
        scale, log_pi0, L, pm_obj = ash_optimisers[prior](x, s, config)
        pi0 = torch.exp(log_pi0)
        Lc = torch.maximum(
            L,
            torch.tensor(config.threshold_loglikelihood, dtype=L.dtype, device=L.device),
        )
        log_lik_rows = torch.logsumexp(Lc + torch.log(torch.clamp(pi0, min=1e-300)).unsqueeze(0), dim=1)
        log_lik = float(log_lik_rows.sum().item())
        return cls(
            post_mean=pm_obj.post_mean,
            post_mean2=pm_obj.post_mean2,
            post_sd=pm_obj.post_sd,
            scale=scale,
            pi0=pi0[0],
            prior=str(prior),
            log_lik=log_lik,
            mode=float(config.mode),
        )


# ---- ASH (Torch) ----
@torch.no_grad()
def ash(
    x: torch.Tensor,
    s: torch.Tensor,
    prior: PriorType = PriorType.NORM,
    mult: float = math.sqrt(2.0),
    penalty: float = 10.,
    verbose: bool = True,
    threshold_loglikelihood: float = -300.0,
    mode: float = 0.0,
    *,
    batch_size: int | None = 128,
    shuffle: bool = False,
    seed: int | None = None,
):
    """
    Adaptive shrinkage with mixture priors ("norm" or "exp") in pure PyTorch.

    Uses EM for π (mini-batch capable via batch_size).
    Returns an ASHResult object with Torch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Observed data.
    s : torch.Tensor
        Standard errors of the observed data.
    prior : PriorType, optional
        Type of prior to use (default: PriorType.NORM).
    mult : float, optional
        Multiplier for scale grid (default: sqrt(2.0)).
    penalty : float, optional
        Penalty for mixture weights (default: 10.0).
    verbose : bool, optional
        Verbosity flag (default: True).
    threshold_loglikelihood : float, optional
        Minimum log-likelihood threshold (default: -300.0).
    mode : float, optional
        Mode parameter (for normal prior only, default: 0.0).
    batch_size : int or None, optional
        Batch size for EM updates (default: 128).
    shuffle : bool, optional
        Whether to shuffle data in EM (default: False).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    ASHResult
        Result object containing posterior summaries and model parameters.
    """

    # choose optimizer mode (EM by default here)
    if prior not in ash_optimisers:
        raise ValueError("prior must be either 'norm' or 'exp'.")

    s.clamp_(min=1e-12)
    config = AshConfig(
        mult=mult,
        penalty=penalty,
        verbose=verbose,
        threshold_loglikelihood=threshold_loglikelihood,
        mode=mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return ASHResult.from_data(x, s, prior, config)
