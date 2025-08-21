
import torch
from torch import Tensor
from dataclasses import dataclass
from .torch_utils_mix import autoselect_scales_mix_norm, autoselect_scales_mix_exp
from .torch_distribution_operation import get_data_loglik_normal, get_data_loglik_exp
from .torch_posterior import posterior_mean_norm, posterior_mean_exp, PosteriorResult
from .torch_mix_opt import optimize_pi_logL_torch
from .torch_utils import logsumexp

@dataclass
class AshResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    scale: Tensor
    pi: Tensor
    prior: str
    log_lik: float
    mode: float = 0.0

def ash(
    betahat: Tensor,
    sebetahat: Tensor,
    prior: str = "norm",
    mult: float = 2.0,
    penalty: float = 10.0,
    verbose: bool = False,
    threshold_loglikelihood: float = -300.0,
    mode: float = 0.0,
    method: str = "adam",  # optimizer for pi
    steps: int = 200,
    batch_size: int = 65536,
    lr: float = 0.05,
) -> AshResult:
    """
    Pure-PyTorch adaptive shrinkage with mixture priors ('norm' or 'exp').
    """
    device = betahat.device
    if prior == "norm":
        scale = autoselect_scales_mix_norm(betahat, sebetahat, mult=mult)
        location = torch.zeros_like(scale) + mode
        L = get_data_loglik_normal(betahat, sebetahat, location, scale)
        pi = optimize_pi_logL_torch(L, penalty=penalty, method=method, steps=steps, batch_size=batch_size, lr=lr)
        res: PosteriorResult = posterior_mean_norm(betahat, sebetahat, torch.log(pi + 1e-32), scale, location)
    elif prior == "exp":
        scale = autoselect_scales_mix_exp(betahat, sebetahat, mult=mult)
        L = get_data_loglik_exp(betahat, sebetahat, scale)
        pi = optimize_pi_logL_torch(L, penalty=penalty, method=method, steps=steps, batch_size=batch_size, lr=lr)
        res: PosteriorResult = posterior_mean_exp(betahat, sebetahat, torch.log(pi + 1e-32), scale)
    else:
        raise ValueError("prior must be 'norm' or 'exp'")

    # Total log-likelihood at optimum pi
    L = torch.clamp(L, min=threshold_loglikelihood)
    L_max = L.max(dim=1, keepdim=True).values
    exp_term = torch.exp(L - L_max) * pi.view(1, -1)
    log_lik = (L_max + torch.log(exp_term.sum(dim=1, keepdim=True)+1e-32)).sum().item()

    return AshResult(
        post_mean=res.post_mean,
        post_mean2=res.post_mean2,
        post_sd=res.post_sd,
        scale=scale,
        pi=pi,
        prior=prior,
        log_lik=float(log_lik),
        mode=mode,
    )
