# torch_convolved_loglik.py
import torch
import math
from .torch_utils_mix import autoselect_scales_mix_exp, autoselect_scales_mix_norm
from .torch_mix_opt import optimize_pi_logL
from .torch_utils import _LOG_SQRT_2PI 
from .torch_posterior import  posterior_mean_norm,  posterior_mean_exp
from cebmf_torch.torch_distribution_operation import get_data_loglik_normal_torch, get_data_loglik_exp_torch

import math
import torch 
from typing import Optional
 
 

class ash_object:
    def __init__(
        self,
        post_mean,
        post_mean2,
        post_sd,
        scale,
        pi0,
        prior,
        log_lik: float = 0.0,
        mode: float = 0.0, 
    ):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.scale = scale
        self.pi0 = pi0
        self.prior = prior
        self.log_lik = log_lik
        self.mode = mode 

# ---- ASH (Torch) ----
@torch.no_grad()
def ash(
    x ,
    s,
    prior: str = "norm",
    mult: float = math.sqrt(2.0),
    penalty: float = 10.0,
    verbose: bool = True,
    threshold_loglikelihood: float = -300.0,
    mode: float = 0.0,
    *, 
    batch_size: Optional[int] = 128,
    shuffle: bool = False,
    seed: Optional[int] = None,
):
    """
    Adaptive shrinkage with mixture priors ("norm" or "exp") in pure PyTorch.
    Uses EM for π (mini-batch capable via batch_size).
    Returns ash_object with Torch tensors.
    """

  
    # choose optimizer mode (EM by default here)
 
    s.clamp_(min=1e-12)
    if prior == "norm":
        scale = autoselect_scales_mix_norm(x, s, mult=mult)  # (K,)
        loc = torch.full((scale.shape[0],), float(mode), dtype=x.dtype, device=x.device)

        L = get_data_loglik_normal_torch(x, s, location=loc, scale=scale)  # (J,K)
        pi0 = optimize_pi_logL (
            L, penalty=penalty, verbose=verbose,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        log_pi0 = torch.log(torch.clamp(pi0, min=1e-32))
        pm_obj = posterior_mean_norm (x, s, log_pi=log_pi0,data_loglik=L, location=loc, scale=scale)
        pm = pm_obj.post_mean
        pm2 = pm_obj.post_mean2
        psd = pm_obj.post_sd

    elif prior == "exp":
        scale = autoselect_scales_mix_exp (x, s, mult=mult)   # (K,) with scale[0]=0 (spike)
        L = get_data_loglik_exp_torch(x, s, scale=scale)           # (J,K)
        pi0 = optimize_pi_logL (
            L, penalty=penalty, verbose=verbose,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        log_pi0 = torch.log(torch.clamp(pi0, min=1e-32))
        pm_obj = posterior_mean_exp (x, s, log_pi=log_pi0, scale=scale ) 
        pm = pm_obj.post_mean
        pm2 = pm_obj.post_mean2
        psd = pm_obj.post_sd


    else:
        raise ValueError("prior must be either 'norm' or 'exp'.")

    # total data log-likelihood: sum_j log sum_k pi_k * exp(L_{jk})
    Lc = torch.maximum(L, torch.tensor(threshold_loglikelihood, dtype=L.dtype, device=L.device))
    log_lik_rows = torch.logsumexp(Lc + torch.log(torch.clamp(pi0, min=1e-300)).unsqueeze(0), dim=1)
    log_lik = float(log_lik_rows.sum().item())

    return ash_object(
        post_mean=pm, post_mean2=pm2, post_sd=psd,
        scale=scale, pi0=pi0[0], prior=prior,
        log_lik=log_lik, mode=float(mode) 
    )
