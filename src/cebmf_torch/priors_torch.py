# priors_torch.py
import torch
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

from cebmf_torch.torch_ash import ash
from cebmf_torch.torch_ebnm_point_laplace import ebnm_point_laplace
from cebmf_torch.torch_ebnm_point_exp import ebnm_point_exp

@dataclass
class PriorResultTorch:
    post_mean: torch.Tensor
    post_mean2: torch.Tensor
    loss: float
    model_param: Optional[Any] = None
    # NEW: optional pruning signals
    pi0_null: Optional[torch.Tensor | float] = None  # P(spike)
    pi_slab:  Optional[torch.Tensor | float] = None  # 1 - pi0_null

def prior_norm_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    obj = ash(betahat, sebetahat, prior="norm")
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss=-float(obj.log_lik), model_param=model_param,
                            
                            pi0_null=obj.pi0)

def prior_exp_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    obj = ash(betahat, sebetahat, prior="exp")
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss= -float(obj.log_lik), model_param=model_param,
                            pi0_null=obj.pi0)

 

def prior_point_laplace_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    obj = ebnm_point_laplace(betahat, sebetahat)
    # obj.w = slab weight in your implementation => pi_slab
  
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss=-float(obj.log_lik), model_param=model_param,
                            pi0_null=float(obj.pi0))

def prior_point_exp_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    obj = ebnm_point_exp(betahat, sebetahat) 
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss= -float(obj.log_lik), model_param=model_param,
                            pi0_null=float(obj.pi0))

PRIOR_REGISTRY: Dict[str, Callable] = {
    "norm": prior_norm_torch,
    "exp": prior_exp_torch,
    "point_laplace": prior_point_laplace_torch,
    "point_exp": prior_point_exp_torch,
}

def get_prior_function_torch(key_or_fn) -> Callable:
    if isinstance(key_or_fn, str):
        if key_or_fn not in PRIOR_REGISTRY:
            raise ValueError(f"Unknown prior '{key_or_fn}'. Available: {list(PRIOR_REGISTRY.keys())}")
        return PRIOR_REGISTRY[key_or_fn]
    if callable(key_or_fn):
        return key_or_fn
    raise ValueError("prior_L/prior_F must be a string key or a callable.")
