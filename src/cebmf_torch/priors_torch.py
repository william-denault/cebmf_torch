# priors_torch.py
import torch
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

# Import your existing solvers/utilities
from cebmf_torch.torch_ash import ash                    # your PyTorch ash(prior="norm"/"exp")
from cebmf_torch.torch_ebnm_point_laplace  import ebnm_point_laplace  # the stable version we wrote earlier

from cebmf_torch.torch_ebnm_point_exp  import ebnm_point_exp   # the stable version we wrote earlier
@dataclass
class PriorResultTorch:
    post_mean: torch.Tensor
    post_mean2: torch.Tensor
    loss: float                      # we store "loss" so it mirrors your NumPy PriorResult
    model_param: Optional[Any] = None

# ---- standard prior wrappers (all same signature) ----
def prior_norm_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    obj = ash(betahat, sebetahat, prior="norm")
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss=-float(obj.log_lik), model_param=model_param)

def prior_exp_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    obj = ash(betahat, sebetahat, prior="exp")
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss=-float(obj.log_lik), model_param=model_param)

def prior_point_laplace_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    # Uses the stable EBNM point-Laplace we wrote in PyTorch earlier
    obj = ebnm_point_laplace(betahat, sebetahat, par_init=[0.0, 2.0, 0.0] )
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss=-float(obj.log_lik), model_param=model_param)

def prior_point_exp_torch(X, betahat: torch.Tensor, sebetahat: torch.Tensor, model_param=None) -> PriorResultTorch:
    # Uses the stable EBNM point-Laplace we wrote in PyTorch earlier
    obj = ebnm_point_exp(betahat, sebetahat)
    return PriorResultTorch(post_mean=obj.post_mean, post_mean2=obj.post_mean2,
                            loss=-float(obj.log_lik), model_param=model_param)

# You can add more (EMDN, CGB, etc.) later with the same signature:
# def prior_emdn_torch(X, betahat, sebetahat, model_param=None) -> PriorResultTorch: ...
# def prior_cgb_torch(X, betahat, sebetahat, model_param=None) -> PriorResultTorch: ...

PRIOR_REGISTRY: Dict[str, Callable] = {
    "norm": prior_norm_torch,
    "exp": prior_exp_torch,
    "point_laplace": prior_point_laplace_torch,
    "point_exp": prior_point_exp_torch,
}

def get_prior_function_torch(key_or_fn) -> Callable:
    """
    If a string, look up in registry. If a callable, return as-is.
    """
    if isinstance(key_or_fn, str):
        if key_or_fn not in PRIOR_REGISTRY:
            raise ValueError(f"Unknown prior '{key_or_fn}'. "
                             f"Available: {list(PRIOR_REGISTRY.keys())}")
        return PRIOR_REGISTRY[key_or_fn]
    if callable(key_or_fn):
        return key_or_fn
    raise ValueError("prior_L/prior_F must be a string key or a callable(X, betahat, sebetahat, model_param).")
