
import torch
from torch import Tensor
from .torch_utils import log_norm_pdf, norm_cdf, safe_log

def get_data_loglik_normal(betahat: Tensor, sebetahat: Tensor, location: Tensor, scale: Tensor) -> Tensor:
    """
    betahat: (n,)
    sebetahat: (n,)
    location: (K,) or (n,K)
    scale: (K,)
    Returns: (n,K) log-likelihood matrix
    """
    n = betahat.shape[0]
    K = scale.shape[0]
    if location.dim() == 1:
        loc = location.view(1, K).expand(n, K)
    else:
        loc = location
    se = sebetahat.view(n, 1).expand(n, K)
    sc = scale.view(1, K).expand(n, K)
    sd = torch.sqrt(se * se + sc * sc)
    x = betahat.view(n, 1).expand(n, K)
    return log_norm_pdf(x, loc, sd).clamp(min=-1e5, max=1e5)

def get_data_loglik_exp(betahat: Tensor, sebetahat: Tensor, scale: Tensor) -> Tensor:
    """
    Exponential (positive) + point-mass at 0 prior convolved with Normal likelihood.
    scale[0] is assumed to be 0 for the spike-at-zero component.
    Returns (n, K) log-likelihood matrix.
    """
    n = betahat.shape[0]
    K = scale.shape[0]
    device = betahat.device

    out = torch.empty((n, K), device=device)
    # spike at zero: Normal(0, se)
    out[:, 0] = log_norm_pdf(betahat, torch.zeros_like(betahat), sebetahat)
    if K == 1:
        return out

    # for k>=1: rate a = 1/scale_k
    rate = 1.0 / scale[1:].view(1, K-1)  # (1, K-1)
    s = sebetahat.view(n,1)              # (n,1)
    x = betahat.view(n,1)                # (n,1)

    # lg = log a + s^2 a^2 / 2 - a x + log CDF( x/s - s a )
    lg = torch.log(rate) + 0.5 * (s*s) * (rate*rate) - rate * x + torch.log(torch.clamp(norm_cdf(x/s - s*rate), min=1e-32))
    out[:, 1:] = lg
    return out
