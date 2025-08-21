
import torch
from torch import Tensor
import math

_TWOPI = 2.0 * math.pi
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EPS = 1e-12

def log_norm_pdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    # log N(x | loc, scale)
    z = (x - loc) / (scale + 1e-32)
    return -0.5 * torch.log(torch.tensor(_TWOPI, device=x.device)) - torch.log(scale + 1e-32) - 0.5 * z * z

def norm_cdf(x: Tensor) -> Tensor:
    # standard normal CDF using torch.distributions
    return torch.distributions.Normal(0.0, 1.0).cdf(x)

def norm_pdf(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT_2PI

def logsumexp(x: Tensor, dim: int=-1, keepdim: bool=False) -> Tensor:
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)

def safe_log(x: Tensor, eps: float = _EPS) -> Tensor:
    return torch.log(torch.clamp(x, min=eps))

def softmax(x: Tensor, dim: int=-1) -> Tensor:
    return torch.softmax(x, dim=dim)

def truncated_normal_moments(a: Tensor, b: Tensor, mu: Tensor, sigma: Tensor):
    """
    Return E[X], E[X^2] for X ~ N(mu, sigma^2) truncated to [a, b].
    Vectorized over all tensors (broadcasting allowed).
    """
    device = mu.device
    alpha = (a - mu) / (sigma + 1e-32)
    beta  = (b - mu) / (sigma + 1e-32)

    Phi = norm_cdf
    phi = norm_pdf

    Phi_a = torch.clamp(Phi(alpha), 0.0, 1.0)
    Phi_b = torch.clamp(Phi(beta),  0.0, 1.0)
    Z = torch.clamp(Phi_b - Phi_a, min=1e-32)

    phi_a = phi(alpha)
    phi_b = phi(beta)

    EZ_std = (phi_a - phi_b) / Z
    # Var[Z] for std normal truncated [a,b]
    VarZ = 1.0 + (alpha * phi_a - beta * phi_b) / Z - EZ_std * EZ_std

    EX = mu + sigma * EZ_std
    EX2 = (sigma * sigma) * VarZ + EX * EX
    return EX, EX2
