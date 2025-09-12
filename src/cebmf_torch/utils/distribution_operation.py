# torch_convolved_loglik.py

import torch

from .maths import _LOG_SQRT_2PI

# ===== numerically-stable primitives =====


def _logpdf_normal(
    x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    z = (x - loc) / scale
    return -0.5 * z.pow(2) - torch.log(scale) - _LOG_SQRT_2PI


def _logcdf_normal(z: torch.Tensor) -> torch.Tensor:
    # stable log Φ(z)
    return torch.special.log_ndtr(z)


# ===== convolved log-pdfs =====


@torch.no_grad()
def convolved_logpdf_normal_torch(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    clamp: float = 1e5,
) -> torch.Tensor:
    """
    Vectorized version of:
        sd = sqrt(se^2 + scale^2)
        logp = N(betahat | location, sd^2)
    Shapes:
        betahat:   (J,)
        sebetahat: (J,)
        location:  (K,)  or (J,K)
        scale:     (K,)
    Returns:
        (J, K) tensor of log-likelihoods.
    """
    betahat = torch.as_tensor(betahat)
    sebetahat = torch.as_tensor(sebetahat)
    scale = torch.as_tensor(scale)
    location = torch.as_tensor(location)

    J = betahat.shape[0]
    K = scale.shape[0]

    # Broadcast sd_jk = sqrt(se_j^2 + scale_k^2)
    se2 = sebetahat.pow(2).unsqueeze(1)  # (J,1)
    sc2 = scale.pow(2).unsqueeze(0)  # (1,K)
    sd = torch.sqrt(se2 + sc2)  # (J,K)

    if location.ndim == 1:
        loc = location.unsqueeze(0).expand(J, K)  # (J,K)
    elif location.ndim == 2:
        assert location.shape == (J, K), "location must be (K,) or (J,K)"
        loc = location
    else:
        raise ValueError("location must be (K,) or (J,K)")

    x = betahat.unsqueeze(1).expand(J, K)  # (J,K)
    logp = _logpdf_normal(x, loc, sd)
    logp = torch.clamp(logp, min=-clamp, max=clamp)
    return logp


@torch.no_grad()
def convolved_logpdf_exp_torch(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized version of your convolved_logpdf_exp.
    Convention:
      - scale[0] corresponds to the point-mass-at-zero component (unused in rate calc)
      - scale[1:] are exponential scales; rate = 1/scale[1:].

    Shapes:
      betahat:   (J,)
      sebetahat: (J,)
      scale:     (K,)  with K>=2, scale[0]=0 recommended.

    Returns:
      L: (J, K) log-likelihood matrix where:
         L[:,0] = N(betahat | 0, se^2)
         L[:,1:] use the analytic Exp⊗Normal convolution:
            log a + 0.5*(s a)^2 - a*x + log Φ(x/s - s a)
         with a = rate = 1/scale[1:].
    """
    betahat = torch.as_tensor(betahat)
    sebetahat = torch.as_tensor(sebetahat)
    scale = torch.as_tensor(scale)

    J = betahat.shape[0]
    K = scale.shape[0]
    if K < 2:
        raise ValueError(
            "scale must have length >= 2 (scale[0] for spike, scale[1:] for Exp)"
        )

    # k=0: spike-at-0 convolved with Normal -> just Normal(0, s^2)
    out0 = _logpdf_normal(betahat, torch.zeros_like(betahat), sebetahat)  # (J,)

    # k>=1: rates and broadcasted formula
    rate = 1.0 / scale[1:]  # (K-1,)
    s = sebetahat.unsqueeze(1)  # (J,1)
    x = betahat.unsqueeze(1)  # (J,1)
    a = rate.unsqueeze(0)  # (1,K-1)

    lg = (
        torch.log(a) + 0.5 * (s * a).pow(2) - a * x + _logcdf_normal(x / s - s * a)
    )  # (J,K-1)

    # Concatenate first column
    L = torch.empty((J, K), dtype=torch.get_default_dtype(), device=betahat.device)
    L[:, 0] = out0
    L[:, 1:] = lg
    return L


# ===== batched wrappers (API parity) =====


@torch.no_grad()
def get_data_loglik_normal_torch(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Torch equivalent of get_data_loglik_normal (returns (J,K) log-lik matrix).
    Accepts location as (K,) or (J,K).
    """
    return convolved_logpdf_normal_torch(betahat, sebetahat, location, scale)


@torch.no_grad()
def get_data_loglik_exp_torch(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Torch equivalent of get_data_loglik_exp (returns (J,K) log-lik matrix).
    """
    return convolved_logpdf_exp_torch(betahat, sebetahat, scale)
