# torch_convolved_loglik.py

import torch

from .maths import _LOG_SQRT_2PI


def _const_like(c, ref: torch.Tensor) -> torch.Tensor:
    """
    Ensure a scalar/0-d tensor constant `c` matches the dtype/device of `ref`.
    """
    if isinstance(c, torch.Tensor):
        return c.to(device=ref.device, dtype=ref.dtype)
    # assume Python scalar
    return torch.tensor(c, device=ref.device, dtype=ref.dtype)


# ===== numerically-stable primitives =====


def _logpdf_normal(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-density of a normal distribution in a numerically stable way.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    loc : torch.Tensor
        Mean of the normal distribution.
    scale : torch.Tensor
        Standard deviation of the normal distribution.

    Returns
    -------
    torch.Tensor
        Log-density evaluated at x.
    """
    z = (x - loc) / scale
    # make sure constant is on the right device/dtype
    log_sqrt_2pi = _const_like(_LOG_SQRT_2PI, scale)
    return -0.5 * z.pow(2) - torch.log(scale) - log_sqrt_2pi


def _logcdf_normal(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the log CDF of the standard normal distribution in a numerically stable way.

    Parameters
    ----------
    z : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Log CDF evaluated at z.
    """
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
    Compute the log-likelihood matrix for a normal prior convolved with normal noise.

    Both ``location`` and ``scale`` may be either shared across observations
    (shape ``(K,)``) or per-observation (shape ``(J, K)``). This makes the
    routine usable both for classical fixed-grid ASH and for neural-net-based
    priors (CASH / EMDN / spiked-EMDN) where the per-observation mixture is
    different.

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates, shape ``(J,)``.
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates, shape ``(J,)``.
    location : torch.Tensor
        Prior means, shape ``(K,)`` or ``(J, K)``.
    scale : torch.Tensor
        Prior standard deviations, shape ``(K,)`` or ``(J, K)``.
    clamp : float, optional
        Clamp log-likelihood values to ``[-clamp, clamp]`` for numerical
        stability (default: ``1e5``).

    Returns
    -------
    torch.Tensor
        Log-likelihood matrix of shape ``(J, K)``.
    """
    betahat = torch.as_tensor(betahat)
    dt, dev = betahat.dtype, betahat.device
    sebetahat = torch.as_tensor(sebetahat, dtype=dt, device=dev)
    scale = torch.as_tensor(scale, dtype=dt, device=dev)
    location = torch.as_tensor(location, dtype=dt, device=dev)

    J = betahat.shape[0]

    # Resolve K from scale (1D or 2D), and shape sc2 to (J, K) or (1, K).
    if scale.ndim == 1:
        K = scale.shape[0]
        sc2 = scale.pow(2).unsqueeze(0)  # (1, K), broadcasts across J
    elif scale.ndim == 2:
        if scale.shape[0] != J:
            raise ValueError(
                f"scale 2D first dim must be J={J}, got shape {tuple(scale.shape)}"
            )
        K = scale.shape[1]
        sc2 = scale.pow(2)  # (J, K)
    else:
        raise ValueError(f"scale must be 1D or 2D, got ndim={scale.ndim}")

    # Broadcast sd_jk = sqrt(se_j^2 + scale_k^2)
    se2 = sebetahat.pow(2).unsqueeze(1)  # (J, 1)
    sd = torch.sqrt(se2 + sc2)  # (J, K)

    if location.ndim == 1:
        if location.shape[0] != K:
            raise ValueError(
                f"location 1D length must be K={K}, got shape {tuple(location.shape)}"
            )
        loc = location.unsqueeze(0).expand(J, K)
    elif location.ndim == 2:
        if location.shape != (J, K):
            raise ValueError(
                f"location 2D shape must be (J={J}, K={K}), got {tuple(location.shape)}"
            )
        loc = location
    else:
        raise ValueError(f"location must be 1D or 2D, got ndim={location.ndim}")

    x = betahat.unsqueeze(1).expand(J, K)  # (J, K)
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
    Compute the log-likelihood matrix for an exponential prior convolved with normal noise.

    Convention:
        - scale[0] corresponds to the point-mass-at-zero component (unused in rate calc)
        - scale[1:] are exponential scales; rate = 1/scale[1:].

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates, shape (J,).
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates, shape (J,).
    scale : torch.Tensor
        Prior scales, shape (K,) with K >= 2, scale[0]=0 recommended.

    Returns
    -------
    torch.Tensor
        Log-likelihood matrix L of shape (J, K), where:
            L[:,0] = N(betahat | 0, se^2)
            L[:,1:] use the analytic Exp⊗Normal convolution.
    """
    betahat = torch.as_tensor(betahat)
    sebetahat = torch.as_tensor(sebetahat, dtype=betahat.dtype, device=betahat.device)
    scale = torch.as_tensor(scale, dtype=betahat.dtype, device=betahat.device)

    J = betahat.shape[0]
    K = scale.shape[0]
    if K < 2:
        raise ValueError("scale must have length >= 2 (scale[0] for spike, scale[1:] for Exp)")

    # k=0: spike-at-0 convolved with Normal -> just Normal(0, s^2)
    out0 = _logpdf_normal(betahat, torch.zeros_like(betahat), sebetahat)  # (J,)

    # k>=1: rates and broadcasted formula
    rate = 1.0 / scale[1:]  # (K-1,)
    s = sebetahat.unsqueeze(1)  # (J,1)
    x = betahat.unsqueeze(1)  # (J,1)
    a = rate.unsqueeze(0)  # (1,K-1)

    lg = torch.log(a) + 0.5 * (s * a).pow(2) - a * x + _logcdf_normal(x / s - s * a)  # (J,K-1)

    # Concatenate first column
    L = torch.empty((J, K), dtype=betahat.dtype, device=betahat.device)
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
    Compute the data log-likelihood matrix for a normal prior (batched wrapper).

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates, shape (J,).
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates, shape (J,).
    location : torch.Tensor
        Prior means, shape (K,) or (J, K).
    scale : torch.Tensor
        Prior standard deviations, shape (K,).

    Returns
    -------
    torch.Tensor
        Log-likelihood matrix of shape (J, K).
    """
    return convolved_logpdf_normal_torch(betahat, sebetahat, location, scale)


@torch.no_grad()
def get_data_loglik_exp_torch(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the data log-likelihood matrix for an exponential prior (batched wrapper).

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates, shape (J,).
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates, shape (J,).
    scale : torch.Tensor
        Prior scales, shape (K,) with K >= 2, scale[0]=0 recommended.

    Returns
    -------
    torch.Tensor
        Log-likelihood matrix of shape (J, K).
    """
    return convolved_logpdf_exp_torch(betahat, sebetahat, scale)
