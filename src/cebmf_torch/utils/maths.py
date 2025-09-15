import math

import torch
from torch import Tensor

_TWOPI = 2.0 * math.pi
_SQRT_2PI = math.sqrt(_TWOPI)
_EPS = 1e-12
_LOG_2PI = math.log(_TWOPI)
_LOG_SQRT_2PI = 0.5 * _LOG_2PI


def log_norm_pdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    # log N(x | loc, scale)
    z = (x - loc) / (scale + 1e-32)
    return (
        -0.5 * torch.log(torch.tensor(_TWOPI, device=x.device))
        - torch.log(scale + 1e-32)
        - 0.5 * z * z
    )


def norm_cdf(x: Tensor) -> Tensor:
    # standard normal CDF using torch.distributions
    return torch.distributions.Normal(0.0, 1.0).cdf(x)


def norm_pdf(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT_2PI


def logsumexp(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)


def safe_log(x: Tensor, eps: float = _EPS) -> Tensor:
    return torch.log(torch.clamp(x, min=eps))


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return torch.softmax(x, dim=dim)


# ------------------------
# helpers
# ------------------------
def logphi(z: torch.Tensor) -> torch.Tensor:
    """log pdf φ(z) = exp(-z^2/2)/√(2π)."""
    return -0.5 * z.pow(2) - _LOG_SQRT_2PI


def logPhi(z: torch.Tensor) -> torch.Tensor:
    """Stable log Φ(z)."""
    return torch.special.log_ndtr(z)


def logscale_sub(logx: torch.Tensor, logy: torch.Tensor) -> torch.Tensor:
    """
    Compute log(exp(logx) - exp(logy)) stably.
    Requires logx >= logy.
    """
    max_log = torch.maximum(logx, logy)
    return max_log + torch.log(torch.exp(logx - max_log) - torch.exp(logy - max_log))


def logscale_add(logx: Tensor, logy: Tensor) -> Tensor:
    """Stable log(exp(logx) + exp(logy))."""
    return torch.logaddexp(logx, logy)


def do_truncnorm_argchecks(a: torch.Tensor, b: torch.Tensor):
    """Clamp and sanity check bounds."""
    # If a >= b, invalid; we just return as-is
    return a, b


# ------------------------
# E[Z | a<Z<b]  and E[Z^2 | a<Z<b] for Z~N(mean, sd^2)
# ------------------------


def my_etruncnorm(a, b, mean=0.0, sd=1.0):
    a, b = do_truncnorm_argchecks(torch.as_tensor(a), torch.as_tensor(b))
    mean = torch.as_tensor(mean, dtype=torch.float64)
    sd = torch.as_tensor(sd, dtype=torch.float64)

    alpha = (a - mean) / sd
    beta = (b - mean) / sd

    flip = ((alpha > 0) & (beta > 0)) | (beta > alpha.abs())
    orig_alpha = alpha.clone()
    alpha = torch.where(flip, -beta, alpha)
    beta = torch.where(flip, -orig_alpha, beta)

    dnorm_diff = logscale_sub(logphi(beta), logphi(alpha))
    pnorm_diff = logscale_sub(logPhi(beta), logPhi(alpha))

    scaled_res = -torch.exp(torch.clamp(dnorm_diff - pnorm_diff, max=700.0))

    # endpoints equal
    endpts_equal = torch.isinf(pnorm_diff)
    scaled_res = torch.where(endpts_equal, (alpha + beta) / 2, scaled_res)

    lower_bd = torch.maximum(beta + 1.0 / beta, (alpha + beta) / 2)
    bad_idx = (
        (~torch.isnan(beta))
        & (beta < 0)
        & ((scaled_res < lower_bd) | (scaled_res > beta))
    )
    scaled_res = torch.where(bad_idx, lower_bd, scaled_res)

    scaled_res = torch.where(flip, -scaled_res, scaled_res)

    res = mean + sd * scaled_res

    if (sd == 0).any():
        a_rep = a.expand_as(res)
        b_rep = b.expand_as(res)
        mean_rep = mean.expand_as(res)
        sd_zero = sd == 0

        cond1 = sd_zero & (b_rep <= mean_rep)
        cond2 = sd_zero & (a_rep >= mean_rep)
        cond3 = sd_zero & (a_rep < mean_rep) & (b_rep > mean_rep)

        res = torch.where(cond1, b_rep, res)
        res = torch.where(cond2, a_rep, res)
        res = torch.where(cond3, mean_rep, res)

    return res


def my_e2truncnorm(a, b, mean=0.0, sd=1.0):
    a, b = do_truncnorm_argchecks(torch.as_tensor(a), torch.as_tensor(b))
    mean = torch.as_tensor(mean, dtype=torch.float64)
    sd = torch.as_tensor(sd, dtype=torch.float64)

    alpha = (a - mean) / sd
    beta = (b - mean) / sd

    flip = (alpha > 0) & (beta > 0)
    orig_alpha = alpha.clone()
    alpha = torch.where(flip, -beta, alpha)
    beta = torch.where(flip, -orig_alpha, beta)

    # absolute mean handling
    if not torch.all(mean == 0):
        mean = mean.abs()

    pnorm_diff = logscale_sub(logPhi(beta), logPhi(alpha))

    alpha_frac = alpha * torch.exp(torch.clamp(logphi(alpha) - pnorm_diff, max=300.0))
    beta_frac = beta * torch.exp(torch.clamp(logphi(beta) - pnorm_diff, max=300.0))

    # handle nan/inf
    alpha_frac = torch.where(
        ~torch.isfinite(alpha_frac), torch.zeros_like(alpha_frac), alpha_frac
    )
    beta_frac = torch.where(
        ~torch.isfinite(beta_frac), torch.zeros_like(beta_frac), beta_frac
    )

    scaled_res = torch.ones_like(alpha)

    alpha_idx = torch.isfinite(alpha)
    scaled_res = torch.where(alpha_idx, 1 + alpha_frac, scaled_res)
    beta_idx = torch.isfinite(beta)
    scaled_res = torch.where(beta_idx, scaled_res - beta_frac, scaled_res)

    endpts_equal = torch.isinf(pnorm_diff)
    scaled_res = torch.where(endpts_equal, ((alpha + beta) ** 2) / 4, scaled_res)

    upper_bd1 = beta**2 + 2 * (1 + 1 / beta**2)
    upper_bd2 = (alpha**2 + alpha * beta + beta**2) / 3
    upper_bd = torch.minimum(upper_bd1, upper_bd2)

    bad_idx = (
        (~torch.isnan(beta))
        & (beta < 0)
        & ((scaled_res < beta**2) | (scaled_res > upper_bd))
    )
    scaled_res = torch.where(bad_idx, upper_bd, scaled_res)

    res = mean**2 + 2 * mean * sd * my_etruncnorm(alpha, beta) + sd**2 * scaled_res

    if (sd == 0).any():
        a_rep = a.expand_as(res)
        b_rep = b.expand_as(res)
        mean_rep = mean.expand_as(res)
        sd_zero = sd == 0

        cond1 = sd_zero & (b_rep <= mean_rep)
        cond2 = sd_zero & (a_rep >= mean_rep)
        cond3 = sd_zero & (a_rep < mean_rep) & (b_rep > mean_rep)

        res = torch.where(cond1, b_rep**2, res)
        res = torch.where(cond2, a_rep**2, res)
        res = torch.where(cond3, mean_rep**2, res)

    return res
