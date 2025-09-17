import math

import torch
from torch import Tensor

_TWOPI = 2.0 * math.pi
_SQRT_2PI = math.sqrt(_TWOPI)
_EPS = 1e-12
_LOG_2PI = math.log(_TWOPI)
_LOG_SQRT_2PI = 0.5 * _LOG_2PI


def log_norm_pdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """
    Compute the log-density of a normal distribution.

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
    z = (x - loc) / (scale + 1e-32)
    return -0.5 * torch.log(torch.tensor(_TWOPI, device=x.device)) - torch.log(scale + 1e-32) - 0.5 * z * z


def norm_cdf(x: Tensor) -> Tensor:
    """
    Compute the standard normal cumulative distribution function (CDF).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        CDF evaluated at x.
    """
    return torch.distributions.Normal(0.0, 1.0).cdf(x)


def norm_pdf(x: Tensor) -> Tensor:
    """
    Compute the standard normal probability density function (PDF).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        PDF evaluated at x.
    """
    return torch.exp(-0.5 * x * x) / _SQRT_2PI


def logsumexp(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """
    Compute the log of the sum of exponentials of input elements along a given dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int, optional
        Dimension along which to operate. Default is -1.
    keepdim : bool, optional
        Whether the output tensor has dim retained or not. Default is False.

    Returns
    -------
    torch.Tensor
        Result of log-sum-exp operation.
    """
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)


def safe_log(x: Tensor, eps: float = _EPS) -> Tensor:
    """
    Compute the logarithm of x with clamping for numerical stability.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    eps : float, optional
        Minimum value to clamp x to. Default is 1e-12.

    Returns
    -------
    torch.Tensor
        Logarithm of clamped x.
    """
    return torch.log(torch.clamp(x, min=eps))


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute the softmax of input tensor along the specified dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int, optional
        Dimension along which softmax will be computed. Default is -1.

    Returns
    -------
    torch.Tensor
        Softmax of the input tensor.
    """
    return torch.softmax(x, dim=dim)


# ------------------------
# helpers
# ------------------------
def logphi(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the log of the standard normal PDF φ(z) = exp(-z^2/2)/√(2π).

    Parameters
    ----------
    z : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Log PDF evaluated at z.
    """
    return -0.5 * z.pow(2) - _LOG_SQRT_2PI


def logPhi(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the stable log CDF of the standard normal distribution Φ(z).

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


def logscale_sub(logx: torch.Tensor, logy: torch.Tensor) -> torch.Tensor:
    """
    Compute log(exp(logx) - exp(logy)) in a numerically stable way.

    Requires logx >= logy.

    Parameters
    ----------
    logx : torch.Tensor
        Logarithm of x.
    logy : torch.Tensor
        Logarithm of y.

    Returns
    -------
    torch.Tensor
        Logarithm of (exp(logx) - exp(logy)).
    """
    max_log = torch.maximum(logx, logy)
    return max_log + torch.log(torch.exp(logx - max_log) - torch.exp(logy - max_log))


def logscale_add(logx: Tensor, logy: Tensor) -> Tensor:
    """
    Compute log(exp(logx) + exp(logy)) in a numerically stable way.

    Parameters
    ----------
    logx : torch.Tensor
        Logarithm of x.
    logy : torch.Tensor
        Logarithm of y.

    Returns
    -------
    torch.Tensor
        Logarithm of (exp(logx) + exp(logy)).
    """
    return torch.logaddexp(logx, logy)


def do_truncnorm_argchecks(a: torch.Tensor, b: torch.Tensor):
    """
    Clamp and sanity check bounds for truncated normal arguments.

    Parameters
    ----------
    a : torch.Tensor
        Lower bound(s).
    b : torch.Tensor
        Upper bound(s).

    Returns
    -------
    tuple of torch.Tensor
        (a, b) after checks.
    """
    # If a >= b, invalid; we just return as-is
    return a, b


def safe_tensor_to_float(
    value: torch.Tensor | float | None, null_value: float = float("-inf"), reduction: str = "min"
) -> float:
    """
    Convert tensor, float, or None to float with safe handling.

    Parameters
    ----------
    value : torch.Tensor, float, or None
        Value to convert.
    null_value : float, optional
        Value to return if input is None or empty. Default is -inf.
    reduction : str, optional
        Reduction to apply if input is a tensor ("min" or "max"). Default is "min".

    Returns
    -------
    float
        Converted float value.
    """
    if value is None:
        return null_value
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return null_value
        if reduction == "min":
            return float(value.min().item())
        elif reduction == "max":
            return float(value.max().item())
        # ... other reductions
    return float(value)


# ------------------------
# E[Z | a<Z<b]  and E[Z^2 | a<Z<b] for Z~N(mean, sd^2)
# ------------------------


def my_etruncnorm(a, b, mean=0.0, sd=1.0):
    """
    Compute E[Z | a < Z < b] for Z ~ N(mean, sd^2), the mean of a truncated normal.

    Parameters
    ----------
    a : float or torch.Tensor
        Lower truncation bound.
    b : float or torch.Tensor
        Upper truncation bound.
    mean : float or torch.Tensor, optional
        Mean of the normal distribution. Default is 0.0.
    sd : float or torch.Tensor, optional
        Standard deviation of the normal distribution. Default is 1.0.

    Returns
    -------
    torch.Tensor
        Mean of the truncated normal distribution.
    """
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
    bad_idx = (~torch.isnan(beta)) & (beta < 0) & ((scaled_res < lower_bd) | (scaled_res > beta))
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
    """
    Compute E[Z^2 | a < Z < b] for Z ~ N(mean, sd^2), the second moment of a truncated normal.

    Parameters
    ----------
    a : float or torch.Tensor
        Lower truncation bound.
    b : float or torch.Tensor
        Upper truncation bound.
    mean : float or torch.Tensor, optional
        Mean of the normal distribution. Default is 0.0.
    sd : float or torch.Tensor, optional
        Standard deviation of the normal distribution. Default is 1.0.

    Returns
    -------
    torch.Tensor
        Second moment of the truncated normal distribution.
    """
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
    alpha_frac = torch.where(~torch.isfinite(alpha_frac), torch.zeros_like(alpha_frac), alpha_frac)
    beta_frac = torch.where(~torch.isfinite(beta_frac), torch.zeros_like(beta_frac), beta_frac)

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

    bad_idx = (~torch.isnan(beta)) & (beta < 0) & ((scaled_res < beta**2) | (scaled_res > upper_bd))
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
