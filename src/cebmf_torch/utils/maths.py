import math

import torch
from torch import Tensor

_TWOPI = 2.0 * math.pi
_SQRT_2PI = math.sqrt(_TWOPI)
_EPS = 1e-12
_LOG_2PI = math.log(_TWOPI)
_LOG_SQRT_2PI = 0.5 * _LOG_2PI


def _like(x: Tensor, val) -> Tensor:
    """Create a scalar tensor `val` on x's device/dtype."""
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)


def _logpdf_normal(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """
    Numerically-clean log-density of a normal distribution.

    log N(x | loc, scale^2) = -0.5*((x-loc)/scale)^2 - log(scale) - 0.5*log(2*pi)

    Callers are expected to ensure ``scale > 0`` (typically by clamping standard
    errors). This is the canonical implementation used across the package; the
    module-level aliases in ``utils/posterior.py`` and ``utils/distribution_operation.py``
    re-export this one to avoid drifting copies.
    """
    z = (x - loc) / scale
    return -0.5 * z.pow(2) - torch.log(scale) - _LOG_SQRT_2PI


def log_norm_pdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """Backward-compatible alias for :func:`_logpdf_normal` with epsilon-padded scale.

    Adds a tiny epsilon to ``scale`` before the log/division to tolerate degenerate
    inputs (e.g., ``scale==0``). Prefer :func:`_logpdf_normal` and clamp ``scale``
    upstream when the caller guarantees positivity.
    """
    eps = _like(scale, 1e-32)
    safe_scale = scale + eps
    return _logpdf_normal(x, loc, safe_scale)


def _logcdf_normal(z: Tensor) -> Tensor:
    """Numerically-stable log Φ(z) (standard normal CDF). Wraps torch.special.log_ndtr."""
    return torch.special.log_ndtr(z)


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
    return torch.distributions.Normal(_like(x, 0.0), _like(x, 1.0)).cdf(x)


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
    return torch.exp(-0.5 * x * x) / _like(x, _SQRT_2PI)


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
    return torch.log(torch.clamp(x, min=_like(x, eps)))


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
    return -0.5 * z.pow(2) - _like(z, _LOG_SQRT_2PI)


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
    # Ensure a and b share device/dtype
    a = torch.as_tensor(a)
    b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
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
    return float(value)


# ------------------------
# E[Z | a<Z<b]  and E[Z^2 | a<Z<b] for Z~N(mean, sd^2)
# ------------------------


def _resolve_truncnorm_dtype(precision: str, *tensors: torch.Tensor) -> torch.dtype:
    """Pick the internal compute dtype for the truncnorm helpers.

    ``precision`` is one of:

    - ``"auto"`` (default): use the highest-precision floating dtype among the
      provided tensors, defaulting to ``float32``. This keeps everything on the
      caller's device/dtype, which on CUDA consumer cards is ~30x faster than
      float64 with no observable accuracy loss for typical EBNM workloads.
    - ``"float64"``: force ``torch.float64``. Slower on CUDA but matches the
      historical behaviour of these helpers exactly. Use this when you have
      extreme bounds (``|alpha|, |beta| > 10``) that benefit from the extra
      precision.
    """
    if precision == "float64":
        return torch.float64
    if precision != "auto":
        raise ValueError(f"precision must be 'auto' or 'float64', got {precision!r}")
    # Pick the highest-precision floating dtype from inputs, default float32.
    best = torch.float32
    rank = {torch.float16: 0, torch.bfloat16: 0, torch.float32: 1, torch.float64: 2}
    for t in tensors:
        if t.dtype.is_floating_point and rank.get(t.dtype, 1) > rank.get(best, 1):
            best = t.dtype
    return best


def my_etruncnorm(a, b, mean=0.0, sd=1.0, precision: str = "auto"):
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
    precision : str, optional
        Internal compute dtype. ``"auto"`` (default) follows the input dtype
        (typically ``float32``), which is fast on CUDA. ``"float64"`` forces
        double precision — slower on CUDA but matches the pre-2026 behaviour.

    Returns
    -------
    torch.Tensor
        Mean of the truncated normal distribution. Dtype matches the chosen
        ``precision``.
    """
    a, b = do_truncnorm_argchecks(torch.as_tensor(a), torch.as_tensor(b))
    device = a.device
    mean_t = torch.as_tensor(mean)
    sd_t = torch.as_tensor(sd)
    work_dtype = _resolve_truncnorm_dtype(precision, a, b, mean_t, sd_t)

    mean = mean_t.to(dtype=work_dtype, device=device)
    sd = sd_t.to(dtype=work_dtype, device=device)

    alpha = (a.to(dtype=work_dtype, device=device) - mean) / sd
    beta = (b.to(dtype=work_dtype, device=device) - mean) / sd

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

    # Branchless degenerate-sd handling. Previous code wrapped these where()
    # calls in `if (sd == 0).any():`, which forced a host sync on every call.
    # The where-chain is a no-op when no entries have sd=0, so we just always
    # run it (a tiny constant cost on healthy inputs, vs. a real GPU stall).
    res = _apply_degenerate_sd_first_moment(res, a, b, mean, sd)

    return res


def _apply_degenerate_sd_first_moment(
    res: torch.Tensor, a: torch.Tensor, b: torch.Tensor, mean: torch.Tensor, sd: torch.Tensor
) -> torch.Tensor:
    """Replace entries of `res` corresponding to sd==0 with the limiting first moment."""
    a_rep = a.expand_as(res).to(res)
    b_rep = b.expand_as(res).to(res)
    mean_rep = mean.expand_as(res)
    sd_zero = sd == 0
    cond1 = sd_zero & (b_rep <= mean_rep)
    cond2 = sd_zero & (a_rep >= mean_rep)
    cond3 = sd_zero & (a_rep < mean_rep) & (b_rep > mean_rep)
    res = torch.where(cond1, b_rep, res)
    res = torch.where(cond2, a_rep, res)
    res = torch.where(cond3, mean_rep, res)
    return res


def my_e2truncnorm(a, b, mean=0.0, sd=1.0, precision: str = "auto"):
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
    precision : str, optional
        Internal compute dtype. See :func:`my_etruncnorm` for the contract.
        ``"auto"`` (default) keeps the caller's dtype (fast on CUDA);
        ``"float64"`` matches the pre-2026 behaviour.

    Returns
    -------
    torch.Tensor
        Second moment of the truncated normal distribution. Dtype matches the
        chosen ``precision``.
    """
    a, b = do_truncnorm_argchecks(torch.as_tensor(a), torch.as_tensor(b))
    device = a.device
    mean_t = torch.as_tensor(mean)
    sd_t = torch.as_tensor(sd)
    work_dtype = _resolve_truncnorm_dtype(precision, a, b, mean_t, sd_t)

    mean = mean_t.to(dtype=work_dtype, device=device)
    sd = sd_t.to(dtype=work_dtype, device=device)

    alpha = (a.to(dtype=work_dtype, device=device) - mean) / sd
    beta = (b.to(dtype=work_dtype, device=device) - mean) / sd

    flip = (alpha > 0) & (beta > 0)
    orig_alpha = alpha.clone()
    alpha = torch.where(flip, -beta, alpha)
    beta = torch.where(flip, -orig_alpha, beta)

    # Absolute mean handling. `mean.abs()` is a no-op for mean==0, so we don't
    # need the previous `if not torch.all(mean == 0): mean = mean.abs()` guard
    # — that guard fired a host sync on every call.
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

    # NOTE: my_etruncnorm expects (a,b,mean,sd). For standardized alpha/beta, use mean=0, sd=1.
    # Forward the same ``precision`` so the inner call doesn't silently re-upcast back to float64.
    res = mean**2 + 2 * mean * sd * my_etruncnorm(alpha, beta, 0.0, 1.0, precision=precision) + sd**2 * scaled_res

    # Branchless degenerate-sd handling — see _apply_degenerate_sd_first_moment
    # for the rationale (no host sync per call).
    a_rep = a.expand_as(res).to(res)
    b_rep = b.expand_as(res).to(res)
    mean_rep = mean.expand_as(res)
    sd_zero = sd == 0
    cond1 = sd_zero & (b_rep <= mean_rep)
    cond2 = sd_zero & (a_rep >= mean_rep)
    cond3 = sd_zero & (a_rep < mean_rep) & (b_rep > mean_rep)
    res = torch.where(cond1, b_rep**2, res)
    res = torch.where(cond2, a_rep**2, res)
    res = torch.where(cond3, mean_rep**2, res)

    return res
