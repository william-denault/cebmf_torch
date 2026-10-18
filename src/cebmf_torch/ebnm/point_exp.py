# torch_only_point_exponential_stable.py
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,
    logg_exp_convolved_with_normal,
    posterior_moments_exp,
)


def _const_like(x: Tensor, val) -> Tensor:
    """Create a scalar tensor `val` on x's device/dtype."""
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)


# =========================
# Core pieces
# =========================


def _loglik_spike(xc: Tensor, s: Tensor) -> Tensor:
    # log N(xc | 0, s^2)
    c = _LOG_SQRT_2PI if isinstance(_LOG_SQRT_2PI, torch.Tensor) else _const_like(s, _LOG_SQRT_2PI)
    return -_const_like(s, 0.5) * (xc / s) ** 2 - torch.log(s) - c


def _loglik_exp_convolved(xc: Tensor, s: Tensor, a: Tensor) -> Tensor:
    # lg = log a + (s a)^2 / 2 - a * xc + log Φ(xc/s - s a), θ_c ≥ 0
    return logg_exp_convolved_with_normal(xc, s, a)


def _posterior_moments_exp_branch(xc: Tensor, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute the first and second moments for the Exp branch using the tilted Normal.

    The random variable Z ~ N(m_tilt, s^2) truncated to [0, +inf),
    where m_tilt = xc - s^2 * a.

    Parameters
    ----------
    xc : torch.Tensor
        Centered data (x - mu).
    s : torch.Tensor
        Standard deviation of the observation noise.
    a : torch.Tensor
        Exponential rate parameter.

    Returns
    -------
    EZ : torch.Tensor
        First moment E[Z].
    EZ2 : torch.Tensor
        Second moment E[Z^2].
    """
    return posterior_moments_exp(xc, s, a)


# =========================
# Public EBNM interface
# =========================


class EBNMPointExp:
    def __init__(
        self,
        post_mean: Tensor,
        post_mean2: Tensor,
        post_sd: Tensor,
        scale: Tensor,
        pi_slab: Tensor,
        log_lik: Tensor,
        mode: Tensor,
    ):
        """
        Container for the results of the point-exponential EBNM posterior estimation.

        All scalar fields are kept as 0-d tensors on the same device as the
        input data so cEBMF can fold them into the ELBO without forcing a
        host sync per factor update. Use ``float(field)`` if you need a
        Python scalar.

        Parameters
        ----------
        post_mean : torch.Tensor
            Posterior means for each observation.
        post_mean2 : torch.Tensor
            Posterior second moments for each observation.
        post_sd : torch.Tensor
            Posterior standard deviations for each observation.
        scale : torch.Tensor (0-d)
            Estimated exponential rate parameter (a).
        pi_slab : torch.Tensor (0-d)
            Estimated mixture weight for the Exp (slab) branch. Equal to
            ``1 - pi_null``.
        log_lik : torch.Tensor (0-d)
            Final marginal log-likelihood value.
        mode : torch.Tensor (0-d)
            Estimated mode (mu).
        """
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.scale = scale
        self.pi_slab = pi_slab
        self.log_lik = log_lik
        self.mode = mode


def ebnm_point_exp(
    x: Tensor,
    s: Tensor,
    par_init=None,  # (alpha, log_a, mu). If None, choose safely inside.
    fix_par=(
        False,
        False,
        True,
    ),  # [w_logit, log_a, mu]; default keeps mu fixed like your Laplace
    max_iter: int = 20,
    tol: float = 1e-6,
    a_bounds=(1e-2, 1e2),  # bounds for the estimated rate a; fixed rates are used unchanged
    loga_l2: float = 0.0,  # optional ridge on a's unconstrained param; default is maximum likelihood
    tresh_pi0: float = 1e-3,  # retained for API compatibility; the boundary uses likelihoods
    eps: float = 1e-12,  # retained for API compatibility; log weights no longer need clamping
) -> EBNMPointExp:
    """
    Direct maximization (no EM) of the observed marginal log-likelihood for a point-Exponential EBNM.

    Prior on θ: (1 - pi_slab) δ_μ + pi_slab [μ + Exp(a)], with support θ ≥ μ.
    Returns pure marginal log-likelihood (no penalties).

    As in R ebnm, infinite standard errors contribute no information to
    fitting and return prior moments. Fixed parameters are respected when
    checking the all-spike boundary. ``scale`` in the result is the rate a;
    R's exponential scale is 1/a. Set ``loga_l2`` explicitly to request a ridge
    penalty; the default fit is unpenalized.
    """
    device, dtype = x.device, x.dtype
    x = torch.as_tensor(x, device=device, dtype=dtype)
    # Use ``clamp`` (out-of-place) — ``torch.as_tensor`` returns the same tensor
    # when ``s`` already has the matching dtype/device, so ``clamp_`` would
    # mutate the caller's tensor across repeated calls.
    s = torch.as_tensor(s, device=device, dtype=dtype)
    if not torch.isfinite(x).all() or torch.isnan(s).any() or (s < 0).any():
        raise ValueError("x must be finite and s must be nonnegative (positive infinity is allowed).")
    x, s = torch.broadcast_tensors(x, s.clamp(min=_const_like(x, 1e-6)))
    uninformative = torch.isinf(s)
    s = torch.where(uninformative, torch.ones_like(s), s)

    # -------- init (keep same API but use a smooth bounded map for 'a') --------
    a_lo, a_hi = a_bounds
    a_lo_t = _const_like(x, a_lo)
    a_hi_t = _const_like(x, a_hi)

    if par_init is None:
        # alpha ~ logit(pi_slab), log_a ~ log(a), mu
        par_init = (0.9, 1.0, 0.0)  # pi_slab≈0.71, a≈e^1≈2.72, mu=0.0

    # Prepare a *logit* parameter for a in (a_lo, a_hi):  a = a_lo + (a_hi-a_lo) * sigmoid(v)
    initial_rate = math.exp(float(par_init[1]))
    fixed_rate = _const_like(x, initial_rate) if fix_par[1] else None
    a_init = float(min(max(initial_rate, a_lo), a_hi))
    # inverse-sigmoid safely
    r = (a_init - a_lo) / (a_hi - a_lo)
    r = min(max(r, 1e-8), 1 - 1e-8)
    v0 = math.log(r) - math.log(1 - r)

    alpha = torch.nn.Parameter(torch.as_tensor(par_init[0], dtype=dtype, device=device), requires_grad=not fix_par[0])  # noqa: E501
    a_logit = torch.nn.Parameter(torch.as_tensor(v0, dtype=dtype, device=device), requires_grad=not fix_par[1])  # noqa: E501
    mu = torch.nn.Parameter(torch.as_tensor(par_init[2], dtype=dtype, device=device), requires_grad=not fix_par[2])  # noqa: E501

    params = [p for p in (alpha, a_logit, mu) if p.requires_grad]

    def closure():
        opt.zero_grad(set_to_none=True)

        # Smooth transforms (no hard clamps on the objective)
        log_slab, log_spike = F.logsigmoid(alpha), F.logsigmoid(-alpha)
        sig = torch.sigmoid(a_logit)
        a = fixed_rate if fixed_rate is not None else a_lo_t + (a_hi_t - a_lo_t) * sig
        xc = torch.where(uninformative, torch.zeros_like(x), x - mu)

        # log-likelihood pieces
        lf = _loglik_spike(xc, s)  # spike: N(xc|0,s^2)
        lg = _loglik_exp_convolved(xc, s, a)  # slab: Exp ⊗ Normal (Z≥0)

        # mixture log-likelihood per datum
        llik_i = torch.logaddexp(log_spike + lf, log_slab + lg)
        llik_sum = torch.where(uninformative, torch.zeros_like(llik_i), llik_i).sum()

        # OPTIONAL tiny penalty on the unconstrained 'a_logit' to tame extremes (off by default)
        penalty = _const_like(x, 0.0)
        if loga_l2 != 0.0:
            penalty = penalty + _const_like(x, loga_l2) * (a_logit**2)

        loss = -(llik_sum - penalty)  # maximize llik_sum -> minimize negative

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite point-exponential objective.")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError("Non-finite point-exponential gradient.")
        return loss

    if params and max_iter > 0:
        opt = torch.optim.LBFGS(
            params,
            max_iter=max_iter,
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
            history_size=20,
        )
        opt.step(closure)

    # ===== Final posterior & summaries =====
    with torch.no_grad():
        pi_slab = torch.sigmoid(alpha)
        log_slab, log_spike = F.logsigmoid(alpha), F.logsigmoid(-alpha)
        sig = torch.sigmoid(a_logit)
        a = fixed_rate if fixed_rate is not None else a_lo_t + (a_hi_t - a_lo_t) * sig

        xc = torch.where(uninformative, torch.zeros_like(x), x - mu)
        lf = _loglik_spike(xc, s)
        lg = _loglik_exp_convolved(xc, s, a)

        log_num = log_slab + lg
        log_den = torch.logaddexp(log_spike + lf, log_num)
        gamma = torch.exp(log_num - log_den).clamp(_const_like(x, 0.0), _const_like(x, 1.0))

        EZ, EZ2 = _posterior_moments_exp_branch(xc, s, a)
        post_mean_c = gamma * EZ
        post_mean2_c = torch.maximum(gamma * EZ2, (post_mean_c**2))
        post_mean_c = torch.where(uninformative, pi_slab / a, post_mean_c)
        post_mean2_c = torch.where(uninformative, 2 * pi_slab / a.square(), post_mean2_c)

        variance = (post_mean2_c - post_mean_c.square()).clamp_min(0)
        post_mean = post_mean_c + mu
        post_mean2 = variance + post_mean.square()

        # pure observed marginal log-likelihood (no penalty)
        llik = torch.where(uninformative, torch.zeros_like(log_den), log_den).sum()

        # Spike-only fallback values. Matches Stephens-lab `ebnm` exactly: the
        # posterior under pi_slab = 0 is a point mass at mu, so E[θ] = mu and
        # E[θ²] = mu² (no +1e-4 fudge — that was an unprincipled offset that
        # biased `tau` in cEBMF).
        post_mean_so = torch.zeros_like(x) + mu
        post_mean2_so = torch.zeros_like(x) + mu * mu
        llik_so = torch.where(uninformative, torch.zeros_like(lf), lf).sum()

        # Boundary check, ported from R's `pe_postcomp`: if the all-spike model
        # (pi_slab = 0) has a strictly higher marginal log-likelihood than the
        # LBFGS solution, prefer the all-spike solution. This is the principled
        # replacement for the old `pi_slab < tresh_pi0` shortcut. As in R,
        # changing the mixture weight is permitted only when it is estimated
        # and the location is fixed. Prediction with fixed parameters must
        # retain the requested prior even when another prior fits better.
        if not torch.isfinite(llik) or not torch.isfinite(post_mean).all() or not torch.isfinite(post_mean2).all():
            raise FloatingPointError("Non-finite point-exponential posterior or likelihood.")
        prefer_spike = (llik_so > llik) & (not fix_par[0] and fix_par[2])
        post_mean = torch.where(prefer_spike, post_mean_so, post_mean)
        post_mean2 = torch.where(prefer_spike, post_mean2_so, post_mean2)
        llik = torch.where(prefer_spike, llik_so, llik)
        # Set pi_slab to 0 when we picked the spike branch so downstream
        # cEBMF factor pruning / pi0_null reporting sees the right thing.
        pi_slab = torch.where(prefer_spike, torch.zeros_like(pi_slab), pi_slab)
        post_sd = torch.where(prefer_spike, torch.zeros_like(variance), variance).sqrt()

    # Keep scalar fields on-device as 0-d tensors — `cebmf.py` consumes them
    # via `priors/point.py` which now stays in tensor space.
    return EBNMPointExp(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        scale=a.detach(),  # 'a' is the *rate*; field name kept as 'scale' for compatibility
        pi_slab=pi_slab.detach(),
        log_lik=llik.detach(),
        mode=mu.detach(),
    )
