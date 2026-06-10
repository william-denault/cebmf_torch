# torch_only_point_exponential_stable.py
import math

import torch
from torch import Tensor

from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,
    logg_exp_convolved_with_normal,
    my_e2truncnorm,
    my_etruncnorm,
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
    m_tilt = xc - (s * s) * a
    zero = torch.zeros_like(xc)
    pinf = torch.full_like(xc, float("inf"))
    EZ = my_etruncnorm(zero, pinf, m_tilt, s)
    EZ2 = my_e2truncnorm(zero, pinf, m_tilt, s)
    return EZ, EZ2


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
    a_bounds=(1e-2, 1e2),  # bounded rate a
    loga_l2: float = 1e-3,  # ridge on a's unconstrained param (optimization only; 0 = off)
    tresh_pi0: float = 1e-3,  # legacy name; slab-weight threshold for spike-only shortcut
    eps: float = 1e-12,
) -> EBNMPointExp:
    """
    Direct maximization (no EM) of the observed marginal log-likelihood for a point-Exponential EBNM.

    Prior on θ: (1 - pi_slab) δ_μ + pi_slab [μ + Exp(a)], with support θ ≥ μ.
    Returns pure marginal log-likelihood (no penalties).
    """
    device, dtype = x.device, x.dtype
    x = torch.as_tensor(x, device=device, dtype=dtype)
    # Use ``clamp`` (out-of-place) — ``torch.as_tensor`` returns the same tensor
    # when ``s`` already has the matching dtype/device, so ``clamp_`` would
    # mutate the caller's tensor across repeated calls.
    s = torch.as_tensor(s, device=device, dtype=dtype).clamp(min=_const_like(x, 1e-6))

    # -------- init (keep same API but use a smooth bounded map for 'a') --------
    a_lo, a_hi = a_bounds
    a_lo_t = _const_like(x, a_lo)
    a_hi_t = _const_like(x, a_hi)

    if par_init is None:
        # alpha ~ logit(pi_slab), log_a ~ log(a), mu
        par_init = (0.9, 1.0, 0.0)  # pi_slab≈0.71, a≈e^1≈2.72, mu=0.0

    # Prepare a *logit* parameter for a in (a_lo, a_hi):  a = a_lo + (a_hi-a_lo) * sigmoid(v)
    a_init = float(min(max(math.exp(float(par_init[1])), a_lo), a_hi))
    # inverse-sigmoid safely
    r = (a_init - a_lo) / (a_hi - a_lo)
    r = min(max(r, 1e-8), 1 - 1e-8)
    v0 = math.log(r) - math.log(1 - r)

    alpha = torch.nn.Parameter(torch.as_tensor(par_init[0], dtype=dtype, device=device), requires_grad=not fix_par[0])  # noqa: E501
    a_logit = torch.nn.Parameter(torch.as_tensor(v0, dtype=dtype, device=device), requires_grad=not fix_par[1])  # noqa: E501
    mu = torch.nn.Parameter(torch.as_tensor(par_init[2], dtype=dtype, device=device), requires_grad=not fix_par[2])  # noqa: E501

    params = [p for p in (alpha, a_logit, mu) if p.requires_grad]
    opt = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn="strong_wolfe",
        history_size=20,
    )

    eps_t = _const_like(x, eps)

    def closure():
        opt.zero_grad(set_to_none=True)

        # Smooth transforms (no hard clamps on the objective)
        pi_slab = torch.sigmoid(alpha).clamp(eps_t, 1 - eps_t)  # (0,1)
        sig = torch.sigmoid(a_logit)
        a = a_lo_t + (a_hi_t - a_lo_t) * sig  # (a_lo, a_hi)
        xc = x - mu

        # log-likelihood pieces
        lf = _loglik_spike(xc, s)  # spike: N(xc|0,s^2)
        lg = _loglik_exp_convolved(xc, s, a)  # slab: Exp ⊗ Normal (Z≥0)

        # mixture log-likelihood per datum
        llik_i = torch.logaddexp(torch.log1p(-pi_slab) + lf, torch.log(pi_slab) + lg)
        llik_sum = llik_i.sum()

        # OPTIONAL tiny penalty on the unconstrained 'a_logit' to tame extremes (off by default)
        penalty = _const_like(x, 0.0)
        if loga_l2 != 0.0:
            penalty = penalty + _const_like(x, loga_l2) * (a_logit**2)

        loss = -(llik_sum - penalty)  # maximize llik_sum -> minimize negative

        # Strict: don't mask NaNs on the value; let failures surface. Gradients only from autograd.
        loss.backward()
        return loss

    if params:
        try:
            opt.step(closure)
        except RuntimeError:
            # Fallback: freeze 'a' if line search gets cranky; continue on remaining params
            if a_logit.requires_grad:
                a_logit.requires_grad_(False)
                params2 = [p for p in (alpha, mu) if p.requires_grad]
                if params2:
                    torch.optim.LBFGS(
                        params2,
                        max_iter=max_iter,
                        tolerance_grad=tol,
                        tolerance_change=tol,
                        line_search_fn="strong_wolfe",
                        history_size=20,
                    ).step(closure)

    # ===== Final posterior & summaries =====
    with torch.no_grad():
        pi_slab = torch.sigmoid(alpha).clamp(eps_t, 1 - eps_t)
        sig = torch.sigmoid(a_logit)
        a = a_lo_t + (a_hi_t - a_lo_t) * sig

        xc = x - mu
        lf = _loglik_spike(xc, s)
        lg = _loglik_exp_convolved(xc, s, a)

        log_num = torch.log(pi_slab) + lg
        log_den = torch.logaddexp(torch.log1p(-pi_slab) + lf, log_num)
        gamma = torch.exp(log_num - log_den).clamp(_const_like(x, 0.0), _const_like(x, 1.0))

        EZ, EZ2 = _posterior_moments_exp_branch(xc, s, a)
        post_mean_c = gamma * EZ
        post_mean2_c = torch.maximum(gamma * EZ2, (post_mean_c**2))

        post_mean = post_mean_c + mu
        post_mean2 = post_mean2_c + _const_like(x, 2.0) * mu * post_mean_c + mu * mu

        # pure observed marginal log-likelihood (no penalty)
        llik = torch.logaddexp(torch.log1p(-pi_slab) + lf, torch.log(pi_slab.clamp_min(eps_t)) + lg).sum()

        # Optional spike-only shortcut — branchless to avoid a per-call host sync.
        tresh_pi0_t = _const_like(x, tresh_pi0)
        spike_only = pi_slab < tresh_pi0_t  # 0-d bool tensor
        post_mean_so = torch.zeros_like(x) + mu
        post_mean2_so = torch.zeros_like(x) + mu * mu + _const_like(x, 1e-4)
        llik_so = lf.sum()
        post_mean = torch.where(spike_only, post_mean_so, post_mean)
        post_mean2 = torch.where(spike_only, post_mean2_so, post_mean2)
        post_sd = (post_mean2 - post_mean**2).clamp_min(_const_like(x, 0.0)).sqrt()
        llik = torch.where(spike_only, llik_so, llik)

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
