import math
from dataclasses import dataclass

import torch
from torch import Tensor

from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,
    logPhi,
    my_e2truncnorm,
    my_etruncnorm,
    safe_log,
)

def _const_like(x: Tensor, val) -> Tensor:
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)

def logg_laplace_convolved_with_normal(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """
    log (Laplace(0, rate=a) ⊗ Normal(0, s^2)) at x.
    = log(a/2) + 0.5*(s a)^2 + log( Φ((x - s^2 a)/s) e^{-a x} + Φ(-(x + s^2 a)/s) e^{a x} )
    """
    x = torch.as_tensor(x, device=x.device, dtype=x.dtype)
    s = torch.as_tensor(s, device=x.device, dtype=x.dtype).clamp_(min=_const_like(x, 1e-12))
    a = torch.as_tensor(a, device=x.device, dtype=x.dtype)

    z1 = (x - (s * s) * a) / s
    z2 = -(x + (s * s) * a) / s

    lg1 = -a * x + logPhi(z1)
    lg2 =  a * x + logPhi(z2)
    lsum = torch.logaddexp(lg1, lg2)
    return safe_log(a / _const_like(x, 2.0)) + _const_like(x, 0.5) * (s * a) ** 2 + lsum

@dataclass
class EBNMLaplaceResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    pi0: float   # mixture weight of the Laplace branch (slab)
    a: float     # Laplace rate (1/scale)
    mu: float
    log_lik: float  # pure marginal log-likelihood (no penalties)

def ebnm_point_laplace(
    x: Tensor,
    s: Tensor,
    par_init=None,                     # None by default; choose safely inside
    fix_par=(False, False, True),      # [w_logit, a_logit, mu]; mu fixed at 0 by default
    max_iter: int = 20,
    tol: float = 1e-6,
    a_bounds=(1e-2, 1e2),              # bounds for Laplace rate a
    loga_l2: float = 1e-3,             # ridge on a's unconstrained logit (optimization only; 0=off)
    tresh_pi0: float = 1e-3,           # spike-only shortcut (post-processing only)
    eps: float = 1e-12,
    pen_pi0: float = 0.0,              # optional symmetric prior on pi0 (size-independent); 0=off
) -> EBNMLaplaceResult:
    """
    Direct maximization (no EM) of the observed marginal log-likelihood for a point-Laplace EBNM.
    Uses AdamW warm-start + short LBFGS polish for speed. Same signature & return type.
    """
    # ---- setup ----
    device, dtype = x.device, x.dtype
    x = torch.as_tensor(x, device=device, dtype=dtype)
    s = torch.as_tensor(s, device=device, dtype=dtype).clamp_min(_const_like(x, 1e-6))
    inv_s  = 1.0 / s
    inv_s2 = inv_s * inv_s

    a_lo, a_hi = a_bounds
    a_lo_t = _const_like(x, a_lo)
    a_hi_t = _const_like(x, a_hi)

    # ---- defaults ----
    if par_init is None:
        par_init = (2.0, 0.0, 0.0)  # (logit(w), log(a_init), mu)

    # Smooth bounded map for a: a = a_lo + (a_hi - a_lo) * sigmoid(v)
    a_init = float(min(max(math.exp(float(par_init[1])), a_lo), a_hi))
    r = (a_init - a_lo) / (a_hi - a_lo); r = min(max(r, 1e-8), 1 - 1e-8)
    v0 = math.log(r) - math.log(1 - r)

    w_logit = torch.nn.Parameter(torch.as_tensor(par_init[0], dtype=dtype, device=device), requires_grad=not fix_par[0])
    a_logit = torch.nn.Parameter(torch.as_tensor(v0,             dtype=dtype, device=device), requires_grad=not fix_par[1])
    mu      = torch.nn.Parameter(torch.as_tensor(par_init[2],    dtype=dtype, device=device), requires_grad=not fix_par[2])

    params = [p for p in (w_logit, a_logit, mu) if p.requires_grad]

    eps_t = _const_like(x, eps)
    c_norm = _LOG_SQRT_2PI if isinstance(_LOG_SQRT_2PI, torch.Tensor) else _const_like(s, _LOG_SQRT_2PI)

    def loss_and_stats():
        # transforms
        w   = torch.sigmoid(w_logit).clamp(eps_t, 1 - eps_t)
        sig = torch.sigmoid(a_logit)
        a   = a_lo_t + (a_hi_t - a_lo_t) * sig  # smooth (a_lo, a_hi)
        xc  = x - mu

        # spike: log N(xc | 0, s^2) using cached inv_s/ inv_s2
        lf = -_const_like(x, 0.5) * (xc * inv_s)**2 - torch.log(s) - c_norm

        # slab: Laplace ⊗ Normal (fused helper already stable)
        lg = logg_laplace_convolved_with_normal(xc, s, a)

        # mixture log-likelihood per datum
        llik_i   = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg)
        llik_sum = llik_i.sum()

        # OPTIONAL tiny penalties (off by default)
        penalty = _const_like(x, 0.0)
        if loga_l2 != 0.0:
            penalty = penalty + _const_like(x, loga_l2) * (a_logit**2)
        if pen_pi0 != 0.0:
            penalty = penalty - _const_like(x, pen_pi0) * (torch.log(w) + torch.log1p(-w))

        loss = -(llik_sum - penalty)
        return loss, llik_sum, w, a, xc, lf, lg

    # ---- fast phase: AdamW warm-start ----
    if params:
        adam_steps = max(60, 6 * max_iter)  # short but effective
        opt_adam = torch.optim.AdamW(params, lr=0.05, betas=(0.9, 0.999), weight_decay=0.0)
        for _ in range(adam_steps):
            opt_adam.zero_grad(set_to_none=True)
            loss, _, _, _, _, _, _ = loss_and_stats()
            loss.backward()
            opt_adam.step()

    # ---- polish: short LBFGS (few closure evals) ----
    if params:
        opt_lbfgs = torch.optim.LBFGS(
            params,
            max_iter=max_iter,              # keep user control here
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
            history_size=10,                # smaller history = fewer matvecs
        )
        def closure():
            opt_lbfgs.zero_grad(set_to_none=True)
            loss, _, _, _, _, _, _ = loss_and_stats()
            loss.backward()
            return loss
        try:
            opt_lbfgs.step(closure)
        except RuntimeError:
            pass  # fall through with AdamW solution

    # ---- posterior & summaries (no penalties) ----
    with torch.no_grad():
        _, _, pi0, a, xc, lf, lg = loss_and_stats()

        # posterior inclusion prob for slab
        log_num   = torch.log(pi0) + lg
        log_denom = torch.logaddexp(torch.log1p(-pi0) + lf, log_num)
        gamma     = torch.exp(log_num - log_denom).clamp(_const_like(x, 0.0), _const_like(x, 1.0))

        # sign mixture within slab
        z1 = (xc - (s * s) * a) / s
        z2 = -(xc + (s * s) * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 =  a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lam  = torch.exp(lg1 - lsum)
        lam  = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5))

        # truncated-normal moments
        m_pos = xc - s * s * a
        m_neg = xc + s * s * a
        infp = torch.full_like(x, float("inf")); infn = -infp

        EX_pos  = my_etruncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EX2_pos = my_e2truncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EX_neg  = my_etruncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)
        EX2_neg = my_e2truncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)

        EX  = lam * EX_pos  + (1 - lam) * EX_neg
        EX2 = lam * EX2_pos + (1 - lam) * EX2_neg

        post_mean  = gamma * (EX + mu) + (1 - gamma) * mu
        post_mean2 = gamma * (EX2 + _const_like(x, 2.0) * mu * EX + mu * mu) + (1 - gamma) * (mu * mu)
        post_sd    = (post_mean2 - post_mean**2).clamp_min(_const_like(x, 0.0)).sqrt()

        # PURE marginal log-likelihood
        llik = torch.logaddexp(torch.log1p(-pi0) + lf, torch.log(pi0.clamp_min(eps_t)) + lg).sum()

        if float(pi0.item()) < tresh_pi0:
            post_mean  = torch.zeros_like(x) + mu
            post_mean2 = torch.zeros_like(x) + mu * mu + _const_like(x, 1e-4)
            post_sd    = (post_mean2 - post_mean**2).clamp_min(_const_like(x, 0.0)).sqrt()
            llik       = lf.sum()
            # (leave pi0 tiny or set to 0.0 if you prefer)

    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi0=float(pi0.item()),
        a=float(a.item()),
        mu=float(mu.item()),
        log_lik=float(llik.item()),
    )
