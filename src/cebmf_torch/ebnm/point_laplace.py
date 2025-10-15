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
    # Assume x, s, a already on correct device/dtype; s already clamped outside
    z1 = (x - (s * s) * a) / s
    z2 = -(x + (s * s) * a) / s
    lg1 = -a * x + logPhi(z1)
    lg2 =  a * x + logPhi(z2)
    lsum = torch.logaddexp(lg1, lg2)
    half = torch.tensor(0.5, device=x.device, dtype=x.dtype)
    two  = torch.tensor(2.0, device=x.device, dtype=x.dtype)
    return safe_log(a / two) + half * (s * a) ** 2 + lsum

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
    a_bounds=(1e-1, 1e2),              # bounds for Laplace rate a
    loga_l2: float = 1e-5,             # ridge on a's unconstrained logit (optimization only; 0=off)
    tresh_pi0: float = 1e-3,           # spike-only shortcut (post-processing only)
    eps: float = 1e-12,
    pen_pi0: float = 0.0,              # optional symmetric prior on pi0 (size-independent); 0=off
    use_adam_warmstart: bool = False,  # off by default; set True to use a short warm-up
    adam_steps: int = 8,
    adam_lr: float = 1e-2,
) -> EBNMLaplaceResult:
    """
    Efficient direct maximization of the observed marginal log-likelihood for a point-Laplace EBNM.
    Optimizer: LBFGS only (AdamW warm-start optional and short). GPU-safe (no host<->device hops).
    """
    # ---- setup & hoisted constants (device/dtype safe) ----
    device, dtype = x.device, x.dtype
    x = torch.as_tensor(x, device=device, dtype=dtype)
    s = torch.as_tensor(s, device=device, dtype=dtype).clamp_min(_const_like(x, 1e-6))

    # Precompute once; reused in closure & posterior
    inv_s  = 1.0 / s
    inv_s2 = inv_s * inv_s
    log_s  = torch.log(s)
    s2     = s * s

    # scalar constants as tensors on the right device/dtype
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    one  = torch.tensor(1.0, device=device, dtype=dtype)
    half = torch.tensor(0.5, device=device, dtype=dtype)
    two  = torch.tensor(2.0, device=device, dtype=dtype)
    eps_t = torch.tensor(eps, device=device, dtype=dtype)
    thresh_pi0_t = torch.tensor(tresh_pi0, device=device, dtype=dtype)

    # normalizing constant (reuse tensor if given, else make one)
    c_norm = _LOG_SQRT_2PI if isinstance(_LOG_SQRT_2PI, torch.Tensor) else _const_like(s, _LOG_SQRT_2PI)

    # bounds for 'a' via smooth map a = a_lo + (a_hi - a_lo) * sigmoid(v)
    a_lo, a_hi = a_bounds
    a_lo_t = torch.tensor(a_lo, device=device, dtype=dtype)
    a_hi_t = torch.tensor(a_hi, device=device, dtype=dtype)
    a_span = a_hi_t - a_lo_t

    # ---- defaults & parameter tensors ----
    if par_init is None:
        par_init = (2.0, 0.0, 0.0)  # (logit(w), log(a_init), mu)

    # map provided log(a_init) into v0 for the bounded-sigmoid parameterization
    a_init = float(min(max(math.exp(float(par_init[1])), a_lo), a_hi))
    r = (a_init - a_lo) / (a_hi - a_lo)
    r = min(max(r, 1e-8), 1 - 1e-8)
    v0 = math.log(r) - math.log(1 - r)

    w_logit = torch.nn.Parameter(torch.as_tensor(par_init[0], dtype=dtype, device=device), requires_grad=not fix_par[0])
    a_logit = torch.nn.Parameter(torch.as_tensor(v0,             dtype=dtype, device=device), requires_grad=not fix_par[1])
    mu      = torch.nn.Parameter(torch.as_tensor(par_init[2],    dtype=dtype, device=device), requires_grad=not fix_par[2])

    params = [p for p in (w_logit, a_logit, mu) if p.requires_grad]

    # ---- tiny optional warm-start (kept short) ----
    if use_adam_warmstart and params:
        opt_adam = torch.optim.AdamW(params, lr=adam_lr, betas=(0.9, 0.999), weight_decay=0.0)
        for _ in range(int(adam_steps)):
            opt_adam.zero_grad(set_to_none=True)
            # lean forward: only loss for speed
            w   = torch.sigmoid(w_logit).clamp(eps_t, one - eps_t)
            sig = torch.sigmoid(a_logit)
            a   = a_lo_t + a_span * sig
            xc  = x - mu

            # spike log-lik
            lf = -(half * (xc * inv_s) ** 2) - log_s - c_norm

            # slab log-lik (fused helper)
            lg = logg_laplace_convolved_with_normal(xc, s, a)

            llik = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg).sum()

            penalty = zero
            if loga_l2 != 0.0:
                penalty = penalty + torch.tensor(loga_l2, device=device, dtype=dtype) * (a_logit**2)
            if pen_pi0 != 0.0:
                penalty = penalty - torch.tensor(pen_pi0, device=device, dtype=dtype) * (
                    torch.log(w) + torch.log1p(-w)
                )
            loss = -(llik - penalty)
            loss.backward()
            opt_adam.step()

    # ---- LBFGS polish (closure computes ONLY the scalar loss) ----
    if params:
        opt_lbfgs = torch.optim.LBFGS(
            params,
            max_iter=max_iter,
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
            history_size=10,
        )

        def closure():
            opt_lbfgs.zero_grad(set_to_none=True)
            w   = torch.sigmoid(w_logit).clamp(eps_t, one - eps_t)
            sig = torch.sigmoid(a_logit)
            a   = a_lo_t + a_span * sig
            xc  = x - mu

            # spike log-lik (reuses inv_s, log_s, c_norm)
            lf = -(half * (xc * inv_s) ** 2) - log_s - c_norm

            # slab log-lik (reuses s2 via s and fused helper)
            lg = logg_laplace_convolved_with_normal(xc, s, a)

            llik = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg).sum()

            # penalties as tensors on-device
            penalty = zero
            if loga_l2 != 0.0:
                penalty = penalty + torch.tensor(loga_l2, device=device, dtype=dtype) * (a_logit**2)
            if pen_pi0 != 0.0:
                penalty = penalty - torch.tensor(pen_pi0, device=device, dtype=dtype) * (
                    torch.log(w) + torch.log1p(-w)
                )

            loss = -(llik - penalty)
            # guard for NaN/Inf without host-device traffic
            huge = torch.tensor(1e30, device=device, dtype=dtype)
            loss = torch.nan_to_num(loss, nan=huge, posinf=huge, neginf=huge)
            loss.backward()
            return loss

        try:
            opt_lbfgs.step(closure)
        except RuntimeError:
            # fallback: freeze 'a' if line search fails; keep everything on-device
            if a_logit.requires_grad:
                a_logit.requires_grad_(False)
                params2 = [p for p in (w_logit, mu) if p.requires_grad]
                if params2:
                    torch.optim.LBFGS(
                        params2,
                        max_iter=max_iter,
                        tolerance_grad=tol,
                        tolerance_change=tol,
                        line_search_fn="strong_wolfe",
                        history_size=10,
                    ).step(closure)

    # ---- posterior & summaries (no penalties; single no_grad block) ----
    with torch.no_grad():
        w   = torch.sigmoid(w_logit).clamp(eps_t, one - eps_t)
        sig = torch.sigmoid(a_logit)
        a   = a_lo_t + a_span * sig
        xc  = x - mu

        # spike
        lf = -(half * (xc * inv_s) ** 2) - log_s - c_norm

        # slab (reuse helper)
        lg = logg_laplace_convolved_with_normal(xc, s, a)

        # posterior inclusion prob (slab)
        log_num   = torch.log(w) + lg
        log_denom = torch.logaddexp(torch.log1p(-w) + lf, log_num)
        gamma     = torch.exp(log_num - log_denom).clamp(zero, one)

        # sign-mixture inside slab
        z1 = (xc - s2 * a) / s
        z2 = -(xc + s2 * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 =  a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lam  = torch.exp(lg1 - lsum)
        lam  = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5))

        # truncated-normal moments
        m_pos = xc - s2 * a
        m_neg = xc + s2 * a
        infp  = torch.full_like(x, float("inf"))
        infn  = -infp

        EX_pos  = my_etruncnorm(zero, infp, mean=m_pos, sd=s)
        EX2_pos = my_e2truncnorm(zero, infp, mean=m_pos, sd=s)
        EX_neg  = my_etruncnorm(infn, zero, mean=m_neg, sd=s)
        EX2_neg = my_e2truncnorm(infn, zero, mean=m_neg, sd=s)

        EX  = lam * EX_pos  + (one - lam) * EX_neg
        EX2 = lam * EX2_pos + (one - lam) * EX2_neg

        post_mean  = gamma * (EX + mu) + (one - gamma) * mu
        post_mean2 = gamma * (EX2 + two * mu * EX + mu * mu) + (one - gamma) * (mu * mu)
        post_sd    = (post_mean2 - post_mean**2).clamp_min(zero).sqrt()

        # PURE marginal log-likelihood
        llik = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w.clamp_min(eps_t)) + lg).sum()

        # spike-only shortcut if pi0 (slab weight) tiny
        if float(w.item()) < float(thresh_pi0_t.item()):
            post_mean  = torch.zeros_like(x) + mu
            post_mean2 = torch.zeros_like(x) + mu * mu + torch.tensor(1e-4, device=device, dtype=dtype)
            post_sd    = (post_mean2 - post_mean**2).clamp_min(zero).sqrt()
            llik       = lf.sum()
            # (leave w as-is, or set to 0 if you prefer)

    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi0=float(w.item()),     # mixture weight of the Laplace branch (slab)
        a=float(a.item()),
        mu=float(mu.item()),
        log_lik=float(llik.item()),
    )
