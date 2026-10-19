import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,
    logg_exp_convolved_with_normal,
    posterior_moments_exp,
)


def _const_like(x: Tensor, val) -> Tensor:
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)


def _laplace_slab_terms(x: Tensor, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Shared terms for the Laplace ⊗ Normal slab marginal.

    Returns ``(lg, lg1, lsum)`` where ``lg`` is the slab log-density and
    ``lg1``/``lsum`` are the positive-branch log-term and the log-sum of the
    two sign branches. The posterior computation reuses ``lg1``/``lsum`` to
    form the within-slab sign-mixture weight ``lam = exp(lg1 - lsum)``.

    Assumes x, s, a already on the correct device/dtype; s already clamped.
    """
    # A Laplace slab is an equal mixture of positive and negative
    # exponentials. Use their stable convolutions: the direct Laplace
    # formula subtracts terms of size (s*a)^2 when noise dominates.
    lg1 = logg_exp_convolved_with_normal(x, s, a)
    lg2 = logg_exp_convolved_with_normal(-x, s, a)
    lsum = torch.logaddexp(lg1, lg2)
    lg = lsum - math.log(2.0)
    return lg, lg1, lsum


def logg_laplace_convolved_with_normal(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """
    log (Laplace(0, rate=a) ⊗ Normal(0, s^2)) at x.
    = log(a/2) + 0.5*(s a)^2 + log( Φ((x - s^2 a)/s) e^{-a x} + Φ(-(x + s^2 a)/s) e^{a x} )
    """
    lg, _, _ = _laplace_slab_terms(x, s, a)
    return lg


@dataclass
class EBNMLaplaceResult:
    """All scalar fields are 0-d tensors on the input device (no host syncs)."""

    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    pi_slab: Tensor  # mixture weight of the Laplace branch (slab); = 1 - pi_null
    a: Tensor  # Laplace rate (1/scale)
    mu: Tensor
    log_lik: Tensor  # pure marginal log-likelihood (no penalties)
    regularization_loss: Tensor | float = 0.0


def ebnm_point_laplace(
    x: Tensor,
    s: Tensor,
    par_init=None,  # None by default; choose safely inside
    fix_par=(False, False, True),  # [w_logit, a_logit, mu]; mu fixed at 0 by default
    max_iter: int = 20,
    tol: float = 1e-6,
    a_bounds=(1e-2, 1e2),  # bounds for Laplace rate a
    loga_l2: float = 0.0,  # ridge on a's unconstrained logit (optimization only; 0=off)
    tresh_pi0: float = 1e-3,  # retained for compatibility; the spike boundary uses likelihoods
    eps: float = 1e-12,  # retained for compatibility; log weights do not need clamping
    pen_pi0: float = 0.0,  # legacy name; optional symmetric prior on slab-weight parameter w
    use_adam_warmstart: bool = False,  # default OFF for speed; set True to enable short warm-up
    adam_steps: int = 8,
    adam_lr: float = 1e-2,
    weight_decay: float = 0.0,
) -> EBNMLaplaceResult:
    """
    Efficient direct maximisation of the observed marginal log-likelihood for a
    point-Laplace EBNM. GPU-resident throughout — no host<->device syncs in
    either the inner objective or the posterior summary.

    Optimizer: LBFGS-only (AdamW warm-start optional and short). The four
    scalar fields of the result (``pi_slab``, ``a``, ``mu``, ``log_lik``) are
    returned as 0-d tensors on the input device so the cEBMF ELBO accumulator
    can fold them in without forcing a host sync per factor update. Call
    ``float(field)`` at your own boundary if you need a Python scalar.
    """
    # ---- setup & hoisted constants (device/dtype safe) ----
    device, dtype = x.device, x.dtype
    x = torch.as_tensor(x, device=device, dtype=dtype)
    s = torch.as_tensor(s, device=device, dtype=dtype).clamp_min(_const_like(x, 1e-6))

    # Precompute once; reused in closure & posterior
    log_s = torch.log(s)

    # scalar constants as tensors on the right device/dtype
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    half = torch.tensor(0.5, device=device, dtype=dtype)
    two = torch.tensor(2.0, device=device, dtype=dtype)

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

    w_logit = torch.nn.Parameter(
        torch.as_tensor(par_init[0], dtype=dtype, device=device), requires_grad=not fix_par[0]
    )  # noqa: E501
    a_logit = torch.nn.Parameter(torch.as_tensor(v0, dtype=dtype, device=device), requires_grad=not fix_par[1])  # noqa: E501
    mu = torch.nn.Parameter(torch.as_tensor(par_init[2], dtype=dtype, device=device), requires_grad=not fix_par[2])  # noqa: E501

    params = [p for p in (w_logit, a_logit, mu) if p.requires_grad]

    # ---- optional warm-start (kept very short) ----
    if use_adam_warmstart and params:
        opt_adam = torch.optim.AdamW(params, lr=adam_lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        for _ in range(int(adam_steps)):
            opt_adam.zero_grad(set_to_none=True)
            log_slab, log_spike = F.logsigmoid(w_logit), F.logsigmoid(-w_logit)
            a = a_lo_t + a_span * torch.sigmoid(a_logit)
            xc = x - mu

            # spike log-lik
            lf = -(half * (xc / s) ** 2) - log_s - c_norm

            # slab log-lik
            lg, _, _ = _laplace_slab_terms(xc, s, a)

            llik = torch.logaddexp(log_spike + lf, log_slab + lg).sum()

            penalty = zero
            if loga_l2 != 0.0:
                penalty = penalty + torch.tensor(loga_l2, device=device, dtype=dtype) * (a_logit**2)
            if pen_pi0 != 0.0:
                penalty = penalty - torch.tensor(pen_pi0, device=device, dtype=dtype) * (log_slab + log_spike)
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
            log_slab, log_spike = F.logsigmoid(w_logit), F.logsigmoid(-w_logit)
            a = a_lo_t + a_span * torch.sigmoid(a_logit)
            xc = x - mu

            # spike log-lik (reuses log_s, c_norm)
            lf = -(half * (xc / s) ** 2) - log_s - c_norm

            # slab log-lik
            lg, _, _ = _laplace_slab_terms(xc, s, a)

            llik = torch.logaddexp(log_spike + lf, log_slab + lg).sum()

            # penalties as tensors on-device
            penalty = zero
            if loga_l2 != 0.0:
                penalty = penalty + torch.tensor(loga_l2, device=device, dtype=dtype) * (a_logit**2)
            if pen_pi0 != 0.0:
                penalty = penalty - torch.tensor(pen_pi0, device=device, dtype=dtype) * (log_slab + log_spike)

            loss = -(llik - penalty)
            loss = torch.nan_to_num(
                loss,
                nan=torch.tensor(1e30, device=device, dtype=dtype),
                posinf=torch.tensor(1e30, device=device, dtype=dtype),
                neginf=torch.tensor(1e30, device=device, dtype=dtype),
            )
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
        w = torch.sigmoid(w_logit)
        log_slab, log_spike = F.logsigmoid(w_logit), F.logsigmoid(-w_logit)
        a = a_lo_t + a_span * torch.sigmoid(a_logit)
        xc = x - mu

        # spike
        lf = -(half * (xc / s) ** 2) - log_s - c_norm

        # slab
        lg, lg1, lsum = _laplace_slab_terms(xc, s, a)

        # posterior inclusion prob (slab)
        log_num = log_slab + lg
        log_denom = torch.logaddexp(log_spike + lf, log_num)
        gamma = torch.exp(log_num - log_denom).clamp(zero, one)

        # sign-mixture inside slab
        lam = torch.exp(lg1 - lsum)
        lam = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5))

        # Stable branch moments avoid cancellation between the tilted mean
        # +/- s^2*a and the truncated-normal correction for nearly null factors.
        EX_pos, EX2_pos = posterior_moments_exp(xc, s, a)
        EX_neg_abs, EX2_neg = posterior_moments_exp(-xc, s, a)
        EX_neg = -EX_neg_abs

        EX = lam * EX_pos + (one - lam) * EX_neg
        EX2 = lam * EX2_pos + (one - lam) * EX2_neg

        post_mean = gamma * (EX + mu) + (one - gamma) * mu
        post_mean2 = gamma * (EX2 + two * mu * EX + mu * mu) + (one - gamma) * (mu * mu)
        post_sd = (post_mean2 - post_mean**2).clamp_min(zero).sqrt()

        # PURE marginal log-likelihood
        llik = log_denom.sum()

        # Only choose the all-spike boundary if it improves the fitted
        # likelihood and the mixture weight is free. A weight threshold can
        # discard a better fit; an added variance would not be a point mass
        # and would make the posterior moments inconsistent with its likelihood.
        post_mean_so = torch.zeros_like(x) + mu
        post_mean2_so = torch.zeros_like(x) + mu * mu
        llik_so = lf.sum()
        spike_only = (llik_so > llik) & (not fix_par[0] and fix_par[2] and pen_pi0 == 0.0)
        post_mean = torch.where(spike_only, post_mean_so, post_mean)
        post_mean2 = torch.where(spike_only, post_mean2_so, post_mean2)
        post_sd = (post_mean2 - post_mean**2).clamp_min(zero).sqrt()
        llik = torch.where(spike_only, llik_so, llik)
        w = torch.where(spike_only, torch.zeros_like(w), w)
        regularization = loga_l2 * a_logit.square() if loga_l2 != 0 else zero
        if pen_pi0 != 0:
            regularization = regularization - pen_pi0 * (log_slab + log_spike)

    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi_slab=w.detach(),  # mixture weight of the Laplace branch (slab)
        a=a.detach(),
        mu=mu.detach(),
        log_lik=llik.detach(),
        regularization_loss=regularization.detach(),
    )
