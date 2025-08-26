import math
import torch
from dataclasses import dataclass
from torch import Tensor

from cebmf_torch.torch_utils import (
    my_etruncnorm,
    my_e2truncnorm,
    log_norm_pdf,   # kept for API parity (unused here)
    logPhi,
    safe_log,
    _TWOPI,
    _LOG_SQRT_2PI,
)


def logg_laplace_convolved_with_normal(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """
    log p(x | theta ~ Laplace(0, 1/a), noise ~ N(0, s^2)) as a function of x.

    Closed form:
        log(a/2) + 0.5*(s*a)^2
        + log( Φ((x - s^2 a)/s) * e^{-a x} + Φ(-(x + s^2 a)/s) * e^{a x} )

    Implemented in log-space with logaddexp.
    """
    s = torch.clamp(s, min=1e-12)
    z1 = (x - (s * s) * a) / s
    z2 = -(x + (s * s) * a) / s

    # log of each branch safely
    lg1 = -a * x + logPhi(z1)
    lg2 = a * x + logPhi(z2)

    lsum = torch.logaddexp(lg1, lg2)  # stable log(exp(lg1) + exp(lg2))
    return safe_log(a / 2.0) + 0.5 * (s * a) ** 2 + lsum


@dataclass
class EBNMLaplaceResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    w: float
    a: float
    mu: float
    log_lik: float


_LOG_2PI = math.log(2 * math.pi)


def ebnm_point_laplace(
    x: Tensor,
    s: Tensor,
    par_init=None,                       # None by default; choose safely inside
    fix_par=(False, False, True),        # [w_logit, log_a, mu]; mu fixed at 0
    max_iter: int = 20,
    tol: float = 1e-6,
    a_bounds=(1e-1, 1e1),               # slightly tighter; adjust if needed
    loga_l2: float = 1e-2,
    tresh_pi0: float = 1e-4,
    eps: float = 1e-12,
) -> EBNMLaplaceResult:

    device, dtype = x.device, x.dtype
    x = x.to(dtype)
    s = torch.clamp(s.to(dtype), min=1e-6)

    # ---- choose robust defaults if None ----
    if par_init is None:
        par_init = (0.0, 2.0, 0.0)  # heuristic init (logit(w), log(a), mu)

    w_logit = torch.nn.Parameter(
        torch.tensor(par_init[0], dtype=dtype, device=device),
        requires_grad=not fix_par[0],
    )
    log_a = torch.nn.Parameter(
        torch.tensor(par_init[1], dtype=dtype, device=device),
        requires_grad=not fix_par[1],
    )
    mu = torch.nn.Parameter(
        torch.tensor(par_init[2], dtype=dtype, device=device),
        requires_grad=not fix_par[2],
    )

    params = [p for p in (w_logit, log_a, mu) if p.requires_grad]
    opt = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn="strong_wolfe",  # steadier
        history_size=20,
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        w = torch.sigmoid(w_logit)

        # bounded a
        log_a_eff = log_a.clamp(
            min=math.log(a_bounds[0]),
            max=math.log(a_bounds[1]),
        )
        a = log_a_eff.exp()

        xc = x - mu

        # spike likelihood
        lf = -0.5 * ((xc / s) ** 2) - torch.log(s) - 0.5 * _LOG_2PI

        # slab log-likelihood (Laplace convolved with Normal)
        z1 = (xc - (s * s) * a) / s
        z2 = -(xc + (s * s) * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 = a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lg = torch.log(a / 2) + 0.5 * (s * a) ** 2 + lsum

        llik_i = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg)

        loss = -llik_i.sum() + loga_l2 * (log_a ** 2)

        # graph-preserving guard
        loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=1e30)
        loss.backward()
        return loss

    if params:
        try:
            opt.step(closure)
        except RuntimeError:
            # fallback: fix 'a' if line search still blows up
            if log_a.requires_grad:
                log_a.requires_grad_(False)
                params2 = [p for p in (w_logit, mu) if p.requires_grad]
                if params2:
                    torch.optim.LBFGS(
                        params2,
                        max_iter=max_iter,
                        tolerance_grad=tol,
                        tolerance_change=tol,
                        line_search_fn="strong_wolfe",
                    ).step(closure)

    # ---- posterior (same bounded a) ----
    with torch.no_grad():
        pi0 = torch.sigmoid(w_logit).clamp(eps, 1 - eps)

        log_a_eff = log_a.clamp(
            min=math.log(a_bounds[0]),
            max=math.log(a_bounds[1]),
        )
        a = log_a_eff.exp()
        mu_v = float(mu)

        xc = x - mu

        # spike loglik
        lf = -0.5 * ((xc / s) ** 2) - torch.log(s) - 0.5 * _LOG_2PI

        # slab loglik
        z1 = (xc - (s * s) * a) / s
        z2 = -(xc + (s * s) * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 = a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lg = torch.log(a / 2) + 0.5 * (s * a) ** 2 + lsum

        # posterior inclusion prob for slab
        log_num = torch.log(pi0) + lg
        log_denom = torch.logaddexp(torch.log1p(-pi0) + lf, log_num)
        gamma = torch.exp(log_num - log_denom).clamp(0, 1)

        # mixture weight within the slab (sign branch)
        lam = torch.exp(lg1 - lsum)
        lam = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5))

        # truncated-normal moments for Z given sign branch
        m_pos = xc - s * s * a
        m_neg = xc + s * s * a
        infp = torch.full_like(x, float("inf"))
        infn = -infp

        EX_pos = my_etruncnorm(0.0, infp, mean=m_pos, sd=s)
        EX2_pos = my_e2truncnorm(0.0, infp, mean=m_pos, sd=s)
        EX_neg = my_etruncnorm(infn, 0.0, mean=m_neg, sd=s)
        EX2_neg = my_e2truncnorm(infn, 0.0, mean=m_neg, sd=s)

        EX = lam * EX_pos + (1 - lam) * EX_neg
        EX2 = lam * EX2_pos + (1 - lam) * EX2_neg

        # combine spike/slab
        post_mean = gamma * (EX + mu) + (1 - gamma) * mu
        post_mean2 = gamma * (EX2 + 2 * mu * EX + mu * mu) + (1 - gamma) * (mu * mu)
        post_sd = (post_mean2 - post_mean ** 2).clamp_min(0).sqrt()

        # mixture log-likelihood (no hard overrides)
        log_lik = torch.logaddexp(
            torch.log1p(-pi0) + lf,
            torch.log(pi0.clamp_min(eps)) + lg,
        ).sum().item()

        # Optional early-exit guard; if keeping, ensure tensor ops
        if float(pi0) < tresh_pi0:
            post_mean = torch.zeros_like(x)
            post_mean2 = torch.zeros_like(x)  +0.0001
            post_sd =   torch.sqrt(post_mean2)
            # consistent spike-only log-lik:
            log_lik = (torch.log1p(-pi0) + lf).sum().item()

    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        w=float(pi0),
        a=float(a),
        mu=mu_v,
        log_lik=float(log_lik),
    )
