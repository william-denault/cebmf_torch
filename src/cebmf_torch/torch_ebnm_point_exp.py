# torch_only_point_exponential_stable.py
import math
import torch
from torch import Tensor

from cebmf_torch.torch_utils import (
    my_etruncnorm,
    my_e2truncnorm,
    logPhi,
    _LOG_SQRT_2PI,
)

_LOG_2PI = math.log(2 * math.pi)


# =========================
# Core pieces
# =========================

def _loglik_spike(xc: Tensor, s: Tensor) -> Tensor:
    # log N(xc | 0, s^2)
    return -0.5 * (xc / s) ** 2 - torch.log(s) - 0.5 * _LOG_2PI


def _loglik_exp_convolved(xc: Tensor, s: Tensor, a: Tensor) -> Tensor:
    # lg = log a + (s a)^2 / 2 - a * xc + log Φ(xc/s - s a), θ_c ≥ 0
    z = xc / s - s * a
    return torch.log(a / 1.0) + 0.5 * (s * a) ** 2 - a * xc + logPhi(z)


def _posterior_moments_exp_branch(xc: Tensor, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
    """
    Moments for the Exp branch using the tilted Normal:
    Z ~ N(m_tilt, s^2) truncated to [0, +inf),
    where m_tilt = xc - s^2 * a.
    Returns (E[Z], E[Z^2]).
    """
    m_tilt = xc - (s * s) * a
    zero = torch.zeros_like(xc)
    inf = torch.full_like(xc, float("inf"))
    EZ  = my_etruncnorm(zero, inf, m_tilt, s)
    EZ2 = my_e2truncnorm(zero, inf, m_tilt, s)
    return EZ, EZ2


# =========================
# Public EBNM interface
# =========================

class EBNMPointExp:
    def __init__(self,
                 post_mean: Tensor,
                 post_mean2: Tensor,
                 post_sd: Tensor,
                 scale: float,
                 pi0: float,
                 log_lik: float,
                 mode: float):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.scale = scale
        self.pi0 = pi0
        self.log_lik = log_lik
        self.mode = mode


def ebnm_point_exp(
    x: Tensor,
    s: Tensor,
    par_init=None,                    # (alpha, log_a, mu). If None, choose safely inside.
    fix_par=(False, False, True),     # [w_logit, log_a, mu]; default keeps mu fixed like your Laplace
    max_iter: int = 20,
    tol: float = 1e-6,
    a_bounds=(1e-2, 1e2),             # bounded scale (like Laplace)
    loga_l2: float = 1e-4,            # ridge on log a
    tresh_pi0: float = 1e-3,          # zero-out when pi0 tiny
    eps: float = 1e-12,
) -> EBNMPointExp:
    """
    Torch-only point-exponential EBNM with the same stability tricks as your point-Laplace:
      - clamps log a to [log(a_min), log(a_max)]
      - L2 penalty on log a
      - thresholds pi0 to spike-only if below tresh_pi0
      - robust LBFGS closure with NaN/Inf guards
    Prior on θ: (1 - pi0) δ_μ + pi0 [μ + Exp(a)], support θ ≥ μ.
    """
    device, dtype = x.device, x.dtype
    x = x.to(dtype)
    s = torch.clamp(s.to(dtype), min=1e-6)

    if par_init is None:
        # alpha ~ logit(pi0), log_a ~ log(a), mu
        par_init = (0.9, 1.0, 0.0)  # pi0≈0.71, a≈1.0, mu=0.0

    alpha = torch.nn.Parameter(torch.tensor(par_init[0], dtype=dtype, device=device),
                               requires_grad=not fix_par[0])
    log_a = torch.nn.Parameter(torch.tensor(par_init[1], dtype=dtype, device=device),
                               requires_grad=not fix_par[1])
    mu    = torch.nn.Parameter(torch.tensor(par_init[2], dtype=dtype, device=device),
                               requires_grad=not fix_par[2])

    params = [p for p in (alpha, log_a, mu) if p.requires_grad]
    opt = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn="strong_wolfe",
        history_size=20,
    )

    a_lo, a_hi = a_bounds
    log_a_lo = math.log(a_lo)
    log_a_hi = math.log(a_hi)

    def closure():
        opt.zero_grad(set_to_none=True)

        # parameters with transforms
        pi0 = torch.sigmoid(alpha)                        # in (0,1)
        log_a_eff = log_a.clamp(min=log_a_lo, max=log_a_hi)
        a = log_a_eff.exp()
        xc = x - mu

        # log-likelihood pieces
        lf = _loglik_spike(xc, s)
        lg = _loglik_exp_convolved(xc, s, a)

        # mixture log-likelihood per datum
        llik_i = torch.logaddexp(torch.log1p(-pi0) + lf, torch.log(pi0) + lg)

        # ridge on log a (like Laplace)
        penalty = loga_l2 * (log_a ** 2)

        nll = -(llik_i.sum() - penalty)

        # guards
        nll = torch.nan_to_num(nll, nan=1e30, posinf=1e30, neginf=1e30)
        nll.backward()
        return nll

    if params:
        try:
            opt.step(closure)
        except RuntimeError:
            # Fallback: freeze log_a if line search is problematic
            if log_a.requires_grad:
                log_a.requires_grad_(False)
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
        pi0 = torch.sigmoid(alpha).clamp(eps, 1 - eps)
        log_a_eff = log_a.clamp(min=log_a_lo, max=log_a_hi)
        a = log_a_eff.exp()
        mu_v = float(mu)

        xc = x - mu

        # log-lik pieces
        lf = _loglik_spike(xc, s)
        lg = _loglik_exp_convolved(xc, s, a)

        # posterior inclusion prob for the Exp branch
        log_num = torch.log(pi0) + lg
        log_den = torch.logaddexp(torch.log1p(-pi0) + lf, log_num)
        gamma = torch.exp(log_num - log_den).clamp(0, 1)

        # moments under Exp branch
        EZ, EZ2 = _posterior_moments_exp_branch(xc, s, a)

        # combine spike/exp on centered variable θ_c
        post_mean_c = gamma * EZ
        post_mean2_c = gamma * EZ2

        # numerical monotonicity: E[X^2] >= (E[X])^2
        post_mean2_c = torch.maximum(post_mean2_c, post_mean_c ** 2)

        # back to θ
        post_mean = post_mean_c + mu
        post_mean2 = post_mean2_c + 2.0 * mu * post_mean_c + mu * mu
        post_sd = (post_mean2 - post_mean ** 2).clamp_min(0).sqrt()

        # mixture log-likelihood
        log_lik = torch.logaddexp(torch.log1p(-pi0) + lf, torch.log(pi0) + lg).sum().item()

        # Threshold to spike-only if pi0 tiny (match Laplace behavior)
        if float(pi0) < tresh_pi0:
            post_mean = torch.zeros_like(x)
            post_mean2 = torch.zeros_like(x) + 1e-4
            post_sd = (post_mean2).sqrt()
            log_lik = (torch.log1p(-pi0) + lf).sum().item()

    return EBNMPointExp(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        scale=float(a),
        pi0=float(pi0),
        log_lik=float(log_lik),
        mode=mu_v,
    )