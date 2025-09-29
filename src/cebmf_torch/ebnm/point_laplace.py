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
    """Create a scalar tensor `val` on x's device/dtype."""
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)


def logg_laplace_convolved_with_normal(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """
    Compute log p(x | theta ~ Laplace(0, 1/a), noise ~ N(0, s^2)) as a function of x.

    Closed form:
        log(a/2) + 0.5*(s*a)^2
        + log( Φ((x - s^2 a)/s) * e^{-a x} + Φ(-(x + s^2 a)/s) * e^{a x} )

    Implemented in log-space with logaddexp for numerical stability.

    Parameters
    ----------
    x : torch.Tensor
        Observed data.
    s : torch.Tensor
        Standard deviation of the noise.
    a : torch.Tensor
        Laplace scale parameter (1/a).

    Returns
    -------
    torch.Tensor
        Log-likelihood values for each observation.
    """
    x = torch.as_tensor(x, device=x.device, dtype=x.dtype)
    s = torch.as_tensor(s, device=x.device, dtype=x.dtype).clamp_(min=_const_like(x, 1e-12))
    a = torch.as_tensor(a, device=x.device, dtype=x.dtype)

    z1 = (x - (s * s) * a) / s
    z2 = -(x + (s * s) * a) / s

    # log of each branch safely
    lg1 = -a * x + logPhi(z1)
    lg2 = a * x + logPhi(z2)

    lsum = torch.logaddexp(lg1, lg2)  # stable log(exp(lg1) + exp(lg2))
    return safe_log(a / _const_like(x, 2.0)) + _const_like(x, 0.5) * (s * a) ** 2 + lsum


@dataclass
class EBNMLaplaceResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    pi0: float
    a: float
    mu: float
    log_lik: float
    """
    Container for the results of the point-Laplace EBNM posterior estimation.

    Attributes
    ----------
    post_mean : torch.Tensor
        Posterior means for each observation.
    post_mean2 : torch.Tensor
        Posterior second moments for each observation.
    post_sd : torch.Tensor
        Posterior standard deviations for each observation.
    pi0 : float
        Estimated mixture weight for the Laplace branch.
    a : float
        Estimated Laplace scale parameter.
    mu : float
        Estimated mode (mu).
    log_lik : float
        Final log-likelihood value.
    """


def ebnm_point_laplace(
    x: Tensor,
    s: Tensor,
    par_init=None,  # None by default; choose safely inside
    fix_par=(False, False, True),  # [w_logit, log_a, mu]; mu fixed at 0
    max_iter: int = 50,
    tol: float = 1e-3,
    a_bounds=(1e-2, 1e2),  # slightly tighter; adjust if needed
    loga_l2: float = 1e-2,
    tresh_pi0: float = 1e-3,
    eps: float = 1e-12,
    pen_pi0=1,
) -> EBNMLaplaceResult:
    """
    Fit a point-Laplace Empirical Bayes Normal Means (EBNM) model using PyTorch.

    This implementation uses stability tricks:
      - clamps log a to [log(a_min), log(a_max)]
      - L2 penalty on log a
      - thresholds pi0 to spike-only if below tresh_pi0
      - robust LBFGS closure with NaN/Inf guards

    The prior on θ is: (1 - pi0) δ_μ + pi0 * Laplace(μ, 1/a), with support θ ∈ ℝ.

    Parameters
    ----------
    x : torch.Tensor
        Observed data.
    s : torch.Tensor
        Standard errors of the observed data.
    par_init : tuple or None, optional
        Initial values for (w_logit, log_a, mu). If None, defaults are used.
    fix_par : tuple of bool, optional
        Which parameters to fix during optimization (default: (False, False, True)).
    max_iter : int, optional
        Maximum number of LBFGS iterations (default: 50).
    tol : float, optional
        Tolerance for optimizer (default: 1e-3).
    a_bounds : tuple, optional
        Bounds for the Laplace scale parameter a (default: (1e-2, 1e2)).
    loga_l2 : float, optional
        L2 penalty on log a (default: 1e-2).
    tresh_pi0 : float, optional
        Threshold for pi0 below which the solution is set to spike-only (default: 1e-3).
    eps : float, optional
        Small value to avoid numerical issues (default: 1e-12).
    pen_pi0 : float, optional
        Penalty on pi0 (default: 1).

    Returns
    -------
    EBNMLaplaceResult
        Container with posterior means, standard deviations, and model parameters.
    """
    device, dtype = x.device, x.dtype
    x = torch.as_tensor(x, device=device, dtype=dtype)
    s = torch.as_tensor(s, device=device, dtype=dtype).clamp_(min=_const_like(x, 1e-6))

    # ---- choose robust defaults if None ----
    if par_init is None:
        par_init = (2, 2.0, 0.0)  # heuristic init (logit(w), log(a), mu)

    w_logit = torch.nn.Parameter(
        torch.as_tensor(par_init[0], dtype=dtype, device=device), requires_grad=not fix_par[0]
    )
    log_a = torch.nn.Parameter(torch.as_tensor(par_init[1], dtype=dtype, device=device), requires_grad=not fix_par[1])
    mu = torch.nn.Parameter(torch.as_tensor(par_init[2], dtype=dtype, device=device), requires_grad=not fix_par[2])

    params = [p for p in (w_logit, log_a, mu) if p.requires_grad]
    opt = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn="strong_wolfe",  # steadier
        history_size=20,
    )

    log_a_lo = math.log(a_bounds[0])
    log_a_hi = math.log(a_bounds[1])
    eps_t = _const_like(x, eps)

    def closure():
        opt.zero_grad(set_to_none=True)
        w = torch.sigmoid(w_logit)

        pen = -_const_like(x, pen_pi0) * torch.log((1 - w).clamp(min=eps_t, max=1 - eps_t))

        # bounded a
        log_a_eff = log_a.clamp(min=log_a_lo, max=log_a_hi)
        a = log_a_eff.exp()

        xc = x - mu

        # spike likelihood
        c = _LOG_SQRT_2PI if isinstance(_LOG_SQRT_2PI, torch.Tensor) else _const_like(s, _LOG_SQRT_2PI)
        lf = -_const_like(x, 0.5) * ((xc / s) ** 2) - torch.log(s) - c

        # slab log-likelihood (Laplace convolved with Normal)
        z1 = (xc - (s * s) * a) / s
        z2 = -(xc + (s * s) * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 = a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lg = safe_log(a / _const_like(x, 2.0)) + _const_like(x, 0.5) * (s * a) ** 2 + lsum

        llik_i = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg)

        loss = -llik_i.sum() + _const_like(x, loga_l2) * (log_a**2) + pen

        # graph-preserving guard
        huge = _const_like(x, 1e30)
        loss = torch.nan_to_num(loss, nan=huge, posinf=huge, neginf=huge) + pen
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
        pi0 = torch.sigmoid(w_logit).clamp(eps_t, 1 - eps_t)

        log_a_eff = log_a.clamp(min=log_a_lo, max=log_a_hi)
        a = log_a_eff.exp()
        mu_v = float(mu.item())

        xc = x - mu

        # spike loglik
        c = _LOG_SQRT_2PI if isinstance(_LOG_SQRT_2PI, torch.Tensor) else _const_like(s, _LOG_SQRT_2PI)
        lf = -_const_like(x, 0.5) * ((xc / s) ** 2) - torch.log(s) - c

        # slab loglik
        z1 = (xc - (s * s) * a) / s
        z2 = -(xc + (s * s) * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 = a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lg = safe_log(a / _const_like(x, 2.0)) + _const_like(x, 0.5) * (s * a) ** 2 + lsum

        # posterior inclusion prob for slab
        log_num = torch.log(pi0) + lg
        log_denom = torch.logaddexp(torch.log1p(-pi0) + lf, log_num)
        gamma = torch.exp(log_num - log_denom).clamp(_const_like(x, 0.0), _const_like(x, 1.0))

        # mixture weight within the slab (sign branch)
        lam = torch.exp(lg1 - lsum)
        lam = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5))

        # truncated-normal moments for Z given sign branch
        m_pos = xc - s * s * a
        m_neg = xc + s * s * a
        infp = torch.full_like(x, float("inf"))
        infn = -infp

        EX_pos = my_etruncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EX2_pos = my_e2truncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EX_neg = my_etruncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)
        EX2_neg = my_e2truncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)

        EX = lam * EX_pos + (1 - lam) * EX_neg
        EX2 = lam * EX2_pos + (1 - lam) * EX2_neg

        # combine spike/slab
        post_mean = gamma * (EX + mu) + (1 - gamma) * mu
        post_mean2 = gamma * (EX2 + _const_like(x, 2.0) * mu * EX + mu * mu) + (1 - gamma) * (mu * mu)
        post_sd = (post_mean2 - post_mean**2).clamp_min(_const_like(x, 0.0)).sqrt()

        # mixture log-likelihood (no hard overrides)
        log_lik = torch.logaddexp(torch.log1p(-pi0) + lf, torch.log(pi0.clamp_min(eps_t)) + lg).sum().item()

        # Optional early-exit guard; keep semantics
        if float(pi0.item()) < tresh_pi0:
            post_mean = torch.zeros_like(x)
            post_mean2 = torch.zeros_like(x) + _const_like(x, 1e-4)
            post_sd = post_mean2.sqrt()
            # consistent spike-only log-lik:
            log_lik = (torch.log1p(-pi0) + lf).sum().item()

    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi0=float(1 - pi0),  # kept semantics of original return
        a=float(a),
        mu=mu_v,
        log_lik=float(log_lik),
    )
