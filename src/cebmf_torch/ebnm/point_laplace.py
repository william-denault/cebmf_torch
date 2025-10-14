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

def _loglik_spike(xc: Tensor, s: Tensor) -> Tensor:
    c = _LOG_SQRT_2PI if isinstance(_LOG_SQRT_2PI, torch.Tensor) else _const_like(s, _LOG_SQRT_2PI)
    return -_const_like(s, 0.5) * (xc / s) ** 2 - torch.log(s) - c

def _loglik_laplace_convolved(xc: Tensor, s: Tensor, a: Tensor) -> Tensor:
    # log (Laplace ⊗ Normal) at xc with Laplace rate a and noise sd s
    z1 = (xc - (s * s) * a) / s
    z2 = -(xc + (s * s) * a) / s
    lg1 = -a * xc + logPhi(z1)
    lg2 =  a * xc + logPhi(z2)
    lsum = torch.logaddexp(lg1, lg2)
    return safe_log(a / _const_like(xc, 2.0)) + _const_like(xc, 0.5) * (s * a) ** 2 + lsum

@dataclass
class EBNMLaplaceResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    pi0: float      # mixture weight of the Laplace slab
    a: float        # Laplace rate (1/scale)
    mu: float
    log_lik: float  # observed marginal log-likelihood (no penalties)
    history: dict   # {'loglik': [...], 'Q': [...], 'pi0': [...], 'a': [...], 'mu': [...]}

def ebnm_point_laplace(
    x: Tensor,
    s: Tensor,
    par_init=None,                 # (logit(pi0), log(a), mu)
    fix_par=(False, False, True),  # [w_logit, log_a, mu]; default fixes mu
    max_iter: int = 200,
    tol: float = 1e-8,
    a_bounds=(1e-6, 1e6),
    eps: float = 1e-12,
    alpha_pi: float = 1.0,         # Beta prior on pi0: alpha
    beta_pi: float = 1.0,          # Beta prior on pi0: beta (alpha=1,beta=1 is flat)
) -> EBNMLaplaceResult:
    """
    Monotone-EM for point–Laplace EBNM.
    Prior: theta ~ (1-pi0) δ_mu + pi0 * Laplace(mu, rate=a).
    Guarantees non-decreasing EM auxiliary function Q; also tracks observed log-likelihood.
    """
    # ---- setup (float64 for stability) ----
    device = x.device
    x = torch.as_tensor(x, device=device, dtype=torch.float64)
    s = torch.as_tensor(s, device=device, dtype=torch.float64).clamp_min(1e-12)
    n = x.numel()

    if par_init is None:
        par_init = (0.0, 0.0, 0.0)  # pi0≈0.5, a≈1.0, mu=0
    w_logit0, log_a0, mu0 = par_init
    w_logit = torch.as_tensor(w_logit0, device=device, dtype=torch.float64)
    log_a   = torch.as_tensor(log_a0,   device=device, dtype=torch.float64)
    mu      = torch.as_tensor(mu0,      device=device, dtype=torch.float64)

    # honor fixes
    fix_pi  = fix_par[0]
    fix_a   = fix_par[1]
    fix_mu  = fix_par[2]

    # bounds for a (apply softly via clamping after closed-form update)
    log_a_lo = math.log(a_bounds[0])
    log_a_hi = math.log(a_bounds[1])

    # history
    hist_loglik, hist_Q, hist_pi, hist_a, hist_mu = [], [], [], [], []

    def e_step(mu, log_a, pi0):
        """Compute responsibilities and required moments at current params."""
        a  = torch.clamp(log_a, min=log_a_lo, max=log_a_hi).exp()
        xc = x - mu

        # spike/slab log-lik
        lf = _loglik_spike(xc, s)
        lg = _loglik_laplace_convolved(xc, s, a)

        # posterior slab prob
        log_num   = torch.log(pi0) + lg
        log_denom = torch.logaddexp(torch.log1p(-pi0) + lf, log_num)
        gamma     = torch.exp(log_num - log_denom).clamp(0.0, 1.0)  # shape (n,)

        # within-slab sign mixture weight lam = P(S=+ | slab,x)
        z1 = (xc - (s * s) * a) / s
        z2 = -(xc + (s * s) * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 =  a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lam  = torch.exp(lg1 - lsum)  # P(positive branch)
        lam  = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5)).clamp(0.0, 1.0)

        # truncated-normal moments for Z = S*U (centered around mu)
        m_pos = xc - s * s * a
        m_neg = xc + s * s * a
        infp = torch.full_like(x, float("inf"))
        infn = -infp

        EZ_pos  = my_etruncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EZ2_pos = my_e2truncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EZ_neg  = my_etruncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)
        EZ2_neg = my_e2truncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)

        # E[Z | slab], E[Z^2 | slab], E[|Z| | slab]
        EZ_slab  = lam * EZ_pos + (1.0 - lam) * EZ_neg
        EZ2_slab = lam * EZ2_pos + (1.0 - lam) * EZ2_neg
        EabsZ    = lam * EZ_pos - (1.0 - lam) * EZ_neg  # since EZ_neg <= 0

        return a, xc, lf, lg, gamma, EZ_slab, EZ2_slab, EabsZ

    def observed_loglik(lf, lg, pi0):
        return torch.logaddexp(torch.log1p(-pi0) + lf, torch.log(pi0) + lg).sum()

    def q_function(xc, s, gamma, EZ_slab, EZ2_slab, EabsZ, pi0, a, mu):
        """
        EM auxiliary function Q(θ | θ_old) up to constants.
        """
        w = 1.0 / (s * s)
        # Gaussian term: -0.5 * Σ w_i * E[ (x - μ - Z)^2 ] = -0.5 Σ w_i [ (x-μ)^2 - 2(x-μ)E[Z] + E[Z^2] ]
        # where E[Z] = γ * EZ_slab, E[Z^2] = γ * EZ2_slab (spike contributes zero Z)
        EZ   = gamma * EZ_slab
        EZ2  = gamma * EZ2_slab

        term_gauss = -0.5 * (w * ((xc - EZ)**2 + (EZ2 - EZ**2))).sum()  # algebraically same, μ only in xc = x-μ

        # Prior/mixing terms:
        # spike/slab mixing with Beta(α,β) prior on pi0
        term_mix = (gamma.sum()    + (alpha_pi - 1.0)) * torch.log(pi0) \
                 + ((n - gamma.sum()) + (beta_pi - 1.0)) * torch.log1p(-pi0)

        # Laplace slab prior: C=1 adds log(a/2) - a * U, with E[U]=E|Z|
        term_slab = gamma.sum() * (torch.log(a) - math.log(2.0)) - a * (gamma * EabsZ).sum()

        return term_gauss + term_mix + term_slab

    # ---- EM loop ----
    for it in range(max_iter):
        # E-step at current params
        pi0 = torch.sigmoid(w_logit)
        a, xc, lf, lg, gamma, EZ_slab, EZ2_slab, EabsZ = e_step(mu, log_a, pi0)

        # Evaluate diagnostics BEFORE M-step
        L  = observed_loglik(lf, lg, pi0)
        Q  = q_function(xc, s, gamma, EZ_slab, EZ2_slab, EabsZ, pi0, a, mu)

        hist_loglik.append(float(L.item()))
        hist_Q.append(float(Q.item()))
        hist_pi.append(float(pi0.item()))
        hist_a.append(float(a.item()))
        hist_mu.append(float(mu.item()))

        # M-step: closed-form (with optional parameter fixing)
        # π update (MAP with Beta prior α,β)
        if not fix_pi:
            sum_gamma = gamma.sum()
            pi0_new = (sum_gamma + (alpha_pi - 1.0)) / (n + (alpha_pi + beta_pi - 2.0))
            # project to (eps, 1-eps)
            pi0_new = torch.clamp(pi0_new, eps, 1.0 - eps)
            w_logit = torch.log(pi0_new) - torch.log1p(-pi0_new)
        # a update (closed form): a* = (Σ γ_i) / (Σ γ_i E|Z|_i)
        if not fix_a:
            denom = (gamma * EabsZ).sum()
            # guard: if denom is ~0, keep previous a
            if float(denom.item()) > 0.0:
                a_new = (gamma.sum() / denom).clamp_min(torch.as_tensor(a_bounds[0], dtype=torch.float64))
                log_a = torch.log(torch.clamp(a_new, min=a_bounds[0], max=a_bounds[1]))
        # μ update (weighted least squares): μ* = [Σ w_i (x_i - EZ_i)] / [Σ w_i], with EZ_i = γ_i E[Z|slab]
        if not fix_mu:
            w = 1.0 / (s * s)
            EZ = gamma * EZ_slab
            num = (w * (x - EZ)).sum()
            den = w.sum().clamp_min(1e-24)
            mu = num / den

        # Check Q monotonicity (optional early stop)
        if it > 0:
            inc = hist_Q[-1] - hist_Q[-2]
            if abs(inc) < tol:
                break

    # ---- final posterior at last params ----
    with torch.no_grad():
        pi0 = torch.sigmoid(w_logit).clamp(eps, 1.0 - eps)
        a   = torch.clamp(log_a, min=log_a_lo, max=log_a_hi).exp()
        xc  = x - mu

        lf = _loglik_spike(xc, s)
        lg = _loglik_laplace_convolved(xc, s, a)

        # posterior slab prob
        log_num   = torch.log(pi0) + lg
        log_denom = torch.logaddexp(torch.log1p(-pi0) + lf, log_num)
        gamma     = torch.exp(log_num - log_denom).clamp(0.0, 1.0)

        # within-slab sign prob
        z1 = (xc - (s * s) * a) / s
        z2 = -(xc + (s * s) * a) / s
        lg1 = -a * xc + logPhi(z1)
        lg2 =  a * xc + logPhi(z2)
        lsum = torch.logaddexp(lg1, lg2)
        lam  = torch.exp(lg1 - lsum)
        lam  = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5)).clamp(0.0, 1.0)

        # truncated-normal moments
        m_pos = xc - s * s * a
        m_neg = xc + s * s * a
        infp = torch.full_like(x, float("inf"))
        infn = -infp

        EX_pos  = my_etruncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EX2_pos = my_e2truncnorm(_const_like(x, 0.0), infp, mean=m_pos, sd=s)
        EX_neg  = my_etruncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)
        EX2_neg = my_e2truncnorm(infn, _const_like(x, 0.0), mean=m_neg, sd=s)

        EX  = lam * EX_pos  + (1.0 - lam) * EX_neg
        EX2 = lam * EX2_pos + (1.0 - lam) * EX2_neg

        # combine spike/slab back to θ
        post_mean_c  = gamma * EX
        post_mean2_c = gamma * EX2
        post_mean    = post_mean_c + mu
        post_mean2   = post_mean2_c + _const_like(x, 2.0) * mu * post_mean_c + mu * mu
        post_sd      = (post_mean2 - post_mean**2).clamp_min(0.0).sqrt()

        llik = observed_loglik(lf, lg, pi0)

    history = {
        "loglik": hist_loglik,
        "Q": hist_Q,
        "pi0": hist_pi,
        "a": hist_a,
        "mu": hist_mu,
    }
    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi0=float(pi0.item()),
        a=float(a.item()),
        mu=float(mu.item()),
        log_lik=float(llik.item()),
        history=history,
    )
