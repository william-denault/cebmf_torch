import math
from dataclasses import dataclass
import torch
from torch import Tensor
from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,  # log(sqrt(2π))
    logPhi,         # stable log Φ
    my_etruncnorm,  # E[X | a < X < b] for Normal(mean, sd) via truncation
    my_e2truncnorm, # E[X^2 | a < X < b] for Normal(mean, sd) via truncation
)

@dataclass
class EBNMGBResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    pi: float        # slab weight π
    mu: float        # learned μ ≥ 0
    omega: float     # fixed ω (σ = ω μ)
    log_lik: float

def _log_normal_pdf(x: Tensor, mean: Tensor, sd: Tensor) -> Tensor:
    sd = sd.clamp_min(1e-12)
    z = (x - mean) / sd
    return -0.5 * z**2 - torch.log(sd) - _LOG_SQRT_2PI

def ebnm_gb(
    x: Tensor,
    s: Tensor,
    omega: float = 0.2,          # fixed ω (σ = ω μ), typically small
    par_init_mu: float = 1.0,    # μ initialization (on the original scale)
    par_init_pi: float = 0.2,    # π initialization
    max_em: int = 200,
    tol_em: float = 1e-5,
    max_lbfgs: int = 200,
    tol_lbfgs: float = 1e-6,
    eps: float = 1e-12,
) -> EBNMGBResult:
    """
    EBNM with Generalized Binary prior:
      θ ~ (1-π) δ0 + π N_+(μ, σ^2), with σ = ω μ, μ≥0, ω fixed.
    Follows the EM scheme in Supplementary Note (eqs. (16)-(27)). :contentReference[oaicite:1]{index=1}
    """
    device, dtype = x.device, x.dtype
    x = x.to(dtype)
    s = torch.clamp(s.to(dtype), min=1e-6)

    # Initialize hyperparameters
    mu = torch.tensor(float(max(par_init_mu, 1e-6)), device=device, dtype=dtype)
    pi = torch.tensor(float(min(max(par_init_pi, 1e-6), 1-1e-6)), device=device, dtype=dtype)
    omega_t = torch.tensor(float(omega), device=device, dtype=dtype)

    # Precompute spike log-likelihood terms: N(x; 0, s^2)
    lf = _log_normal_pdf(x, torch.zeros_like(x), s)

    # Helper to compute ζ (E-step) and optionally pieces for M-step
    def _E_step(mu_val: Tensor, pi_val: Tensor):
        # σ = ω μ
        sigma = omega_t * mu_val.clamp_min(0.0)

        # slab marginal density from eq. (18):
        # N(x; μ, σ^2 + s^2) * Φ(μ̃/σ̃) / Φ(μ/σ)
        var_sum = s*s + sigma*sigma
        lg0 = _log_normal_pdf(x, mu_val, var_sum.sqrt())

        # σ̃_i^2 and μ̃_i (eqs. (19)-(20) with σ=ω μ)
        denom = 1.0/(sigma*sigma + eps) + 1.0/(s*s)
        sig_tilde2 = 1.0 / denom
        mu_tilde = ( (s*s)*mu_val + (sigma*sigma)*x ) / (s*s + sigma*sigma)

        # log Φ(μ̃/σ̃) − log Φ(μ/σ). Note μ/σ = 1/ω is constant in μ’s optimization. :contentReference[oaicite:2]{index=2}
        log_norm_cdf_ratio = logPhi(mu_tilde / sig_tilde2.sqrt()) - logPhi(torch.tensor(1.0/float(omega), device=device, dtype=dtype))

        lg = lg0 + log_norm_cdf_ratio  # log slab marginal per i

        # ζ_i = posterior slab prob (eq. for E[zi | ...])
        # ζ_i = softmax in log-space: lognum = log π + lg, logden = logaddexp(log(1-π)+lf, log π + lg)
        log_num = torch.log(pi_val.clamp_min(eps)) + lg
        log_denom = torch.logaddexp(torch.log1p(-pi_val).clamp_min(-50) + lf, log_num)
        zeta = torch.exp(log_num - log_denom).clamp(0.0, 1.0)

        return zeta, lg, lf, mu_tilde, sig_tilde2

    # M-step for μ: maximize sum_i ζ_i [ log N(x; μ, ω^2 μ^2 + s^2) + log Φ(μ̃/σ̃) ], no closed form. :contentReference[oaicite:3]{index=3}
    # We optimize an unconstrained η with μ = softplus(η).
    def _optimize_mu(zeta: Tensor, mu_init: Tensor):
        eta = torch.nn.Parameter(torch.log(torch.expm1(mu_init.clamp_min(1e-8))))
        opt = torch.optim.LBFGS([eta],
                                max_iter=max_lbfgs,
                                tolerance_grad=tol_lbfgs,
                                tolerance_change=tol_lbfgs,
                                line_search_fn="strong_wolfe",
                                history_size=20)

        def closure():
            opt.zero_grad(set_to_none=True)
            mu_pos = torch.nn.functional.softplus(eta) + 0.0  # ensure ≥ 0
            sigma = omega_t * mu_pos
            var_sum = s*s + sigma*sigma
            lg0 = _log_normal_pdf(x, mu_pos, var_sum.sqrt())

            # μ̃_i, σ̃_i as functions of μ (via σ)
            denom = 1.0/(sigma*sigma + eps) + 1.0/(s*s)
            sig_tilde2 = 1.0 / denom
            mu_tilde = ( (s*s)*mu_pos + (sigma*sigma)*x ) / (s*s + sigma*sigma)

            obj_terms = lg0 + logPhi(mu_tilde / sig_tilde2.sqrt())  # drop constant −logΦ(1/ω)
            loss = -(zeta * obj_terms).sum()
            loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=1e30)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except RuntimeError:
            # Fallback: small gradient steps if LBFGS fails
            adam = torch.optim.Adam([eta], lr=1e-2)
            for _ in range(200):
                adam.zero_grad(set_to_none=True)
                mu_pos = torch.nn.functional.softplus(eta)
                sigma = omega_t * mu_pos
                var_sum = s*s + sigma*sigma
                lg0 = _log_normal_pdf(x, mu_pos, var_sum.sqrt())
                denom = 1.0/(sigma*sigma + eps) + 1.0/(s*s)
                sig_tilde2 = 1.0 / denom
                mu_tilde = ( (s*s)*mu_pos + (sigma*sigma)*x ) / (s*s + sigma*sigma)
                obj_terms = lg0 + logPhi(mu_tilde / sig_tilde2.sqrt())
                loss = -(zeta * obj_terms).sum()
                loss.backward()
                adam.step()

        with torch.no_grad():
            return torch.nn.functional.softplus(eta).clamp_min(1e-8)

    # ---- EM loop ----
    prev_ll = -float("inf")
    for _ in range(max_em):
        # E-step
        zeta, lg, lf, mu_tilde, sig_tilde2 = _E_step(mu, pi)

        # M-step π (eq. (22)): average ζ
        pi_new = zeta.mean().clamp(1e-8, 1-1e-8)

        # M-step μ: optimize expected complete log-lik (eq. (23); constant −logΦ(1/ω) dropped). :contentReference[oaicite:4]{index=4}
        mu_new = _optimize_mu(zeta, mu)

        # Evaluate marginal log-likelihood with updated params (for convergence check; eq. (18))
        with torch.no_grad():
            sigma = omega_t * mu_new
            var_sum = s*s + sigma*sigma
            lg0 = _log_normal_pdf(x, mu_new, var_sum.sqrt())
            denom = 1.0/(sigma*sigma + eps) + 1.0/(s*s)
            sig_tilde2 = 1.0 / denom
            mu_tilde = ( (s*s)*mu_new + (sigma*sigma)*x ) / (s*s + sigma*sigma)
            log_norm_cdf_ratio = logPhi(mu_tilde / sig_tilde2.sqrt()) - logPhi(torch.tensor(1.0/float(omega), device=device, dtype=dtype))
            lg_marg = lg0 + log_norm_cdf_ratio

            # log ∏ [ (1-π)N(x;0,s^2) + π * slab ]
            ll = torch.logaddexp(torch.log1p(-pi_new) + lf, torch.log(pi_new) + lg_marg).sum().item()

        # Check convergence
        if ll - prev_ll < tol_em:
            pi, mu = pi_new, mu_new
            break
        pi, mu = pi_new, mu_new
        prev_ll = ll

    # ---- Posterior moments (eqs. (25)-(27)) ----
    with torch.no_grad():
        # Final E-step for ζ̂ and slab parts
        zeta, lg, lf, mu_tilde, sig_tilde2 = _E_step(mu, pi)

        # Posterior over θ: (1-ζ̂) δ0 + ζ̂ N_+(μ̃, σ̃²).
        # Compute E[θ], E[θ²] from truncated normal moments on [0, ∞).
        a = torch.full_like(x, 0.0)
        b = torch.full_like(x, float("inf"))
        EX = my_etruncnorm(a, b, mean=mu_tilde, sd=sig_tilde2.sqrt())
        EX2 = my_e2truncnorm(a, b, mean=mu_tilde, sd=sig_tilde2.sqrt())

        post_mean = zeta * EX
        post_mean2 = zeta * EX2
        post_sd = (post_mean2 - post_mean**2).clamp_min(0).sqrt()

        # Final marginal log-likelihood
        sigma = omega_t * mu
        var_sum = s*s + sigma*sigma
        lg0 = _log_normal_pdf(x, mu, var_sum.sqrt())
        log_norm_cdf_ratio = logPhi(mu_tilde / sig_tilde2.sqrt()) - logPhi(torch.tensor(1.0/float(omega), device=device, dtype=dtype))
        lg_marg = lg0 + log_norm_cdf_ratio
        log_lik = torch.logaddexp(torch.log1p(-pi) + lf, torch.log(pi) + lg_marg).sum().item()

    return EBNMGBResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi=float(pi),
        mu=float(mu),
        omega=float(omega),
        log_lik=float(log_lik),
    )
