from dataclasses import dataclass

import torch
from torch import Tensor

from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,  # log(sqrt(2π))
    logPhi,  # stable log Φ
    my_e2truncnorm,  # E[X^2 | a < X < b] for Normal(mean, sd) via truncation
    my_etruncnorm,  # E[X | a < X < b] for Normal(mean, sd) via truncation
)


@dataclass
class EBNMGBResult:
    """Generalized-binary EBNM result. Scalar fields are 0-d tensors on the
    input device — no per-call host syncs."""

    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    pi_slab: Tensor  # slab weight π (= 1 - pi_null)
    mode: Tensor  # learned μ (may be any sign; σ = ω·|μ|)
    scale: Tensor  # fixed ω (σ = ω μ)
    log_lik: Tensor


def _log_normal_pdf(x: Tensor, mean: Tensor, sd: Tensor) -> Tensor:
    sd = sd.clamp_min(1e-12)
    z = (x - mean) / sd
    return -0.5 * z**2 - torch.log(sd) - _LOG_SQRT_2PI


def _gb_default_mu_init(x: Tensor, s: Tensor) -> float:
    """Data-driven default for ``par_init_mu``, matching R's
    ``ebnm_generalized_binary_defaults``:

        if any x > 0: mu_init = mean(x[x > 0])
        else:         mu_init = min(s) / 10

    This is just a *starting point* for the LBFGS optimisation — μ itself
    is unconstrained, so the init only sets a sensible direction.
    """
    positive_mask = x > 0
    if bool(positive_mask.any().item()):
        return float(x[positive_mask].mean().item())
    return float(s.min().item()) / 10.0


def ebnm_gb(
    x: Tensor,
    s: Tensor,
    omega: float = 0.1,
    par_init_mu: float | None = None,
    par_init_pi: float | None = None,
    wlist: tuple[float, ...] = (1e-5, 1.0),
    max_em: int = 200,
    tol_em: float = 1e-5,
    max_lbfgs: int = 200,
    tol_lbfgs: float = 1e-6,
    eps: float = 1e-12,
) -> EBNMGBResult:
    """
    EBNM with Generalized Binary prior:
      θ ~ (1-π) δ0 + π N_+(μ, σ^2), with σ = ω·|μ|, ω fixed, μ ∈ ℝ.

    Parameters
    ----------
    omega : float, optional
        Coefficient of variation σ/|μ| for the slab. Default ``0.1`` matches
        Stephens-lab ``ebnm`` (``ebnm_generalized_binary(scale = 0.1)``).
    par_init_mu, par_init_pi : float or None, optional
        Initial values for μ and π. ``None`` (default) uses R's data-driven
        defaults: ``par_init_mu = mean(x[x>0])`` (or ``min(s)/10`` if no
        positive entries), and ``par_init_pi`` set to the midpoint of the
        first ``wlist`` interval. μ itself is *unconstrained* in the LBFGS
        step — these are only starting points that nudge the optimiser in
        a sensible direction.
    wlist : tuple of float, optional
        Boundary points for the multi-start EM. With ``len(wlist) == k+1``,
        EM is run once from the centre of each ``[wlist[i], wlist[i+1]]``
        interval and the result with the highest marginal log-likelihood
        is kept. Within each run, π is clamped to its interval at every
        M-step. Default ``(1e-5, 1.0)`` reproduces R's behaviour: a single
        EM run with π ∈ ``[1e-5, 1]``.

    Notes
    -----
    Closely mirrors ``ebnm::ebnm_generalized_binary`` (Liu / Willwerscheid)
    on the E- and M-step math (E-step ζ, M-step π = mean(ζ), M-step μ
    objective). Unlike R, μ is *not* bounded to a positive range — the
    optimiser sees an unconstrained scalar and σ = ω·|μ| keeps the slab
    standard deviation non-negative regardless of μ's sign. This matters
    when the EM machinery wants to push μ slightly into the negative
    half-line; the slab is still a valid truncated normal on ``[0, ∞)``.
    """
    device, dtype = x.device, x.dtype
    x = x.to(dtype)
    s = torch.clamp(s.to(dtype), min=1e-6)

    # Data-driven μ init (R's default). μ itself is unconstrained.
    if par_init_mu is None:
        mu_init_val = _gb_default_mu_init(x, s)
    else:
        mu_init_val = float(par_init_mu)

    # Validate wlist
    if len(wlist) < 2:
        raise ValueError(f"wlist must have at least 2 elements, got {wlist!r}")
    if any(wlist[i] >= wlist[i + 1] for i in range(len(wlist) - 1)):
        raise ValueError(f"wlist must be strictly increasing, got {wlist!r}")

    omega_t = torch.tensor(float(omega), device=device, dtype=dtype)
    log_phi_const_t = logPhi(torch.tensor(1.0 / float(omega), device=device, dtype=dtype))

    # Spike log-likelihood N(x; 0, s^2) — independent of (μ, π), so hoisted.
    lf = _log_normal_pdf(x, torch.zeros_like(x), s)

    # ---- E-step: posterior responsibility ζ for the slab component ----
    def _E_step(mu_val: Tensor, pi_val: Tensor):
        # σ = ω·|μ| keeps σ ≥ 0 for any μ ∈ ℝ. (R uses `abs(mu*scale)`.)
        sigma = omega_t * mu_val.abs()
        var_sum = s * s + sigma * sigma
        lg0 = _log_normal_pdf(x, mu_val, var_sum.sqrt())

        denom = 1.0 / (sigma * sigma + eps) + 1.0 / (s * s)
        sig_tilde2 = 1.0 / denom
        mu_tilde = ((s * s) * mu_val + (sigma * sigma) * x) / (s * s + sigma * sigma)

        log_norm_cdf_ratio = logPhi(mu_tilde / sig_tilde2.sqrt()) - log_phi_const_t
        lg = lg0 + log_norm_cdf_ratio

        log_num = torch.log(pi_val.clamp_min(eps)) + lg
        log_denom = torch.logaddexp(torch.log1p(-pi_val).clamp_min(-50) + lf, log_num)
        zeta = torch.exp(log_num - log_denom).clamp(0.0, 1.0)

        return zeta, lg, mu_tilde, sig_tilde2

    # ---- M-step μ via unconstrained LBFGS (η = μ directly; no bounds) ----
    def _optimize_mu(zeta: Tensor, mu_warmstart: Tensor) -> Tensor:
        eta = torch.nn.Parameter(mu_warmstart.detach().clone())

        opt = torch.optim.LBFGS(
            [eta],
            max_iter=max_lbfgs,
            tolerance_grad=tol_lbfgs,
            tolerance_change=tol_lbfgs,
            line_search_fn="strong_wolfe",
            history_size=20,
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            mu_curr = eta  # unconstrained
            sigma = omega_t * mu_curr.abs()
            var_sum = s * s + sigma * sigma
            lg0 = _log_normal_pdf(x, mu_curr, var_sum.sqrt())

            denom = 1.0 / (sigma * sigma + eps) + 1.0 / (s * s)
            sig_tilde2 = 1.0 / denom
            mu_tilde = ((s * s) * mu_curr + (sigma * sigma) * x) / (s * s + sigma * sigma)

            obj_terms = lg0 + logPhi(mu_tilde / sig_tilde2.sqrt())  # drop const −logΦ(1/ω)
            loss = -(zeta * obj_terms).sum()
            loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=1e30)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except RuntimeError:
            adam = torch.optim.Adam([eta], lr=1e-2)
            for _ in range(200):
                adam.zero_grad(set_to_none=True)
                mu_curr = eta
                sigma = omega_t * mu_curr.abs()
                var_sum = s * s + sigma * sigma
                lg0 = _log_normal_pdf(x, mu_curr, var_sum.sqrt())
                denom = 1.0 / (sigma * sigma + eps) + 1.0 / (s * s)
                sig_tilde2 = 1.0 / denom
                mu_tilde = ((s * s) * mu_curr + (sigma * sigma) * x) / (s * s + sigma * sigma)
                obj_terms = lg0 + logPhi(mu_tilde / sig_tilde2.sqrt())
                loss = -(zeta * obj_terms).sum()
                loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=1e30)
                loss.backward()
                adam.step()

        with torch.no_grad():
            return eta.detach().clone()

    # ---- Marginal log-likelihood under (μ, π) for convergence / scoring ----
    def _marginal_loglik(mu_val: Tensor, pi_val: Tensor) -> Tensor:
        sigma = omega_t * mu_val.abs()
        var_sum = s * s + sigma * sigma
        lg0 = _log_normal_pdf(x, mu_val, var_sum.sqrt())
        denom = 1.0 / (sigma * sigma + eps) + 1.0 / (s * s)
        sig_tilde2 = 1.0 / denom
        mu_tilde = ((s * s) * mu_val + (sigma * sigma) * x) / (s * s + sigma * sigma)
        log_norm_cdf_ratio = logPhi(mu_tilde / sig_tilde2.sqrt()) - log_phi_const_t
        lg_marg = lg0 + log_norm_cdf_ratio
        return torch.logaddexp(torch.log1p(-pi_val) + lf, torch.log(pi_val) + lg_marg).sum()

    # ---- One EM run constrained to a [w_lo, w_hi] interval ----
    def _run_em_in_interval(w_lo_val: float, w_hi_val: float) -> tuple[Tensor, Tensor, Tensor]:
        # π start: user override (if supplied) else interval midpoint
        if par_init_pi is None:
            w_start = (w_lo_val + w_hi_val) / 2.0
        else:
            w_start = float(par_init_pi)
        w_start = min(max(w_start, w_lo_val), w_hi_val)
        # Keep π strictly inside (0, 1) for log1p / logaddexp stability.
        w_start = min(max(w_start, 1e-8), 1.0 - 1e-8)

        mu_local = torch.tensor(mu_init_val, device=device, dtype=dtype)
        pi_local = torch.tensor(w_start, device=device, dtype=dtype)
        # Hard floor/ceiling on π in this interval (also inside (0, 1)).
        w_lo_t = torch.tensor(max(w_lo_val, 1e-8), device=device, dtype=dtype)
        w_hi_t = torch.tensor(min(w_hi_val, 1.0 - 1e-8), device=device, dtype=dtype)

        check_every = 5
        prev_ll = -float("inf")
        ll_local = torch.tensor(prev_ll, device=device, dtype=dtype)
        for it in range(max_em):
            zeta, _, _, _ = _E_step(mu_local, pi_local)

            # M-step π: clamp the mean of ζ into the current wlist interval.
            pi_new = zeta.mean().clamp(min=w_lo_t, max=w_hi_t)

            # M-step μ: unconstrained LBFGS warm-started from current μ.
            mu_new = _optimize_mu(zeta, mu_local)

            with torch.no_grad():
                ll_local = _marginal_loglik(mu_new, pi_new)

            pi_local, mu_local = pi_new, mu_new

            if (it + 1) % check_every == 0:
                ll_val = ll_local.item()
                if ll_val - prev_ll < tol_em:
                    break
                prev_ll = ll_val

        return pi_local, mu_local, ll_local

    # ---- Multi-start over wlist intervals; pick best by marginal log-lik ----
    best_ll_val = -float("inf")
    best_pi: Tensor | None = None
    best_mu: Tensor | None = None
    best_ll_t: Tensor | None = None
    for k in range(len(wlist) - 1):
        w_lo, w_hi = float(wlist[k]), float(wlist[k + 1])
        pi_k, mu_k, ll_k_t = _run_em_in_interval(w_lo, w_hi)
        ll_k = ll_k_t.item()
        if ll_k > best_ll_val:
            best_ll_val = ll_k
            best_pi = pi_k
            best_mu = mu_k
            best_ll_t = ll_k_t

    assert best_pi is not None and best_mu is not None and best_ll_t is not None
    pi, mu = best_pi, best_mu

    # ---- Posterior moments under the best (π, μ) ----
    with torch.no_grad():
        zeta, _, mu_tilde, sig_tilde2 = _E_step(mu, pi)

        a = torch.full_like(x, 0.0)
        b = torch.full_like(x, float("inf"))
        EX = my_etruncnorm(a, b, mean=mu_tilde, sd=sig_tilde2.sqrt())
        EX2 = my_e2truncnorm(a, b, mean=mu_tilde, sd=sig_tilde2.sqrt())

        post_mean = zeta * EX
        post_mean2 = zeta * EX2
        post_sd = (post_mean2 - post_mean**2).clamp_min(0).sqrt()

        log_lik = best_ll_t

    scale_t = torch.as_tensor(1.0 / (omega + 1e-8), device=device, dtype=dtype)
    return EBNMGBResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi_slab=pi.detach(),
        mode=mu.detach(),
        scale=scale_t,
        log_lik=log_lik.detach(),
    )
