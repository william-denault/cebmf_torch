"""
Point-Laplace empirical-Bayes Normal-Means solver (PyTorch).

Numerically-equivalent port of the R `stephenslab/ebnm` solver
(`R/point_laplace.R`): same marginal log-likelihood and posterior moments
(via ``wpost_laplace`` and ``lambda``), same boundary check, and same
handling of ``s = inf``.

The optimiser here is L-BFGS with autograd-derived gradients instead of R's
``nlm`` with analytic gradient + Hessian.

Internal parameterisation (differs from R, retained for stability inside
cEBMF, where this solver is called repeatedly with warm-starts and the
synthetic data ``lhat`` can have arbitrarily small SEs):

- ``alpha = logit(pi_slab)`` (unbounded; pi_slab in (0,1))
- ``a`` is mapped into a finite interval via a smooth sigmoid
  (``a = a_lo + (a_hi - a_lo) * sigmoid(a_logit)``), so L-BFGS cannot push
  the slab into the degenerate ``a → ∞`` region where ``0.5*(s*a)^2``
  blows up in float32 (cEBMF casts internally to float32) and the slab is
  numerically indistinguishable from the spike. The public ``par_init``
  API still takes ``beta = log(a)`` for backward compatibility; it is
  squashed into the bounded representation internally.
- ``mu`` (unbounded)

The function signature is kept compatible with previous cebmf_torch callers
— the legacy kwargs ``loga_l2``, ``pen_pi0``, ``tresh_pi0``,
``use_adam_warmstart``, and the related Adam knobs are accepted but
ignored (they have no equivalents in R `ebnm` and biased the optimum).
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from cebmf_torch.utils.maths import (
    _LOG_SQRT_2PI,
    logPhi,
    my_e2truncnorm,
    my_etruncnorm,
)


def _const_like(x: Tensor, val) -> Tensor:
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)


# ----------------------------------------------------------------------
# R port: logg_laplace (R/point_laplace.R line 382)
# ----------------------------------------------------------------------
def logg_laplace(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """log of g = Laplace(rate=a) ⊗ N(0, s²), evaluated at x.

    Port of R::``logg_laplace`` in ``stephenslab/ebnm``:

        lg1 = -a*x + log Φ((x - s²a)/s)
        lg2 =  a*x + log Q((x + s²a)/s)         # Q = upper tail = Φ(-·)
        lg  = log(a/2) + s²a²/2 + logsumexp(lg1, lg2)
    """
    sa = s * a
    z_right = x / s - sa
    z_left = x / s + sa
    lg1 = -a * x + logPhi(z_right)
    lg2 = a * x + logPhi(-z_left)
    return torch.log(a / 2.0) + 0.5 * (sa * sa) + torch.logaddexp(lg1, lg2)


# Backward-compatible alias for callers that imported the previous name.
def logg_laplace_convolved_with_normal(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """Alias for :func:`logg_laplace` — preserved for API compatibility."""
    return logg_laplace(x, s, a)


# ----------------------------------------------------------------------
# R port: lambda (R/point_laplace.R line 394) — posterior P(theta>0 | non-null)
# ----------------------------------------------------------------------
def _lambda(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """Posterior probability of being positive given a non-null effect.

    R::``lambda``:

        lm1 = -a*x + log Φ(x/s - s*a)
        lm2 =  a*x + log Q(x/s + s*a)
        lm  = 1 / (1 + exp(lm2 - lm1))

    Implemented as ``sigmoid(lm1 - lm2)`` for numerical stability.
    """
    z_right = x / s - s * a
    z_left = x / s + s * a
    lm1 = -a * x + logPhi(z_right)
    lm2 = a * x + logPhi(-z_left)
    return torch.sigmoid(lm1 - lm2)


# ----------------------------------------------------------------------
# R port: wpost_laplace (R/point_laplace.R line 362)
# ----------------------------------------------------------------------
def _wpost_laplace(lf: Tensor, lg: Tensor, w: Tensor) -> Tensor:
    """Posterior weight on the slab branch.

    R::``wpost_laplace`` returns ``w / (w + (1-w) exp(lf - lg))``. Implemented
    here as ``sigmoid(log(w/(1-w)) + (lg - lf))`` to stay on the log-scale
    until the very last step.
    """
    log_odds_w = torch.log(w) - torch.log1p(-w)
    return torch.sigmoid(log_odds_w + (lg - lf))


@dataclass
class EBNMLaplaceResult:
    """Container for point-Laplace EBNM results.

    All scalar fields are 0-d tensors on the input device so the cEBMF ELBO
    accumulator can fold them in without per-call host syncs. Call
    ``float(field)`` at your own boundary if you need a Python scalar.

    Attributes
    ----------
    post_mean, post_mean2, post_sd : Tensor
        Posterior first/second/SD per observation.
    pi_slab : Tensor (0-d)
        Mixture weight of the Laplace branch (slab) — ``1 - pi_null``.
    a : Tensor (0-d)
        Laplace rate (``= 1/scale``).
    mu : Tensor (0-d)
        Estimated mode.
    log_lik : Tensor (0-d)
        Pure marginal log-likelihood (no penalties).
    """

    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    pi_slab: Tensor
    a: Tensor
    mu: Tensor
    log_lik: Tensor


def _initial_beta(x: Tensor, eps: Tensor) -> float:
    """R `pl_initpar`: ``beta = -0.5 * log(mean(x^2)/2)`` when scale='estimate'."""
    with torch.no_grad():
        mean_xsq = (x.detach().double() ** 2).mean().clamp_min(eps.double())
        return float(-0.5 * torch.log(mean_xsq / 2.0).item())


def _initial_mu(x: Tensor, fix_mu: bool) -> float:
    """R `pl_initpar`: when mode='estimate', ``mu = mean(x)``; else mu = mode (here 0)."""
    if fix_mu:
        return 0.0
    with torch.no_grad():
        return float(x.detach().double().mean().item())


def ebnm_point_laplace(
    x: Tensor,
    s: Tensor,
    par_init=None,  # (alpha, beta, mu) = (logit(pi_slab), log(a), mu); None → defaults
    fix_par=(False, False, True),  # [w, a, mu]; mu fixed by default
    max_iter: int = 20,
    tol: float = 1e-8,
    eps: float = 1e-12,
    # Smooth bounds on `a` to keep the slab away from the degenerate
    # ``a → ∞`` region. These bounds are wide enough not to bias R-style
    # one-shot use (where the optimum is order 1), but tight enough to keep
    # ``0.5*(s*a)^2`` representable in float32 inside cEBMF stress tests.
    a_bounds: tuple[float, float] = (1e-2, 1e2),
    # ---- legacy kwargs kept for API compatibility (the R reference has no
    # equivalents; they are intentionally ignored to avoid biasing the fit).
    loga_l2: float = 0.5,
    tresh_pi0: float = 1e-3,
    pen_pi0: float = 0.0,
    use_adam_warmstart: bool = False,
    adam_steps: int = 0,
    adam_lr: float = 0.0,
    weight_decay: float = 0.0,
) -> EBNMLaplaceResult:
    """Empirical-Bayes Normal-Means under a point-Laplace prior.

    The prior on θ is

        (1 - pi_slab) δ_μ + pi_slab · Laplace(μ, scale = 1/a)

    The ``par_init`` tuple uses R's ``ebnm`` convention so warm-starts from
    R or from previous cebmf iterations remain valid:

    - ``alpha = logit(pi_slab)``
    - ``beta  = log(a)``     (squashed into ``a_bounds`` internally)
    - ``mu``                  (unbounded)

    Internally, ``a`` is constrained to the interval ``a_bounds = (a_lo, a_hi)``
    via a smooth sigmoid map (same trick as ``ebnm_point_exp``). The defaults
    ``(1e-3, 1e3)`` are wide enough to leave the R-port optimum interior for
    typical use, while excluding the degenerate ``a → ∞`` region that
    destabilises L-BFGS inside cEBMF when the slab should be empty.

    The marginal log-likelihood, posterior moments, and boundary check are
    line-for-line ports of ``stephenslab/ebnm`` ``R/point_laplace.R``
    (``pl_nllik``, ``pl_summres_untransformed``, ``pl_postcomp``).
    """
    # Silence ruff / lint warnings about deliberately-unused legacy kwargs.
    del loga_l2, tresh_pi0, pen_pi0
    del use_adam_warmstart, adam_steps, adam_lr, weight_decay

    device, dtype = x.device, x.dtype
    x = torch.as_tensor(x, device=device, dtype=dtype)
    s = torch.as_tensor(s, device=device, dtype=dtype).clamp(min=_const_like(x, 1e-6))

    eps_t = _const_like(x, eps)
    log_sqrt_2pi = _const_like(x, _LOG_SQRT_2PI)

    fix_pi0 = bool(fix_par[0])
    fix_a = bool(fix_par[1])
    fix_mu = bool(fix_par[2])

    # ---- Bounded `a` parameterisation ----------------------------------------
    # `a` lives in (a_lo, a_hi) via a smooth sigmoid map. Internally we
    # optimise `a_logit ∈ ℝ` with ``a = a_lo + (a_hi - a_lo) * sigmoid(a_logit)``.
    # This mirrors `ebnm_point_exp` and is what keeps the slab branch
    # numerically stable when the data has full mass at 0 — the L-BFGS
    # search can no longer drift `a` to extreme values where
    # ``0.5*(s*a)^2`` underflows / overflows in float32 and the slab is
    # numerically indistinguishable from the spike.
    a_lo, a_hi = float(a_bounds[0]), float(a_bounds[1])
    if not (0.0 < a_lo < a_hi):
        raise ValueError(f"a_bounds must satisfy 0 < a_lo < a_hi, got {a_bounds!r}")
    a_lo_t = _const_like(x, a_lo)
    a_hi_t = _const_like(x, a_hi)

    def _loga_to_logit(log_a_val: float) -> float:
        """Map a user-facing ``log(a)`` to the internal logit param."""
        a_val = float(min(max(math.exp(float(log_a_val)), a_lo), a_hi))
        r = (a_val - a_lo) / (a_hi - a_lo)
        r = min(max(r, 1e-8), 1.0 - 1e-8)
        return math.log(r) - math.log1p(-r)

    # ---- Defaults match R `pl_initpar` ---------------------------------------
    if par_init is None:
        alpha_init = 0.5  # pi_slab = 0.5  (R: alpha=0 when pointmass)
        beta_init = _initial_beta(x, eps_t)  # R: -0.5*log(mean(x^2)/2)
        mu_init = _initial_mu(x, fix_mu)  # R: mean(x) if estimating, else 0
        par_init = (alpha_init, beta_init, mu_init)

    alpha = torch.nn.Parameter(
        torch.as_tensor(par_init[0], dtype=dtype, device=device),
        requires_grad=not fix_pi0,
    )
    # par_init[1] is interpreted as `log(a)` for backward compatibility with
    # callers (and the R-port semantics); we squash it into the bounded
    # logit representation here.
    a_logit = torch.nn.Parameter(
        torch.as_tensor(_loga_to_logit(par_init[1]), dtype=dtype, device=device),
        requires_grad=not fix_a,
    )
    mu = torch.nn.Parameter(
        torch.as_tensor(par_init[2], dtype=dtype, device=device),
        requires_grad=not fix_mu,
    )

    params = [p for p in (alpha, a_logit, mu) if p.requires_grad]

    def _a_from_logit() -> Tensor:
        return a_lo_t + (a_hi_t - a_lo_t) * torch.sigmoid(a_logit)

    # ---- Negative log marginal likelihood (R `pl_nllik`, line 122) ----------
    def _nllik() -> Tensor:
        # R: w = 1 - 1/(1+exp(alpha)) = sigmoid(alpha); a = exp(beta).
        # Here `a` is sigmoid-bounded — same objective, restricted to a
        # finite interval that excludes numerically-degenerate values.
        w = torch.sigmoid(alpha).clamp(eps_t, 1.0 - eps_t)
        a = _a_from_logit()
        xc = x - mu

        # Spike: lf = log N(xc | 0, s²)
        lf = -0.5 * (xc / s).pow(2) - torch.log(s) - log_sqrt_2pi

        # Slab: lg = log(Laplace ⊗ Normal). R uses the lgleft / lgright
        # split; logsumexp gives the same value as the helper.
        sa = s * a
        xleft = xc / s + sa
        xright = xc / s - sa
        common = torch.log(a / 2.0) + 0.5 * (sa * sa)
        lgleft = common + a * xc + logPhi(-xleft)  # log Q(xleft)
        lgright = common - a * xc + logPhi(xright)  # log Φ(xright)
        lg = torch.logaddexp(lgleft, lgright)

        llik_i = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg)
        return -llik_i.sum()

    # ---- L-BFGS optimisation -------------------------------------------------
    if params:
        opt = torch.optim.LBFGS(
            params,
            max_iter=max_iter,
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
            history_size=20,
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = _nllik()
            # Strong-Wolfe line search dies on non-finite probes; route those
            # to a large finite penalty so the search can back off cleanly.
            loss = torch.nan_to_num(
                loss,
                nan=_const_like(loss, 1e30),
                posinf=_const_like(loss, 1e30),
                neginf=_const_like(loss, 1e30),
            )
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except RuntimeError:
            # If line search fails, freeze 'a' and retry with the remaining
            # free parameters — same fallback as the previous implementation.
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

    # ---- Posterior summaries (R `pl_summres_untransformed`, line 324) -------
    with torch.no_grad():
        w = torch.sigmoid(alpha).clamp(eps_t, 1.0 - eps_t)
        a = _a_from_logit()
        xc = x - mu

        # Densities
        lf = -0.5 * (xc / s).pow(2) - torch.log(s) - log_sqrt_2pi
        lg = logg_laplace(xc, s, a)

        # Marginal log-lik (R `pl_nllik`'s positive of the converged objective)
        llik = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg).sum()

        # Posterior weight on slab (R `wpost_laplace`)
        wpost = _wpost_laplace(lf, lg, w)

        # Sign-mixture inside slab (R `lambda`)
        lam = _lambda(xc, s, a)

        # Truncated-normal moments — R uses `ashr::my_etruncnorm`.
        m_pos = xc - (s * s) * a  # mean of right-truncated normal (>= 0)
        m_neg = xc + (s * s) * a  # mean of left-truncated normal  (<= 0)
        zero = torch.zeros_like(x)
        inf_p = torch.full_like(x, float("inf"))
        inf_n = torch.full_like(x, -float("inf"))

        EX_pos = my_etruncnorm(zero, inf_p, mean=m_pos, sd=s)
        EX2_pos = my_e2truncnorm(zero, inf_p, mean=m_pos, sd=s)
        EX_neg = my_etruncnorm(inf_n, zero, mean=m_neg, sd=s)
        EX2_neg = my_e2truncnorm(inf_n, zero, mean=m_neg, sd=s)

        # Cast back to caller dtype — my_e[2]truncnorm may up-cast internally.
        EX_pos = EX_pos.to(dtype)
        EX2_pos = EX2_pos.to(dtype)
        EX_neg = EX_neg.to(dtype)
        EX2_neg = EX2_neg.to(dtype)

        post_mean_c = wpost * (lam * EX_pos + (1.0 - lam) * EX_neg)
        post_mean2_c = wpost * (lam * EX2_pos + (1.0 - lam) * EX2_neg)

        # R: handle s = inf (completely uninformative observation).
        # post_mean -> 0, post_mean2 -> 2*w/a²  (centred moments).
        s_is_inf = torch.isinf(s)
        if bool(torch.any(s_is_inf)):
            # `torch.where` broadcasts 0-d tensors automatically — no
            # explicit `expand_as` needed (and `expand_as` from a 0-d source
            # is brittle across torch versions).
            post_mean_c = torch.where(s_is_inf, torch.zeros_like(post_mean_c), post_mean_c)
            post_mean2_c = torch.where(s_is_inf, 2.0 * w / a.pow(2), post_mean2_c)

        # Var ≥ 0 floor — matches R `pmax(post$mean2, post$mean^2)`.
        post_mean2_c = torch.maximum(post_mean2_c, post_mean_c.pow(2))
        post_sd = (post_mean2_c - post_mean_c.pow(2)).clamp_min(0.0).sqrt()

        # Re-apply mu-shift (R: post$mean2 + mu² + 2 mu post$mean; post$mean + mu)
        post_mean = post_mean_c + mu
        post_mean2 = post_mean2_c + mu * mu + 2.0 * mu * post_mean_c

        # ---- Boundary check (R `pl_postcomp`, line 291) ---------------------
        # R: only triggered when !fix_pi0 && fix_mu. Uses `par_init$mu`
        # (the initial mu) — not the optimised mu — for the spike fallback.
        if (not fix_pi0) and fix_mu:
            mu_init_t = _const_like(x, par_init[2])
            lf_init = (
                -0.5 * ((x - mu_init_t) / s).pow(2) - torch.log(s) - log_sqrt_2pi
            )
            # R: sum only over finite x (matches `sum(is.finite(x))`).
            finite_x = torch.isfinite(x)
            llik_so = torch.where(finite_x, lf_init, torch.zeros_like(lf_init)).sum()

            llik_finite = torch.isfinite(llik)
            prefer_spike = (~llik_finite) | (llik_so > llik)
            if bool(prefer_spike):
                # Spike-fallback: report the spike-only posterior & log-lik
                # (so the ELBO/KL accumulator in cEBMF sees the validated
                # value), but DO NOT reset ``a`` to 1 or ``mu`` to
                # ``mu_init`` — those are the parameters we'll warm-start
                # the next iteration's L-BFGS from, and resetting them
                # discards a perfectly good search direction. ``point_exp``
                # already does it this way; matching it here is what
                # restores ELBO monotonicity in the full-mass-at-0 regime,
                # where the slab branch is genuinely flat in ``a`` and
                # L-BFGS may converge to a slab-likelihood that's within
                # ULP of ``llik_so`` (and so flips above/below it from
                # iteration to iteration).
                w = eps_t
                # Posterior is the point mass at the current ``mu``
                # (mu_init_t equals mu when fix_mu=True, which is the cEBMF
                # default; when fix_mu=False, mu_init is the principled
                # spike location used in the boundary likelihood ``llik_so``).
                post_mean_c0 = torch.zeros_like(x)
                post_mean = post_mean_c0 + mu_init_t
                post_mean2 = post_mean_c0 + mu_init_t * mu_init_t
                post_sd = torch.zeros_like(x)
                # Keep the L-BFGS-found ``mu`` so warm-starts stay coherent
                # in the (rare) free-mu case. With fix_mu=True (cEBMF
                # default) ``mu.detach() == mu_init_t`` so this is a no-op.
                mu_out = mu.detach()
                llik = llik_so
            else:
                mu_out = mu.detach()
        else:
            mu_out = mu.detach()

        # Per-element NaN guard — float32 underflow inside my_etruncnorm can
        # still produce a few non-finite posterior moments when the aggregate
        # llik is healthy. Route those entries to the (validated) spike values.
        # `torch.where` broadcasts 0-d ``mu_out`` automatically.
        bad = ~torch.isfinite(post_mean) | ~torch.isfinite(post_mean2)
        if bool(torch.any(bad)):
            post_mean = torch.where(bad, mu_out, post_mean)
            post_mean2 = torch.where(bad, mu_out * mu_out, post_mean2)
            post_sd = (post_mean2 - post_mean.pow(2)).clamp_min(0.0).sqrt()

    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi_slab=w.detach(),
        a=a.detach(),
        mu=mu_out.detach(),
        log_lik=llik.detach(),
    )



__all__ = [
    "EBNMLaplaceResult",
    "ebnm_point_laplace",
    "logg_laplace",
    "logg_laplace_convolved_with_normal",
    "math",
]
