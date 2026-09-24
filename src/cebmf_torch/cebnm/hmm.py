"""Adaptive-shrinkage HMM normal means, ported from fSuSiE's exact EM solver.

Reference: fsusieR/R/hmm_routines.R, version 3.0.0-variational, commit
9f0fd55. This module implements the exact, equally spaced, single-sequence
model, including automatic grids, center learning and checked state pruning.
The optional R variational approximation, BIC gate and step decoder are not
part of this EBNM interface. See docs/source/hmm_priors.rst.

Adapted from fsusieR (BSD-3-Clause), copyright 2023-2025 William R.P.
Denault, Peter Carbonetto, Gao Wang and Matthew Stephens. See
THIRD_PARTY_NOTICES.md for the license.
"""

import math
from types import SimpleNamespace
from warnings import warn

import torch
from torch import Tensor


def _normalize(x: Tensor, fallback: Tensor) -> Tensor:
    mass = x.sum(-1, keepdim=True)
    return torch.where(mass > 0, x / torch.where(mass > 0, mass, 1.0), fallback)


def _objective(loglik, transition, init, rho, penalty, estimate_init, null_state):
    """Dirichlet log penalty on zero-state probabilities, as in fSuSiE."""
    if penalty == 1:
        return loglik
    log_null = transition[:, 0].log().sum()
    if estimate_init:
        log_null = log_null + init[0].log()
    if null_state == "adaptive":
        log_null = log_null + rho[0, 0].log()
    return loglik + (penalty - 1) * log_null


def _component_log_emission(y: Tensor, se: Tensor, mu: Tensor, sd: Tensor, positive: bool) -> Tensor:
    """Shape (positions, states, components); zero scales are exact atoms."""
    y, se, mu, sd = y[:, None, None], se[:, None, None], mu[None, :, None], sd[None, None, :]
    variance = se.square() + sd.square()
    ans = -0.5 * (math.log(2 * math.pi) + variance.log() + (y - mu).square() / variance)
    if positive:
        # Avoid division by zero even in the unused point-mass branch.
        safe_sd = torch.where(sd > 0, sd, torch.ones_like(sd))
        post_mu = (se.square() * mu + sd.square() * y) / variance
        post_sd = safe_sd * se / variance.sqrt()
        correction = torch.special.log_ndtr(post_mu / post_sd) - torch.special.log_ndtr(mu / safe_sd)
        ans = ans + torch.where(sd > 0, correction, 0.0)
    return ans


@torch.jit.script
def _scaled_smoother(log_emission: Tensor, transition: Tensor, init: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """fSuSiE's scaled forward/backward recursions and BLAS count update."""
    shift = log_emission.max(dim=1).values
    emission = (log_emission - shift[:, None]).exp()
    n = emission.size(0)
    alpha = torch.empty_like(emission)
    beta = torch.ones_like(emission)
    scale = torch.empty_like(shift)
    forward = init * emission[0]
    scale[0] = forward.sum()
    alpha[0] = forward / scale[0]
    for t in range(1, n):
        forward = (alpha[t - 1] @ transition) * emission[t]
        scale[t] = forward.sum()
        alpha[t] = forward / scale[t]
    for t in range(n - 2, -1, -1):
        beta[t] = (transition @ (emission[t + 1] * beta[t + 1])) / scale[t + 1]
    gamma = alpha * beta
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    left = alpha[:-1]
    right = emission[1:] * beta[1:]
    right = right / ((left @ transition) * right).sum(dim=1, keepdim=True)
    counts = transition * (left.T @ right)
    return (shift + scale.log()).sum(), gamma, counts


@torch.jit.script
def _log_smoother(log_emission: Tensor, transition: Tensor, init: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Log-domain fallback for unreachable maxima / scaled underflow."""
    n = log_emission.size(0)
    log_a = transition.log()
    alpha = torch.empty_like(log_emission)
    beta = torch.zeros_like(log_emission)
    alpha[0] = init.log() + log_emission[0]
    for t in range(1, n):
        alpha[t] = log_emission[t] + torch.logsumexp(alpha[t - 1, :, None] + log_a, dim=0)
    for t in range(n - 2, -1, -1):
        beta[t] = torch.logsumexp(log_a + (log_emission[t + 1] + beta[t + 1])[None, :], dim=1)
    gamma = torch.softmax(alpha + beta, dim=1)
    counts = torch.zeros_like(transition)
    for t in range(n - 1):
        xi = alpha[t, :, None] + log_a + (log_emission[t + 1] + beta[t + 1])[None, :]
        counts += (xi - torch.logsumexp(xi.flatten(), dim=0)).exp()
    return torch.logsumexp(alpha[-1], dim=0), gamma, counts


def _smooth(log_emission: Tensor, transition: Tensor, init: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    result = _scaled_smoother(log_emission, transition, init)
    if not all(torch.isfinite(x).all() for x in result):
        result = _log_smoother(log_emission, transition, init)
    if not all(torch.isfinite(x).all() for x in result):
        raise ValueError("The HMM assigns zero probability to the observed sequence.")
    return result


def _sigma_grid(y: Tensor, se: Tensor) -> Tensor:
    minimum = float(se.min()) / 10
    signal_var = float((y.square() - se.square()).max())
    maximum = 2 * math.sqrt(signal_var) if signal_var > 0 else 8 * minimum
    # fSuSiE's formula can give maximum < minimum for near-null signals.
    maximum = max(minimum, maximum)
    steps = math.ceil(math.log(maximum / minimum) / math.log(math.sqrt(2)))
    positive = minimum * math.sqrt(2) ** torch.arange(steps + 1, device=y.device, dtype=y.dtype)
    positive[-1] = maximum
    return torch.cat((y.new_zeros(1), positive)).unique(sorted=True)


def _mean_grid(y: Tensor, half_grid: int, shape: float, expansion: float, maximum, positive: bool) -> Tensor:
    maximum = float(y.abs().max()) if maximum is None else maximum
    if maximum <= 0:
        maximum = 1.0
    grid = torch.linspace(0, 1, half_grid, device=y.device, dtype=y.dtype).pow(1 / shape) * expansion * maximum
    return grid if positive else torch.cat((grid, -grid[1:]))


def _screen(y: Tensor, se: Tensor, mu: Tensor, min_count: float, min_fraction: float) -> Tensor:
    counts = torch.zeros_like(mu)
    for first in range(0, y.numel(), 5000):
        score = -0.5 * ((y[first : first + 5000, None] - mu) / se[first : first + 5000, None]).square()
        counts += score.softmax(1).sum(0)
    keep = counts >= max(min_count, y.numel() * min_fraction)
    keep[0] = True
    return mu[keep]


def _bounds(anchor: Tensor) -> tuple[Tensor, Tensor]:
    order = anchor.argsort()
    midpoint = (anchor[order][1:] + anchor[order][:-1]) / 2
    lower, upper = torch.full_like(anchor, -torch.inf), torch.full_like(anchor, torch.inf)
    lower[order[1:]], upper[order[:-1]] = midpoint, midpoint
    return lower, upper


def _golden_maximum(fn, lower: float, upper: float) -> float:
    """Bounded scalar conditional maximization without a SciPy dependency."""
    ratio = (math.sqrt(5) - 1) / 2
    a, b = lower, upper
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(128):
        if b - a <= 1e-8 * (1 + abs(c) + abs(d)):
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = fn(d)
    return c if fc > fd else d


def _update_means(y, se, mu, sd, anchor, joint, eligible, damping, positive, minimum_count, mean_bounds):
    new_mu = mu.clone()
    lower, upper = (
        _bounds(anchor)
        if mean_bounds == "voronoi"
        else (torch.full_like(mu, -torch.inf), torch.full_like(mu, torch.inf))
    )
    eligible = eligible & (joint.sum((0, 2)) >= minimum_count) & (damping > 0)
    eligible[0] = False
    if not positive:
        precision = 1 / (se[:, None, None].square() + sd.square())
        weight = joint * precision
        target = (weight * y[:, None, None]).sum((0, 2)) / weight.sum((0, 2)).clamp_min(torch.finfo(y.dtype).tiny)
        target = target.clamp(min=lower, max=upper)
        new_mu[eligible] += damping[eligible] * (target[eligible] - mu[eligible])
        return new_mu

    # Same conditional emission criterion, eligibility, bounds and backtracking
    # as .ash_hmm_update_truncated_means; only the scalar optimizer differs.
    median_se = float(se.quantile(0.5))
    lower = lower.clamp_min(math.sqrt(torch.finfo(torch.float64).eps) * max(1, median_se))
    lower[0] = 0
    pad = max(1e-8, median_se)
    cap = max(float(mu.max()), float(anchor.max()), float((y + 8 * se).max()), float(lower.max()) + pad)
    upper = torch.where(upper.isfinite(), upper, torch.maximum(lower + pad, upper.new_tensor(cap)))
    for state in eligible.nonzero().flatten().tolist():
        weights = joint[:, state : state + 1, :]
        active = weights > 0

        def criterion(center, weights=weights, active=active):
            terms = _component_log_emission(y, se, mu.new_tensor([center]), sd, True)
            return float((weights[active] * terms[active]).sum())

        lo, hi = float(lower[state]), float(upper[state])
        if hi <= lo:
            continue
        current = min(hi, max(lo, float(mu[state])))
        before = criterion(current)
        optimum = _golden_maximum(criterion, lo, hi)
        target = max((current, lo, optimum, hi), key=criterion)
        if criterion(target) <= before + 1e-10 * (1 + abs(before)):
            continue
        fraction = float(damping[state])
        proposed = current + fraction * (target - current)
        while fraction > 2**-20 and criterion(proposed) < before - 1e-10 * (1 + abs(before)):
            fraction /= 2
            proposed = current + fraction * (target - current)
        if criterion(proposed) >= before - 1e-10 * (1 + abs(before)):
            new_mu[state] = proposed
    return new_mu


def _posterior_moments(y, se, mu, sd, joint, positive):
    y, se, mu, sd = y[:, None, None], se[:, None, None], mu[None, :, None], sd[None, None, :]
    variance = se.square() + sd.square()
    mean = (se.square() * mu + sd.square() * y) / variance
    var = sd.square() * se.square() / variance
    if positive:
        post_sd = var.sqrt()
        z = mean / torch.where(post_sd > 0, post_sd, 1.0)
        # Match fSuSiE's tail expansions, avoiding cancellation in the mean.
        extreme = z < -10
        zz = z.clamp_min(-10)
        mills = (-0.5 * (math.log(2 * math.pi) + zz.square()) - torch.special.log_ndtr(zz)).exp()
        inv = 1 / (-z).clamp_min(10)
        residual = inv - 2 * inv.pow(3) + 10 * inv.pow(5) - 74 * inv.pow(7)
        truncated_mean = torch.where(extreme, post_sd * residual, mean + post_sd * mills)
        var_factor = torch.where(
            extreme, inv.square() - 6 * inv.pow(4) + 50 * inv.pow(6), 1 - zz * mills - mills.square()
        ).clamp_min(0)
        mean = torch.where(sd > 0, truncated_mean.clamp_min(0), mean)
        var = torch.where(sd > 0, var * var_factor, var)
    post_mean = (joint * mean).sum((1, 2))
    post_mean2 = (joint * (var + mean.square())).sum((1, 2))
    prob_zero = (joint * ((mu == 0) & (sd == 0))).sum((1, 2)).clamp(0, 1)
    if positive:
        lfsr = prob_zero
    else:
        post_sd = var.sqrt()
        z = mean / torch.where(post_sd > 0, post_sd, 1.0)
        # Include atoms at zero in both tails, as in fSuSiE. Evaluate each
        # Gaussian tail separately in log space to retain tiny probabilities.
        prob_ge = torch.where(sd > 0, torch.special.log_ndtr(z).exp(), (mu >= 0).to(y.dtype))
        prob_le = torch.where(sd > 0, torch.special.log_ndtr(-z).exp(), (mu <= 0).to(y.dtype))
        lfsr = torch.minimum((joint * prob_ge).sum((1, 2)), (joint * prob_le).sum((1, 2))).clamp(0, 1)
    return post_mean, torch.maximum(post_mean2, post_mean.square()), prob_zero, lfsr


@torch.no_grad()
def fit_ash_hmm(
    y: Tensor,
    se: Tensor,
    *,
    mu=None,
    prior_sd=None,
    nonnegative_state_means: bool = False,
    half_grid: int = 50,
    grid_shape: float = 3,
    grid_expansion: float = 3,
    grid_max_abs: float | None = None,
    prefilter: bool | None = None,
    min_state_count: float = 0.5,
    min_state_fraction: float = 1e-5,
    topology: str = "full",
    init_transition=None,
    init_prob=None,
    init_rho=None,
    stay_probability: float = 0.98,
    null_state: str = "pointmass",
    penalty: float = 1.0,
    shared_mixture: bool = False,
    estimate_init: bool = False,
    learn_state_means: bool = True,
    mean_update_start: int = 3,
    mean_min_effective_count: float = 2,
    mean_min_pointmass_weight: float = 0,
    mean_min_self_transition: float = 0.93,
    mean_damping: float = 1,
    mean_bounds: str = "voronoi",
    prune_states: bool = True,
    prune_start: int = 5,
    prune_every: int = 5,
    prune_min_state_count: float = 1,
    prune_min_state_fraction: float = 1e-4,
    prune_max_fraction: float = 0.25,
    prune_max_loglik_loss: float = 0.05,
    merge_distance: float | None = None,
    maxiter: int = 20,
    tolerance: float = 1e-5,
    device: torch.device | None = None,
):
    """Fit fSuSiE's adaptive-shrinkage HMM to one ordered sequence.

    ``y[t] | beta[t] ~ N(beta[t], se[t]**2)``. A Markov state selects a
    state-specific mixture of normals centered at ``mu[state]`` on the scale
    grid ``prior_sd``. The first scale is zero (an atom at the state center).
    The first state is centered at zero and is an exact null by default.
    ``nonnegative_state_means=True`` truncates continuous components at zero.

    Inputs must be nonempty finite 1-D tensors, with strictly positive SEs;
    a scalar SE is broadcast. Order defines unit-spaced adjacency. Defaults
    follow the local fSuSiE exact solver. ``mu`` and ``prior_sd`` may be supplied
    explicitly; initial state-sized probabilities require explicit ``mu``.
    ``maxiter=0`` evaluates a specified model without fitting it.

    ``penalty`` must be finite and >= 1. The default 1 is unpenalized;
    larger values add ``penalty - 1`` pseudo-counts to transitions into the
    zero state, to its initial probability when ``estimate_init=True``, and
    to its zero-scale component when ``null_state='adaptive'``. This favors
    zero effects, matching the other priors' penalty convention. Convergence
    uses the penalized objective; ``loss`` excludes the penalty.

    Returns a namespace with ``post_mean``, ``post_mean2``, ``pi0_null``
    (posterior probability of an exact zero), ``lfsr`` (local false sign rate,
    min{P(beta[t] <= 0 | y), P(beta[t] >= 0 | y)}), ``loss`` (negative full-sequence
    marginal log likelihood), ``state_probability``, ``log_likelihood``,
    ``history`` (log likelihood), ``objective``, ``objective_history``
    (penalized log likelihood), ``converged``, ``iterations``, and fitted
    ``model_param``.
    Posterior outputs preserve input dtype/device; calculations and stored
    parameters use float64 for R-compatible numerical accuracy. See the HMM
    prior guide for controls.
    """
    y = torch.as_tensor(y, device=device)
    dtype = y.dtype if y.is_floating_point() else torch.get_default_dtype()
    y = y.to(dtype=torch.float64)
    se = torch.as_tensor(se, device=y.device, dtype=y.dtype)
    if y.ndim != 1 or not y.numel() or not y.isfinite().all():
        raise ValueError("y must be a nonempty finite one-dimensional vector.")
    if se.ndim == 0 or (se.ndim == 1 and se.numel() == 1):
        se = se.expand_as(y)
    if se.shape != y.shape or not se.isfinite().all() or (se <= 0).any():
        raise ValueError("se must be finite and strictly positive, with one value per observation (or a scalar).")
    if isinstance(penalty, bool) or not math.isfinite(penalty) or penalty < 1:
        raise ValueError("penalty must be finite and >= 1 (1 means no penalty).")
    penalty = float(penalty)
    for name, value in (
        ("nonnegative_state_means", nonnegative_state_means),
        ("shared_mixture", shared_mixture),
        ("estimate_init", estimate_init),
        ("learn_state_means", learn_state_means),
        ("prune_states", prune_states),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")
    if prefilter is not None and not isinstance(prefilter, bool):
        raise ValueError("prefilter must be a boolean or None.")
    for name, value, minimum in (
        ("half_grid", half_grid, 1),
        ("maxiter", maxiter, 0),
        ("mean_update_start", mean_update_start, 1),
        ("prune_start", prune_start, 1),
        ("prune_every", prune_every, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name, value in (
        ("grid_shape", grid_shape),
        ("grid_expansion", grid_expansion),
        ("tolerance", tolerance),
        ("mean_damping", mean_damping),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in (
        ("min_state_count", min_state_count),
        ("min_state_fraction", min_state_fraction),
        ("mean_min_effective_count", mean_min_effective_count),
        ("prune_min_state_count", prune_min_state_count),
        ("prune_min_state_fraction", prune_min_state_fraction),
        ("prune_max_loglik_loss", prune_max_loglik_loss),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    for name, value in (
        ("stay_probability", stay_probability),
        ("mean_damping", mean_damping),
        ("min_state_fraction", min_state_fraction),
        ("prune_min_state_fraction", prune_min_state_fraction),
        ("mean_min_self_transition", mean_min_self_transition),
        ("mean_min_pointmass_weight", mean_min_pointmass_weight),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")
    if not 0 < prune_max_fraction < 1:
        raise ValueError("prune_max_fraction must be in (0, 1).")
    if topology not in ("full", "hub") or null_state not in ("pointmass", "adaptive"):
        raise ValueError("topology must be 'full' or 'hub'; null_state must be 'pointmass' or 'adaptive'.")
    if mean_bounds not in ("voronoi", "none"):
        raise ValueError("mean_bounds must be 'voronoi' or 'none'.")
    if grid_max_abs is not None and (not math.isfinite(grid_max_abs) or grid_max_abs <= 0):
        raise ValueError("grid_max_abs must be finite and positive.")
    if merge_distance is None:
        merge_distance = 0.05 * float(se.quantile(0.5))
    if not math.isfinite(merge_distance) or merge_distance < 0:
        raise ValueError("merge_distance must be finite and nonnegative.")

    def tensor(value):
        return torch.as_tensor(value, device=y.device, dtype=y.dtype).clone()

    automatic_mu = mu is None
    if prefilter is None:
        prefilter = automatic_mu
    if (automatic_mu or prefilter) and any(v is not None for v in (init_transition, init_prob, init_rho)):
        raise ValueError("Supply mu explicitly and disable prefilter when supplying initial probabilities.")
    mu = (
        _mean_grid(y, half_grid, grid_shape, grid_expansion, grid_max_abs, nonnegative_state_means)
        if mu is None
        else tensor(mu)
    )
    if mu.ndim != 1 or not mu.numel() or not mu.isfinite().all() or mu[0] != 0:
        raise ValueError("mu must be a finite nonempty vector with the zero state first.")
    if nonnegative_state_means and (mu[1:] <= 0).any():
        raise ValueError("Non-null state means must be strictly positive for the nonnegative HMM.")
    if learn_state_means and mean_bounds == "voronoi" and mu.unique().numel() != mu.numel():
        raise ValueError("State means must be distinct for Voronoi bounds.")
    if prefilter:
        mu = _screen(y, se, mu, min_state_count, min_state_fraction)
    sd = _sigma_grid(y, se) if prior_sd is None else tensor(prior_sd)
    if sd.ndim != 1 or not sd.numel() or not sd.isfinite().all() or sd[0] != 0 or (sd < 0).any():
        raise ValueError("prior_sd must be finite, nonnegative and start with zero.")
    if (sd[1:] < sd[:-1]).any():
        raise ValueError("prior_sd must be sorted in nondecreasing order.")
    n, m, components = y.numel(), mu.numel(), sd.numel()
    mask = torch.ones((m, m), device=y.device, dtype=torch.bool)
    if topology == "hub":
        mask = torch.eye(m, device=y.device, dtype=torch.bool)
        mask[0, :], mask[:, 0] = True, True
    if init_transition is None:
        off_diagonal = mask & ~torch.eye(m, device=y.device, dtype=torch.bool)
        outgoing = off_diagonal.sum(1).to(y.dtype).clamp_min(1)
        transition = off_diagonal.to(y.dtype) * ((1 - stay_probability) / outgoing)[:, None]
        transition.diagonal().copy_(
            torch.where(off_diagonal.any(1), y.new_tensor(stay_probability), y.new_tensor(1.0))
        )
    else:
        transition = tensor(init_transition)
        if (
            transition.shape != (m, m)
            or not transition.isfinite().all()
            or (transition < 0).any()
            or (transition.sum(1) <= 0).any()
            or (transition[~mask] > 1e-10).any()
        ):
            raise ValueError("init_transition must have positive row sums and respect the transition topology.")
        transition[~mask] = 0
        transition /= transition.sum(1, keepdim=True)
    if init_prob is None:
        init = y.new_full((m,), 1 / m)
        for _ in range(100000):
            new_init = init @ transition
            delta = (new_init - init).abs().max()
            init = new_init
            if delta < 1e-13:
                break
        else:
            warn("HMM stationary-distribution iteration did not converge; using its last iterate.", stacklevel=2)
        init /= init.sum()
    else:
        init = tensor(init_prob)
        if init.shape != (m,) or not init.isfinite().all() or (init < 0).any() or init.sum() <= 0:
            raise ValueError("init_prob must be a finite nonnegative state vector with positive mass.")
        init /= init.sum()
    rho = y.new_full((m, components), 1 / components) if init_rho is None else tensor(init_rho)
    if rho.shape != (m, components) or not rho.isfinite().all() or (rho < 0).any() or (rho.sum(1) <= 0).any():
        raise ValueError("init_rho must be a finite nonnegative state-by-scale matrix with positive row sums.")
    rho /= rho.sum(1, keepdim=True)
    if null_state == "pointmass":
        rho[0] = 0
        rho[0, 0] = 1
    anchor = mu.clone()
    enabled = torch.zeros(m, device=y.device, dtype=torch.bool)
    component_log = _component_log_emission(y, se, mu, sd, nonnegative_state_means)
    emission = torch.logsumexp(component_log + rho.log(), 2)
    loglik, gamma, transition_counts = _smooth(emission, transition, init)
    objective = _objective(loglik, transition, init, rho, penalty, estimate_init, null_state)
    history, state_counts = [loglik], [m]
    objective_history = [objective]
    converged = False

    for iteration in range(1, maxiter + 1):
        joint = (component_log + rho.log() - emission[:, :, None]).exp() * gamma[:, :, None]
        counts = joint.sum(0)
        if penalty > 1 and null_state == "adaptive":
            counts[0, 0] += penalty - 1
        rho_new = _normalize(counts, rho)
        if shared_mixture:
            first = int(null_state == "pointmass")
            if first < m:
                rho_new[first:] = _normalize(counts[first:].sum(0), rho[first:].mean(0))
        if null_state == "pointmass":
            rho_new[0] = rho[0]
        if penalty > 1:
            transition_counts = transition_counts.clone()
            transition_counts[:, 0] += penalty - 1
        transition_new = _normalize(transition_counts, transition)
        gate = (rho_new[:, 0] >= mean_min_pointmass_weight) & (transition_new.diag() >= mean_min_self_transition)
        if iteration == mean_update_start:
            enabled = gate
        elif iteration > mean_update_start:
            enabled &= gate
        damping = mean_damping * (
            (transition_new.diag() - mean_min_self_transition) / max(1e-8, 1 - mean_min_self_transition)
        ).clamp(0, 1)
        mu_new = mu
        if learn_state_means and iteration >= mean_update_start:
            mu_new = _update_means(
                y,
                se,
                mu,
                sd,
                anchor,
                joint,
                enabled,
                damping,
                nonnegative_state_means,
                mean_min_effective_count,
                mean_bounds,
            )
        init_new = gamma[0].clone() if estimate_init else init
        if penalty > 1 and estimate_init:
            init_new[0] += penalty - 1
            init_new = _normalize(init_new, init)
        # Reuse emissions when centers are unchanged, as in the reference cache.
        changed = mu_new != mu
        component_new = component_log.clone() if changed.any() else component_log
        if changed.any():
            component_new[:, changed] = _component_log_emission(y, se, mu_new[changed], sd, nonnegative_state_means)
        emission_new = torch.logsumexp(component_new + rho_new.log(), 2)
        ll_new, gamma_new, counts_new = _smooth(emission_new, transition_new, init_new)
        objective_new = _objective(ll_new, transition_new, init_new, rho_new, penalty, estimate_init, null_state)
        delta_em = objective_new - objective
        if delta_em < -1e-7 * (1 + objective.abs()):
            warn("The fixed-dimension HMM EM objective decreased beyond numerical tolerance.", stacklevel=2)

        pruning_due = (
            prune_states and iteration >= prune_start and (iteration - prune_start) % prune_every == 0 and m > 1
        )
        pruned = False
        if pruning_due:
            occupancy = gamma_new.sum(0)
            candidates = set(
                (occupancy < max(prune_min_state_count, n * prune_min_state_fraction)).nonzero().flatten().tolist()
            )
            order = mu_new.argsort().tolist()
            for a, b in zip(order[:-1], order[1:], strict=True):
                if mu_new[b] - mu_new[a] <= merge_distance:
                    candidates.add(b if a == 0 else a if b == 0 or occupancy[a] <= occupancy[b] else b)
            candidates.discard(0)
            candidates = sorted(candidates, key=lambda i: (float(occupancy[i]), i))
            number = min(max(1, math.floor(prune_max_fraction * m)), len(candidates), m - 1)
            while number >= 1:
                keep = torch.ones(m, device=y.device, dtype=torch.bool)
                keep[candidates[:number]] = False
                a_try = transition_new[keep][:, keep]
                if (a_try.sum(1) > 0).all() and init_new[keep].sum() > 0:
                    a_try = a_try / a_try.sum(1, keepdim=True)
                    init_try = init_new[keep] / init_new[keep].sum()
                    trial = _smooth(emission_new[:, keep], a_try, init_try)
                    if ll_new - trial[0] <= prune_max_loglik_loss:
                        mu_new, rho_new = mu_new[keep], rho_new[keep]
                        transition_new, init_new = a_try, init_try
                        anchor, enabled = anchor[keep], enabled[keep]
                        component_new, emission_new = component_new[:, keep], emission_new[:, keep]
                        ll_new, gamma_new, counts_new = trial
                        objective_new = _objective(
                            ll_new, transition_new, init_new, rho_new, penalty, estimate_init, null_state
                        )
                        m = mu_new.numel()
                        pruned = True
                        break
                number //= 2
        mu, rho, transition, init = mu_new, rho_new, transition_new, init_new
        component_log, emission = component_new, emission_new
        loglik, gamma, transition_counts = ll_new, gamma_new, counts_new
        objective = objective_new
        history.append(loglik)
        objective_history.append(objective)
        state_counts.append(m)
        minimum_iterations = max(mean_update_start if learn_state_means else 1, prune_start if prune_states else 1)
        if (
            iteration >= minimum_iterations
            and (not prune_states or pruning_due)
            and not pruned
            and delta_em.abs() <= tolerance * (1 + objective.abs())
        ):
            converged = True
            break

    joint = (component_log + rho.log() - emission[:, :, None]).exp() * gamma[:, :, None]
    mean, second, p0, lfsr = _posterior_moments(y, se, mu, sd, joint, nonnegative_state_means)
    params = {
        "mu": mu,
        "prior_sd": sd,
        "transition": transition,
        "init_prob": init,
        "mixture_weight": rho,
        "nonnegative_state_means": nonnegative_state_means,
        "effect_support": "nonnegative" if nonnegative_state_means else "real",
        "penalty": penalty,
    }
    return SimpleNamespace(
        post_mean=mean.to(dtype),
        post_mean2=second.to(dtype),
        pi0_null=p0.to(dtype),
        lfsr=lfsr.to(dtype),
        loss=-loglik.to(dtype),
        log_likelihood=loglik.to(dtype),
        state_probability=gamma.to(dtype),
        model_param=params,
        history=torch.stack(history).to(dtype),
        objective=objective.to(dtype),
        objective_history=torch.stack(objective_history).to(dtype),
        state_counts=state_counts,
        iterations=len(history) - 1,
        converged=converged,
    )


def hmm_posterior_means(
    X: Tensor | None,
    betahat: Tensor,
    sebetahat: Tensor,
    *,
    model_param: dict | None = None,
    n_epochs: int | None = None,
    maxiter: int | None = None,
    device: torch.device | None = None,
    **kwargs,
):
    """cEBNM adapter for the signed HMM; ``X`` is deliberately ignored.

    Refit grids from the current normal-means data on every call, like fSuSiE.
    ``model_param`` is accepted for the builder contract but is not reused:
    irreversible state pruning and factor rescaling make stale grids unsafe.
    ``maxiter`` overrides ``n_epochs`` (cEBMF's ``internal_epoch``); otherwise
    the standalone default is 20. Remaining options go to :func:`fit_ash_hmm`.
    Initial state probabilities are learned by default to avoid unsupported
    boundary spikes; pass ``estimate_init=False`` to keep them fixed.
    The result includes ``lfsr``, one local false sign rate per observation.
    """
    kwargs.setdefault("estimate_init", True)
    return fit_ash_hmm(
        betahat,
        sebetahat,
        maxiter=maxiter if maxiter is not None else (n_epochs if n_epochs is not None else 20),
        device=device,
        **kwargs,
    )


def hmm_pos_posterior_means(X: Tensor | None, betahat: Tensor, sebetahat: Tensor, **kwargs):
    """Nonnegative HMM with exact truncated-normal emissions and moments."""
    if kwargs.pop("nonnegative_state_means", True) is not True:
        raise ValueError("hmm_pos requires nonnegative_state_means=True.")
    return hmm_posterior_means(X, betahat, sebetahat, nonnegative_state_means=True, **kwargs)


def hmm_neg_posterior_means(X: Tensor | None, betahat: Tensor, sebetahat: Tensor, **kwargs):
    """Nonpositive HMM obtained by fitting the nonnegative model to -betahat.

    Explicit ``mu`` must start at zero and have strictly negative non-null
    centers. Second moments, zero probabilities, lfsr and likelihood are invariant
    under reflection; returned centers and posterior means are negative.
    """
    if kwargs.get("mu") is not None:
        kwargs["mu"] = -torch.as_tensor(kwargs["mu"], dtype=torch.float64)
    result = hmm_pos_posterior_means(X, -betahat, sebetahat, **kwargs)
    result.post_mean = -result.post_mean
    result.model_param["mu"] = -result.model_param["mu"]
    result.model_param["nonnegative_state_means"] = False
    result.model_param["effect_support"] = "nonpositive"
    return result
