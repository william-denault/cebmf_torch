import torch

# Re-export the canonical normal-density helpers so existing imports keep working.
# The single source of truth lives in utils/maths.py.
from .maths import _logcdf_normal, _logpdf_normal, my_e2truncnorm, my_etruncnorm  # noqa: F401


class PosteriorMean:
    """
    Container for posterior mean, second moment, and standard deviation.

    Parameters
    ----------
    post_mean : torch.Tensor
        Posterior mean.
    post_mean2 : torch.Tensor
        Posterior second moment.
    post_sd : torch.Tensor
        Posterior standard deviation.
    """

    def __init__(self, post_mean, post_mean2, post_sd):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd


@torch.no_grad()
def wpost_exp(x: torch.Tensor, s: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Compute responsibilities for a spike+exponential mixture prior on (theta >= 0).

    Parameters
    ----------
    x : torch.Tensor
        Observed betahat (scalar tensor or shape ()).
    s : torch.Tensor
        Standard error (scalar tensor or shape ()).
    w : torch.Tensor
        (K,) mixture weights (sum to 1), w[0] for spike at 0, w[1:] for Exp scales.
    scale : torch.Tensor
        (K,) with scale[0]=0 for spike, scale[1:]>0 as Exp scales (rate = 1/scale).

    Returns
    -------
    torch.Tensor
        (K,) posterior responsibilities.
    """
    # ensure tensors stay on the same device/dtype
    x = torch.as_tensor(x)
    s = torch.as_tensor(s, dtype=x.dtype, device=x.device)
    w = torch.as_tensor(w, dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)

    # spike log-lik
    lf = _logpdf_normal(x, torch.as_tensor(0.0, dtype=x.dtype, device=x.device), s)

    # exp components
    a = 1.0 / scale[1:]  # rates
    lg = torch.log(a) + 0.5 * (s * a).pow(2) - a * x + _logcdf_normal(x / s - s * a)

    log_prob = torch.empty_like(scale)
    log_prob[0] = lf
    log_prob[1:] = lg

    # posterior responsibilities with log-sum-exp stabilization
    bmax = torch.max(log_prob)
    num = w * torch.exp(log_prob - bmax)
    r = num / torch.clamp(num.sum(), min=1e-300)

    # Degenerate case: w[0] >= 1 means all mass on the spike. Replace with the
    # one-hot response, branchless to avoid `if torch.all(...):` host sync.
    spike_only = w[0] >= 1.0
    r_spike = torch.zeros_like(scale)
    r_spike[0] = 1.0
    r = torch.where(spike_only, r_spike, r)
    return r


@torch.no_grad()
def posterior_mean_exp(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    log_pi: torch.Tensor,
    scale: torch.Tensor,
) -> PosteriorMean:
    """
    Vectorised posterior mean and second moment for a spike+exponential mixture
    prior.

    Replaces the previous per-observation Python loop with a single (J, K)
    tensor pipeline. The math is unchanged: the prior is

        theta ~ pi_0 * delta_0 + sum_{k>=1} pi_k * Exp(rate=1/scale[k]),

    and the likelihood is ``x | theta ~ N(theta, s^2)``.

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect-size estimates, shape ``(J,)``.
    sebetahat : torch.Tensor
        Standard errors of the effect-size estimates, shape ``(J,)``.
    log_pi : torch.Tensor
        Log mixture weights, shape ``(K,)``. The mixture is shared across
        observations.
    scale : torch.Tensor
        Mixture scales, shape ``(K,)``. ``scale[0]`` should be ``0`` (spike)
        and ``scale[1:] > 0`` are the Exp scales (rate ``= 1/scale``).

    Returns
    -------
    PosteriorMean
        Container with posterior mean, second moment, and standard deviation
        (each of shape ``(J,)``).
    """
    betahat = torch.as_tensor(betahat)
    dt, dev = betahat.dtype, betahat.device
    sebetahat = torch.as_tensor(sebetahat, dtype=dt, device=dev)
    log_pi = torch.as_tensor(log_pi, dtype=dt, device=dev)
    scale = torch.as_tensor(scale, dtype=dt, device=dev)

    # Normalise pi out of log-space (still shared across observations).
    assignment = torch.exp(log_pi)
    assignment = assignment / torch.clamp(assignment.sum(), min=1e-300)  # (K,)

    # Build the (J, K) log-probability matrix:
    #   col 0: log N(x | 0, s^2)            (spike)
    #   col k>=1: log[ a_k * exp(0.5 (s a_k)^2) * exp(-a_k x) * Phi(x/s - s a_k) ]
    x_col = betahat.unsqueeze(1)  # (J, 1)
    s_col = sebetahat.unsqueeze(1)  # (J, 1)

    lf = _logpdf_normal(betahat, torch.zeros_like(betahat), sebetahat)  # (J,)

    a = 1.0 / scale[1:]  # (K-1,) — rates of the Exp components
    a_row = a.unsqueeze(0)  # (1, K-1)

    lg = (
        torch.log(a_row) + 0.5 * (s_col * a_row).pow(2) - a_row * x_col + _logcdf_normal(x_col / s_col - s_col * a_row)
    )  # (J, K-1)

    log_prob = torch.cat([lf.unsqueeze(1), lg], dim=1)  # (J, K)

    # Posterior responsibilities, with stable normalisation.
    bmax = log_prob.max(dim=1, keepdim=True).values  # (J, 1)
    num = assignment.unsqueeze(0) * torch.exp(log_prob - bmax)  # (J, K)
    post_assign = num / torch.clamp(num.sum(dim=1, keepdim=True), min=1e-300)  # (J, K)

    # First/second moments of theta | data on the Exp branch via tilted truncnorm.
    # m_tilt = x - s^2 * a_k  with theta_c >= 0.
    m_tilt = x_col - (s_col * s_col) * a_row  # (J, K-1)
    zero = torch.zeros_like(m_tilt)
    pinf = torch.full_like(m_tilt, float("inf"))

    # The truncnorm helpers broadcast over arbitrary input shapes; passing
    # s_col (shape (J, 1)) lets them broadcast the per-observation SD over K-1
    # components without a Python loop.
    e1 = my_etruncnorm(zero, pinf, m_tilt, s_col).to(dtype=dt)  # (J, K-1)
    e2 = my_e2truncnorm(zero, pinf, m_tilt, s_col).to(dtype=dt)  # (J, K-1)

    r_exp = post_assign[:, 1:]  # (J, K-1) — responsibilities of the Exp components
    post_mean = (r_exp * e1).sum(dim=1)  # (J,)
    post_mean2 = (r_exp * e2).sum(dim=1)  # (J,)
    # Numerical floor: E[theta^2] >= E[theta]^2 by variance non-negativity
    # (Jensen), so this only catches float error. Guarding against
    # ``post_mean`` (rather than ``post_mean**2``) wrongly inflated the
    # second moment whenever post_mean < 1.
    post_mean2 = torch.maximum(post_mean2, post_mean.pow(2))

    # Infinite-s rows: collapse to the *prior* mixture mean / second moment.
    # With s = inf the data contributes no information, so the posterior
    # equals the prior. We must use the prior weights ``assignment`` here, not
    # ``post_assign`` — the latter is NaN at inf-s rows because the per-row
    # log-prob terms (``lf`` from ``log N(x|0, inf^2)`` and ``lg`` from the
    # Exp⊗Normal convolution at s=inf) are ill-defined. Both summaries are
    # scalars (0-d) and broadcast cleanly through ``torch.where``.
    a_flat = a  # (K-1,) Exp rates
    prior_slab = assignment[1:]  # (K-1,) prior weights on the Exp components
    inf_post_mean = (prior_slab / a_flat).sum()  # scalar
    inf_post_mean2 = (2.0 * prior_slab / a_flat.pow(2)).sum()  # scalar
    inf_mask = torch.isinf(sebetahat)  # (J,)
    post_mean = torch.where(inf_mask, inf_post_mean, post_mean)
    post_mean2 = torch.where(inf_mask, inf_post_mean2, post_mean2)

    post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean.pow(2), min=0.0))

    # mu is fixed at 0 in this prior; kept here for parity with the previous
    # implementation in case a non-zero spike location is added later.
    return PosteriorMean(post_mean, post_mean2, post_sd)


@torch.no_grad()
def apply_log_sum_exp(data_loglik: torch.Tensor, assignment_loglik: torch.Tensor) -> torch.Tensor:
    """
    Row-wise: (L + log_pi) - logsumexp(L + log_pi, axis=1).

    Parameters
    ----------
    data_loglik : torch.Tensor
        Data log-likelihood matrix (J, K).
    assignment_loglik : torch.Tensor
        Log mixture weights, either shared across observations (shape ``(K,)``)
        or per-observation (shape ``(J, K)``).

    Returns
    -------
    torch.Tensor
        Log posterior assignment matrix (J, K).
    """
    if assignment_loglik.ndim == 1:
        assignment_loglik = assignment_loglik.unsqueeze(0)  # (1, K) -> broadcasts
    combined = data_loglik + assignment_loglik  # (J, K)
    norm = torch.logsumexp(combined, dim=1, keepdim=True)  # (J, 1)
    return combined - norm  # (J, K)


def _broadcast_to_jk(t: torch.Tensor, J: int, K: int, name: str) -> torch.Tensor:
    """
    Broadcast a (K,) or (J, K) tensor to (J, K). Used internally by the
    posterior-mean routines so that mixture parameters (log_pi, scale, location)
    can be either shared across observations or per-observation.
    """
    if t.ndim == 1:
        if t.shape[0] != K:
            raise ValueError(f"{name} 1D length must be K={K}, got {tuple(t.shape)}")
        return t.unsqueeze(0).expand(J, K)
    if t.ndim == 2:
        if t.shape != (J, K):
            raise ValueError(f"{name} 2D shape must be (J={J}, K={K}), got {tuple(t.shape)}")
        return t
    raise ValueError(f"{name} must be 1D or 2D, got ndim={t.ndim}")


@torch.no_grad()
def posterior_mean_norm(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    log_pi: torch.Tensor,
    data_loglik: torch.Tensor,
    scale: torch.Tensor,
    location: torch.Tensor | None = None,
) -> PosteriorMean:
    """
    Compute posterior mean and second moment for a normal mixture prior.

    All mixture parameters (``log_pi``, ``scale``, ``location``) may be either
    shared across observations (1D, shape ``(K,)``) or per-observation
    (2D, shape ``(J, K)``). This is what makes the function usable both for
    classical ASH (one shared prior over a batch) and for covariate-adaptive
    methods like CASH/EMDN where the neural network emits per-observation
    mixture parameters.

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates, shape ``(J,)``.
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates, shape ``(J,)``.
    log_pi : torch.Tensor
        Log mixture weights, shape ``(K,)`` or ``(J, K)``.
    data_loglik : torch.Tensor
        Data log-likelihood matrix, shape ``(J, K)``.
    scale : torch.Tensor
        Prior standard deviations, shape ``(K,)`` or ``(J, K)``. Components
        with ``scale == 0`` are treated as a point mass at ``location``.
    location : torch.Tensor or None, optional
        Prior means, shape ``(K,)`` or ``(J, K)``. If ``None``, uses zeros.

    Returns
    -------
    PosteriorMean
        Container with posterior mean, second moment, and standard deviation
        (each of shape ``(J,)``).
    """
    betahat = torch.as_tensor(betahat)
    dt, dev = betahat.dtype, betahat.device
    sebetahat = torch.as_tensor(sebetahat, dtype=dt, device=dev)
    log_pi = torch.as_tensor(log_pi, dtype=dt, device=dev)
    scale = torch.as_tensor(scale, dtype=dt, device=dev)
    data_loglik = torch.as_tensor(data_loglik, dtype=dt, device=dev)

    if data_loglik.ndim != 2:
        raise ValueError(f"data_loglik must be 2D (J, K), got shape {tuple(data_loglik.shape)}")

    J = betahat.shape[0]
    K = data_loglik.shape[1]

    if location is None:
        location = torch.zeros(K, dtype=dt, device=dev)
    else:
        location = torch.as_tensor(location, dtype=dt, device=dev)

    log_pi_jk = _broadcast_to_jk(log_pi, J, K, "log_pi")
    scale_jk = _broadcast_to_jk(scale, J, K, "scale")
    loc_jk = _broadcast_to_jk(location, J, K, "location")

    # Posterior responsibilities (J, K) via stable log-domain normalisation.
    combined = data_loglik + log_pi_jk
    log_norm = torch.logsumexp(combined, dim=1, keepdim=True)
    resp = torch.exp(combined - log_norm)

    # Per-component posterior moments under x ~ N(theta, s^2), theta ~ N(loc, t^2).
    s2 = sebetahat.pow(2).unsqueeze(1)  # (J, 1)
    t2 = scale_jk.pow(2)  # (J, K)
    has_slab = t2 > 0

    inv_t2 = torch.where(has_slab, 1.0 / t2, torch.zeros_like(t2))
    denom = 1.0 / s2 + inv_t2  # (J, K)
    var_post = torch.where(has_slab, 1.0 / denom, torch.zeros_like(denom))  # (J, K)

    loc_over_t2 = torch.where(has_slab, loc_jk * inv_t2, torch.zeros_like(loc_jk))
    rhs = var_post * (betahat.unsqueeze(1) / s2 + loc_over_t2)
    m_comp = torch.where(has_slab, rhs, loc_jk)  # spike returns the location itself

    post_mean = (resp * m_comp).sum(dim=1)
    post_mean2 = (resp * (var_post + m_comp.pow(2))).sum(dim=1)
    post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean.pow(2), min=0.0))

    return PosteriorMean(post_mean, post_mean2, post_sd)


# --- point-mass + normal prior posterior (Torch) ---
@torch.no_grad()
def posterior_point_mass_normal(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    pi: float | torch.Tensor,
    mu0: float,
    mu1: float,
    sigma_0: float,
):
    """
    Compute posterior mean and variance for a point-mass + normal prior.

    Prior: with prob pi, theta = mu0 (point mass); else theta ~ N(mu1, sigma_0^2).
    Likelihood: x ~ N(theta, se^2).

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates (vectorized).
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates (vectorized).
    pi : float or torch.Tensor
        Probability of point mass at mu0.
    mu0 : float
        Location of the point mass.
    mu1 : float
        Mean of the normal component.
    sigma_0 : float
        Standard deviation of the normal component.

    Returns
    -------
    tuple of torch.Tensor
        post_mean (J,), post_var (J,)
    """
    x = torch.as_tensor(betahat)
    se = torch.as_tensor(sebetahat, dtype=x.dtype, device=x.device)
    pi = torch.as_tensor(pi, dtype=x.dtype, device=x.device)

    sigma0 = torch.tensor(max(float(sigma_0), 1e-8), dtype=x.dtype, device=x.device)
    se = torch.clamp(se, min=1e-8)

    # marginal likelihoods
    mlik = _logpdf_normal(
        x,
        torch.as_tensor(mu1, dtype=x.dtype, device=x.device),
        torch.sqrt(se**2 + sigma0**2),
    ).exp()
    lpm = _logpdf_normal(x, torch.as_tensor(mu0, dtype=x.dtype, device=x.device), se).exp()

    denom = torch.clamp(pi * lpm + (1.0 - pi) * mlik, min=1e-12)

    w0 = torch.clamp(pi * lpm / denom, min=0.0, max=1.0)
    w1 = 1.0 - w0

    # posterior for normal component
    mu_post = (mu1 / sigma0**2 + x / se**2) / (1.0 / sigma0**2 + 1.0 / se**2)
    sigma_post2 = 1.0 / (1.0 / sigma0**2 + 1.0 / se**2)

    post_mean = w0 * mu0 + w1 * mu_post
    post_var = w0 * (mu0 - post_mean).pow(2) + w1 * (sigma_post2 + (mu_post - post_mean).pow(2))

    return post_mean, post_var
