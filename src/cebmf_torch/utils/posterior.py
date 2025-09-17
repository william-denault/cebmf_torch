import torch

from .maths import _LOG_SQRT_2PI, my_e2truncnorm, my_etruncnorm


def _logpdf_normal(x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = (x - loc) / scale
    return -0.5 * z.pow(2) - torch.log(scale) - _LOG_SQRT_2PI


def _logcdf_normal(z: torch.Tensor) -> torch.Tensor:
    return torch.special.log_ndtr(z)


# --- imports you said are available ---
# from cebmf.routines.numerical_routine import my_etruncnorm, my_e2truncnorm, log_sum_exp
# from cebmf.routines.distribution_operation import get_data_loglik_normal


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
    # ensure tensors
    x = torch.as_tensor(x)
    s = torch.as_tensor(s, dtype=x.dtype, device=x.device)
    w = torch.as_tensor(w, dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)

    if torch.all(w[0] == 1):
        out = torch.zeros_like(scale)
        out[0] = 1.0
        return out

    # spike log-lik
    lf = _logpdf_normal(x, torch.tensor(0.0, dtype=x.dtype, device=x.device), s)

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
    return r


@torch.no_grad()
def posterior_mean_exp(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    log_pi: torch.Tensor,
    scale: torch.Tensor,
) -> PosteriorMean:
    """
    Compute posterior mean and second moment for a spike+exponential mixture prior.

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates, shape (J,).
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates, shape (J,).
    log_pi : torch.Tensor
        Log mixture weights, shape (K,).
    scale : torch.Tensor
        (K,) with scale[0]=0 (spike), scale[1:]>0 (Exp scales).

    Returns
    -------
    PosteriorMean
        Container with posterior mean, second moment, and standard deviation.
    """
    betahat = torch.as_tensor(betahat)
    sebetahat = torch.as_tensor(sebetahat, dtype=betahat.dtype, device=betahat.device)
    log_pi = torch.as_tensor(log_pi, dtype=betahat.dtype, device=betahat.device)
    scale = torch.as_tensor(scale, dtype=betahat.dtype, device=betahat.device)

    J = betahat.shape[0]
    K = scale.shape[0]
    mu = 0.0

    # Normalize π from log-space
    assignment = torch.exp(log_pi)
    assignment = assignment / torch.clamp(assignment.sum(), min=1e-300)

    # Responsibilities per datum (J,K)
    post_assign = torch.empty((J, K), dtype=betahat.dtype, device=betahat.device)
    for i in range(J):
        post_assign[i] = wpost_exp(betahat[i], sebetahat[i], assignment, scale)

    # Component expectations for Exp part
    a = 1.0 / scale[1:]  # (K-1,)
    post_mean = torch.zeros(J, dtype=betahat.dtype, device=betahat.device)
    post_mean2 = torch.zeros(J, dtype=betahat.dtype, device=betahat.device)

    # Use your truncnorm helpers (assumed Torch implementations)
    # E[theta_c | data] with tilt: N(m_tilt, s^2) truncated to [0, +inf),
    # where m_tilt = x - s^2 * a
    for i in range(J):
        x_i = betahat[i]
        s_i = sebetahat[i]
        m_tilt = x_i - (s_i**2) * a  # (K-1,)

        e1 = my_etruncnorm(torch.zeros_like(m_tilt), torch.full_like(m_tilt, float("inf")), m_tilt, s_i)
        e2 = my_e2truncnorm(torch.zeros_like(m_tilt), torch.full_like(m_tilt, 99999.0), m_tilt, s_i)

        # Mix only over Exp components (skip spike at 0)
        r_exp = post_assign[i, 1:]  # (K-1,)
        post_mean[i] = torch.sum(r_exp * e1)
        post_mean2[i] = torch.sum(r_exp * e2)
        post_mean2[i] = torch.maximum(post_mean2[i], post_mean[i])  # mimic original guard

    # Handle infinite s
    if torch.isinf(sebetahat).any():
        inf_mask = torch.isinf(sebetahat)
        a = 1.0 / scale[1:]
        # post_mean[inf] = sum_k r_k / a_k
        post_mean[inf_mask] = torch.sum(post_assign[inf_mask, 1:] / a, dim=1)
        # post_mean2[inf] = sum_k 2 r_k / a_k^2
        post_mean2[inf_mask] = torch.sum(2.0 * post_assign[inf_mask, 1:] / (a**2), dim=1)

    post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean**2, min=0.0))

    # add mu (mu=0 here keeps values as-is; kept for parity)
    post_mean2 = post_mean2 + mu**2 + 2 * mu * post_mean
    post_mean = post_mean + mu

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
        Log mixture weights (K,).

    Returns
    -------
    torch.Tensor
        Log posterior assignment matrix (J, K).
    """
    combined = data_loglik + assignment_loglik.unsqueeze(0)  # (J,K)
    norm = torch.logsumexp(combined, dim=1, keepdim=True)  # (J,1)
    return combined - norm  # (J,K)


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

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates, shape (J,).
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates, shape (J,).
    log_pi : torch.Tensor
        Log mixture weights, shape (K,).
    data_loglik : torch.Tensor
        Data log-likelihood matrix (J, K).
    scale : torch.Tensor
        Prior standard deviations, shape (K,).
    location : torch.Tensor or None, optional
        Prior means, shape (K,) or (J, K). If None, uses zeros like scale.

    Returns
    -------
    PosteriorMean
        Container with posterior mean, second moment, and standard deviation.
    """
    betahat = torch.as_tensor(betahat)
    sebetahat = torch.as_tensor(sebetahat, dtype=betahat.dtype, device=betahat.device)
    log_pi = torch.as_tensor(log_pi, dtype=betahat.dtype, device=betahat.device)
    scale = torch.as_tensor(scale, dtype=betahat.dtype, device=betahat.device)

    J = betahat.shape[0]
    K = scale.shape[0]

    if location is None:
        location = torch.zeros(K, dtype=betahat.dtype, device=betahat.device)
    else:
        location = torch.as_tensor(location, dtype=betahat.dtype, device=betahat.device)
        if location.ndim == 1:
            pass
        elif location.ndim == 2:
            assert location.shape == (J, K), "location must be (K,) or (J,K)"
        else:
            raise ValueError("location must be (K,) or (J,K)")

    # data log-likelihood and posterior assignment in log-space
    # data_loglik = get_data_loglik_normal_torch(betahat, sebetahat, location, scale)  # (J,K)
    log_post_assignment = apply_log_sum_exp(data_loglik, log_pi)  # (J,K)
    resp = torch.exp(log_post_assignment)  # (J,K)

    # per-component posterior variance v_{jk} = 1 / (1/s^2 + 1/t^2)
    s2 = sebetahat.pow(2).unsqueeze(1)  # (J,1)
    t2 = scale.pow(2).unsqueeze(0)  # (1,K)

    # start with full formula, then patch spike columns where scale==0 -> variance 0
    with torch.no_grad():
        denom = (1.0 / s2) + torch.where(t2 > 0, 1.0 / t2, torch.zeros_like(t2))
        t_ind_Var = torch.where(t2 > 0, 1.0 / denom, torch.zeros_like(denom))  # (J,K)

    # posterior mean per component:
    # m_{jk} = v_{jk} * (x_j/s_j^2 + loc_{jk}/t_k^2), spike handled by t2==0 -> m=loc
    if location.ndim == 1:
        loc = location.unsqueeze(0).expand(J, K)  # (J,K)
    else:
        loc = location

    rhs = t_ind_Var * (betahat.unsqueeze(1) / s2 + loc / t2)  # [J, K]

    mask_spike = (t2 == 0.0).expand(J, K)
    m_comp = torch.where(mask_spike, loc, rhs)
    post_mean = torch.sum(resp * m_comp, dim=1)
    post_mean2 = torch.sum(resp * (t_ind_Var + m_comp.pow(2)), dim=1)
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

    sigma_0 = max(float(sigma_0), 1e-8)
    se = torch.clamp(se, min=1e-8)

    # marginal likelihoods
    mlik = _logpdf_normal(
        x,
        torch.tensor(mu1, dtype=x.dtype, device=x.device),
        torch.sqrt(se**2 + sigma_0**2),
    ).exp()
    lpm = _logpdf_normal(x, torch.tensor(mu0, dtype=x.dtype, device=x.device), se).exp()

    denom = torch.clamp(pi * lpm + (1.0 - pi) * mlik, min=1e-12)

    w0 = torch.clamp(pi * lpm / denom, min=0.0, max=1.0)
    w1 = 1.0 - w0

    # posterior for normal component
    mu_post = (mu1 / sigma_0**2 + x / se**2) / (1.0 / sigma_0**2 + 1.0 / se**2)
    sigma_post2 = 1.0 / (1.0 / sigma_0**2 + 1.0 / se**2)

    post_mean = w0 * mu0 + w1 * mu_post
    post_var = w0 * (mu0 - post_mean).pow(2) + w1 * (sigma_post2 + (mu_post - post_mean).pow(2))

    return post_mean, post_var
