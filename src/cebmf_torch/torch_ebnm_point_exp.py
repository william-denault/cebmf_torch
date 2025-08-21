# torch_only_point_exponential.py
import math
import torch
from torch import Tensor

# =========================
# Utilities (Torch only)
# =========================

_TWOPI = 2.0 * math.pi
_LOG_SQRT_2PI = 0.5 * math.log(_TWOPI)

def logphi(z: Tensor) -> Tensor:
    """log pdf of standard normal φ(z) = exp(-z^2/2)/sqrt(2π)."""
    return -0.5 * z.pow(2) - _LOG_SQRT_2PI

def logPhi(z: Tensor) -> Tensor:
    """log CDF of standard normal using a stable implementation."""
    # torch.special.log_ndtr is numerically stable for wide ranges
    return torch.special.log_ndtr(z)

def logscale_add(logx: Tensor, logy: Tensor) -> Tensor:
    """Stable log(exp(logx) + exp(logy))."""
    return torch.logaddexp(logx, logy)

 

# =========================
# Model components
# =========================

def wpost_pe(x: Tensor, s: Tensor, w: Tensor, a: Tensor) -> Tensor:
    """
    Posterior weight for the exponential component under a point-exponential prior.
    x ~ N(θ, s^2) with prior: (1-w) δ0 + w Exp(a) on θ with support θ>=0.
    """
    # handle scalar vs tensor shapes
    x, s = torch.as_tensor(x), torch.as_tensor(s)
    w, a = torch.as_tensor(w), torch.as_tensor(a)

    # quick exits (keep dtype/device consistent)
    if torch.all(w == 0):
        return torch.zeros_like(x)
    if torch.all(w == 1):
        return torch.ones_like(x)

    # log-likelihood under point mass at 0: θ=0 -> N(0, s^2)
    lf = -0.5 * ((x / s) ** 2) - torch.log(s) - _LOG_SQRT_2PI  # Normal(0,s).log_prob(x)

    # log-likelihood under Exp(a) prior integrated out (one-sided)
    # lg = log a + (s a)^2 / 2 - a x + log Φ(x/s - s a)
    xright = x / s - s * a
    lg = torch.log(a) + 0.5 * (s * a) ** 2 - a * x + logPhi(xright)

    # posterior mixing weight: w / (w + (1-w)*exp(lf - lg))
    wpost = w / (w + (1.0 - w) * torch.exp(lf - lg))
    return wpost

class PosteriorMeanPointExp:
    def __init__(self, post_mean: Tensor, post_mean2: Tensor, post_sd: Tensor):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd

def posterior_mean_pe(x: Tensor, s: Tensor, w: Tensor, a: Tensor, mu: Tensor = None) -> PosteriorMeanPointExp:
    """
    Posterior mean and second moment under point-exponential prior on (θ - μ), θ≥μ.

    Prior: (1-w) δ_μ + w [μ + Exp(a)]  (i.e., θ = μ with prob 1-w, otherwise μ + Y, Y~Exp(a), Y≥0)

    Returns post_mean, post_mean2, post_sd for θ.
    """
    x = torch.as_tensor(x)
    s = torch.as_tensor(s)
    w = torch.as_tensor(w)
    a = torch.as_tensor(a)
    if mu is None:
        mu = torch.zeros((), dtype=x.dtype, device=x.device)
    else:
        mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)

    # work on centered variable: x_c = x - μ, θ_c ≥ 0 with prior (1-w) δ0 + w Exp(a)
    xc = x - mu
    wpost = wpost_pe(xc, s, w, a)

    # Truncated Normal moments for the exponential branch:
    # For θ_c | data, its posterior under the Exp prior mixes with truncation at 0 through the likelihood tilt:
    # The posterior of θ_c given we are in the Exp component is proportional to:
    #    exp(-a θ_c) * N(xc | θ_c, s^2) for θ_c ≥ 0
    # which is a tilted (shifted) normal -> compute via truncated normal moments by completing the square:
    # Equivalent to Normal(mean = xc - s^2 a, sd = s) truncated to [0, +∞)
    m_tilt = xc - s**2 * a
    s_tilt = s

    # E[θ_c | Exp-branch, data] and E[θ_c^2 | Exp-branch, data]
    # = E[Z | Z ~ N(m_tilt, s^2), Z in [0, +∞)) etc.
    zero = torch.zeros_like(xc)
    inf = torch.full_like(xc, float("inf"))

    laplace_component_mean = etruncnorm(zero, inf, m_tilt, s_tilt)
    laplace_component_mean2 = e2truncnorm(zero, inf, m_tilt, s_tilt)

    # Combine with posterior mixing weight
    post_mean_c = wpost * laplace_component_mean
    post_mean2_c = wpost * laplace_component_mean2

    # Handle s = +inf edge case, if any
    if torch.isinf(s).any():
        inf_mask = torch.isinf(s)
        post_mean_c = post_mean_c.clone()
        post_mean2_c = post_mean2_c.clone()
        post_mean_c[inf_mask] = wpost[inf_mask] / a
        post_mean2_c[inf_mask] = 2.0 * wpost[inf_mask] / (a**2)

    # Ensure E[θ_c^2] >= (E[θ_c])^2 numerically
    post_mean2_c = torch.maximum(post_mean2_c, post_mean_c ** 2)

    # sd for θ_c (not adding μ here)
    post_sd_c = torch.sqrt(torch.clamp(post_mean2_c - post_mean_c**2, min=0.0))

    # Shift back to θ = θ_c + μ
    post_mean = post_mean_c + mu
    post_mean2 = post_mean2_c + 2.0 * mu * post_mean_c + mu**2
    post_sd = post_sd_c  # sd unaffected by location shift

    return PosteriorMeanPointExp(post_mean=post_mean, post_mean2=post_mean2, post_sd=post_sd)


# =========================
# Negative log-likelihood
# =========================

def pe_nllik_raw(x: Tensor, s: Tensor, alpha: Tensor, beta: Tensor, mu: Tensor) -> Tensor:
    """
    Negative log-likelihood for point-exponential prior with reparameterization:
      w = 1 - 1/(1+exp(alpha))  in (0,1)
      a = exp(beta)             in (0,∞)

    Returns scalar nllik.
    """
    # parameters
    w = 1.0 - 1.0 / (1.0 + torch.exp(alpha))
    a = torch.exp(beta)

    # components
    # lf = log N(x | mu, s^2)
    z = (x - mu) / s
    lf = -0.5 * z.pow(2) - torch.log(s) - _LOG_SQRT_2PI

    # lg = log ∫_{θ≥μ} Exp(a, on θ-μ) * N(x|θ,s^2) dθ
    # analytic form: log a + (s a)^2 / 2 - a (x - μ) + log Φ((x-μ)/s - s a)
    xright = (x - mu) / s - s * a
    lg = torch.log(a) + 0.5 * (s * a) ** 2 - a * (x - mu) + logPhi(xright)

    # mixture log-likelihood per datum: log((1-w) e^{lf} + w e^{lg})
    llik_i = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg)
    nllik = -torch.sum(llik_i)
    return nllik

# =========================
# Optimizer wrapper (Torch)
# =========================

class OptimizePointExponential:
    def __init__(self, w: float, a: float, mu: float, nllik: float):
        self.w = w
        self.a = a
        self.mu = mu
        self.nllik = nllik

def optimize_pe_nllik_with_gradient(
    x: Tensor,
    s: Tensor,
    par_init: list[float],
    fix_par: list[bool],
    max_iter: int = 200,
    line_search_fn: str | None = "strong_wolfe",
    tol: float = 1e-7,
) -> OptimizePointExponential:
    """
    Optimize α, β, μ (with optional fixing) using LBFGS and autograd.
    par_init = [alpha0, beta0, mu0]; fix_par booleans for [w, a, mu] respectively.
    """
    device = x.device
    dtype = x.dtype

    alpha0, beta0, mu0 = [torch.tensor(v, dtype=dtype, device=device) for v in par_init]

    # Create parameters; freeze the fixed ones
    alpha = torch.nn.Parameter(alpha0.clone(), requires_grad=not fix_par[0])
    beta  = torch.nn.Parameter(beta0.clone(),  requires_grad=not fix_par[1])
    mu    = torch.nn.Parameter(mu0.clone(),    requires_grad=not fix_par[2])

    # LBFGS needs a list of parameters (only those requiring grad)
    params = [p for p in [alpha, beta, mu] if p.requires_grad]
    optimizer = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn=line_search_fn, tolerance_grad=tol, tolerance_change=tol)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        nll = pe_nllik_raw(x, s, alpha, beta, mu)
        nll.backward()
        return nll

    # Run optimization
    if len(params) > 0:
        optimizer.step(closure)

    # Final values (transform back)
    with torch.no_grad():
        w = 1.0 - 1.0 / (1.0 + torch.exp(alpha))
        a = torch.exp(beta)
        nll_final = pe_nllik_raw(x, s, alpha, beta, mu).item()

    return OptimizePointExponential(
        w=float(w.item()),
        a=float(a.item()),
        mu=float(mu.item()),
        nllik=nll_final,
    )

# =========================
# Public EBNM interface
# =========================

class EBNMPointExp:
    def __init__(self, post_mean: Tensor, post_mean2: Tensor, post_sd: Tensor, scale: float, pi: float, log_lik: float = 0.0, mode: float = 0.0):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.scale = scale
        self.pi = pi
        self.log_lik = log_lik
        self.mode = mode

def ebnm_point_exp_solver(
    x: Tensor,
    s: Tensor,
    opt_mu: bool = False,
    par_init: list[float] = [0.0, 1.0, 0.0],  # alpha, beta, mu
) -> EBNMPointExp:
    """
    Fit the point-exponential prior and return posterior summaries (Torch only).
    If opt_mu=False, μ is held fixed at par_init[2].
    """
    # fix_par: [fix_w, fix_a, fix_mu] == True means fixed (NOT optimized)
    fix_par = [False, False, not opt_mu]

    # Optimize
    opt = optimize_pe_nllik_with_gradient(x, s, par_init=par_init, fix_par=fix_par)

    # Posterior summaries
    post = posterior_mean_pe(x, s, torch.tensor(opt.w, dtype=x.dtype, device=x.device),
                             torch.tensor(opt.a, dtype=x.dtype, device=x.device),
                             torch.tensor(opt.mu, dtype=x.dtype, device=x.device))

    return EBNMPointExp(
        post_mean=post.post_mean,
        post_mean2=post.post_mean2,
        post_sd=post.post_sd,
        scale=opt.a,
        pi=opt.w,
        log_lik=-opt.nllik,
        mode=opt.mu,
    )

 