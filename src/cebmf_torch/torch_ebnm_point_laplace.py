import math
import torch
from dataclasses import dataclass
from torch import Tensor
from cebmf_torch.torch_utils import my_etruncnorm, my_e2truncnorm, log_norm_pdf, logPhi, safe_log , _TWOPI, _LOG_SQRT_2PI

def logg_laplace_convolved_with_normal(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    """
    log p(x | theta ~ Laplace(0, 1/a), noise ~ N(0, s^2)) as a function of x.
    Uses the standard closed form:
      log(a/2) + 0.5*(s*a)^2 + log( Φ((x - s^2 a)/s) * e^{-a x} + Φ(-(x + s^2 a)/s) * e^{a x} )
    implemented in log-space with logaddexp.
    """
    s = torch.clamp(s, min=1e-12)
    z1 = (x - (s*s)*a) / s
    z2 = -(x + (s*s)*a) / s

    # log of each branch safely
    lg1 = -a * x + logPhi(z1)
    lg2 =  a * x + logPhi(z2)

    lsum = torch.logaddexp(lg1, lg2)                   # stable log(exp(lg1)+exp(lg2))
    return safe_log(a / 2.0) + 0.5 * (s * a) ** 2 + lsum

@dataclass
class EBNMLaplaceResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    w: float
    a: float
    mu: float
    log_lik: float

def ebnm_point_laplace(
    x: Tensor,
    s: Tensor,
    par_init: list = [0.0, 0.0, 0.0],   # [w_logit, log_a, mu]
    fix_par:  list = [False, False,  True],
    max_iter: int = 200,
    tol: float = 1e-6
) -> EBNMLaplaceResult:
    """
    Empirical Bayes normal means with prior: (1-w) * delta_mu + w * Laplace(mu, 1/a)
    """
    # dtype/device
    device, dtype = x.device, x.dtype

    # parameters
    w_logit0, log_a0, mu0 = [torch.tensor(v, dtype=dtype, device=device) for v in par_init]
    w_logit = torch.nn.Parameter(w_logit0.clone(), requires_grad=not fix_par[0])
    log_a   = torch.nn.Parameter(log_a0.clone(),   requires_grad=not fix_par[1])
    mu      = torch.nn.Parameter(mu0.clone(),      requires_grad=not fix_par[2])

    params = [p for p in [w_logit, log_a, mu] if p.requires_grad]
    optimizer = torch.optim.LBFGS(params, max_iter=max_iter, tolerance_grad=tol, tolerance_change=tol)

    x = x.to(dtype)
    s = torch.clamp(s.to(dtype), min=1e-12)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        w = torch.sigmoid(w_logit)
        a = torch.exp(log_a)
        x_c = x - mu

        lf = log_norm_pdf(x_c, loc=torch.zeros_like(x_c), scale=s)                # point-mass at mu branch
        lg = logg_laplace_convolved_with_normal(x_c, s, a)                        # Laplace branch

        # total log-likelihood (sum over observations) in a numerically stable way
        llik_i = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg)
        loss = -llik_i.sum()
        loss.backward()
        return loss

    if len(params) > 0:
        optimizer.step(closure)

    # ======= posterior moments =======
    with torch.no_grad():
        w_t = torch.sigmoid(w_logit)
        a_t = torch.exp(log_a)
        mu_v = float(mu.item())

        x_c = x - mu
        lf  = log_norm_pdf(x_c, loc=torch.zeros_like(x_c), scale=s)
        lg1 = -a_t * x_c + logPhi((x_c - (s*s)*a_t) / s)
        lg2 =  a_t * x_c + logPhi(-(x_c + (s*s)*a_t) / s)
        lsum = torch.logaddexp(lg1, lg2)
        lg   = safe_log(a_t / 2.0) + 0.5 * (s * a_t) ** 2 + lsum

        # mixture responsibility for Laplace branch: gamma = p(Laplace | x)
        log_num   = torch.log(w_t)     + lg
        log_denom = torch.logaddexp(torch.log1p(-w_t) + lf, log_num)
        mix_weight = torch.exp(log_num - log_denom)
        # Handle (pathological) -inf denom: default to 0.5
        mix_weight = torch.where(torch.isfinite(log_denom), mix_weight, torch.full_like(mix_weight, 0.5))

        # split within Laplace into positive/negative sides
        lam = torch.exp(lg1 - lsum)                     # p(positive | Laplace, x)
        lam = torch.where(torch.isfinite(lsum), lam, torch.full_like(lsum, 0.5))

        # truncated normal moments for theta - mu (call them T)
        # Positive side: T ~ N(m_pos, s^2) truncated to [0, +inf)
        # Negative side: T ~ N(m_neg, s^2) truncated to (-inf, 0]
        m_pos = x_c - s * s * a_t
        m_neg = x_c + s * s * a_t
        inf_pos = torch.full_like(x_c, float('inf'))
        inf_neg = -torch.full_like(x_c, float('inf'))

        EX_pos  = my_etruncnorm(0.0,     inf_pos, mean=m_pos.to(torch.float64), sd=s.to(torch.float64)).to(dtype)
        EX2_pos = my_e2truncnorm(0.0,    inf_pos, mean=m_pos.to(torch.float64), sd=s.to(torch.float64)).to(dtype)
        EX_neg  = my_etruncnorm(inf_neg, 0.0,     mean=m_neg.to(torch.float64), sd=s.to(torch.float64)).to(dtype)
        EX2_neg = my_e2truncnorm(inf_neg,0.0,     mean=m_neg.to(torch.float64), sd=s.to(torch.float64)).to(dtype)

        EX_mix  = lam * EX_pos + (1.0 - lam) * EX_neg           # E[T | Laplace, x]
        EX2_mix = lam * EX2_pos + (1.0 - lam) * EX2_neg         # E[T^2 | Laplace, x]

        # Combine with point-mass at mu (i.e., T=0) to get E[theta] and E[theta^2]
        post_mean  = mix_weight * (EX_mix + mu) + (1.0 - mix_weight) * mu
        post_mean2 = mix_weight * ((EX2_mix + 2 * mu * EX_mix + mu*mu)) + (1.0 - mix_weight) * (mu * mu)
        post_sd    = torch.sqrt(torch.clamp(post_mean2 - post_mean * post_mean, min=0.0))

        log_lik = float(torch.sum(torch.logaddexp(torch.log1p(-w_t) + lf, torch.log(w_t) + lg)).item())
        w_val = float(w_t.item())
        a_val = float(a_t.item())

    return EBNMLaplaceResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        w=w_val,
        a=a_val,
        mu=mu_v,
        log_lik=log_lik,
    )
