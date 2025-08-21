
import torch
from torch import Tensor
from dataclasses import dataclass
from .torch_utils import safe_log, norm_cdf, norm_pdf
from .torch_utils import truncated_normal_moments
from .torch_posterior import PosteriorResult

@dataclass
class EBNMLaplaceResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    w: Tensor
    a: Tensor
    mu: Tensor
    log_lik: float

def logg_laplace(x: Tensor, s: Tensor, a: Tensor) -> Tensor:
    # Convolution of Laplace(a) with N(0,s^2)
    # lg1 = -a x + log CDF( (x - s^2 a)/s )
    # lg2 = +a x + log CDF( -(x + s^2 a)/s )
    normal = torch.distributions.Normal(torch.tensor(0.0, device=x.device), torch.tensor(1.0, device=x.device))
    lg1 = -a * x + torch.log(torch.clamp(normal.cdf((x - (s*s)*a)/s), min=1e-32))
    lg2 =  a * x + torch.log(torch.clamp(normal.cdf(-(x + (s*s)*a)/s), min=1e-32))
    lfac = torch.maximum(lg1, lg2)
    return torch.log(a/2.0) + 0.5*(s*s)*(a*a) + lfac + torch.log(torch.exp(lg1 - lfac) + torch.exp(lg2 - lfac))

def ebnm_point_laplace(x: Tensor, s: Tensor, w_init: float = 0.5, a_init: float = 1.0, mu_init: float = 0.0, steps: int = 200, lr: float = 0.05) -> EBNMLaplaceResult:
    """
    Point mass at 0 + Laplace(a) prior, Gaussian likelihood. Pure PyTorch.
    Optimize (w,a,mu) by gradient ascent on marginal likelihood.
    """
    device = x.device

    w_logit = torch.tensor([0.0], device=device, requires_grad=True)
    a_param = torch.tensor([a_init], device=device, requires_grad=True)
    mu = torch.tensor([mu_init], device=device, requires_grad=True)

    opt = torch.optim.Adam([w_logit, a_param, mu], lr=lr)

    for t in range(steps):
        w = torch.sigmoid(w_logit)[0]
        a = torch.exp(a_param)[0]

        # components
        # f = N(x|mu, s)
        lf = -0.5*torch.log(2*torch.pi*s*s) - 0.5*((x-mu)/s)**2
        # g = Laplace conv normal
        lg = logg_laplace(x - mu, s, a)

        llik = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg).mean()
        loss = -llik
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.sigmoid(w_logit)[0].clamp(0.0, 1.0)
        a = torch.exp(a_param)[0].clamp_min(1e-8)
        # posterior weights for nonzero
        lf = -0.5*torch.log(2*torch.pi*s*s) - 0.5*((x-mu)/s)**2
        lg = logg_laplace(x - mu, s, a)
        denom = torch.exp(torch.log1p(-w) + lf) + torch.exp(torch.log(w) + lg)
        r = torch.exp(torch.log(w) + lg) / torch.clamp(denom, min=1e-32)

        # Posterior mean as mixture of truncated normals on +/-
        # Compute lambda = P(positive | nonzero, data)
        normal = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        lm1 = -a*(x-mu) + torch.log(torch.clamp(normal.cdf((x-mu)/s - s*a), min=1e-32))
        lm2 =  a*(x-mu) + torch.log(torch.clamp(normal.cdf(-( (x-mu)/s + s*a )), min=1e-32))
        lam = torch.sigmoid(lm1 - lm2)

        EX_pos, EX2_pos = truncated_normal_moments(torch.zeros_like(x), torch.full_like(x, float('inf')), (x - mu) - (s*s)*a, s)
        EX_neg, EX2_neg = truncated_normal_moments(-torch.full_like(x, float('inf')), torch.zeros_like(x), (x - mu) + (s*s)*a, s)
        EX_mix = lam * EX_pos + (1.0 - lam) * EX_neg
        EX2_mix = lam * EX2_pos + (1.0 - lam) * EX2_neg

        post_mean = r * EX_mix
        post_mean2 = r * EX2_mix
        post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean*post_mean, min=0.0))
        log_lik = llik.item()

    return EBNMLaplaceResult(post_mean, post_mean2, post_sd, w, a, mu, log_lik)
