
import torch
from torch import Tensor
from dataclasses import dataclass
from .torch_utils import safe_log
from .torch_posterior import PosteriorResult
from .torch_posterior import truncated_normal_moments  # re-exported in utils

@dataclass
class EBNMExpResult:
    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    w: Tensor
    a: Tensor
    mu: Tensor
    log_lik: float

def ebnm_point_exp(x: Tensor, s: Tensor, w_init: float = 0.5, a_init: float = 1.0, mu_init: float = 0.0, steps: int = 200, lr: float = 0.05) -> EBNMExpResult:
    """
    Point mass at 0 + Exponential(a) prior, with Gaussian likelihood. Pure PyTorch.
    Optimize (w,a,mu) by gradient ascent on marginal likelihood (mini-batch ready).
    """
    device = x.device
    n = x.numel()

    w_logit = torch.tensor([0.0], device=device, requires_grad=True)  # sigmoid -> w
    a_param = torch.tensor([a_init], device=device, requires_grad=True)  # exp -> a
    mu = torch.tensor([mu_init], device=device, requires_grad=True)

    opt = torch.optim.Adam([w_logit, a_param, mu], lr=lr)

    normal = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

    for t in range(steps):
        w = torch.sigmoid(w_logit)[0]
        a = torch.exp(a_param)[0]

        # log-likelihood per point: log((1-w)*f + w*g)
        # f = N(x|mu, s)
        lf = -0.5*torch.log(2*torch.pi*s*s) - 0.5*((x-mu)/s)**2
        # g = exp conv normal (right tail)
        xright = (x-mu)/s - s*a
        lpnormright = torch.log(torch.clamp(normal.cdf(xright), min=1e-32))
        lg = torch.log(a) + 0.5*s*s*a*a - a*(x-mu) + lpnormright
        llik = torch.logaddexp(torch.log1p(-w) + lf, torch.log(w) + lg).mean()

        loss = -llik
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.sigmoid(w_logit)[0].clamp(0.0, 1.0)
        a = torch.exp(a_param)[0].clamp_min(1e-8)

        # posterior mean via truncated normal (Z>0) with mean (x - s^2 a)
        mu_k = x - (s*s)*a
        EX, EX2 = truncated_normal_moments(torch.zeros_like(mu_k), torch.full_like(mu_k, float('inf')), mu_k, s)
        # mixture with spike at 0
        # compute posterior responsibility of non-zero: r = w*g / ((1-w)f + w g)
        normal = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        lf = -0.5*torch.log(2*torch.pi*s*s) - 0.5*((x-mu)/s)**2
        xright = (x-mu)/s - s*a
        lpnormright = torch.log(torch.clamp(normal.cdf(xright), min=1e-32))
        lg = torch.log(a) + 0.5*s*s*a*a - a*(x-mu) + lpnormright
        denom = torch.exp(torch.log1p(-w) + lf) + torch.exp(torch.log(w) + lg)
        r = torch.exp(torch.log(w) + lg) / torch.clamp(denom, min=1e-32)

        post_mean = r * EX
        post_mean2 = r * EX2
        post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean*post_mean, min=0.0))
        log_lik = llik.item()

    return EBNMExpResult(post_mean, post_mean2, post_sd, w, a, mu, log_lik)
