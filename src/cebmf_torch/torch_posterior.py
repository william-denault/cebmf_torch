
import torch
from torch import Tensor
from .torch_utils import logsumexp, truncated_normal_moments

class PosteriorResult:
    def __init__(self, post_mean: Tensor, post_mean2: Tensor, post_sd: Tensor):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd

def posterior_mean_norm(betahat: Tensor, sebetahat: Tensor, log_pi: Tensor, scale: Tensor, location: Tensor=None) -> PosteriorResult:
    """
    Mixture of normals (including a spike at scale=0).
    Compute posterior mean/second-moment for each datum by mixing component-wise normals.
    """
    n = betahat.shape[0]
    K = scale.shape[0]
    device = betahat.device

    if location is None:
        location = torch.zeros(K, device=device)

    # data log-likelihoods under each component
    # (use local import to avoid circular with distribution module)
    from .torch_distribution_operation import get_data_loglik_normal
    data_loglik = get_data_loglik_normal(betahat, sebetahat, location, scale)  # (n,K)

    log_post = data_loglik + log_pi.view(1, K)
    post = torch.softmax(log_post, dim=1)  # (n,K)

    # Component-wise posterior variance for normal-normal conjugacy
    se2 = sebetahat.view(n,1).pow(2).expand(n,K)
    sc2 = scale.view(1,K).pow(2).expand(n,K)

    v = 1.0 / (1.0/(se2 + 1e-32) + 1.0/torch.clamp(sc2, min=1e-32))
    # spike (scale=0): variance=0, mean=location (0)
    spike_mask = (scale <= 0)
    if spike_mask.any():
        v[:, spike_mask] = 0.0

    # posterior mean for component k
    x = betahat.view(n,1).expand(n,K)
    loc = location.view(1,K).expand(n,K)
    m = (v/(se2 + 1e-32))*x + (1.0 - v/(se2 + 1e-32))*loc

    post_mean = (post * m).sum(dim=1)
    post_mean2 = (post * (v + m*m)).sum(dim=1)
    post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean*post_mean, min=0.0))
    return PosteriorResult(post_mean, post_mean2, post_sd)

def posterior_mean_exp(betahat: Tensor, sebetahat: Tensor, log_pi: Tensor, scale: Tensor) -> PosteriorResult:
    """
    Mixture of [spike at 0] + exponential(a_k) components convolved with Gaussian.
    Uses truncated normal moments to compute E[theta | component k, data] for k>=1.
    """
    from .torch_distribution_operation import get_data_loglik_exp
    device = betahat.device
    n = betahat.shape[0]
    K = scale.shape[0]

    logL = get_data_loglik_exp(betahat, sebetahat, scale)
    log_post = logL + log_pi.view(1, K)
    post = torch.softmax(log_post, dim=1)  # (n,K)

    # For k==0 (spike), contribution to posterior mean is 0.
    # For k>=1, E[theta | data, k] = E[Z | Z ~ N(betahat - se^2*a_k, se), Z > 0]
    if K == 1:
        post_mean = torch.zeros(n, device=device)
        post_mean2 = torch.zeros(n, device=device)
        post_sd = torch.zeros(n, device=device)
        return PosteriorResult(post_mean, post_mean2, post_sd)

    a = (1.0 / scale[1:]).view(1, K-1)  # (1,K-1)
    mu_k = betahat.view(n,1) - (sebetahat.view(n,1)**2) * a  # (n,K-1)
    s = sebetahat.view(n,1)
    # Truncated to [0, +inf)
    EX, EX2 = truncated_normal_moments(torch.zeros_like(mu_k), torch.full_like(mu_k, float('inf')), mu_k, s)

    # Weight by posterior assignments (exclude spike column 0)
    w = post[:, 1:]
    comp_mean = (w * EX).sum(dim=1)
    comp_mean2 = (w * EX2).sum(dim=1)

    post_mean = comp_mean
    post_mean2 = torch.maximum(comp_mean2, post_mean*post_mean)
    post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean*post_mean, min=0.0))
    return PosteriorResult(post_mean, post_mean2, post_sd)
