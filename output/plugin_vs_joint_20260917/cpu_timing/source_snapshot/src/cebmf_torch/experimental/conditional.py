"""Legacy sampler transitions; distributions live in priors.conditional."""

import torch
from torch import Tensor

from cebmf_torch.priors.conditional import SCALAR_PRIORS, ConditionalMixture, GaussianMixture

__all__ = ["SCALAR_PRIORS", "ConditionalMixture", "GaussianMixture", "metropolis_normal_means"]

@torch.no_grad()
def metropolis_normal_means(
    prior: GaussianMixture,
    a: Tensor,
    b: Tensor,
    old: Tensor,
    component: Tensor,
    child_log_prob,
    generator: torch.Generator,
):
    """One augmented Gibbs-within-MH update, simultaneously across independent rows.

    child_log_prob(v) must evaluate every *direct child's* augmented prior
    under the candidate parent value, including its mean/variance dependence.
    Global hyperparameters and all other latent coordinates stay fixed.
    """
    posterior, _ = prior.posterior(a, b)
    proposal, proposed_component = posterior.sample(generator)
    correction = child_log_prob(proposal) - child_log_prob(old)
    if torch.isnan(correction).any():
        raise FloatingPointError("Undefined child-prior ratio; check current state support.")
    accept = torch.rand(len(a), generator=generator, dtype=a.dtype, device=a.device).log() < correction.clamp_max(0)
    return torch.where(accept, proposal, old), torch.where(accept, proposed_component, component), accept
