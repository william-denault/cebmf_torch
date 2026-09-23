"""Explicit mixed-support variational distributions for Monte Carlo ELBOs.

These are diagnostics for the joint sampler, not variational E-steps. Fit
independent augmented Gaussian mixtures to saved scalar marginals. Keep an
HMM axis as a normalized whole-path normal-means posterior at fixed reference
statistics. Evaluate E_q[log p - log q] using new independent draws from q,
never by assigning an empirical/discrete entropy to continuous MCMC draws.
"""

import math
from dataclasses import dataclass

import torch

from .conditional import GaussianMixture, SCALAR_PRIORS
from .joint import sample_hmm


@dataclass(frozen=True)
class ELBOEstimate:
    elbo: float
    regularized_elbo: float
    standard_error: float
    regularized_standard_error: float
    penalty_term: float
    n_draws: int
    phase: str
    approximation: str = "factorized_augmented_mixtures_with_hmm_blocks"

    @property
    def loss(self):
        """Negative regularized ELBO, matching cEBMF's lower-is-better convention."""
        return -self.regularized_elbo


def fit_marginal_mixture(values, labels, reference):
    """Fit normalized labelled marginals, preserving every atomic support.

    Unseen labels receive zero q probability. With fewer than two visits to
    a continuous label, use its reference posterior variance. A variance
    floor keeps a fitted continuous component continuous, including at a
    constant Monte Carlo sample. This approximation does not alter the model.
    """
    count = torch.zeros_like(reference.mean)
    count.scatter_add_(1, labels.T, torch.ones_like(values.T))
    total = torch.zeros_like(count).scatter_add_(1, labels.T, values.T)
    mean = torch.where(count > 0, total / count.clamp_min(1), reference.mean)
    centered = values.T - mean.gather(1, labels.T)
    variance = torch.zeros_like(count).scatter_add_(1, labels.T, centered.square()) / count.clamp_min(1)
    variance = torch.where(count >= 2, variance, reference.variance).clamp_min(1e-8)
    atom = reference.variance == 0
    mean = torch.where(atom, reference.mean, mean)
    variance = torch.where(atom, 0.0, variance)
    return GaussianMixture((count / values.shape[0]).log(), mean, variance)


def hmm_log_prior(value, state, label, parameters):
    p = parameters
    score = p["init_prob"][state[0]].log() + p["transition"][state[:-1], state[1:]].log().sum()
    score += p["mixture_weight"][state, label].log().sum()
    sd, mu = p["prior_sd"][label], p["mu"][state]
    atom = sd == 0
    if (value[atom] != mu[atom]).any():
        return value.new_tensor(-torch.inf)
    sd, mu, v = sd[~atom], mu[~atom], value[~atom]
    density = -0.5 * math.log(2 * math.pi) - sd.log() - 0.5 * ((v - mu) / sd).square()
    if p.get("effect_support") == "nonpositive":
        if (v > 0).any():
            return value.new_tensor(-torch.inf)
        density -= torch.special.log_ndtr(-mu / sd)
    elif p["nonnegative_state_means"]:
        if (v < 0).any():
            return value.new_tensor(-torch.inf)
        density -= torch.special.log_ndtr(mu / sd)
    return score + density.sum()


class HMMVariationalBlock:
    """q(path, components, values) proportional to p_HMM exp(-a*v²/2+b*v)."""

    def __init__(self, a, b, parameters):
        self.a, self.b, self.parameters = a, b, parameters
        mu, sd = parameters["mu"], parameters["prior_sd"]
        positive = parameters["nonnegative_state_means"]
        if parameters.get("effect_support") == "nonpositive":
            mu, b, positive = -mu, -b, True
        mean, variance = mu[None, :, None], sd.square()[None, None, :]
        a, b = a[:, None, None], b[:, None, None]
        denominator = 1 + a * variance
        log_integral = (-0.5 * torch.log1p(a * variance)
                        + (-0.5 * a * mean.square() + b * mean + 0.5 * b.square() * variance) / denominator)
        if positive:
            continuous = variance > 0
            posterior_mean = (mean + b * variance) / denominator
            posterior_sd = torch.where(continuous, variance / denominator, 1.0).sqrt()
            prior_sd = torch.where(continuous, variance, 1.0).sqrt()
            correction = (torch.special.log_ndtr(posterior_mean / posterior_sd)
                          - torch.special.log_ndtr(mean / prior_sd))
            log_integral += torch.where(continuous, correction, 0.0)
        emission = torch.logsumexp(log_integral + parameters["mixture_weight"].log()[None], dim=2)
        filtered = parameters["init_prob"].log() + emission[0]
        log_transition = parameters["transition"].log()
        for j in range(1, len(emission)):
            filtered = emission[j] + torch.logsumexp(filtered[:, None] + log_transition, dim=0)
        self.log_normalizer = torch.logsumexp(filtered, 0)

    def log_prob(self, value, state, label):
        return (hmm_log_prior(value, state, label, self.parameters)
                + (-0.5 * self.a * value.square() + self.b * value).sum() - self.log_normalizer)

    def sample(self, generator):
        value, state, label = sample_hmm(self.a, self.b, self.parameters, generator)
        return value, label, state, self.log_prob(value, state, label)


class MatrixVariationalApproximation:
    """Factorized across axes/columns, with whole HMM paths where applicable."""

    def __init__(self, engine, values, labels):
        reference = [v.mean(0) for v in values]
        residual = engine.y0 - reference[0] @ reference[1].T
        self.blocks = []
        for m, axis in enumerate(engine.axes):
            weight, r = (engine.weight, residual) if m == 0 else (engine.weight.T, residual.T)
            blocks = []
            for k in range(engine.rank):
                other = reference[1 - m][:, k]
                partial = r + torch.outer(reference[m][:, k], other)
                a, b = weight @ other.square(), (weight * partial) @ other
                if axis.name in SCALAR_PRIORS:
                    mixture = axis.priors[k](axis._inputs(reference[m], k))
                elif axis.name == "norm":
                    mixture = axis.fixed[k]
                else:
                    blocks.append(HMMVariationalBlock(a, b, axis.hmm[k]))
                    continue
                posterior, _ = mixture.posterior(a, b)
                blocks.append(fit_marginal_mixture(values[m][:, :, k], labels[m][:, :, k], posterior))
            self.blocks.append(blocks)

    def sample(self, generator):
        values, labels, states = [], [], []
        log_q = 0.0
        for blocks in self.blocks:
            v, z, h = [], [], []
            for block in blocks:
                if isinstance(block, HMMVariationalBlock):
                    value, label, state, score = block.sample(generator)
                else:
                    value, label = block.sample(generator)
                    state, score = None, block.log_prob(value, label).sum()
                v.append(value)
                z.append(label)
                h.append(state)
                log_q += score
            values.append(torch.stack(v, 1))
            labels.append(torch.stack(z, 1))
            states.append(h)
        return values, labels, states, log_q


@torch.no_grad()
def estimate_elbo(engine, values, labels, *, phase="diagnostic"):
    """Monte Carlo estimate of a genuine ELBO for an explicit normalized q.

    The reported standard errors concern iid evaluation draws conditional
    on q and fitted parameters. They do not assess MCMC mixing or the gap
    between q and the posterior. The regularized value bounds the auxiliary
    observation evidence; with penalty=1 it is the ordinary data ELBO.
    Finite Monte Carlo estimates themselves need not lie below the evidence.
    """
    q = MatrixVariationalApproximation(engine, values, labels)
    ordinary, regularized = [], []
    for _ in range(engine.options["elbo_draws"]):
        sample, component, state, log_q = q.sample(engine.elbo_generator)
        likelihood = engine.log_likelihood(sample)
        prior, penalized_prior = 0.0, 0.0
        for m, axis in enumerate(engine.axes):
            prior += axis.log_prior(sample[m], component[m], state[m], regularized=False)
            penalized_prior += axis.log_prior(sample[m], component[m], state[m])
        ordinary.append(likelihood + prior - log_q)
        regularized.append(likelihood + penalized_prior - log_q)
    ordinary, regularized = torch.stack(ordinary), torch.stack(regularized)
    if not torch.isfinite(ordinary).all() or not torch.isfinite(regularized).all():
        raise FloatingPointError("Nonfinite ELBO evaluation; check q support and numerical scales.")
    count = len(ordinary)
    return ELBOEstimate(
        elbo=float(ordinary.mean()), regularized_elbo=float(regularized.mean()),
        standard_error=float(ordinary.std(unbiased=True) / math.sqrt(count)),
        regularized_standard_error=float(regularized.std(unbiased=True) / math.sqrt(count)),
        penalty_term=float((regularized - ordinary).mean()), n_draws=count, phase=phase,
    )
