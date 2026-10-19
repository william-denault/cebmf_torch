"""Independent checks of the normal-means KL and cEBMF evidence bound."""

import importlib
import math

import pytest
import torch
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from cebmf_torch import cEBMF
from cebmf_torch.cebmf.cebmf import normal_means_loglik
from cebmf_torch.ebnm.ash import ash
from cebmf_torch.priors.base import Prior
from cebmf_torch.priors.learned import LearnedBuilder, LearnedPriorType


@pytest.mark.parametrize("side", ["L", "F"])
@pytest.mark.parametrize("regularization", [0.0, 2.3])
def test_factor_kl_and_elbo_match_exact_gaussian_evidence(side, regularization):
    """A known opposite factor leaves a conjugate Gaussian linear model."""
    y = torch.tensor([[0.4, -0.8, 1.1], [1.2, -0.3, 0.5]])
    known_factor = torch.tensor([0.6, -1.0, 1.4])
    prior_sd, noise_sd = 1.3, 0.7
    model = cEBMF(y if side == "L" else y.T, K=1, S=noise_sd, device="cpu")
    unknown = torch.zeros(2, 1)
    known = known_factor[:, None]
    model.initialise_factors(L=unknown if side == "L" else known, F=known if side == "L" else unknown)
    exact_kl = []

    def fit(*, betahat, sebetahat, **kwargs):
        variance = 1 / (prior_sd**-2 + sebetahat.square().reciprocal())
        mean = variance * betahat / sebetahat.square()
        posterior = Normal(mean, variance.sqrt())
        exact_kl.append(kl_divergence(posterior, Normal(0, prior_sd)).sum())
        loss = -Normal(0, (prior_sd**2 + sebetahat.square()).sqrt()).log_prob(betahat).sum()
        return Prior(mean, variance + mean.square(), loss, regularization_loss=regularization)

    getattr(model, f"prior_{side}_fn").fit = fit
    getattr(model, f"_update_{side}_factor")(0, model.Y0, model.tau_map, 1e-12)
    recorded_kl = getattr(model, f"kl_{side.lower()}")[0]
    torch.testing.assert_close(recorded_kl, exact_kl[0], atol=2e-6, rtol=2e-6)
    assert recorded_kl > 0

    model._cal_obj()
    covariance = noise_sd**2 * torch.eye(3) + prior_sd**2 * torch.outer(known_factor, known_factor)
    exact_evidence = MultivariateNormal(torch.zeros(3), covariance).log_prob(y).sum()
    assert model.negative_elbo[-1] == pytest.approx(-exact_evidence.item(), abs=3e-6)
    assert model.obj[-1] == pytest.approx(-exact_evidence.item() + regularization, abs=3e-6)


@pytest.mark.parametrize("prior", ["norm", "exp"])
def test_ash_reports_unclipped_marginal_density(prior):
    x = torch.tensor([-4.0, 0.2, 3.5], dtype=torch.float64)
    s = torch.tensor([0.4, 0.8, 0.6], dtype=torch.float64)
    result = ash(x, s, prior=prior, threshold_loglikelihood=-0.5, verbose=False)
    if prior == "norm":
        log_density = Normal(0, (s[:, None].square() + result.scale.square()).sqrt()).log_prob(x[:, None])
    else:
        rate = result.scale[1:].reciprocal()
        z = x[:, None] / s[:, None] - s[:, None] * rate
        slab = rate.log() - x[:, None] * rate + (s[:, None] * rate).square() / 2 + torch.special.log_ndtr(z)
        log_density = torch.cat((Normal(0, s).log_prob(x)[:, None], slab), dim=1)
    expected = torch.logsumexp(result.pi.log() + log_density, dim=1).sum()
    torch.testing.assert_close(result.log_lik, expected)


@pytest.mark.parametrize("kind", [LearnedPriorType.LCASH, LearnedPriorType.PO_LCASH])
def test_learned_prior_kl_matches_density_integration(kind, monkeypatch):
    """Integrate posterior/prior density ratios, including the discrete spike."""
    learned = importlib.import_module("cebmf_torch.priors.learned")
    solver = learned.builder_functions[kind]
    fitted = []

    def capture(*args, **kwargs):
        result = solver(*args, **kwargs)
        fitted.append(result)
        return result

    monkeypatch.setitem(learned.builder_functions, kind, capture)
    x = torch.tensor([-2.0, -0.4, 0.3, 1.5])
    s = torch.full_like(x, 0.6)
    X = torch.arange(4.0)[:, None]
    result = LearnedBuilder(kind, ash_init=False, verbose=False).fit(X, x, s, internal_epoch=1, device="cpu")
    pi, scale = fitted[0].pi_np.double(), fitted[0].scale.double()
    x, s = x.double(), s.double()
    variance = scale.square()
    component_ll = Normal(0, (s[:, None].square() + variance).sqrt()).log_prob(x[:, None])
    responsibility = torch.softmax(pi.log() + component_ll, dim=1)
    component_variance = variance[1:] * s[:, None].square() / (variance[1:] + s[:, None].square())
    component_mean = component_variance * x[:, None] / s[:, None].square()
    grid = torch.linspace(-8, 8, 20001, dtype=torch.float64)
    log_prior = torch.logsumexp(pi[:, None, 1:].log() + Normal(0, scale[1:]).log_prob(grid[:, None]), dim=2)
    log_posterior = torch.logsumexp(
        responsibility[:, None, 1:].log()
        + Normal(component_mean[:, None, :], component_variance[:, None, :].sqrt()).log_prob(grid[None, :, None]),
        dim=2,
    )
    integrated = torch.trapezoid(log_posterior.exp() * (log_posterior - log_prior), grid, dim=1).sum()
    integrated += (responsibility[:, 0] * (responsibility[:, 0] / pi[:, 0]).log()).sum()
    from_moments = normal_means_loglik(x, s, result.post_mean.double(), result.post_mean2.double()) + result.loss
    assert math.isfinite(integrated.item())
    torch.testing.assert_close(from_moments, integrated, atol=2e-5, rtol=2e-5)
