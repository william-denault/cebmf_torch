"""Independent density, quadrature, detailed-balance and sampling checks."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from cebmf_torch.experimental.conditional import (
    SCALAR_PRIORS,
    ConditionalMixture,
    GaussianMixture,
    metropolis_normal_means,
)
from cebmf_torch.experimental.joint import JointATACRNA


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


@pytest.mark.parametrize("name", SCALAR_PRIORS)
def test_adapter_matches_original_network_and_has_finite_gradients(name):
    torch.manual_seed(14)
    x = torch.randn(19, 3, dtype=torch.float64)
    prior = ConditionalMixture(name, x)
    mix = prior(x)
    original = prior.net(prior.standardize(x))
    if name in ("cash", "lcash", "po_lcash"):
        weight = original
        mean = torch.zeros_like(weight)
        var = prior.grid.square().expand_as(mean)
    elif name in ("cgb", "cgb_sharp"):
        weight = torch.stack(original[:2], 1)
        mean = torch.stack((original[2] * 0, original[2])).expand_as(weight)
        var = torch.cat((tensor([0]), (2 * prior.log_slab_sd).exp())).expand_as(mean)
    elif name == "cgb_sharp_2":
        weight = torch.stack(original[:3], 1)
        mean = torch.stack((original[3] * 0, original[3], original[4])).expand_as(weight)
        var = torch.cat((tensor([0]), (2 * prior.log_slab_sd).exp())).expand_as(mean)
    else:
        weight, mean, log_sd = original
        var = (2 * log_sd).exp()
        if name == "spiked_emdn":
            mean = torch.cat((torch.zeros(19, 1), mean), 1)
            var = torch.cat((torch.zeros(19, 1), var), 1)
    torch.testing.assert_close(mix.log_weight.exp(), weight, atol=2e-8, rtol=2e-7)
    torch.testing.assert_close(mix.mean, mean)
    torch.testing.assert_close(mix.variance, var)
    v, z = mix.sample(torch.Generator().manual_seed(12))
    loss = -mix.log_prob(v.detach(), z).mean()
    loss.backward()
    grads = [p.grad for p in prior.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if name in ("cgb_sharp", "cgb_sharp_2"):
        assert not prior.log_slab_sd.requires_grad


@pytest.mark.parametrize("a,b", [(0.0, 0.0), (0.1, -1.0), (3.0, 2.0), (100.0, -30.0)])
def test_normal_means_integrals_against_quadrature(a, b):
    # Includes nonzero atoms: independent quadrature integrates the original
    # density times exp(-a*v*v/2+b*v), not the implementation's completed square.
    weight, mu, variance = tensor([[0.1, 0.2, 0.3, 0.4]]), tensor([[0, 1.3, -0.4, 0.8]]), tensor([[0, 0, 0.7, 0.2]])
    mix = GaussianMixture(weight.log(), mu, variance)
    post, evidence = mix.posterior(tensor([a]), tensor([b]))
    raw = np.zeros(3)
    masses = []
    for h in range(4):
        location, var, w = float(mu[0, h]), float(variance[0, h]), float(weight[0, h])
        if var == 0:
            moments = np.array([location**r * math.exp(-a * location**2 / 2 + b * location) for r in range(3)])
        else:
            grid = np.linspace(-12, 12, 240001)
            density = np.exp(-0.5 * (grid - location) ** 2 / var) / math.sqrt(2 * math.pi * var)
            density *= np.exp(-a * grid**2 / 2 + b * grid)
            moments = np.array([np.trapezoid(grid**r * density, grid) for r in range(3)])
        raw += w * moments
        masses.append(w * moments[0])
    torch.testing.assert_close(post.log_weight.exp()[0], tensor(masses) / raw[0], atol=1e-9, rtol=1e-8)
    assert float(evidence.exp()) == pytest.approx(raw[0], rel=1e-8)
    assert float((post.log_weight.exp() * post.mean).sum()) == pytest.approx(raw[1] / raw[0], abs=1e-8)
    assert float((post.log_weight.exp() * (post.mean.square() + post.variance)).sum()) == pytest.approx(
        raw[2] / raw[0], abs=1e-8
    )


def test_tiny_variance_missing_likelihood_and_mixed_reference_measure():
    mix = GaussianMixture(tensor([[0.3, 0.7]]).log(), tensor([[0, 1.0]]), tensor([[0, 1e-200]]))
    post, evidence = mix.posterior(tensor([0]), tensor([0]))
    torch.testing.assert_close(post.log_weight, mix.log_weight)
    torch.testing.assert_close(post.mean, mix.mean)
    torch.testing.assert_close(post.variance, mix.variance, atol=0, rtol=0)
    assert abs(float(evidence)) < 1e-15
    # At zero, continuous density cannot be added to the spike probability.
    both_zero = GaussianMixture(tensor([[0.3, 0.7]]).log(), tensor([[0, 0]]), tensor([[0, 1]]))
    assert float(both_zero.marginal_log_prob(tensor([0]))) == pytest.approx(math.log(0.3))
    assert both_zero.log_prob(tensor([1]), torch.tensor([0])).isneginf().all()


def make_graph(parent_name, child_name, n=5):
    """A real two-node graph, with a fixed side covariate and fixed children."""
    torch.manual_seed(35)
    graph = JointATACRNA.__new__(JointATACRNA)
    graph.side_info = torch.ones(n, 1, dtype=torch.float64) * 0.3
    graph.loadings = torch.stack((torch.linspace(0.2, 0.9, n), torch.full((n,), 0.8)), 1).double()
    graph.components = torch.ones(n, 2, dtype=torch.long)
    graph.priors = nn.ModuleList(
        [
            ConditionalMixture(parent_name, tensor([[-1], [1]])),
            ConditionalMixture(child_name, tensor([[-1, -1], [1, 1]])),
        ]
    )
    return graph


@pytest.mark.parametrize("parent", SCALAR_PRIORS)
@pytest.mark.parametrize("child", SCALAR_PRIORS)
def test_mh_ratio_equals_entire_joint_ratio_for_every_family_pair(parent, child):
    g = make_graph(parent, child)
    old = g.loadings[:, 0].clone()
    proposed = tensor([-0.4, 0.0, 0.5, 1.1, 1.7])
    zold = g.components[:, 0].clone()
    znew = zold.clone()
    if g.priors[0].has_spike:
        znew[1] = 0
    own = g.priors[0](g._inputs(g.loadings, 0))
    a, b = tensor([0, 1, 2, 3, 4]), tensor([0, -0.3, 0.2, 2, 1])
    proposal, _ = own.posterior(a, b)

    # Evaluate the complete two-node augmented target independently of the
    # implementation's child_log_prior and normal-means cancellation.
    def joint(v, z):
        x = torch.stack((g.side_info[:, 0], v), 1)
        child_prior = g.priors[1](x)
        return (
            -0.5 * a * v.square()
            + b * v
            + own.log_prob(v, z)
            + child_prior.log_prob(g.loadings[:, 1], g.components[:, 1])
        )

    full_ratio = (
        joint(proposed, znew) - joint(old, zold) + proposal.log_prob(old, zold) - proposal.log_prob(proposed, znew)
    )
    correction = g.child_log_prior(0, proposed) - g.child_log_prior(0, old)
    torch.testing.assert_close(full_ratio, correction, atol=1e-9, rtol=1e-9)
    # Numerical detailed balance, including a transition to the exact atom.
    forward = joint(old, zold) + proposal.log_prob(proposed, znew) + correction.clamp_max(0)
    reverse = joint(proposed, znew) + proposal.log_prob(old, zold) + (-correction).clamp_max(0)
    torch.testing.assert_close(forward, reverse, atol=1e-9, rtol=1e-9)
    assert torch.equal(g.child_log_prior(1, old), torch.zeros_like(old))


@pytest.mark.parametrize("child", SCALAR_PRIORS)
def test_mh_stationary_moments_against_numerical_quadrature(child):
    # 6000 independent chains run in parallel. Sampling errors below are
    # based on the independent final draws, not autocorrelated chain output.
    g = make_graph("cgb", child, n=6000)
    n = len(g.loadings)
    a, b = torch.ones(n, dtype=torch.float64), torch.full((n,), 0.7, dtype=torch.float64)
    prior = g.priors[0](g._inputs(g.loadings, 0))
    proposal, _ = prior.posterior(a, b)
    roots, weights = np.polynomial.hermite.hermgauss(160)
    expectations = torch.zeros(4, dtype=torch.float64)
    # Integrate under the normal-means proposal to resolve narrow slabs;
    # its own mass times the child density is the unnormalized joint target.
    for h in range(2):
        m, v, w = (
            proposal.mean[0, h].detach(),
            proposal.variance[0, h].detach(),
            proposal.log_weight[0, h].exp().detach(),
        )
        points = m + (2 * v).sqrt() * tensor(roots)
        cx = torch.stack((torch.full_like(points, 0.3), points), 1)
        density = (
            g.priors[1](cx)
            .log_prob(torch.full_like(points, 0.8), torch.ones(len(points), dtype=torch.long))
            .detach()
            .exp()
        )
        mass = w * tensor(weights) / math.sqrt(math.pi) * density
        expectations += torch.stack(
            (mass.sum(), (mass * points).sum(), (mass * points.square()).sum(), mass.sum() * (h == 0))
        )
    target_mean, target_second, target_spike = expectations[1:] / expectations[0]
    generator = torch.Generator().manual_seed(840)
    old, z = proposal.sample(generator)
    for _ in range(35):
        old, z, accept = metropolis_normal_means(prior, a, b, old, z, lambda v: g.child_log_prior(0, v), generator)
    assert float(accept.double().mean()) > 0.1
    for samples, truth in ((old, target_mean), (old.square(), target_second), ((z == 0).double(), target_spike)):
        error = 6 * samples.std(unbiased=True) / math.sqrt(n) + 5e-4
        assert abs(samples.mean() - truth) < error


def test_emdn_child_mean_and_variance_are_included_even_with_constant_weights():
    g = make_graph("cgb", "emdn", n=4)
    child = g.priors[1]
    with torch.no_grad():
        child.net.pi.weight.zero_()
        child.net.pi.bias.zero_()
        for layer in (child.net.fc_in, *child.net.hidden_layers, child.net.mu, child.net.log_sigma):
            layer.weight.fill_(0.06)
            layer.bias.fill_(0.1)
    lo, hi = torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
    first = child(torch.stack((g.side_info[:, 0], lo), 1))
    second = child(torch.stack((g.side_info[:, 0], hi), 1))
    torch.testing.assert_close(first.log_weight, second.log_weight)
    assert not torch.allclose(g.child_log_prior(0, hi), g.child_log_prior(0, lo))


def test_standardization_is_fixed_and_rows_do_not_influence_each_other():
    p = ConditionalMixture("spiked_emdn", tensor([[-1, -1], [1, 1]]))
    x = tensor([[0.2, 0.3], [0.5, -0.7]])
    first = p(x)
    x[1] = tensor([200, -700])
    second = p(x)
    for field in ("log_weight", "mean", "variance"):
        torch.testing.assert_close(getattr(first, field)[0], getattr(second, field)[0])


@pytest.mark.parametrize("name", SCALAR_PRIORS)
def test_exact_score_against_centered_finite_differences(name):
    graph = make_graph("cgb", name)
    value, component = tensor([0.21, 0.38, 0.72, 0.91, 1.07]), torch.ones(5, dtype=torch.long)
    a, b = torch.ones(5, dtype=torch.float64), tensor([0, 0.2, 0.1, -0.3, 0.8])
    score = graph.loading_score(0, value, component, a, b)
    eps = 1e-5
    finite_difference = (
        graph.loading_log_conditional(0, value + eps, component, a, b)
        - graph.loading_log_conditional(0, value - eps, component, a, b)
    ) / (2 * eps)
    torch.testing.assert_close(score, finite_difference, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="spike"):
        graph.loading_score(0, value * 0, component * 0, a, b)


def test_proportional_odds_extreme_inputs_and_bad_configuration():
    p = ConditionalMixture("po_lcash", tensor([[-1], [1]]))
    with torch.no_grad():
        p.net.w.fill_(1)
    mix = p(tensor([[-1000], [1000]]))
    assert torch.isfinite(mix.log_weight).all()
    torch.testing.assert_close(mix.log_weight.exp().sum(1), tensor([1, 1]))
    for kwargs in ({"omega": 0}, {"scales": [0, 0, 1]}, {"scales": [0, 2, 1]}):
        with pytest.raises(ValueError):
            ConditionalMixture("cash", tensor([[-1], [1]]), **kwargs)
    with pytest.raises(ValueError, match="scalar learned"):
        ConditionalMixture("hmm", tensor([[-1], [1]]))
    with pytest.raises(ValueError, match="finite"):
        p(tensor([[float("nan")]]))


@pytest.mark.parametrize("rna_observed", [False, True])
def test_joint_missing_modality_marginal_against_quadrature(rna_observed):
    # The child is now sampled too. When its likelihood is absent it must
    # integrate to one, leaving the parent's own marginal posterior intact.
    g = make_graph("cgb", "spiked_emdn", n=5000)
    n = len(g.loadings)
    with torch.no_grad():
        child = g.priors[1]
        child.net.pi.weight.zero_()
        child.net.pi.bias.copy_(tensor([-1, 0, -0.3]))
        for layer in (child.net.fc_in, *child.net.hidden_layers):
            layer.weight.fill_(0.15)
            layer.bias.fill_(0.2)
        child.net.mu.weight.fill_(0.3)
        child.net.mu.bias.fill_(0.05)
    # RNA-only: parent has no direct data. ATAC-only: child has no data.
    a = torch.full((n,), 0.0 if rna_observed else 1.0, dtype=torch.float64)
    b = a * 0.7
    ar = torch.full((n,), 4.0 if rna_observed else 0.0, dtype=torch.float64)
    br = ar * 0.9
    prior = g.priors[0](g._inputs(g.loadings, 0))
    base, _ = prior.posterior(a, b)
    roots, weights = np.polynomial.hermite.hermgauss(160)
    target = tensor([0, 0])
    for h in range(2):
        v = base.mean[0, h].detach() + (2 * base.variance[0, h]).detach().sqrt() * tensor(roots)
        if rna_observed:
            child = g.priors[1](torch.stack((torch.full_like(v, 0.3), v), 1))
            # Independent convolution of the observed RNA likelihood and
            # each Gaussian/atomic child component.
            variance = child.variance + 0.25
            lik = (
                (
                    child.log_weight
                    - 0.5 * ((0.9 - child.mean).square() / variance + variance.log() + math.log(2 * math.pi))
                )
                .exp()
                .sum(1)
                .detach()
            )
        else:
            lik = torch.ones_like(v)
        mass = base.log_weight[0, h].exp().detach() * tensor(weights) / math.sqrt(math.pi) * lik
        target += torch.stack((mass.sum(), (mass * v).sum()))
    expected_mean = target[1] / target[0]
    generator = torch.Generator().manual_seed(918)
    g.loadings[:, 0], g.components[:, 0] = base.sample(generator)
    with torch.no_grad():
        for _ in range(60):
            g.loadings[:, 0], g.components[:, 0], _ = metropolis_normal_means(
                prior, a, b, g.loadings[:, 0], g.components[:, 0], lambda v: g.child_log_prior(0, v), generator
            )
            child, _ = g.priors[1](g._inputs(g.loadings, 1)).posterior(ar, br)
            g.loadings[:, 1], g.components[:, 1] = child.sample(generator)
    values = g.loadings[:, 0]
    assert abs(values.mean() - expected_mean) < 6 * values.std() / math.sqrt(n) + 5e-4


def test_supported_scalar_families_cover_the_registry():
    from cebmf_torch.priors.learned import LearnedPriorType

    assert set(SCALAR_PRIORS) == {str(prior) for prior in LearnedPriorType if not str(prior).startswith("hmm")}
