"""Adversarial checks using analytic posteriors and independent integration."""

import io
import math

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebmf._conditional import Coordinate, LoadingGraph, hermite_rule, tilted_coordinate
from cebmf_torch.priors.conditional import GaussianMixture
from cebmf_torch.utils.posterior import posterior_point_mass_normal
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def profile(mix, a, b, child=lambda v: torch.zeros_like(v), points=48):
    var = mix.variance
    inv = torch.where(var > 0, var.clamp_min(torch.finfo(var.dtype).tiny).reciprocal(), 0.)
    height = mix.log_weight - .5 * (math.log(2 * math.pi) + torch.where(var > 0, var, 1).log())
    height = torch.where(var == 0, mix.log_weight, height)
    return tilted_coordinate(a, b, mix.log_weight, inv, mix.mean, height, var == 0,
                             hermite_rule(points, a), child)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('precision,linear', [(2., 3.), (0., 0.), (1e10, 1.)])
def test_narrow_off_center_slab_matches_analytic_evidence_moments_entropy(dtype, precision, linear):
    # Former float32 implementation: mean=.0125 instead of ~1.59e-29 for
    # a=2, b=3. Natural-parameter cancellation erased the likelihood penalty.
    mix = GaussianMixture(torch.tensor([[.2, .8]], dtype=dtype).log(),
                          torch.tensor([[0., 10.]], dtype=dtype),
                          torch.tensor([[0., 1e-8]], dtype=dtype))
    a, b = torch.tensor([precision], dtype=dtype), torch.tensor([linear], dtype=dtype)
    z, q = profile(mix, a, b)
    exact, exact_z = mix.posterior(a, b)
    w = exact.log_weight.exp()
    exact_mean = (w * exact.mean).sum(1)
    exact_second = (w * (exact.mean.square() + exact.variance)).sum(1)
    entropy = -(w * exact.log_weight).sum(1) + (
        w[:, 1:] * .5 * (math.log(2 * math.pi * math.e) + exact.variance[:, 1:].log())).sum(1)
    tol = 1e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(z, exact_z, rtol=tol, atol=tol)
    torch.testing.assert_close(q.mean, exact_mean, rtol=tol, atol=1e-35)
    torch.testing.assert_close(q.second, exact_second, rtol=tol, atol=1e-35)
    torch.testing.assert_close(q.entropy, entropy, rtol=tol, atol=tol)


def test_child_second_moment_does_not_cancel_for_narrow_coordinate():
    values = torch.tensor([[[9.999, 10.001]]], dtype=torch.float32)
    q = Coordinate(values, torch.full_like(values, .5), torch.zeros(1))
    mass, mean, variance = q.component_centered_moments()
    assert (q.second - q.mean.square()).item() == 0  # raw second moments lose this spread
    assert variance.item() > 0
    torch.testing.assert_close(variance.double(), (values.double() - 10).square().mean(2))
    torch.testing.assert_close(mean, torch.full_like(mean, 10))
    torch.testing.assert_close(mass, torch.ones_like(mass))


def test_uncertain_parent_potential_retains_unnormalized_geometric_mean():
    graph = LoadingGraph.__new__(LoadingGraph)
    def prior(x):
        mu = 10 + .2 * x[:, :1]
        var = .01 * torch.exp(.3 * x[:, :1])
        return GaussianMixture(torch.zeros_like(mu), mu, var)
    graph.priors = [prior]
    inputs = torch.tensor([[[-1.], [2.]]], dtype=torch.float64)
    weights = torch.tensor([[.7, .3]], dtype=torch.float64)
    _, inverse, center, height, _ = graph.coefficients(0, inputs, weights)
    v = torch.tensor([9.5, 10.2, 11.], dtype=torch.float64)
    actual = height[..., None] - .5 * inverse[..., None] * (v - center[..., None]).square()
    mix = prior(inputs.flatten(0, 1))
    expected = (weights.flatten()[:, None] * torch.distributions.Normal(
        mix.mean, mix.variance.sqrt()).log_prob(v)).sum(0)
    torch.testing.assert_close(actual.flatten(), expected)
    assert (height + .5 * (math.log(2 * math.pi) - inverse.log())).item() < -1


def test_nonlinear_child_profile_and_gradient_against_dense_independent_integral():
    # Logistic child probability creates a non-Gaussian tilted posterior.
    # Use a uniform-grid trapezoidal reference, not another Hermite rule.
    theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    mix = GaussianMixture(torch.zeros(1, 1, dtype=theta.dtype), theta.reshape(1, 1),
                          torch.ones(1, 1, dtype=theta.dtype))
    a, b = torch.tensor([1.2], dtype=theta.dtype), torch.tensor([.8], dtype=theta.dtype)
    child = lambda v: 1.7 * torch.nn.functional.logsigmoid(.9 * v - .2)
    z, q = profile(mix, a, b, child, points=64)
    v = torch.linspace(-12, 12, 60001, dtype=theta.dtype)
    psi = -.5 * a * v.square() + b * v + torch.distributions.Normal(theta, 1.).log_prob(v) + child(v)
    density = psi.exp()
    integral = torch.trapezoid(density, v)
    reference_z = integral.log()
    mean = torch.trapezoid(v * density, v) / integral
    entropy = -torch.trapezoid(density * (psi - reference_z), v) / integral
    torch.testing.assert_close(z.squeeze(), reference_z, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(q.mean.squeeze(), mean, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(q.entropy.squeeze(), entropy, atol=1e-10, rtol=1e-10)
    actual_grad, = torch.autograd.grad(z.sum(), theta)
    reference_grad, = torch.autograd.grad(reference_z, theta)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-10, rtol=1e-10)


def test_child_feedback_includes_density_penalty_and_missing_branch_mask():
    graph = LoadingGraph.__new__(LoadingGraph)
    dtype = torch.float64
    def child_prior(x):
        v = x[:, :1]
        logits = .7 * v - .3
        return GaussianMixture(torch.cat((torch.nn.functional.logsigmoid(-logits),
                                           torch.nn.functional.logsigmoid(logits)), 1),
                               torch.cat((torch.zeros_like(v), .4 + .6 * v), 1),
                               torch.cat((torch.zeros_like(v), (.2 * v).exp()), 1))
    own = GaussianMixture(torch.tensor([[.3, .7], [.5, .5]], dtype=dtype).log(),
                          torch.tensor([[0., 1.], [0., -.2]], dtype=dtype),
                          torch.tensor([[0., .2], [0., .7]], dtype=dtype))
    _, child_q = profile(own, torch.zeros(2, dtype=dtype), torch.zeros(2, dtype=dtype))
    graph.q = [child_q, child_q]
    graph.parents, graph.children = [[], [0]], [[1], []]
    graph.external = [torch.empty(2, 0, dtype=dtype), torch.empty(2, 0, dtype=dtype)]
    graph.priors = [None, child_prior]
    graph.training = [{'penalty': 1.}, {'penalty': 1.4}]
    graph.active = [torch.ones(2, dtype=torch.bool), torch.tensor([True, False])]
    candidates = torch.tensor([[[-1., .5, 2.]], [[-.2, .3, 1.]]], dtype=dtype, requires_grad=True)
    actual = graph.child_term(0, torch.arange(2), candidates, draws=None)
    # Integrate child value and label directly, without using component moments.
    expected_rows = []
    for row in range(2):
        scores = []
        for value in candidates[row, 0]:
            mix = child_prior(value.reshape(1, 1))
            atom_score = child_q.weights[row, 0].sum() * mix.log_weight[0, 0]
            slab_logp = torch.distributions.Normal(mix.mean[0, 1], mix.variance[0, 1].sqrt()).log_prob(
                child_q.values[row, 1]) + mix.log_weight[0, 1]
            scores.append(atom_score + (child_q.weights[row, 1] * slab_logp).sum()
                          + .4 * mix.log_weight[0, 0])
        expected_rows.append(torch.stack(scores) * graph.active[1][row])
    expected = torch.stack(expected_rows)[:, None]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    grad, = torch.autograd.grad(actual.sum(), candidates, retain_graph=True)
    ref_grad, = torch.autograd.grad(expected.sum(), candidates)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-12, atol=1e-12)


def test_point_normal_posterior_stays_correct_when_both_densities_underflow():
    # At x=-100, both densities underflow, but spike has essentially all mass.
    x = torch.tensor([-100., 100.], dtype=torch.float64)
    s = torch.ones_like(x)
    mean, var = posterior_point_mass_normal(x, s, .5, 0., 10., torch.tensor(1e-4))
    torch.testing.assert_close(mean[0], torch.zeros_like(mean[0]), atol=1e-100, rtol=0)
    assert mean[1] > 9.9
    assert torch.isfinite(var).all() and (var >= 0).all()


def test_normal_mixture_loglik_does_not_clip_tail_component_odds():
    x = torch.tensor([1000.], dtype=torch.float64)
    scales = torch.tensor([0., .01], dtype=x.dtype)
    actual = get_data_loglik_normal_torch(x, torch.ones_like(x), torch.zeros_like(scales), scales)
    expected = torch.distributions.Normal(torch.zeros_like(scales), (1 + scales.square()).sqrt()).log_prob(x)
    torch.testing.assert_close(actual.squeeze(), expected)
    assert (actual[0, 1] - actual[0, 0]) > 49  # old clipping made these equally likely


@pytest.mark.parametrize('axis', ['row', 'column', 'both'])
def test_serialized_conditional_fit_continues_without_rebuilding_identity_caches(axis):
    torch.manual_seed(8)
    m = cEBMF(torch.randn(10, 7, dtype=torch.float64), K=2, device='cpu',
              prior_L='cgb', prior_F='cgb' if axis != 'row' else 'norm',
              self_row_cov=axis != 'column', self_col_cov=axis != 'row',
              allow_backfitting=False, internal_epoch=1,
              prior_L_kwargs=dict(hidden_dim=3, n_layers=0, penalty=1),
              prior_F_kwargs=dict(hidden_dim=3, n_layers=0, penalty=1) if axis != 'row' else dict(penalty=1),
              conditional_kwargs=dict(quadrature_points=8, parent_samples=8))
    m.initialise_factors()
    m.fit(1)
    file = io.BytesIO()
    torch.save(m, file)
    file.seek(0)
    restored = torch.load(file, weights_only=False)
    m.fit(1)
    restored.fit(1)
    torch.testing.assert_close(restored.L, m.L, atol=0, rtol=0)
    torch.testing.assert_close(restored.F, m.F, atol=0, rtol=0)
    torch.testing.assert_close(restored.obj[-1], m.obj[-1], atol=0, rtol=0)
