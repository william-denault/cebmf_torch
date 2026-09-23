"""Analytic identities and end-to-end checks for local quadratic feedback."""

import io
import math
import warnings
from unittest.mock import patch

import pytest
import torch

from cebmf_torch import cEBMF, fit_joint
from cebmf_torch.cebmf._conditional import hermite_rule, integration_options, tilted_coordinate
from cebmf_torch.cebmf._quadratic import local_feedback, quadratic_coordinate


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def coefficients(weight, mean, variance):
    atoms = variance == 0
    safe = torch.where(atoms, 1, variance)
    inverse = torch.where(atoms, 0, safe.reciprocal())
    height = torch.where(atoms, weight.log(), weight.log() - .5 * (math.log(2 * math.pi) + safe.log()))
    return weight.log(), inverse, mean, height, atoms


@pytest.mark.parametrize('strength', [0., .7, 12.])
def test_concave_quadratic_matches_independent_gaussian_solution_and_gradient(strength):
    dtype = torch.float64
    a, b = torch.tensor([2.], dtype=dtype), torch.tensor([.8], dtype=dtype)
    mean = torch.tensor([[0., -1.3, 2.]], dtype=dtype, requires_grad=True)
    variance = torch.tensor([[0., .4, 1.2]], dtype=dtype)
    weight = torch.tensor([[.2, .3, .5]], dtype=dtype)
    anchor = torch.tensor([[[0.], [-.9], [1.4]]], dtype=dtype)
    function = lambda x: -.3 + .6 * x - .5 * strength * x.square()
    local = local_feedback(function, anchor)
    z, q = quadratic_coordinate(a, b, *coefficients(weight, mean, variance), hermite_rule(8, a), local)

    # Independent completion directly in natural parameters (moderate scales).
    p = a[:, None] + 1 / variance[:, 1:] + strength
    natural_mean = b[:, None] + mean[:, 1:] / variance[:, 1:] + .6
    expected_mean = natural_mean / p
    log_mass = weight[:, 1:].log() - .5 * (variance[:, 1:].log() + p.log()) - .3
    log_mass = log_mass - .5 * mean[:, 1:].square() / variance[:, 1:] + .5 * natural_mean.square() / p
    log_mass = torch.cat((weight[:, :1].log() - .3, log_mass), 1)
    expected_z = log_mass.logsumexp(1)
    mass = torch.softmax(log_mass, 1)
    expected_mean = torch.cat((torch.zeros_like(a[:, None]), expected_mean), 1)
    expected_var = torch.cat((torch.zeros_like(a[:, None]), 1 / p), 1)
    entropy = -(mass * torch.log_softmax(log_mass, 1)).sum(1)
    entropy += (mass[:, 1:] * .5 * (math.log(2 * math.pi * math.e) - p.log())).sum(1)
    for actual, expected in zip(q.component_centered_moments(), (mass, expected_mean, expected_var)):
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(z, expected_z, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(q.entropy, entropy, atol=2e-13, rtol=2e-13)
    actual_gradient, = torch.autograd.grad(z.sum(), mean, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected_z.sum(), mean)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-13, rtol=2e-13)
    # The integration representation must reproduce the stored analytic moments.
    torch.testing.assert_close((q.weights * q.values).sum((1, 2)), q.mean, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close((q.weights * q.values.square()).sum((1, 2)), q.second, atol=2e-13, rtol=2e-13)


def test_logistic_feedback_derivatives_and_exact_spike_mass():
    anchor = torch.tensor([[[0.], [1.2]]], dtype=torch.float64)
    r, slope, intercept = .3, 1.7, -.2
    function = lambda x: r * (slope * x + intercept) - torch.nn.functional.softplus(slope * x + intercept)
    local = local_feedback(function, anchor)
    pi = torch.sigmoid(slope * anchor + intercept)
    torch.testing.assert_close(local.slope, slope * (r - pi))
    torch.testing.assert_close(local.curvature, -slope**2 * pi * (1 - pi))
    assert not local.clipped.any()
    assert all(not v.requires_grad for v in (local.anchor, local.height, local.slope, local.curvature))
    weight = torch.tensor([[.4, .6]], dtype=anchor.dtype)
    mean, variance = torch.tensor([[0., 1.]], dtype=anchor.dtype), torch.tensor([[0., .3]], dtype=anchor.dtype)
    a, b = torch.tensor([2.], dtype=anchor.dtype), torch.tensor([1.], dtype=anchor.dtype)
    z, q = quadratic_coordinate(a, b, *coefficients(weight, mean, variance), hermite_rule(12, a), local)
    torch.testing.assert_close(q.gaussian_moments[0][:, 0].log() + z,
                               weight[:, 0].log() + function(anchor)[:, 0, 0])


def test_positive_curvature_is_clipped_and_posterior_remains_proper():
    a, b = torch.tensor([.01], dtype=torch.float64), torch.tensor([.03], dtype=torch.float64)
    mean, variance = torch.tensor([[0., .3]], dtype=a.dtype), torch.tensor([[0., 10.]], dtype=a.dtype)
    weight = torch.tensor([[.2, .8]], dtype=a.dtype)
    local = local_feedback(lambda x: 100 * x.square(), mean[:, :, None])
    assert local.clipped.all() and (local.curvature == 0).all()
    z, q = quadratic_coordinate(a, b, *coefficients(weight, mean, variance), hermite_rule(8, a), local)
    assert torch.isfinite(z).all() and torch.isfinite(q.second).all()
    torch.testing.assert_close(q.gaussian_moments[2][:, 1], 1 / (a + .1))
    # Constant and affine feedback also work when second derivatives have no graph.
    for function in (lambda x: torch.zeros_like(x), lambda x: 2 * x):
        assert (local_feedback(function, mean[:, :, None]).curvature == 0).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_narrow_slab_preserves_tail_odds(dtype):
    a, b = torch.tensor([2.], dtype=dtype), torch.tensor([3.], dtype=dtype)
    mean, variance = torch.tensor([[0., 10.]], dtype=dtype), torch.tensor([[0., 1e-8]], dtype=dtype)
    weight = torch.tensor([[.2, .8]], dtype=dtype)
    coeff = coefficients(weight, mean, variance)
    local = local_feedback(lambda x: torch.zeros_like(x), mean[:, :, None])
    z, q = quadratic_coordinate(a, b, *coeff, hermite_rule(8, a), local)
    reference_z, reference_q = tilted_coordinate(a, b, *coeff, hermite_rule(24, a), lambda v: torch.zeros_like(v))
    torch.testing.assert_close(q.mean, reference_q.mean, rtol=2e-5, atol=1e-35)
    torch.testing.assert_close(z, reference_z)
    assert q.mean.item() < 1e-27


def model(y, prior='cgb', **kwargs):
    options = dict(K=3, device=y.device, prior_L=prior, prior_F='norm', self_row_cov=True,
                   allow_backfitting=False, internal_epoch=2,
                   prior_L_kwargs=dict(hidden_dim=5, n_layers=0,
                                       penalty=1.1 if prior == 'spiked_emdn' else 1.0511),
                   prior_F_kwargs=dict(penalty=1, scales=y.new_tensor([0., .1, .5, 2., 8.])),
                   conditional_kwargs=dict(approximation='quadratic', quadrature_points=8, parent_samples=8))
    options.update(kwargs)
    m = cEBMF(y, **options)
    m.initialise_factors()
    return m


@pytest.mark.parametrize('options', [None, {}, {'quadrature_points': 8, 'parent_samples': 8}])
def test_default_matches_explicit_quadratic_and_warns_once_with_alternative(options):
    torch.manual_seed(52)
    y = torch.randn(12, 7)
    fitted = []
    for numerical in (options, {**(options or {}), 'approximation': 'quadratic'}):
        torch.manual_seed(23)
        m = model(y, K=2, conditional_kwargs=numerical)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = m.fit(1)
            m.iter_once()
            m.fit(0)
        assert len(caught) == 1
        message = str(caught[0].message)
        assert caught[0].category is UserWarning
        assert 'local quadratic approximation' in message and 'faster processing' in message
        assert "conditional_kwargs={'approximation': 'quadrature'}" in message
        assert result.inference == 'conditional_quadratic'
        fitted.append((m.L, m.L2, m.F, m.F2, torch.stack(m.obj)))
    for actual, expected in zip(*fitted):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_explicit_quadrature_does_not_warn_about_quadratic_approximation():
    m = model(torch.randn(12, 7), K=2, conditional_kwargs={'approximation': 'quadrature'})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = m.fit(1)
    assert not caught
    assert result.inference == 'conditional_variational'


@pytest.mark.parametrize('rank,self_cov', [(1, True), (2, False)])
def test_default_without_latent_edges_does_not_warn(rank, self_cov):
    m = model(torch.randn(12, 7), K=rank, self_row_cov=self_cov, conditional_kwargs=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = m.fit(1)
    assert not caught
    assert result.inference == 'variational' and m.conditional_fit is None


@pytest.mark.parametrize('axis', ['column', 'both', 'coupled'])
def test_default_column_and_coupled_graphs_warn_once(axis):
    kwargs = dict(K=2, conditional_kwargs={}, self_row_cov=axis == 'both',
                  self_col_cov=axis != 'coupled')
    if axis != 'coupled':
        kwargs.update(prior_F='cgb', prior_F_kwargs=dict(hidden_dim=4, n_layers=0, penalty=1))
    m = model(torch.randn(12, 7), **kwargs)
    other = model(torch.randn(12, 6), K=2, conditional_kwargs={}, self_row_cov=False) if axis == 'coupled' else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if other is not None:
            result, second = fit_joint(m, other, maxit=1)
            fit_joint(m, other, maxit=1)
            assert second.inference == 'conditional_quadratic'
        else:
            result = m.fit(1)
            m.fit(1)
    assert len(caught) == 1
    assert 'local quadratic approximation' in str(caught[0].message)
    assert result.inference == 'conditional_quadratic'


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_narrow_child_feedback_preserves_normalizer_not_only_mean(dtype):
    # A broad own posterior and far-away narrow child mode used to yield a
    # plausible mean (10) but a log integral off by 121.5 in float32.
    a, b = torch.tensor([2.], dtype=dtype), torch.tensor([3.], dtype=dtype)
    weight = torch.tensor([[.4, .6]], dtype=dtype)
    mean, variance = torch.tensor([[0., 4.]], dtype=dtype), torch.ones_like(weight)
    sd2, target = 1e-8, 10.
    function = lambda value: -.5 * (math.log(2 * math.pi * sd2) + (value - target).square() / sd2)
    local = local_feedback(function, mean[:, :, None])
    z, q = quadratic_coordinate(a, b, *coefficients(weight, mean, variance), hermite_rule(8, a), local)
    mu = mean.double()
    base_mean, base_var = (mu + 3) / 3, 1 / 3
    base_log_mass = weight.double().log() - .5 * math.log(3) - .5 * mu.square() + (mu + 3).square() / 6
    expected_log_mass = base_log_mass - .5 * (math.log(2 * math.pi * (base_var + sd2))
                                            + (base_mean - target).square() / (base_var + sd2))
    expected_z = expected_log_mass.logsumexp(1)
    torch.testing.assert_close(z.double(), expected_z, rtol=0, atol=5e-6)
    # Component probabilities can be wrong even when every component's mean
    # rounds to the same child mode. Check the tiny component's odds relatively.
    torch.testing.assert_close(q.gaussian_moments[0].double(), expected_log_mass.softmax(1), rtol=5e-6, atol=1e-20)
    expected_mean = (3. + mu + target / sd2) / (3. + 1 / sd2)
    torch.testing.assert_close(q.gaussian_moments[1].double(), expected_mean, rtol=0, atol=5e-7)
    assert local.height.dtype == local.slope.dtype == local.curvature.dtype == torch.float64
    assert q.mean.dtype == q.entropy.dtype == dtype


@pytest.mark.parametrize('prior', ['cgb', 'spiked_emdn'])
def test_feedback_is_cached_through_prior_training_and_respects_frozen_flags(prior):
    torch.manual_seed(14)
    m = model(torch.randn(20, 8), prior)
    m.fit(0)
    graph = m.conditional_fit.row
    frozen = next(graph.priors[1].parameters())
    frozen.requires_grad_(False)
    with patch.object(graph, 'child_term', wraps=graph.child_term) as calls:
        graph.update(0)
    assert calls.call_count == len(graph.batches[0])
    assert not frozen.requires_grad
    assert graph.history[-1]['after'] >= graph.history[-1]['before']
    for tensor in graph.q[0].gaussian_moments:
        assert tensor.grad_fn is None


@pytest.mark.parametrize('prior', ['cgb', 'cgb_sharp', 'spiked_emdn'])
@pytest.mark.parametrize('rank,self_cov', [(1, True), (3, False)])
def test_quadratic_option_with_no_latent_edges_is_original_cebmf(prior, rank, self_cov):
    torch.manual_seed(4)
    y, x = torch.randn(16, 7), torch.randn(16, 2)
    results = []
    for approximation in ('quadrature', 'quadratic'):
        torch.manual_seed(34)
        row = dict(hidden_dim=4, n_layers=0, penalty=1.1 if prior == 'spiked_emdn' else 1.0511)
        if prior == 'cgb_sharp':
            row['omega'] = .01
        m = model(y, prior, K=rank, X_l=x, self_row_cov=self_cov, prior_L_kwargs=row,
                  conditional_kwargs={'approximation': approximation})
        m.fit(2)
        assert m.conditional_fit is None
        results.append((m.L, m.L2, m.F, m.F2, torch.stack(m.obj)))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('prior', ['cgb', 'spiked_emdn'])
def test_joint_and_column_graphs_missing_rows_and_checkpoint_continuation(prior):
    torch.manual_seed(123)
    ya, yr = torch.randn(12, 8), torch.randn(12, 7)
    ya[-3:] = torch.nan
    yr[:3] = torch.nan
    first = model(ya, prior, K=2)
    second = model(yr, prior, K=2, self_col_cov=True, prior_F='cgb',
                   prior_F_kwargs=dict(hidden_dim=4, n_layers=0, penalty=1.0511))
    a, b = fit_joint(first, second, maxit=2)
    assert a.inference == b.inference == 'conditional_quadratic'
    assert torch.isfinite(a.L).all() and torch.isfinite(b.L).all()
    graph = first.conditional_fit.row
    for u in range(2, 4):
        assert not graph.active[u][:3].any()
    stream = io.BytesIO()
    torch.save((first, second), stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    fit_joint(first, second, maxit=1)
    fit_joint(*restored, maxit=1)
    for original, saved in zip((first, second), restored):
        torch.testing.assert_close(original.L, saved.L, atol=0, rtol=0)
        torch.testing.assert_close(original.F, saved.F, atol=0, rtol=0)
        torch.testing.assert_close(torch.stack(original.obj), torch.stack(saved.obj), atol=0, rtol=0)


def test_invalid_approximation_rejected():
    with pytest.raises(ValueError, match='approximation'):
        integration_options({'approximation': 'plug_in'})


@pytest.mark.parametrize('prior', ['cgb', 'cgb_sharp', 'cgb_sharp_2', 'cash', 'lcash', 'po_lcash', 'emdn', 'spiked_emdn'])
def test_all_supported_conditional_families_have_finite_quadratic_updates(prior):
    torch.manual_seed(71)
    row = dict(n_epochs=1, hidden_dim=4, n_layers=0, penalty=1 if prior == 'emdn' else 1.1)
    m = model(torch.randn(12, 6, dtype=torch.float64), prior, K=2, prior_L_kwargs=row)
    result = m.fit(2)
    assert result.inference == 'conditional_quadratic'
    assert torch.isfinite(m.L).all() and torch.isfinite(torch.stack(m.obj)).all()
    for q in m.conditional_fit.row.q:
        assert q.gaussian_moments is not None
        assert (q.gaussian_moments[2] >= 0).all()
