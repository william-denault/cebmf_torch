"""Real CUDA tests are skipped, never simulated, when CUDA is unavailable."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cebmf_torch import ash, cEBMF, fit_joint
from cebmf_torch.cebmf._conditional import tilted_coordinate, hermite_rule
from cebmf_torch.utils.mixture import optimize_pi_logL


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware/runtime unavailable')


class NoHostRead(TorchDispatchMode):
    """Reject host reads, dynamic indices and array transfers.

    Initial uploads of scalar literals are allowed; extracting any scalar
    from a device tensor, or copying it back to CPU, is not.
    """
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        assert func != torch.ops.aten._local_scalar_dense.default, 'Tensor scalar read by Python'
        assert func != torch.ops.aten.nonzero.default, 'Dynamic-size index allocation in fitting loop'
        # fill_(Tensor) extracts its scalar inside ATen, below this dispatch
        # hook. Use fill_(Python literal) or copy_(device Tensor) instead.
        assert func != torch.ops.aten.fill_.Tensor, 'Tensor-valued fill hides a scalar extraction'
        if func == torch.ops.aten._to_copy.default:
            target = kwargs.get('device', args[0].device)
            source = args[0]
            literal_upload = source.device.type == 'cpu' and source.ndim == 0
            assert torch.device(target).type == source.device.type or literal_upload, 'Tensor crossed CPU/CUDA boundary'
        if func == torch.ops.aten.copy_.default and all(isinstance(v, torch.Tensor) for v in args[:2]):
            assert args[0].device.type == args[1].device.type, 'Tensor crossed CPU/CUDA boundary'
        return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_model(y, **kwargs):
    options = dict(K=2, device=y.device, prior_L='cgb', prior_F='norm',
                   self_row_cov=True, allow_backfitting=False, internal_epoch=1,
                   prior_L_kwargs=dict(hidden_dim=4, n_layers=0, penalty=1),
                   prior_F_kwargs=dict(penalty=1, scales=y.new_tensor([0., .1, .5, 2., 8.])),
                   conditional_kwargs=dict(quadrature_points=8, parent_samples=8))
    options.update(kwargs)
    m = cEBMF(y, **options)
    m.initialise_factors()
    return m


def test_em_stops_at_same_iterate_without_any_python_scalar_reads():
    torch.manual_seed(0)
    logl = torch.randn(23, 4, dtype=torch.float64)
    # Independent original full-batch EM with early stopping.
    pi = torch.arange(4, dtype=logl.dtype).neg().exp()
    pi /= pi.sum()
    penalty = torch.tensor([2., 1., 1., 1.], dtype=logl.dtype)
    for step in range(100):
        old = pi
        logits = logl + (pi + 1e-12).log()
        counts = (logits - logits.logsumexp(1, keepdim=True)).exp().sum(0)
        counts = (counts + penalty - 1).clamp_min(1e-12)
        pi = counts / counts.sum()
        if (step + 1) % 10 == 0 and (pi - old).norm() < .01:
            break
    assert step < 99  # actually tests freezing after early convergence
    with NoHostRead():
        result = optimize_pi_logL(logl, penalty, tol=.01)
    torch.testing.assert_close(result, pi, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize('approximation', ['quadrature', 'quadratic'])
def test_profile_math_has_no_python_scalar_reads_on_cpu_either(approximation):
    a, b = torch.ones(2), torch.tensor([-1., 2.])
    lw = torch.tensor([[.3, .7], [.1, .9]]).log()
    inverse, center = torch.tensor([[0., 1.], [0., 1.]]), torch.tensor([[0., 10.], [0., -2.]])
    atoms = inverse == 0
    height = torch.where(atoms, lw, lw - .5 * 1.8378770664093453)
    rule = hermite_rule(12, a)
    from cebmf_torch.cebmf._quadratic import local_feedback, quadratic_coordinate
    feedback = local_feedback(lambda v: -.1 * v.square(), center[:, :, None])
    with NoHostRead():
        if approximation == 'quadratic':
            _, q = quadratic_coordinate(a, b, lw, inverse, center, height, atoms, rule, feedback)
        else:
            _, q = tilted_coordinate(a, b, lw, inverse, center, height, atoms, rule,
                                      lambda v: -.1 * v.square())
        q.component_centered_moments()


@pytest.mark.parametrize('method', ['svd', 'random', 'zero', 'supplied'])
def test_initialization_preserves_dtype_and_handles_entirely_missing_column(method):
    y = torch.randn(8, 5, dtype=torch.float64)
    y[:, 0] = torch.nan
    m = cEBMF(y, K=6, device='cpu', allow_backfitting=False)
    if method == 'supplied':
        m.initialise_factors(L=torch.randn(8, 6), F=torch.randn(5, 6))
    else:
        m.initialise_factors(method)
    assert m.L.dtype == m.F.dtype == y.dtype
    assert torch.isfinite(m.L).all() and torch.isfinite(m.F).all()


def test_fixed_ash_grid_is_preserved_and_weights_refit():
    x, s = torch.linspace(-4., 4., 50), torch.ones(50)
    grid = torch.tensor([0., .5, 2., 8.])
    first = ash(x, s, scales=grid, penalty=1)
    second = ash(x * 0, s, scales=grid, penalty=1)
    assert first.scale.data_ptr() == grid.data_ptr()
    assert not torch.equal(first.pi, second.pi)
    for invalid in [torch.tensor([.1, 1.]), torch.tensor([0., -1.]), torch.tensor([0., 1., 1.])]:
        with pytest.raises(ValueError):
            ash(x, s, scales=invalid)


@CUDA
@pytest.mark.parametrize('prior', ['cgb', 'spiked_emdn'])
@pytest.mark.parametrize('verbose', [True, False])
def test_cuda_default_quadratic_warns_and_keeps_warmed_update_resident(prior, verbose, capsys):
    # Deliberately omit approximation: this exercises the public default.
    m = small_model(torch.randn(14, 9, device='cuda'), prior_L=prior, verbose=verbose)
    with pytest.warns(UserWarning, match="local quadratic approximation.*approximation.*quadrature"):
        result = m.fit(1)
    assert result.inference == 'conditional_quadratic'
    with NoHostRead():
        m.iter_once()
    torch.cuda.synchronize()
    for value in (m.L, m.L2, m.F, m.F2, m.obj[-1]):
        assert value.is_cuda and torch.isfinite(value).all()
    assert len(capsys.readouterr().out.splitlines()) == (2 if verbose else 0)


@CUDA
@pytest.mark.parametrize('approximation', ['quadrature', 'quadratic'])
@pytest.mark.parametrize('self_cov', [False, True])
@pytest.mark.parametrize('prior', ['cgb', 'spiked_emdn'])
def test_cuda_entire_warmed_sweep_has_no_host_reads_or_device_hops(prior, self_cov, approximation, monkeypatch):
    m = small_model(torch.randn(14, 9, device='cuda'), prior_L=prior, self_row_cov=self_cov,
                    conditional_kwargs=dict(approximation=approximation, quadrature_points=8, parent_samples=8))
    m.fit(1)  # validate inputs, construct graph, and initialize CUDA libraries
    optimizers = []
    original_adam = torch.optim.Adam
    def recording_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer
    monkeypatch.setattr(torch.optim, 'Adam', recording_adam)
    with NoHostRead():
        m.iter_once()
    torch.cuda.synchronize()
    for value in [m.Y, m.data, m.L, m.L2, m.F, m.F2, m.tau, m.R, m.obj[-1]]:
        assert value.is_cuda and torch.isfinite(value).all()
    for optimizer in optimizers:
        assert optimizer.defaults['capturable']
        for state in optimizer.state.values():
            assert all(v.is_cuda for v in state.values() if isinstance(v, torch.Tensor))
    if self_cov:
        graph = m.conditional_fit.row
        for prior_module, q in zip(graph.priors, graph.q):
            assert all(p.is_cuda for p in prior_module.parameters())
            assert all(p.is_cuda for p in prior_module.buffers())
            assert q.values.is_cuda and q.weights.is_cuda and q.entropy.is_cuda
            if q.gaussian_moments is not None:
                assert all(t.is_cuda and t.grad_fn is None for t in q.gaussian_moments)


@CUDA
@pytest.mark.parametrize('approximation', ['quadrature', 'quadratic'])
def test_cuda_partially_paired_views_and_column_graph_stay_resident(approximation):
    first, second = torch.randn(12, 8, device='cuda'), torch.randn(12, 7, device='cuda')
    first[-3:] = torch.nan
    second[:3] = torch.nan
    numerical = dict(approximation=approximation, quadrature_points=8, parent_samples=8)
    a = small_model(first, conditional_kwargs=numerical)
    b = small_model(second, prior_F='cgb', self_col_cov=True,
                    prior_F_kwargs=dict(hidden_dim=4, n_layers=0, penalty=1), conditional_kwargs=numerical)
    fit_joint(a, b, maxit=1)
    engine = a.conditional_fit
    table = engine.row.prediction_uniform
    # fit_joint performs boundary validation; measure the established engine.
    with NoHostRead():
        engine.step()
        engine.finish()
    torch.cuda.synchronize()
    assert table is engine.row.prediction_uniform
    for m in (a, b):
        assert torch.isfinite(m.L).all() and m.L.is_cuda
        assert m.obj[-1].is_cuda


@CUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_ash_matches_cpu_with_identical_fixed_grid(dtype):
    torch.manual_seed(17)
    x, s = torch.randn(40, dtype=dtype), torch.full((40,), .5, dtype=dtype)
    scales = torch.tensor([0., .1, .5, 1., 3.], dtype=dtype)
    cpu = ash(x, s, scales=scales, penalty=1)
    gpu = ash(x.cuda(), s.cuda(), scales=scales.cuda(), penalty=1)
    tol = 2e-5 if dtype == torch.float32 else 1e-10
    for field in ('post_mean', 'post_mean2', 'pi', 'log_lik'):
        torch.testing.assert_close(getattr(cpu, field), getattr(gpu, field).cpu(), rtol=tol, atol=tol)


@CUDA
@pytest.mark.parametrize('approximation', ['quadrature', 'quadratic'])
def test_cuda_profile_and_retained_memory_do_not_grow_with_training_graphs(tmp_path, approximation):
    m = small_model(torch.randn(24, 12, device='cuda'),
                    conditional_kwargs=dict(approximation=approximation, quadrature_points=8, parent_samples=8))
    m.fit(2)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(4):
            m.iter_once()
    torch.cuda.synchronize()
    events = [event.key.lower() for event in profiler.key_averages()]
    assert not any('_local_scalar_dense' in key or 'dtoh' in key or 'device to host' in key for key in events)
    profiler.export_chrome_trace(str(tmp_path / f'conditional_{approximation}_cuda_trace.json'))
    # q replaces its predecessor; no backward graph should survive a sweep.
    # Histories intentionally retain small scalar tensors. Allow allocator
    # granularity while rejecting accumulation of the much larger graphs.
    assert torch.cuda.memory_allocated() - baseline < 1024 * 1024
    for q in m.conditional_fit.row.q:
        assert q.values.grad_fn is None and q.weights.grad_fn is None
