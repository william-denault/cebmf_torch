"""Verify the benchmark changes only covariates and labels its score correctly."""

import math
from pathlib import Path
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmarks/tree'))
from benchmark_tree_priors import PluginCGB
from plugin_moments import MomentPluginCEBMF, frozen_elbo_terms


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_model(cls=MomentPluginCEBMF, device='cpu', **kwargs):
    torch.manual_seed(313)
    y = torch.randn(32, 18, device=device)
    model = cls(y, K=3, self_row_cov=True, prior_L='cgb', device=device, verbose=False,
                allow_backfitting=False, prior_L_kwargs={'n_epochs': 1, 'hidden_dim': 4, 'n_layers': 0},
                prior_F_kwargs={'scales': [0., .1, .5, 2., 8.], 'penalty': 1.}, **kwargs)
    model.initialise_factors()
    return model


def test_second_moments_are_raw_posterior_moments_and_only_earlier_factors():
    m = make_model(row_cov_moments='mean_second')
    m.L2 = m.L.square() + .7
    x = m.row_inputs(2)
    torch.testing.assert_close(x[:, :2], m.L[:, :2])
    torch.testing.assert_close(x[:, 2:], m.L2[:, :2])
    assert not torch.equal(x[:, 2:], m.L[:, :2].square())
    m.L[:, 2], m.L2[:, 2] = 1e4, 1e8
    torch.testing.assert_close(x, m.row_inputs(2))
    torch.testing.assert_close(m.row_inputs(0), torch.ones(32, 1))


def test_mean_mode_preserves_original_scalar_fit_exactly():
    baseline, updated = make_model(PluginCGB), make_model(row_cov_moments='mean')
    torch.manual_seed(909)
    baseline.fit(2)
    torch.manual_seed(909)
    updated.fit(2)
    for name in ('L', 'L2', 'F', 'F2', 'tau'):
        torch.testing.assert_close(getattr(baseline, name), getattr(updated, name), rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(baseline.obj), torch.stack(updated.obj), rtol=0, atol=0)


@pytest.mark.parametrize('moments', ['mean', 'mean_second'])
def test_scores_use_current_inputs_after_forward_sweep_and_flag_pruning(moments):
    m = make_model(row_cov_moments=moments)
    m.fit(2)
    score = frozen_elbo_terms(m)
    assert score['current_covariates_match_fitted']
    assert abs(score['package_score_discrepancy']) < .001
    assert score['penalized_elbo'] <= score['elbo']
    m._prune_indices([1])
    assert not frozen_elbo_terms_after_prune_match(m)
    m.fit(1)
    assert frozen_elbo_terms(m)['current_covariates_match_fitted']


def frozen_elbo_terms_after_prune_match(m):
    # Pruning clears the package score; compute it for the retained q/prior KLs.
    m._cal_obj()
    return frozen_elbo_terms(m)['current_covariates_match_fitted']


def test_frozen_elbo_matches_independent_gaussian_entropy_calculation():
    m = make_model()
    m._prune_indices([1, 2])
    ml = torch.linspace(-.5, .8, m.N, dtype=torch.float64)[:, None]
    mf = torch.linspace(-.3, .4, m.P, dtype=torch.float64)[:, None]
    vl, vf = torch.full_like(ml, .2), torch.full_like(mf, .3)
    m.L, m.L2, m.F, m.F2 = ml, ml.square() + vl, mf, mf.square() + vf
    m.tau = ml.new_tensor(1.7)
    m.tau_map = m.tau.expand(m.N, m.P)
    # Independently specified q=N(m,v), p=N(0,1): E log p + entropy.
    entropy_l = .5 * (1 + math.log(2 * math.pi) + vl.log()).sum()
    entropy_f = .5 * (1 + math.log(2 * math.pi) + vf.log()).sum()
    prior_l = -.5 * (math.log(2 * math.pi) + ml.square() + vl).sum()
    prior_f = -.5 * (math.log(2 * math.pi) + mf.square() + vf).sum()
    m.kl_l = (-(prior_l + entropy_l)).reshape(1)
    m.kl_f = (-(prior_f + entropy_f)).reshape(1)
    m.pi0_L = [torch.full((m.N,), .5)]
    m.regularization_F = torch.zeros(1)
    ll = ml.new_zeros(())
    for i in range(m.N):
        for j in range(m.P):
            e2 = (m.Y0[i, j] - ml[i, 0] * mf[j, 0]).square()
            e2 += vl[i, 0] * vf[j, 0] + vl[i, 0] * mf[j, 0].square() + vf[j, 0] * ml[i, 0].square()
            ll += -.5 * (math.log(2 * math.pi / 1.7) + 1.7 * e2)
    expected = ll + prior_l + prior_f + entropy_l + entropy_f
    m._cal_obj()
    actual = frozen_elbo_terms(m)
    assert actual['elbo'] == pytest.approx(float(expected), abs=1e-9)


class NoTransferOrScalarRead(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        assert func != torch.ops.aten._local_scalar_dense.default
        if func == torch.ops.aten._to_copy.default:
            assert torch.device(kwargs.get('device', args[0].device)) == args[0].device
        return func(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires actual CUDA/ROCm')
@pytest.mark.parametrize('moments', ['mean', 'mean_second'])
def test_gpu_covariates_and_full_fit_stay_on_device(moments):
    m = make_model(device='cuda', row_cov_moments=moments)
    with NoTransferOrScalarRead():
        x = m._build_covariate_matrix(None, True, m.L, 2, m.N)
    assert x.device.type == 'cuda'
    m.fit(2)
    assert all(getattr(m, key).device.type == 'cuda' for key in ('L', 'L2', 'F', 'F2', 'tau'))
    assert all(t.device.type == 'cuda' for state in m.model_state_L for t in state.values())
    assert all(x.device.type == 'cuda' for x in m.fitted_row_inputs)
    assert frozen_elbo_terms(m)['current_covariates_match_fitted']
