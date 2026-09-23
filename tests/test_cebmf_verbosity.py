"""Sweep progress across ordinary, conditional and coupled fitting routes."""

import pytest
import torch

from cebmf_torch import cEBMF, fit_joint


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_model(method, **kwargs):
    torch.manual_seed(9)
    settings = dict(K=2, device='cpu', allow_backfitting=False, internal_epoch=1,
                    prior_L='norm' if method == 'ordinary' else 'cgb', prior_F='norm',
                    self_row_cov=method != 'ordinary',
                    prior_F_kwargs=dict(scales=torch.tensor([0., .2, 1., 4.]), penalty=1))
    if method != 'ordinary':
        settings.update(prior_L_kwargs=dict(hidden_dim=4, n_layers=0, penalty=1),
                        conditional_kwargs=dict(approximation=method, quadrature_points=6, parent_samples=6))
    settings.update(kwargs)
    model = cEBMF(torch.randn(10, 7), **settings)
    model.initialise_factors()
    return model


@pytest.mark.parametrize('method', ['ordinary', 'quadratic', 'quadrature'])
def test_default_progress_once_per_completed_sweep_and_continuation(method, capsys):
    model = make_model(method)
    model.fit(0)
    assert capsys.readouterr().out == ''
    model.fit(2)
    model.iter_once()
    assert capsys.readouterr().out.splitlines() == [
        f'cEBMF sweep {i} completed.' for i in (1, 2, 3)
    ]


@pytest.mark.parametrize('method', ['ordinary', 'quadratic', 'quadrature'])
def test_quiet_progress_counts_sweeps_and_preserves_fitted_values(method, capsys):
    loud = make_model(method)
    loud.fit(2)
    assert len(capsys.readouterr().out.splitlines()) == 2
    quiet = make_model(method, verbose=False)
    quiet.fit(2)
    assert capsys.readouterr().out == ''
    for field in ('L', 'L2', 'F', 'F2', 'tau'):
        torch.testing.assert_close(getattr(quiet, field), getattr(loud, field), rtol=0, atol=0)
    quiet.verbose = True
    quiet.iter_once()
    assert capsys.readouterr().out == 'cEBMF sweep 3 completed.\n'
    quiet.initialise_factors()
    quiet.fit(1)
    assert capsys.readouterr().out == 'cEBMF sweep 1 completed.\n'


@pytest.mark.parametrize('first_verbose,second_verbose', [(True, True), (False, False), (True, False), (False, True)])
def test_joint_progress_prints_one_line_if_either_model_is_verbose(first_verbose, second_verbose, capsys):
    first = make_model('quadratic', self_row_cov=False, verbose=first_verbose)
    second = make_model('quadratic', self_row_cov=False, verbose=second_verbose)
    fit_joint(first, second, maxit=0)
    assert capsys.readouterr().out == ''
    fit_joint(first, second, maxit=2)
    fit_joint(first, second, maxit=1)
    expected = ([f'cEBMF joint sweep {i} completed.' for i in (1, 2, 3)]
                if first_verbose or second_verbose else [])
    assert capsys.readouterr().out.splitlines() == expected
    assert first._sweeps_completed == second._sweeps_completed == 3


def test_failed_sweep_does_not_print_a_completion(monkeypatch, capsys):
    model = make_model('ordinary')
    def fail(*args, **kwargs):
        raise RuntimeError('prior failed')
    monkeypatch.setattr(model.prior_L_fn, 'fit', fail)
    with pytest.raises(RuntimeError, match='prior failed'):
        model.fit(1)
    assert capsys.readouterr().out == ''
    assert model._sweeps_completed == 0
