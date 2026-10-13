"""Regression contracts for the ASH handoff and direct log probabilities."""

import importlib
import io
from types import SimpleNamespace

import pytest
import torch

from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means
from cebmf_torch.ebnm.ash import ash
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.posterior import posterior_mean_norm

lcash = importlib.import_module("cebmf_torch.cebnm.lcash")
MODELS = [LcashNet, PropOddsLcashNet]
SOLVERS = [lcash_posterior_means, po_lcash_posterior_means]


@pytest.fixture(params=[torch.float32, torch.float64])
def precision(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("weights", [[0.8, 0.1999999, 1e-7], [0.8, 1e-12, 0.199999999999]])
def test_handoff_preserves_small_log_weights(model_class, weights, precision):
    log_weights = torch.tensor(weights).log()
    model = model_class(3, len(weights), log_pi_init=log_weights)
    x = torch.tensor([[-3.0, 0.5, 2.0], [7.0, -4.0, 1.0]])
    expected = log_weights.double() - torch.logsumexp(log_weights.double(), 0)
    actual = model.log_probs(x)
    tolerance = 1e-5 if precision == torch.float32 else 1e-12
    torch.testing.assert_close(actual.double(), expected.expand_as(actual), atol=tolerance, rtol=0)
    torch.testing.assert_close(model(x), actual.exp(), atol=0, rtol=0)
    assert torch.equal(actual[0], actual[1])


@pytest.mark.parametrize("model_class,solver", zip(MODELS, SOLVERS, strict=True))
@pytest.mark.parametrize("penalty", [1.0, 1.5, 10.0])
def test_public_handoff_matches_retained_ash10_for_every_conditional_penalty(model_class, solver, penalty):
    x = torch.linspace(-6, 6, 128)
    se = torch.linspace(0.8, 1.2, len(x))
    reference = ash(x.double(), se.double(), optimizer="lbfgs", penalty=10.0, zero_threshold=0)
    active = reference.pi > 1e-6
    scale = reference.scale[active]
    log_pi = reference.pi[active].double().log()
    log_pi -= torch.logsumexp(log_pi, 0)
    log_lik = get_data_loglik_normal_torch(x.double(), se.double(), torch.zeros_like(scale), scale)
    expected = posterior_mean_norm(x.double(), se.double(), log_pi, log_lik, scale.double())
    expected_mean_loss = -torch.logsumexp(log_lik + log_pi, dim=1).mean()
    actual = solver(x[:, None], x, se, max_iter=0, penalty=penalty, verbose=False)

    torch.testing.assert_close(actual.scale, scale.float(), rtol=0, atol=0)
    model = model_class(1, actual.scale.numel()).double()
    model.load_state_dict(actual.model_param["state_dict"])
    actual_log_pi = model.log_probs(torch.tensor([[-2.0], [2.0]], dtype=torch.float64))
    torch.testing.assert_close(actual_log_pi.double(), log_pi.expand_as(actual_log_pi), rtol=0, atol=1e-5)

    tolerance = 64 * torch.finfo(torch.float32).eps
    for field, units in (("post_mean", se), ("post_mean2", se.square()), ("post_sd", se)):
        torch.testing.assert_close(
            getattr(actual, field).double() / units.double(),
            getattr(expected, field) / units.double(),
            atol=tolerance,
            rtol=tolerance,
        )
    assert actual.loss / len(x) == pytest.approx(expected_mean_loss.item(), abs=tolerance, rel=tolerance)


@pytest.mark.parametrize("solver", SOLVERS)
def test_single_spike_round_trips(solver, monkeypatch):
    monkeypatch.setattr(lcash, "ash", lambda *args, **kwargs: SimpleNamespace(pi=torch.ones(1), scale=torch.zeros(1)))
    x = torch.linspace(-2, 2, 32)
    first = solver(x[:, None], x, torch.ones_like(x), max_iter=3, verbose=False)
    stream = io.BytesIO()
    torch.save(first.model_param, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    for iterations in (0, 2):
        result = solver(x[:, None], x, torch.ones_like(x), model_param=saved, max_iter=iterations, verbose=False)
        assert torch.equal(result.pi_np, torch.ones((32, 1)))
        for field in ("post_mean", "post_mean2", "post_sd"):
            assert torch.equal(getattr(result, field), torch.zeros_like(x))
        expected_loss = -torch.distributions.Normal(0, 1).log_prob(x).sum().item()
        assert result.loss == pytest.approx(expected_loss, abs=1e-5)
