"""Numerical and public-call contracts for the conditional solvers."""

import importlib
import math
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
import torch

from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means

lcash = importlib.import_module("cebmf_torch.cebnm.lcash")
MODELS = [LcashNet, PropOddsLcashNet]


@pytest.fixture
def fixed_grid(monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(
            pi=torch.tensor([0.6, 0.2, 0.15, 0.05]), scale=torch.tensor([0.0, 1.0, 2.0, 4.0])
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gap_switch_matches_decimal_values_and_derivatives(dtype):
    eps = torch.finfo(dtype).eps
    switch = torch.tensor(math.log(eps), dtype=dtype)
    neighbors = [torch.nextafter(switch, torch.tensor(target, dtype=dtype)).item() for target in (-math.inf, math.inf)]
    high = 80.0 if dtype == torch.float32 else 700.0
    values = [-1000.0, -90.0, -20.0, neighbors[0], switch.item(), neighbors[1], -1.0, 0.0, 2.0, 5.0, high]
    h = torch.tensor(values, dtype=dtype, requires_grad=True)
    actual = lcash._log_gap_mass(h)
    derivative = torch.autograd.grad(actual.sum(), h)[0]
    references = []
    with localcontext() as context:
        context.prec = 120
        one = Decimal(1)
        for value in h[1:-1].detach().tolist():
            gap = Decimal.from_float(value).exp()
            references.append((float((one - (-gap).exp()).ln()), float(gap / (gap.exp() - one))))
    reference = torch.tensor(references, dtype=dtype)
    torch.testing.assert_close(actual[1:-1], reference[:, 0], atol=8 * eps, rtol=8 * eps)
    torch.testing.assert_close(derivative[1:-1], reference[:, 1], atol=8 * eps, rtol=8 * eps)
    torch.testing.assert_close(actual[[0, -1]], torch.tensor([-1000.0, 0.0], dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(derivative[[0, -1]], torch.tensor([1.0, 0.0], dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gap_range", ["large", "cumulative_overflow", "exp_overflow"])
def test_po_large_finite_gaps_keep_probabilities_gradients_and_posteriors_finite(dtype, gap_range):
    log_gap = {
        "large": 80 if dtype == torch.float32 else 700,
        "cumulative_overflow": math.log(torch.finfo(dtype).max * 0.75),
        "exp_overflow": 1000.0,
    }[gap_range]
    model = PropOddsLcashNet(1, 4).to(dtype=dtype)
    with torch.no_grad():
        model.w.zero_()
        model.delta_1.zero_()
        model.delta_gaps.fill_(log_gap)
    x = torch.zeros(2, dtype=dtype)
    cuts = model._get_cutpoints()
    assert torch.isfinite(cuts).all()
    assert torch.all(cuts[1:] > cuts[:-1])
    log_pi = model.log_probs(x[:, None])
    log_lik = torch.tensor([[-2.0, -1.0, -3.0, -4.0]], dtype=dtype)
    loss = -torch.logsumexp(log_pi + log_lik, dim=1).sum()
    loss.backward()
    assert torch.isfinite(log_pi).all() and torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    torch.testing.assert_close(model.delta_gaps.grad, torch.zeros_like(model.delta_gaps), rtol=0, atol=0)
    posterior = lcash._compute_posteriors(
        model, x[:, None], x, torch.ones_like(x), torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=dtype), log_lik
    )
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in posterior)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_po_gap_cap_preserves_representable_cutpoints_and_derivatives(dtype):
    model = PropOddsLcashNet(1, 4).to(dtype=dtype)
    cap = math.log(torch.finfo(dtype).max) - math.log(4) - 1
    with torch.no_grad():
        model.delta_1.fill_(0.3)
        model.delta_gaps.copy_(torch.tensor([-2.0, cap - 1], dtype=dtype))
    expected = torch.cat([model.delta_1, model.delta_1 + model.delta_gaps.exp().cumsum(0)])
    actual = model._get_cutpoints()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual.sum(), model.delta_gaps)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), model.delta_gaps)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    "model_class,solver", zip(MODELS, [lcash_posterior_means, po_lcash_posterior_means], strict=True)
)
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nonfinite_trial_gradient_restores_model_and_does_not_return_a_fit(
    model_class, solver, value, monkeypatch, fixed_grid
):
    original_log_probs = model_class.log_probs
    calls = []
    handles = []
    initial_parameters = []
    models = []

    def poison_first_gradient(gradient):
        calls.append(True)
        return torch.full_like(gradient, value) if len(calls) == 2 else gradient

    def attach_hook(model, features):
        result = original_log_probs(model, features)
        if not handles:
            assert torch.isfinite(result).all()
            models.append(model)
            initial_parameters.extend(parameter.detach().clone() for parameter in model.parameters())
            handles.append(next(model.parameters()).register_hook(poison_first_gradient))
        return result

    monkeypatch.setattr(model_class, "log_probs", attach_hook)
    x = torch.linspace(-1.0, 1.0, 5)
    try:
        with pytest.raises(FloatingPointError, match="gradient.*finite"):
            solver(x[:, None], x, torch.ones_like(x), max_iter=3, verbose=False)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 2
    for parameter, initial in zip(models[0].parameters(), initial_parameters, strict=True):
        torch.testing.assert_close(parameter, initial, rtol=0, atol=0)


@pytest.mark.parametrize(
    "solver,value",
    [
        (lcash_posterior_means, math.nan),
        (lcash_posterior_means, math.inf),
        (po_lcash_posterior_means, math.nan),
        (po_lcash_posterior_means, math.inf),
        (po_lcash_posterior_means, 710.0),
        (po_lcash_posterior_means, math.log(torch.finfo(torch.float64).max * 0.75)),
    ],
)
def test_final_lbfgs_step_rejects_nonfinite_parameters_but_accepts_finite_gaps(solver, value, monkeypatch, fixed_grid):
    original_step = torch.optim.LBFGS.step

    def final_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            # The last parameter is the LC bias or the PO log gap.
            optimizer.param_groups[0]["params"][-1].fill_(value)
        return result

    monkeypatch.setattr(torch.optim.LBFGS, "step", final_step)
    x = torch.linspace(-2.0, 2.0, 8)
    if not math.isfinite(value):
        with pytest.raises(FloatingPointError, match="Conditional .*finite"):
            solver(x[:, None], x, torch.ones_like(x), max_iter=1)
    else:
        result = solver(x[:, None], x, torch.ones_like(x), max_iter=1)
        assert math.isfinite(result.loss)
        for field in ("post_mean", "post_mean2", "post_sd", "pi_np"):
            assert torch.isfinite(getattr(result, field)).all()


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
def test_output_cast_overflow_is_not_returned_as_a_fit(solver, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(
            pi=torch.tensor([0.5, 0.5], dtype=torch.float64),
            scale=torch.tensor([0.0, 1e20], dtype=torch.float64),
        ),
    )
    observed = torch.full((2,), 1e20, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="Returned conditional post_mean2"):
        solver(torch.zeros(2, 1), observed, torch.ones_like(observed), max_iter=0, verbose=False)


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
def test_output_cast_must_preserve_distinct_component_scales(solver, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.5, 0.5]), scale=torch.tensor([0.0, 1.0])),
    )
    features = torch.zeros(2, 1)
    state = solver(features, torch.zeros(2), torch.ones(2), max_iter=0, verbose=False).model_param
    state["scale"] = torch.tensor([0.0, 1e-50], dtype=torch.float64)
    observed = torch.full((2,), 1e-49, dtype=torch.float64)
    standard_error = torch.full((2,), 1e-50, dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        solver(features, observed, standard_error, model_param=state, max_iter=0, verbose=False)


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
def test_float32_moments_imply_nonnegative_variance(solver, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.5, 0.5]), scale=torch.tensor([0.0, 1.0])),
    )
    observed = torch.tensor([1.0008100271224976])
    result = solver(torch.zeros(1, 1), observed, torch.full_like(observed, 1e-4), max_iter=0)
    assert torch.all(result.post_mean2 >= result.post_mean.square())
    # The separately computed SD retains uncertainty too small for float32 moments.
    torch.testing.assert_close(result.post_sd, torch.full_like(observed, 1e-4))
