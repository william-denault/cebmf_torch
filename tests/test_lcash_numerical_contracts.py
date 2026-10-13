"""Numerical and public-call contracts for the conditional solvers."""

import importlib
import math
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
import torch

from cebmf_torch.cebnm.lcash import PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means

lcash = importlib.import_module("cebmf_torch.cebnm.lcash")


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

    model = PropOddsLcashNet(1, 3, dtype=dtype)
    with torch.no_grad():
        model.delta_gaps.fill_(-1000)
    interior_log_mass = model.log_probs(torch.zeros(2, 1, dtype=dtype))[:, 1].sum()
    gap_gradient = torch.autograd.grad(interior_log_mass, model.delta_gaps)[0]
    assert torch.isfinite(interior_log_mass)
    torch.testing.assert_close(gap_gradient, torch.full_like(gap_gradient, 2.0), rtol=0, atol=8 * eps)


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


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_final_lbfgs_step_rejects_nonfinite_parameters(solver, value, monkeypatch, fixed_grid):
    original_step = torch.optim.LBFGS.step

    def final_step(optimizer, closure):
        result = original_step(optimizer, closure)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][-1].fill_(value)
        return result

    monkeypatch.setattr(torch.optim.LBFGS, "step", final_step)
    x = torch.linspace(-2.0, 2.0, 8)
    with pytest.raises(FloatingPointError):
        solver(x[:, None], x, torch.ones_like(x), max_iter=1)


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


@pytest.mark.parametrize(
    "observed,standard_error,error",
    [(1e20, 1.0, FloatingPointError), (1e-49, 1e-50, ValueError)],
)
def test_output_cast_rejects_unrepresentable_results(observed, standard_error, error):
    y = torch.full((2,), observed, dtype=torch.float64)
    se = torch.full_like(y, standard_error)
    with pytest.raises(error):
        lcash_posterior_means(torch.zeros(2, 1), y, se, ash_init=False, max_iter=0)
