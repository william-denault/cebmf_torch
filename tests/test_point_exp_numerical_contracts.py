"""R parity at ordinary inputs and independent quadrature in extreme tails."""

import json
import math
from pathlib import Path

import pytest
import torch

from cebmf_torch import cEBMF, ebnm_point_exp
from cebmf_torch.utils.maths import logg_exp_convolved_with_normal, posterior_moments_exp
from cebmf_torch.utils.posterior import posterior_mean_exp

FIXTURES = Path(__file__).with_name("fixtures")
R_REFERENCE = json.loads((FIXTURES / "point_exp_r_reference.json").read_text())
TAIL_REFERENCE = json.loads((FIXTURES / "exp_tail_reference.json").read_text())
DTYPES = (torch.float32, torch.float64)


def logit(weight):
    if weight == 0:
        return -math.inf
    if weight == 1:
        return math.inf
    return math.log(weight / (1 - weight))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", R_REFERENCE["fixed"], ids=lambda case: case["name"])
def test_fixed_prior_matches_r_ebnm(case, dtype):
    x, s = (torch.tensor(case[key], dtype=dtype) for key in ("x", "s"))
    result = ebnm_point_exp(
        x,
        s,
        par_init=(logit(case["weight"]), math.log(case["rate"]), case["mode"]),
        fix_par=(True, True, True),
    )
    tolerance = 2e-4 if dtype == torch.float32 else 2e-8
    for field, expected in (("post_mean", "mean"), ("post_mean2", "mean2"), ("post_sd", "sd")):
        torch.testing.assert_close(
            getattr(result, field), torch.tensor(case[expected], dtype=dtype), rtol=tolerance, atol=1e-7
        )
    assert result.log_lik.item() == pytest.approx(case["log_lik"], rel=tolerance, abs=1e-7)
    assert result.pi_slab.item() == pytest.approx(case["weight"], abs=1e-7)


def test_unpenalized_fit_matches_r_ebnm():
    case = R_REFERENCE["fit"]
    x, s = (torch.tensor(case[key], dtype=torch.float64) for key in ("x", "s"))
    result = ebnm_point_exp(x, s, max_iter=200, tol=1e-10)
    assert result.log_lik.item() == pytest.approx(case["log_lik"], abs=2e-5)
    assert result.scale.item() == pytest.approx(case["rate"], rel=2e-4)
    assert result.pi_slab.item() == pytest.approx(case["weight"], abs=2e-5)
    torch.testing.assert_close(result.post_mean, torch.tensor(case["mean"], dtype=x.dtype), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", TAIL_REFERENCE)
def test_density_moments_and_gradients_match_independent_quadrature(case, dtype):
    x, s, rate = (torch.tensor(case[key], dtype=dtype, requires_grad=True) for key in ("x", "s", "rate"))
    density = logg_exp_convolved_with_normal(x, s, rate)
    mean, mean2 = posterior_moments_exp(x, s, rate)
    gradients = torch.autograd.grad(density, (x, s, rate))
    tolerance = 1e-4 if dtype == torch.float32 else 2e-10
    for actual, key in ((density, "log_density"), (mean, "mean"), (mean2, "mean2")):
        assert actual.detach().item() == pytest.approx(case[key], rel=tolerance, abs=1e-12)
    # A tiny *relative* tolerance matters here: the old tail calculation could
    # give a finite, wildly wrong gradient, or erase a small true derivative.
    for actual, key in zip(gradients, ("gradient_x", "gradient_s", "gradient_rate"), strict=True):
        # At |x/s|=1 the two noise-density derivatives cancel. Autograd may
        # lose a few ulps of 1/s there; keep the rate-gradient check relative
        # even when the true derivative is as small as 1e-24.
        absolute = 4 * torch.finfo(dtype).eps / case["s"] if key == "gradient_s" else 1e-30
        assert actual.item() == pytest.approx(case[key], rel=tolerance, abs=absolute)


@pytest.mark.parametrize("dtype", DTYPES)
def test_issue_84_likelihood_and_moment_bounds(dtype):
    x = torch.zeros(1, dtype=dtype)
    s = torch.full_like(x, 500000)
    result = ebnm_point_exp(x, s)
    likelihood_bound = -(s.log() + 0.5 * math.log(2 * math.pi)).sum()
    moment_bound = 2 * result.pi_slab / result.scale.square()
    assert result.log_lik <= likelihood_bound + 2e-6
    assert (result.post_mean2 <= moment_bound * (1 + 2e-6)).all()
    assert (result.post_mean2 >= result.post_mean.square()).all()
    assert torch.isfinite(result.post_sd).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_large_noise_fixed_prior_recovers_prior_moments(dtype):
    x = torch.zeros(1, dtype=dtype)
    result = ebnm_point_exp(
        x, torch.full_like(x, 500000), par_init=(logit(0.6), math.log(2), 0), fix_par=(True, True, True)
    )
    torch.testing.assert_close(result.post_mean, torch.full_like(x, 0.3))
    torch.testing.assert_close(result.post_mean2, torch.full_like(x, 0.3))


def test_fixed_weight_is_not_replaced_by_all_spike_boundary():
    result = ebnm_point_exp(
        torch.tensor([-3.0, -2.0], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        par_init=(0.0, 0.0, 0.0),
        fix_par=(True, False, True),
    )
    assert result.pi_slab == 0.5
    assert (result.post_mean > 0).all()


@pytest.mark.parametrize("rate", [0.001, 0.01, 100.0, 1000.0])
def test_fixed_rate_is_not_changed_by_optimizer_bounds(rate):
    result = ebnm_point_exp(
        torch.zeros(1, dtype=torch.float64),
        torch.full((1,), 500000.0, dtype=torch.float64),
        par_init=(0.0, math.log(rate), 0.0),
        fix_par=(True, True, True),
    )
    assert result.scale.item() == pytest.approx(rate, rel=1e-14)
    assert result.pi_slab == 0.5


def test_infinite_standard_errors_do_not_affect_fit_and_return_prior_moments():
    x = torch.tensor([1.0, 1.0, -0.5], dtype=torch.float64)
    s = torch.ones_like(x)
    baseline = ebnm_point_exp(x, s)
    with_missing = ebnm_point_exp(torch.cat((x, x.new_tensor([10.0]))), torch.cat((s, s.new_tensor([math.inf]))))
    for name in ("pi_slab", "scale", "log_lik"):
        torch.testing.assert_close(getattr(with_missing, name), getattr(baseline, name))
    torch.testing.assert_close(with_missing.post_mean[:3], baseline.post_mean)
    torch.testing.assert_close(with_missing.post_mean[-1], baseline.pi_slab / baseline.scale)
    torch.testing.assert_close(with_missing.post_mean2[-1], 2 * baseline.pi_slab / baseline.scale.square())


@pytest.mark.parametrize("dtype", DTYPES)
def test_exponential_mixture_uses_the_same_stable_tail_moments(dtype):
    x = torch.zeros(3, dtype=dtype)
    s = torch.tensor([500000.0, 1e12, math.inf], dtype=dtype)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=dtype)
    scales = torch.tensor([0.0, 0.5, 2.0], dtype=dtype)
    result = posterior_mean_exp(x, s, weights.log(), scales)
    torch.testing.assert_close(result.post_mean, (weights * scales).sum().expand_as(x))
    torch.testing.assert_close(result.post_mean2, (2 * weights * scales.square()).sum().expand_as(x))


def test_retained_near_zero_factor_has_valid_kl_and_evidence_bound():
    generator = torch.Generator().manual_seed(84)
    y = 0.5 * torch.randn(24, 18, generator=generator)
    model = cEBMF(y, K=1, prior_L="exp", prior_F="exp", allow_backfitting=False, S=1.0, device="cpu")
    model.initialise_factors(L=torch.ones(24, 1), F=torch.full((18, 1), 1e-7))
    result = model.fit(maxit=4)
    assert model.model.K == 1
    assert torch.isfinite(result.L).all() and torch.isfinite(result.F).all()
    assert (model.kl_l >= -1e-4).all() and (model.kl_f >= -1e-4).all()
    # With unit noise, even the maximum possible likelihood is bounded.
    assert min(result.history_obj) >= y.numel() * 0.5 * math.log(2 * math.pi) - 1e-3


def test_nonfinite_optimizer_failure_is_reported(monkeypatch):
    def invalid_step(optimizer, closure):
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(math.nan)
        closure()

    monkeypatch.setattr(torch.optim.LBFGS, "step", invalid_step)
    with pytest.raises(FloatingPointError, match="objective"):
        ebnm_point_exp(torch.ones(2), torch.ones(2))
