"""L-BFGS controls, deterministic fitting, and informative-covariate recovery."""

import importlib
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means

lcash = importlib.import_module("cebmf_torch.cebnm.lcash")
SOLVERS = [lcash_posterior_means, po_lcash_posterior_means]


@pytest.fixture
def fixed_grid(monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.6, 0.4]), scale=torch.tensor([0.0, 3.0])),
    )


@pytest.mark.parametrize(
    "options",
    [
        {"max_iter": -1},
        {"max_iter": 1.5},
        {"max_iter": math.inf},
        {"penalty": 0.5},
        {"penalty": math.nan},
        {"coefficient_prior_scale": 0},
        {"coefficient_prior_scale": math.inf},
        {"coefficient_prior_df": -1},
        {"coefficient_prior_df": math.nan},
    ],
)
def test_invalid_fit_options_are_rejected(options):
    x = torch.linspace(-2, 2, 7)
    with pytest.raises((TypeError, ValueError)):
        lcash_posterior_means(x[:, None], x, torch.ones_like(x), **options)


@pytest.mark.parametrize("solver", SOLVERS)
def test_informative_covariate_recovery_with_few_observations(solver, fixed_grid):
    generator = torch.Generator().manual_seed(410)
    x = torch.randn(500, generator=generator)
    x = (x - x.mean()) / x.std(correction=0)
    slab = torch.rand(500, generator=generator) < torch.sigmoid(1.5 * x - 0.5)
    truth = 3 * slab * torch.randn(500, generator=generator)
    observed = truth + torch.randn(500, generator=generator)
    before = torch.random.get_rng_state().clone()
    first = solver(x[:, None], observed, torch.ones_like(x), max_iter=0, verbose=False)
    fitted = solver(x[:, None], observed, torch.ones_like(x), model_param=first.model_param, verbose=False)
    assert torch.equal(torch.random.get_rng_state(), before)
    state = fitted.model_param["state_dict"]
    if solver is lcash_posterior_means:
        slope = (state["linear.weight"][1] - state["linear.weight"][0]).item()
    else:
        slope = state["w"].item()
    # This range admits sampling error but rejects a fit that barely leaves zero.
    assert 0.8 < slope < 2.2
    assert fitted.loss < first.loss - 5
    assert torch.mean((fitted.post_mean - truth).square()) < torch.mean((first.post_mean - truth).square())


@pytest.mark.parametrize("model_class", [LcashNet, PropOddsLcashNet])
@pytest.mark.parametrize("penalty", [1.0, 1.5, 10.0])
@pytest.mark.parametrize("prior_scale,prior_df", [(None, 5.0), (1.0, 5.0), (0.5, 3.0)])
def test_full_data_objective_matches_null_and_student_t_priors(model_class, penalty, prior_scale, prior_df):
    model = model_class(1, 3, log_pi_init=torch.tensor([0.6, 0.3, 0.1]).log()).double()
    with torch.no_grad():
        slopes = model.linear.weight if isinstance(model, LcashNet) else model.w
        slopes.fill_(0.5)
        if isinstance(model, LcashNet):
            slopes[0].zero_()
    x = torch.linspace(-2, 2, 7, dtype=torch.float64)
    scale = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    log_lik = lcash.get_data_loglik_normal_torch(x, torch.ones_like(x), torch.zeros_like(scale), scale)

    coefficient_prior = None
    if prior_scale is not None:
        coefficient_prior = torch.distributions.StudentT(x.new_tensor(prior_df), scale=prior_scale)
    actual = lcash._conditional_loss(model, x[:, None], log_lik, penalty, coefficient_prior)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))
    log_pi = model.log_probs(x[:, None])
    expected = -torch.logsumexp(log_pi + log_lik, dim=1).sum()
    expected -= ((penalty - 1) / len(x)) * log_pi[:, 0].sum()
    if prior_scale is not None:
        coefficients = model.linear.weight[1:] - model.linear.weight[:1] if isinstance(model, LcashNet) else model.w
        # Student-t(df, 0, scale) log density, including its normalizing constant.
        normalizer = (
            math.lgamma((prior_df + 1) / 2)
            - math.lgamma(prior_df / 2)
            - 0.5 * math.log(prior_df * math.pi)
            - math.log(prior_scale)
        )
        log_prior = normalizer - (prior_df + 1) / 2 * torch.log1p(coefficients.square() / (prior_df * prior_scale**2))
        expected -= log_prior.sum()
    expected /= len(x)
    gradients = torch.autograd.grad(expected, tuple(model.parameters()))
    torch.testing.assert_close(actual, expected)
    for actual_gradient, gradient in zip(actual_gradients, gradients, strict=True):
        torch.testing.assert_close(actual_gradient, gradient)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_training_state_keeps_float64_and_outputs_keep_requested_device(solver, device, fixed_grid):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    x = torch.linspace(-2, 2, 16, device=device)
    result = solver(x[:, None], x, torch.ones_like(x), max_iter=np.int64(2), device=device, verbose=False)
    for key in ("scale", "feature_mean", "feature_sd"):
        assert result.model_param[key].dtype == torch.float64
    for value in result.model_param["state_dict"].values():
        assert value.dtype == torch.float64
        assert value.device.type == ("cpu" if device == "mps" else device)
    for field in ("post_mean", "post_mean2", "post_sd", "pi_np", "scale"):
        value = getattr(result, field)
        assert value.dtype == torch.float32
        assert value.device.type == device


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("ash_init", [False, True])
def test_mps_default_device_is_preserved_after_cpu_fit(solver, ash_init):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    previous = torch.get_default_device()
    try:
        torch.set_default_device("mps")
        x = torch.linspace(-4, 4, 32)
        result = solver(x[:, None], x, torch.ones_like(x), ash_init=ash_init, max_iter=2)
        assert torch.get_default_device().type == "mps"
        assert result.post_mean.device.type == "mps"
        assert result.model_param["scale"].device.type == "cpu"
        assert math.isfinite(result.loss)
    finally:
        torch.set_default_device(previous)


def test_python_inputs_retain_float64_precision():
    value = 1 + 2**-40
    features, observed, standard_error = lcash._prepare_inputs([value], [value], [value], torch.device("cpu"))
    for tensor in (features, observed, standard_error):
        assert tensor.dtype == torch.float64
        assert tensor.item() == value


def test_native_po_line_search_recovers_from_large_gap_trial():
    # This heteroscedastic normal-means problem produces an overflowing trial
    # under the native strong-Wolfe line search when gaps are unbounded.
    rng = np.random.default_rng(96153)
    n = 1000
    raw_features = rng.normal(size=(n, 2))
    features = torch.tensor((raw_features - raw_features.mean(0)) / raw_features.std(0), dtype=torch.float64)
    truth = np.where(features[:, 0].numpy() > 0, rng.normal(size=n) * 3, 0)
    se = np.exp(rng.uniform(np.log(0.01), np.log(0.8), n))
    observed = torch.tensor(truth + rng.normal(size=n) * se, dtype=torch.float64)
    se = torch.tensor(se, dtype=torch.float64)
    scale = torch.tensor([0, 0.01, 0.05, 0.2, 0.5, 1.0, 3.0, 10.0], dtype=torch.float64)
    model = PropOddsLcashNet(2, len(scale), torch.tensor(rng.normal(size=len(scale)), dtype=torch.float64))
    log_lik = lcash.get_data_loglik_normal_torch(observed, se, torch.zeros_like(scale), scale)
    lcash._train_model(model, features, log_lik, 500, 1, False, "PO-LC-ASH")
    assert all(
        torch.isfinite(parameter).all() and torch.isfinite(parameter.grad).all() for parameter in model.parameters()
    )
    posterior = lcash._compute_posteriors(model, features, observed, se, scale, log_lik)
    for value in posterior:
        assert torch.isfinite(torch.as_tensor(value)).all()


@pytest.mark.parametrize("model_class", [LcashNet, PropOddsLcashNet])
def test_iteration_budget_warns_and_returns_finite_parameters(model_class):
    model = model_class(1, 2, log_pi_init=torch.tensor([0.6, 0.4]).log()).double()
    x = torch.linspace(-3, 3, 100, dtype=torch.float64)
    scale = torch.tensor([0.0, 3.0], dtype=torch.float64)
    log_lik = lcash.get_data_loglik_normal_torch(x, torch.ones_like(x), torch.zeros_like(scale), scale)
    with pytest.warns(RuntimeWarning):
        lcash._train_model(model, x[:, None], log_lik, 1, 10, False, "test")
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("solver", SOLVERS)
def test_default_fit_is_quiet_and_verbose_reports_final_diagnostics(solver, fixed_grid, capsys):
    x = torch.linspace(-2, 2, 32)
    solver(x[:, None], x, torch.ones_like(x), max_iter=2)
    assert not capsys.readouterr().out
    solver(x[:, None], x, torch.ones_like(x), max_iter=2, verbose=True)
    output = capsys.readouterr().out
    assert output.strip()


def test_near_optimum_intercept_reaches_gradient_tolerance_without_warning():
    model = LcashNet(1, 2, dtype=torch.float64)
    with torch.no_grad():
        # This intercept has a maximum gradient of 3e-7 for the data below.
        model.linear.bias.copy_(torch.tensor([-0.3269970844990212, 0.0], dtype=torch.float64))
    features = torch.zeros((100, 1), dtype=torch.float64)
    y = torch.cat([torch.zeros(50), torch.full((50,), 3.0)]).double()
    scale = torch.tensor([0.0, 3.0], dtype=torch.float64)
    log_lik = lcash.get_data_loglik_normal_torch(y, torch.ones_like(y), torch.zeros_like(scale), scale)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        lcash._train_model(model, features, log_lik, 500, 10, False, "test")
    assert not captured
    assert max(parameter.grad.abs().max().item() for parameter in model.parameters()) <= 1e-7


def test_two_component_prior_matches_po_and_is_invariant_to_common_lc_slopes(fixed_grid):
    generator = torch.Generator().manual_seed(410)
    features = torch.randn(500, 2, generator=generator)
    slab = torch.rand(500, generator=generator) < torch.sigmoid(1.5 * features[:, 0] - 0.5)
    truth = 3 * slab * torch.randn(500, generator=generator)
    observed = truth + torch.randn(500, generator=generator)
    se = torch.ones_like(observed)
    first = lcash_posterior_means(features, observed, se, max_iter=0)
    lc = lcash_posterior_means(features, observed, se, model_param=first.model_param)
    state = first.model_param
    state["state_dict"]["linear.weight"] += torch.tensor([2.0, -3.0])
    shifted = lcash_posterior_means(features, observed, se, model_param=state)
    po = po_lcash_posterior_means(features, observed, se)
    for fitted in (shifted, po):
        torch.testing.assert_close(fitted.pi_np, lc.pi_np, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(fitted.post_mean, lc.post_mean, atol=2e-5, rtol=2e-5)
    contrast = lc.model_param["state_dict"]["linear.weight"][1] - lc.model_param["state_dict"]["linear.weight"][0]
    torch.testing.assert_close(po.model_param["state_dict"]["w"], contrast, atol=2e-5, rtol=2e-5)
