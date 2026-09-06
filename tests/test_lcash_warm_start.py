"""Fitted LC-ASH state preserves component and feature coordinates."""

import copy
import importlib

import pytest
import torch

from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means

lcash_module = importlib.import_module("cebmf_torch.cebnm.lcash")
SOLVERS = [lcash_posterior_means, po_lcash_posterior_means]


@pytest.fixture
def fit_inputs():
    betahat = torch.linspace(-3, 3, 32)
    X = torch.stack((betahat, betahat.square()), dim=1)
    se = torch.ones(32)
    options = {"n_epochs": 0, "ash_init": False, "verbose": False, "device": torch.device("cpu")}
    return X, betahat, se, options


def assert_same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_same_state(actual[key], value)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value


def no_grid_selection(*args, **kwargs):
    pytest.fail("A warm start must not rebuild the mixture grid or repeat ASH.")


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("ash_init", [False, True])
def test_zero_epoch_warm_start_restores_full_fit(solver, ash_init, fit_inputs, monkeypatch):
    X, betahat, se, options = fit_inputs
    trained = solver(X, betahat, se, **(options | {"n_epochs": 2, "ash_init": ash_init}))
    saved = copy.deepcopy(trained.model_param)
    monkeypatch.setattr(lcash_module, "_select_grid", no_grid_selection)
    loaded = solver(
        X,
        betahat,
        se,
        model_param=trained.model_param,
        **(options | {"ash_init": True, "ash_threshold": 1.0, "mult": 2.0}),
    )
    assert_same_state(loaded.model_param, saved)
    assert_same_state(trained.model_param, saved)
    for name in ("scale", "pi_np", "post_mean", "post_mean2", "post_sd"):
        torch.testing.assert_close(getattr(loaded, name), getattr(trained, name), rtol=0, atol=0)
    assert loaded.loss == trained.loss


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("change", ["same_size_grid", "different_size_grid"])
def test_changed_observations_use_saved_scales(solver, change, fit_inputs, monkeypatch):
    X, betahat, se, options = fit_inputs
    first = solver(X, betahat, se, **options)
    if change == "same_size_grid":
        betahat, se = 2 * betahat, 2 * se
    else:
        betahat = 10 * betahat
    fresh = solver(X, betahat, se, **options)
    if change == "same_size_grid":
        assert fresh.scale.shape == first.scale.shape
        assert not torch.equal(fresh.scale, first.scale)
    else:
        assert fresh.scale.shape != first.scale.shape

    monkeypatch.setattr(lcash_module, "_select_grid", no_grid_selection)
    loaded = solver(X, betahat, se, model_param=first.model_param, **options)
    torch.testing.assert_close(loaded.scale, first.scale, rtol=0, atol=0)
    torch.testing.assert_close(loaded.pi_np, first.pi_np, rtol=0, atol=0)
    # Evaluate the normal mixture independently using the saved prior.
    variance = first.scale.square()[None, :] + se.square()[:, None]
    log_density = torch.distributions.Normal(0, variance.sqrt()).log_prob(betahat[:, None])
    responsibilities = torch.softmax(log_density + first.pi_np.log(), dim=1)
    component_mean = betahat[:, None] * first.scale.square()[None, :] / variance
    torch.testing.assert_close(loaded.post_mean, (responsibilities * component_mean).sum(1), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("solver,model_class", zip(SOLVERS, [LcashNet, PropOddsLcashNet], strict=True))
def test_new_rows_use_training_feature_transform(solver, model_class, fit_inputs):
    X, betahat, se, options = fit_inputs
    X = torch.cat((X, torch.ones(32, 1), torch.full((32, 1), float("nan"))), dim=1)
    X[:3, 0] = float("nan")
    first = solver(X, betahat, se, **(options | {"n_epochs": 2}))
    state = first.model_param
    new_X = X[:7].nan_to_num() + 4
    new_X[0, 0] = float("nan")
    new_X[:, 2:] = torch.arange(14).reshape(7, 2)
    loaded = solver(new_X, betahat[:7], se[:7], model_param=state, **options)

    # Columns that were constant/all-missing in training remain disabled.
    expected_X = torch.zeros_like(new_X)
    for column in range(2):
        observed = X[:, column][~X[:, column].isnan()]
        expected_X[:, column] = ((new_X[:, column] - observed.mean()) / observed.std(correction=0)).nan_to_num()
    model = model_class(X.shape[1], first.scale.numel())
    model.load_state_dict(state["state_dict"])
    torch.testing.assert_close(loaded.pi_np, model(expected_X), rtol=1e-6, atol=1e-7)
    assert_same_state(loaded.model_param, state)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("mismatch", ["legacy", "solver", "version", "features", "parameters", "scales"])
def test_incompatible_state_raises_without_mutation(solver, mismatch, fit_inputs):
    X, betahat, se, options = fit_inputs
    state = solver(X, betahat, se, **options).model_param
    error, message = ValueError, "model_param"
    if mismatch == "legacy":
        state = state["state_dict"]
        message = "Older flat state dictionaries"
    elif mismatch == "solver":
        state["solver"] = "another solver"
    elif mismatch == "version":
        state["version"] += 1
    elif mismatch == "features":
        X = X[:, :1]
        message = "covariate columns"
    elif mismatch == "parameters":
        key = next(iter(state["state_dict"]))
        state["state_dict"][key] = torch.zeros(123)
        error, message = RuntimeError, "size mismatch"
    else:
        state["scale"][0] = 0.01
        message = "spike at zero"
    snapshot = copy.deepcopy(state)
    with pytest.raises(error, match=message):
        solver(X, betahat, se, model_param=state, **options)
    assert_same_state(state, snapshot)


@pytest.mark.parametrize("solver", SOLVERS)
def test_further_training_does_not_mutate_previous_state(solver, fit_inputs):
    X, betahat, se, options = fit_inputs
    first = solver(X, betahat, se, **options)
    snapshot = copy.deepcopy(first.model_param)
    updated = solver(X, betahat, se, model_param=first.model_param, **(options | {"n_epochs": 2}))
    assert_same_state(first.model_param, snapshot)
    assert any(
        not torch.equal(value, updated.model_param["state_dict"][key])
        for key, value in first.model_param["state_dict"].items()
    )
