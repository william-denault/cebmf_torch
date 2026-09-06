"""LC-ASH warm starts use PyTorch's strict state-dict loading."""

import pytest
import torch

from cebmf_torch.cebnm.lcash import lcash_posterior_means, po_lcash_posterior_means


@pytest.fixture
def fit_inputs():
    betahat = torch.linspace(-3, 3, 32)
    X = torch.stack((betahat, betahat.square()), dim=1)
    se = torch.ones(32)
    options = {"n_epochs": 0, "ash_init": False, "verbose": False, "device": torch.device("cpu")}
    return X, betahat, se, options


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
@pytest.mark.parametrize("mismatch", ["features", "grid"])
def test_incompatible_warm_start_raises(solver, mismatch, fit_inputs):
    X, betahat, se, options = fit_inputs
    previous = solver(X, betahat, se, **options)
    if mismatch == "features":
        X = X[:, :1]
    else:
        betahat = betahat * 10
        fresh = solver(X, betahat, se, **options)
        assert previous.scale.numel() != fresh.scale.numel()

    with pytest.raises(RuntimeError, match="size mismatch"):
        solver(X, betahat, se, model_param=previous.model_param, **options)


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
def test_compatible_warm_start_loads_all_parameters(solver, fit_inputs):
    X, betahat, se, options = fit_inputs
    fresh = solver(X, betahat, se, **options)
    trained = solver(X, betahat, se, **(options | {"n_epochs": 1}))
    assert any(not torch.equal(trained.model_param[key], value) for key, value in fresh.model_param.items())

    loaded = solver(X, betahat, se, model_param=trained.model_param, **options)
    for key, value in trained.model_param.items():
        torch.testing.assert_close(loaded.model_param[key], value, rtol=0, atol=0)
