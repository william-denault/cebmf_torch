"""Fitted LC-ASH state survives real matrix-factorization updates."""

import math

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebnm.lcash import _select_grid
from cebmf_torch.priors.learned import LearnedBuilder, LearnedPriorType

PRIOR_TYPES = (LearnedPriorType.LCASH, LearnedPriorType.PO_LCASH)


def make_model(prior, **kwargs):
    generator = torch.Generator().manual_seed(123)
    data = torch.randn(24, 18, generator=generator)
    prior_kwargs = {"n_epochs": 2, "ash_init": False, "verbose": False, "seed": 17}
    model = cEBMF(
        data,
        K=2,
        prior_L=prior,
        prior_F=prior,
        X_l=torch.randn(24, 2, generator=generator),
        X_f=torch.randn(18, 3, generator=generator),
        prior_L_kwargs=prior_kwargs,
        prior_F_kwargs=prior_kwargs,
        S=1.0,
        device=torch.device("cpu"),
        **kwargs,
    )
    model.initialise_factors()
    return model


@pytest.mark.parametrize("prior", PRIOR_TYPES)
def test_cebmf_repeated_updates_retain_each_factor_grid(prior, monkeypatch):
    model = make_model(prior, allow_backfitting=False)
    calls = []
    real_fit = LearnedBuilder.fit

    def record_fit(builder, **kwargs):
        result = real_fit(builder, **kwargs)
        calls.append((kwargs, result.model_param))
        return result

    # Observe the actual builder boundary; every prior fit and cEBMF update runs.
    monkeypatch.setattr(LearnedBuilder, "fit", record_fit)
    result = model.fit(maxit=3)

    assert len(calls) == 12  # Three iterations, two factors, both matrix sides.
    assert len(result.history_obj) == 3
    assert torch.isfinite(torch.tensor(result.history_obj)).all()
    assert torch.isfinite(result.L).all()
    assert torch.isfinite(result.F).all()

    different_grid_size = False
    for index, (inputs, state) in enumerate(calls):
        if index < 4:
            assert inputs["model_param"] is None
            continue

        previous = calls[index - 4][1]
        first = calls[index % 4][1]
        assert inputs["model_param"] is previous
        for field in ("scale", "feature_mean", "feature_sd"):
            torch.testing.assert_close(state[field], first[field], rtol=0, atol=0)

        # These are the actual changing pseudo-observations seen by the solver.
        # A fresh automatic grid changes even when its component count matches.
        fresh_grid, _ = _select_grid(
            inputs["betahat"],
            inputs["sebetahat"],
            mult=math.sqrt(2),
            ash_init=False,
            ash_threshold=1e-6,
            device=model.device,
        )
        assert not torch.equal(fresh_grid, first["scale"])
        different_grid_size |= fresh_grid.shape != first["scale"].shape

    assert different_grid_size
    for factor in range(2):
        assert model.model_state_L[factor] is calls[8 + 2 * factor][1]
        assert model.model_state_F[factor] is calls[9 + 2 * factor][1]


@pytest.mark.parametrize("prior", PRIOR_TYPES)
@pytest.mark.parametrize("side", ("row", "col"))
def test_cebmf_pruning_rejects_changed_self_covariate_count(prior, side):
    # A zero pruning threshold deterministically retains only the first factor.
    model = make_model(prior, prune_thresh=0.0, **{f"self_{side}_cov": True})
    model.fit(maxit=1)
    assert model.model.K == 1
    saved = model.model_state_L[0] if side == "row" else model.model_state_F[0]
    external = model.covariate.X_l if side == "row" else model.covariate.X_f
    assert saved["feature_mean"].numel() == external.shape[1] + 1

    # Pruning removed the other-factor covariate. Its old fitted coefficients
    # cannot be reinterpreted as coefficients for the remaining columns.
    with pytest.raises(ValueError, match="covariate columns"):
        model.fit(maxit=1)
