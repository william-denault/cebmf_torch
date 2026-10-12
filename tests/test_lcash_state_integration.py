"""Fitted LC-ASH state survives real matrix-factorization updates."""

import math

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebnm.lcash import _select_grid
from cebmf_torch.priors.learned import LearnedBuilder, LearnedPriorType

PRIOR_TYPES = (LearnedPriorType.LCASH, LearnedPriorType.PO_LCASH)


def make_model(prior, **kwargs):
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(123)
    data = torch.randn(24, 18, generator=generator)
    prior_kwargs = {"ash_init": False, "verbose": False}
    model = cEBMF(
        data,
        K=2,
        prior_L=prior,
        prior_F=prior,
        X_l=kwargs.pop("X_l", torch.randn(24, 2, generator=generator)),
        X_f=kwargs.pop("X_f", torch.randn(18, 3, generator=generator)),
        prior_L_kwargs=prior_kwargs,
        prior_F_kwargs=prior_kwargs,
        internal_epoch=2,
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
@pytest.mark.parametrize(
    "sides,external,prune",
    [
        (("row",), True, True),
        (("col",), True, True),
        (("row", "col"), True, True),
        (("row",), False, True),
        (("col",), False, True),
        (("row", "col"), False, True),
        (("row", "col"), True, False),
    ],
)
def test_cebmf_pruning_resets_only_changed_self_covariate_states(prior, sides, external, prune, monkeypatch):
    options = {f"self_{side}_cov": True for side in sides}
    if not external:
        options.update({"X_l" if side == "row" else "X_f": None for side in sides})
    # Zero retains only the first factor; 1.1 retains both factors.
    model = make_model(prior, prune_thresh=0.0 if prune else 1.1, **options)
    calls = []
    real_fit = LearnedBuilder.fit

    def record_fit(builder, **kwargs):
        result = real_fit(builder, **kwargs)
        calls.append((kwargs, result.model_param))
        return result

    monkeypatch.setattr(LearnedBuilder, "fit", record_fit)
    model.fit(maxit=1)
    assert model.model.K == (1 if prune else 2)
    for factor in range(model.model.K):
        for index, side in enumerate(("row", "col")):
            saved = model.model_state_L[factor] if side == "row" else model.model_state_F[factor]
            changed = prune and side in sides
            assert saved is (None if changed else calls[2 * factor + index][1])

    result = model.fit(maxit=1)
    assert len(calls) == 4 + 2 * model.model.K
    assert torch.isfinite(result.L).all()
    assert torch.isfinite(result.F).all()
    assert torch.isfinite(torch.tensor(result.history_obj)).all()
    for index, (continued, continued_state) in enumerate(calls[4:]):
        side = ("row", "col")[index % 2]
        first, first_state = calls[index]
        changed = prune and side in sides
        assert continued["model_param"] is (None if changed else first_state)
        assert continued_state["feature_mean"].numel() == continued["X"].shape[1]
        if changed:
            if external:
                assert continued["X"].shape[1] == first["X"].shape[1] - 1
            else:
                # One other-factor feature becomes one constant feature.
                assert first["X"].shape == continued["X"].shape
                assert torch.std(first["X"]) > 0
                torch.testing.assert_close(continued["X"], torch.ones_like(continued["X"]))
                torch.testing.assert_close(
                    continued_state["feature_mean"], torch.ones_like(continued_state["feature_mean"])
                )
