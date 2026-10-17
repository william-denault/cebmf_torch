"""Fitted conditional priors remain usable through matrix-factorization updates."""

from copy import deepcopy

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.priors.learned import LearnedPriorType

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
def test_cebmf_continues_from_copied_fitted_states(prior):
    model = make_model(prior, allow_backfitting=False, prune_thresh=1.1)
    model.fit(maxit=1)
    copied = make_model(prior, allow_backfitting=False, prune_thresh=1.1)
    copied.fit(maxit=1)
    copied.model_state_L = deepcopy(model.model_state_L)
    copied.model_state_F = deepcopy(model.model_state_F)
    result = model.fit(maxit=2)
    restored = copied.fit(maxit=2)
    for field in ("L", "F", "history_obj"):
        actual = torch.as_tensor(getattr(result, field))
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, torch.as_tensor(getattr(restored, field)))


def predict_null(builder, X, state):
    return builder.fit(
        X=X,
        betahat=torch.ones(X.shape[0]),
        sebetahat=torch.ones(X.shape[0]),
        model_param=state,
        internal_epoch=0,
    ).pi0_null


@pytest.mark.parametrize("prior", PRIOR_TYPES)
@pytest.mark.parametrize(
    "sides,external,prune",
    [
        (("row",), True, True),
        (("col",), True, True),
        (("row", "col"), True, True),
        (("row", "col"), False, True),
        (("row", "col"), True, False),
        ((), True, True),
    ],
)
def test_cebmf_pruning_preserves_covariate_meaning(prior, sides, external, prune):
    options = {f"self_{side}_cov": True for side in sides}
    if not external:
        options.update({"X_l" if side == "row" else "X_f": None for side in sides})
    # Zero retains only the first factor; 1.1 retains both factors.
    model = make_model(prior, prune_thresh=0.0 if prune else 1.1, **options)
    reference = make_model(prior, prune_thresh=1.1, **options)
    reference.fit(maxit=1)
    model.fit(maxit=1)
    assert model.model.K == (1 if prune else 2)

    for side, suffix, X in (("row", "L", model.covariate.X_l), ("col", "F", model.covariate.X_f)):
        if side not in sides:
            # Removing other factors must not change a prior with fixed external covariates.
            builder = getattr(model, f"prior_{suffix}_fn")
            actual = predict_null(builder, X, getattr(model, f"model_state_{suffix}")[0])
            expected = predict_null(builder, X, getattr(reference, f"model_state_{suffix}")[0])
            torch.testing.assert_close(actual, expected)

    result = model.fit(maxit=1)
    for field in ("L", "F", "history_obj"):
        assert torch.isfinite(torch.as_tensor(getattr(result, field))).all()
    if prune and not external:
        # One other-factor feature becomes a constant, with the same column count.
        # A prior fitted without informative covariates must have no covariate response.
        for suffix in ("L", "F"):
            builder = getattr(model, f"prior_{suffix}_fn")
            state = getattr(model, f"model_state_{suffix}")[0]
            pi0 = predict_null(builder, torch.tensor([[-10.0], [0.0], [10.0]]), state)
            torch.testing.assert_close(pi0, pi0[0].expand_as(pi0))
