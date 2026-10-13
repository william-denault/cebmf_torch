from types import SimpleNamespace

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.priors import PRIOR_REGISTRY, learned


@pytest.fixture(autouse=True)
def restore_registry_kwargs(monkeypatch):
    for builder in PRIOR_REGISTRY.registry.values():
        monkeypatch.setattr(builder, "kwargs", {})


@pytest.mark.parametrize(
    ("prior_name", "L_kwargs", "F_kwargs"),
    [
        ("norm", {"penalty": 1.0}, {"penalty": 2.0}),
        ("laplace", {"max_iter": 10}, {"max_iter": 20}),
        ("cgb", {"n_layers": 3, "penalty": 1.2}, {"penalty": 1.0}),
        ("lcash", {"penalty": 1.0}, {"penalty": 2.0}),
        ("po_lcash", {"penalty": 1.0}, {"penalty": 2.0}),
    ],
)
def test_same_prior_keeps_separate_side_options(prior_name, L_kwargs, F_kwargs):
    model = cEBMF(
        torch.ones(6, 4),
        K=1,
        prior_L=prior_name,
        prior_F=prior_name,
        prior_L_kwargs=L_kwargs,
        prior_F_kwargs=F_kwargs,
        device="cpu",
    )

    assert model.prior_L_fn.kwargs == L_kwargs
    assert model.prior_F_fn.kwargs == F_kwargs
    assert PRIOR_REGISTRY.get_builder(prior_name).kwargs == {}


@pytest.mark.parametrize("configured_side", ["L", "F"])
def test_omitted_side_options_remain_defaults(configured_side):
    options = {"penalty": 1.2}
    model = cEBMF(
        torch.ones(6, 4),
        K=1,
        prior_L="lcash",
        prior_F="lcash",
        device="cpu",
        **{f"prior_{configured_side}_kwargs": options},
    )

    assert model.prior_L_fn.kwargs == (options if configured_side == "L" else {})
    assert model.prior_F_fn.kwargs == (options if configured_side == "F" else {})


@pytest.mark.parametrize("prior_name", ["cgb", "lcash", "po_lcash"])
def test_solver_options_survive_another_model(monkeypatch, prior_name):
    calls = []

    def record_solver(X, betahat, sebetahat, *, model_param, device, **kwargs):
        calls.append((betahat.numel(), kwargs))
        pi = torch.full_like(betahat, 0.5)
        return SimpleNamespace(
            post_mean=betahat,
            post_mean2=betahat.square() + sebetahat.square(),
            loss=betahat.new_zeros(()),
            model_param=model_param,
            pi=pi,
            pi_np=torch.stack((pi, pi), dim=1),
        )

    monkeypatch.setitem(learned.builder_functions, learned.LearnedPriorType(prior_name), record_solver)
    data = torch.arange(1, 25, dtype=torch.float32).reshape(6, 4)
    L_kwargs = {"penalty": 1.2}
    if prior_name == "cgb":
        L_kwargs["n_layers"] = 3
    F_kwargs = {"penalty": 1.0}
    first = cEBMF(
        data,
        K=1,
        prior_L=prior_name,
        prior_F=prior_name,
        prior_L_kwargs=L_kwargs,
        prior_F_kwargs=F_kwargs,
        X_l=torch.ones(6, 1),
        X_f=torch.ones(4, 1),
        allow_backfitting=False,
        internal_epoch=2,
        device="cpu",
    )
    first.initialise_factors()
    first.fit(maxit=1)
    calls.clear()

    cEBMF(
        data,
        K=1,
        prior_L=prior_name,
        prior_F=prior_name,
        prior_L_kwargs={"penalty": 3.0},
        prior_F_kwargs={"penalty": 4.0},
        device="cpu",
    )
    first.fit(maxit=1)

    budget = {"n_epochs": 2} if prior_name == "cgb" else {"max_iter": 4}
    expected = {6: {**L_kwargs, **budget}, 4: {**F_kwargs, **budget}}
    assert {size for size, _ in calls} == expected.keys()
    assert all(options == expected[size] for size, options in calls)


@pytest.mark.parametrize(
    "prior_type,internal_epoch,options,expected",
    [
        (prior_type, internal_epoch, options, expected)
        for prior_type in (learned.LearnedPriorType.LCASH, learned.LearnedPriorType.PO_LCASH)
        for internal_epoch, options, expected in [
            (10, {}, {"max_iter": 20}),
            (10, {"max_iter": None}, {"max_iter": 20}),
            (10, {"max_iter": 0}, {"max_iter": 0}),
            (10, {"max_iter": 7}, {"max_iter": 7}),
            (0, {}, {"max_iter": 0}),
            (None, {}, {}),
        ]
    ]
    + [(learned.LearnedPriorType.CGB, 10, {}, {"n_epochs": 10})],
)
def test_iteration_budget_precedence(monkeypatch, prior_type, internal_epoch, options, expected):
    calls = []

    def record_solver(X, betahat, sebetahat, *, model_param, device, **kwargs):
        calls.append(kwargs)
        pi = torch.full_like(betahat, 0.5)
        return SimpleNamespace(
            post_mean=betahat,
            post_mean2=betahat.square() + sebetahat.square(),
            loss=betahat.new_zeros(()),
            model_param=model_param,
            pi=pi,
            pi0_null=pi,
            pi_np=torch.stack((pi, pi), dim=1),
        )

    monkeypatch.setitem(learned.builder_functions, prior_type, record_solver)
    builder = learned.LearnedBuilder(prior_type, **options)
    builder.fit(torch.zeros(2, 1), torch.zeros(2), torch.ones(2), internal_epoch=internal_epoch)

    assert calls and all(options == expected for options in calls)
    assert builder.kwargs == options
