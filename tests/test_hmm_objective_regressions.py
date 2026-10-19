"""End-to-end regressions missed by standalone R/Python solver agreement."""

import copy
import math

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebnm.hmm import (
    fit_ash_hmm,
    hmm_neg_posterior_means,
    hmm_pos_posterior_means,
    hmm_posterior_means,
)
from cebmf_torch.priors.learned import LearnedBuilder, LearnedPriorType


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_pruning_cannot_spend_likelihood_gain_on_a_worse_objective():
    # Both states explain this observation. Deleting the non-null state used
    # to pass the default 0.05 deletion budget and lose about 0.0025 at step 5.
    fit = fit_ash_hmm(
        torch.tensor([0.26], dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        mu=[0, 0.5],
        prior_sd=[0],
        estimate_init=True,
        learn_state_means=False,
        maxiter=10,
    )
    assert (fit.objective_history.diff() >= 0).all()
    assert fit.state_counts[-1] == 2


def test_adapter_keeps_previous_prior_as_a_candidate_on_new_data():
    y = torch.tensor([0.0] * 8 + [2.0] * 12 + [0.0] * 8, dtype=torch.float64)
    first = hmm_posterior_means(None, y, torch.full_like(y, 0.3), maxiter=15)
    state = copy.deepcopy(first.model_param)
    # A small new fitting budget must not throw away an already useful fit.
    updated_y, se = 0.9 * y + 0.05, torch.full_like(y, 0.35)
    fixed = fit_ash_hmm(
        updated_y,
        se,
        mu=state["mu"],
        prior_sd=state["prior_sd"],
        init_transition=state["transition"],
        init_prob=state["init_prob"],
        init_rho=state["mixture_weight"],
        maxiter=0,
    )
    updated = hmm_posterior_means(None, updated_y, se, model_param=state, maxiter=0)
    assert updated.objective >= fixed.objective - 1e-10
    for field in ("mu", "prior_sd", "transition", "init_prob", "mixture_weight"):
        torch.testing.assert_close(state[field], first.model_param[field], rtol=0, atol=0)


def test_rejected_em_update_retains_parameters_and_actual_likelihood(monkeypatch):
    y = torch.full((20,), 2.0, dtype=torch.float64)
    se = torch.full_like(y, 0.2)
    options = {"mu": [0, 2], "prior_sd": [0, 0.2], "mean_update_start": 1, "prune_states": False}
    original = fit_ash_hmm(y, se, maxiter=0, **options)
    monkeypatch.setattr(
        "cebmf_torch.cebnm.hmm._update_means",
        lambda *args: y.new_tensor([0, 100]),
    )
    with pytest.warns(UserWarning, match="Rejected a decreasing"):
        result = fit_ash_hmm(y, se, maxiter=1, **options)
    assert result.iterations == 0 and not result.converged
    for field in ("post_mean", "post_mean2", "loss", "objective"):
        torch.testing.assert_close(getattr(result, field), getattr(original, field), atol=0, rtol=0)
    for field in ("mu", "transition", "init_prob", "mixture_weight"):
        torch.testing.assert_close(result.model_param[field], original.model_param[field], atol=0, rtol=0)


@pytest.mark.parametrize("solver,sign", [(hmm_pos_posterior_means, 1), (hmm_neg_posterior_means, -1)])
def test_one_sided_continuation_preserves_state_and_honors_explicit_initialization(solver, sign):
    y = sign * torch.tensor([0.0] * 8 + [2.0] * 12, dtype=torch.float64)
    se = torch.full_like(y, 0.3)
    options = {"mu": [0, sign * 2], "prior_sd": [0, 0.3], "learn_state_means": False, "prune_states": False}
    first = solver(None, y, se, maxiter=8, **options)
    before = copy.deepcopy(first.model_param)
    continued = solver(None, y, se, model_param=first.model_param, maxiter=0, **options)
    torch.testing.assert_close(continued.loss, first.loss, atol=1e-10, rtol=0)
    for key in ("mu", "transition", "init_prob", "mixture_weight"):
        torch.testing.assert_close(first.model_param[key], before[key], atol=0, rtol=0)
    explicit = {**options, "init_prob": [1, 0], "init_transition": [[1, 0], [0, 1]], "maxiter": 0}
    reset = solver(None, y, se, model_param=first.model_param, **explicit)
    direct = solver(None, y, se, **explicit)
    torch.testing.assert_close(reset.post_mean, direct.post_mean, atol=0, rtol=0)
    assert (reset.post_mean == 0).all()


def test_fresh_candidate_can_restore_states_after_previous_pruning():
    y = torch.zeros(30, dtype=torch.float64)
    se = torch.full_like(y, 0.3)
    first = hmm_posterior_means(None, y, se, maxiter=15)
    y[10:20] = 3
    result = hmm_posterior_means(None, y, se, model_param=first.model_param, maxiter=15)
    assert result.post_mean[10:20].mean() > 2
    assert result.model_param["mu"].numel() > 1


def test_unchanged_initial_options_do_not_discard_a_better_fitted_prior():
    y = torch.tensor([0.0] * 8 + [2.0] * 12, dtype=torch.float64)
    se = torch.full_like(y, 0.3)
    options = {
        "mu": [0, 2],
        "prior_sd": [0, 0.3],
        "init_prob": [0.5, 0.5],
        "init_transition": [[0.98, 0.02], [0.02, 0.98]],
        "learn_state_means": False,
        "prune_states": False,
    }
    first = hmm_posterior_means(None, y, se, maxiter=8, **options)
    continued = hmm_posterior_means(None, y, se, model_param=first.model_param, maxiter=0, **options)
    torch.testing.assert_close(continued.loss, first.loss, atol=1e-10, rtol=0)


def test_changed_support_does_not_reuse_an_incompatible_saved_prior():
    y = torch.full((12,), -2.0, dtype=torch.float64)
    se = torch.full_like(y, 0.3)
    signed = hmm_posterior_means(None, y, se, maxiter=5)
    positive = hmm_pos_posterior_means(None, y, se, model_param=signed.model_param, maxiter=0)
    direct = hmm_pos_posterior_means(None, y, se, maxiter=0)
    torch.testing.assert_close(positive.post_mean, direct.post_mean, atol=0, rtol=0)


@pytest.mark.parametrize("prior", [LearnedPriorType.HMM, LearnedPriorType.HMM_POS, LearnedPriorType.HMM_NEG])
@pytest.mark.parametrize("penalty", [1.0, 3.0])
def test_builder_reports_penalty_separately_from_marginal_loss(prior, penalty):
    y = torch.tensor([0.1, -0.3, 1.8, 1.3, 0.2, 0.4], dtype=torch.float64)
    mu = [0, -1.4] if prior == LearnedPriorType.HMM_NEG else [0, 1.4]
    fit = LearnedBuilder(
        prior,
        mu=mu,
        prior_sd=[0, 0.3, 1],
        null_state="adaptive",
        estimate_init=True,
        penalty=penalty,
        maxiter=8,
    ).fit(None, y, torch.full_like(y, 0.5))
    p = fit.model_param
    log_null = p["transition"][:, 0].log().sum() + p["init_prob"][0].log()
    log_null += p["mixture_weight"][0, 0].log()
    count = (len(p["mu"]) + 1 if len(p["mu"]) > 1 else 0) + 1
    expected_penalty = -(penalty - 1) * log_null / count
    torch.testing.assert_close(torch.as_tensor(fit.regularization_loss).to(y), expected_penalty)
    # Independent re-evaluation of the fitted prior must still give pure
    # marginal likelihood, even when the fitting penalty is nonzero.
    negative = prior == LearnedPriorType.HMM_NEG
    fixed = fit_ash_hmm(
        -y if negative else y,
        torch.full_like(y, 0.5),
        mu=-p["mu"] if negative else p["mu"],
        prior_sd=p["prior_sd"],
        init_transition=p["transition"],
        init_prob=p["init_prob"],
        init_rho=p["mixture_weight"],
        null_state="adaptive",
        nonnegative_state_means=prior != LearnedPriorType.HMM,
        maxiter=0,
    )
    torch.testing.assert_close(fit.loss, fixed.loss)


@pytest.mark.parametrize(
    "prior,side,seed,penalty",
    [
        ("hmm", "F", 1, 1),
        ("hmm", "F", 4, 1),
        ("hmm_pos", "F", 1, 1),
        ("hmm_neg", "F", 1, 1),
        ("hmm", "L", 4, 2),
        ("hmm_pos", "L", 1, 2),
        ("hmm_neg", "F", 4, 2),
        ("hmm", "both", 1, 1),
    ],
)
def test_outer_hmm_objective_is_nonincreasing_and_matches_fitted_state(prior, side, seed, penalty):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(30, generator=g, dtype=torch.float64)
    v = torch.rand(40, generator=g, dtype=torch.float64)
    y = torch.outer(u, v) + torch.randn(30, 40, generator=g, dtype=torch.float64)
    options = {"prior_L_kwargs": {"penalty": 1}, "prior_F_kwargs": {"penalty": 1}}
    for s in ("L", "F") if side == "both" else (side,):
        options[f"prior_{s}"] = prior
        options[f"prior_{s}_kwargs"] = {"penalty": penalty}
    model = cEBMF(y, K=3, allow_backfitting=False, device="cpu", **options)
    model.initialise_factors()
    for _ in range(20):
        model.fit(1)
        expected_sse = (y - model.L @ model.F.T).square().sum()
        for k in range(3):
            expected_sse += (
                torch.outer(model.L2[:, k], model.F2[:, k])
                - torch.outer(model.L[:, k].square(), model.F[:, k].square())
            ).sum()
        nll = 0.5 * (y.numel() * (math.log(2 * math.pi) - model.tau.log()) + model.tau * expected_sse)
        regularization = y.new_zeros(())
        if penalty > 1:
            for s in ("L", "F") if side == "both" else (side,):
                for state in getattr(model, f"model_state_{s}"):
                    count = len(state["mu"]) + 1 if len(state["mu"]) > 1 else 1
                    regularization -= (
                        (penalty - 1) / count * (state["transition"][:, 0].log().sum() + state["init_prob"][0].log())
                    )
        expected = nll + model.kl_l.sum() + model.kl_f.sum() + regularization
        assert model.obj[-1] == pytest.approx(expected.item(), abs=1e-9, rel=0)
        for s in ("L", "F"):
            mean, second = getattr(model, s), getattr(model, s + "2")
            assert torch.isfinite(mean).all() and torch.isfinite(second).all()
            assert (second >= mean.square() - 1e-10).all()
    history = torch.tensor(model.obj, dtype=torch.float64)
    assert history.numel() == 20 and torch.isfinite(history).all()
    assert history.diff().max() <= 1e-9
