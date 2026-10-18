"""ASH coordinate updates retain a feasible prior and optimize the intended objective."""

import math

import pytest
import torch

from cebmf_torch import ash, cEBMF
from cebmf_torch.priors.mixture import ASHBuilder, ASHState, MixturePriorType
from cebmf_torch.utils.distribution_operation import get_data_loglik_exp_torch
from cebmf_torch.utils.mixture import mixture_log_penalty, optimize_pi_logL


def normal_loglik(x, s, scale, pi):
    component_sd = (s[:, None].square() + scale.square()).sqrt()
    log_density = torch.distributions.Normal(0, component_sd).log_prob(x[:, None])
    return torch.logsumexp(log_density + pi.log(), dim=1).sum()


@pytest.mark.parametrize("optimizer", ["em", "lbfgs"])
@pytest.mark.parametrize("prior", ["norm", "exp"])
def test_fixed_grid_warm_start_improves_marginal_likelihood(prior, optimizer):
    x = torch.tensor([-1.0, 0.2, 2.0, 0.6], dtype=torch.float64)
    s = torch.tensor([0.5, 0.9, 0.4, 0.7], dtype=x.dtype)
    scale = x.new_tensor([0.0, 0.3, 1.0, 3.0])
    pi = x.new_tensor([0.6, 0.2, 0.1, 0.1])
    original_pi, original_scale = pi.clone(), scale.clone()
    if prior == "norm":
        initial_loglik = normal_loglik(x, s, scale, pi)
    else:
        initial_loglik = torch.logsumexp(get_data_loglik_exp_torch(x, s, scale) + pi.log(), dim=1).sum()
    result = ash(x, s, prior=prior, penalty=1, scale=scale, pi_init=pi, optimizer=optimizer, zero_threshold=0)
    assert result.log_lik >= initial_loglik - 1e-10
    torch.testing.assert_close(result.scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(pi, original_pi, rtol=0, atol=0)
    torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)


def test_em_zero_iterations_preserves_requested_weights():
    loglik = torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float64)
    initial = torch.tensor([1.0, 8.0, 1.0], dtype=loglik.dtype)
    result = optimize_pi_logL(loglik, penalty=1, max_iters=0, pi_init=initial)
    torch.testing.assert_close(result, initial / initial.sum(), rtol=0, atol=0)


@pytest.mark.parametrize("penalty", [1.0, 10.0])
def test_builder_keeps_previous_prior_feasible_without_mutating_state(penalty):
    x = torch.linspace(-2.0, 2.0, 30, dtype=torch.float64)
    s = torch.full_like(x, 0.7)
    builder = ASHBuilder(MixturePriorType.NORM, penalty=penalty)
    first = builder.fit(None, x, s)
    state = first.model_param
    assert isinstance(state, ASHState)
    saved_scale, saved_pi = state.scale.clone(), state.pi.clone()
    new_x, new_s = x * 1.4, s * 0.8
    previous_criterion = normal_loglik(new_x, new_s, state.scale, state.pi) + (penalty - 1) * state.pi[0].log()
    updated = builder.fit(None, new_x, new_s, model_param=state)
    updated_criterion = -updated.loss + (penalty - 1) * updated.pi0_null.log()
    assert updated_criterion >= previous_criterion - 1e-10
    torch.testing.assert_close(state.scale, saved_scale, rtol=0, atol=0)
    torch.testing.assert_close(state.pi, saved_pi, rtol=0, atol=0)
    assert updated.model_param is not state


@pytest.mark.parametrize("prior", [MixturePriorType.NORM, MixturePriorType.EXP])
def test_builder_preserves_standalone_ash_penalty_default(prior):
    x = torch.linspace(-2.0, 2.0, 30, dtype=torch.float64)
    s = torch.full_like(x, 0.7)
    result = ASHBuilder(prior).fit(None, x, s)
    reference = ash(x, s, prior=str(prior), penalty=10)
    torch.testing.assert_close(result.post_mean, reference.post_mean)
    torch.testing.assert_close(result.loss, -reference.log_lik)
    torch.testing.assert_close(result.regularization_loss, -9 * reference.pi0.log())


@pytest.mark.parametrize("scale", [[], [1.0, 2.0], [0.0, -1.0], [0.0, math.nan], [[0.0, 1.0]]])
def test_invalid_fixed_grid_is_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        ash(torch.ones(3), torch.ones(3), scale=torch.tensor(scale))


def test_initial_weights_require_an_explicit_grid():
    with pytest.raises(ValueError, match="explicit scale"):
        ash(torch.ones(3), torch.ones(3), pi_init=torch.tensor([0.5, 0.5]))


def test_zero_weights_with_no_penalty_have_zero_regularization():
    pi = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    assert mixture_log_penalty(pi, 1.0) == 0
    assert mixture_log_penalty(pi, torch.ones_like(pi)) == 0


def test_per_side_penalties_are_counted_once_and_follow_pruned_factors():
    generator = torch.Generator().manual_seed(17)
    y = torch.rand(12, 10, generator=generator, dtype=torch.float64)
    scale = y.new_tensor([0.0, 0.1, 0.5, 2.0])
    alpha_l = y.new_tensor([3.0, 1.0, 2.0, 1.0])
    model = cEBMF(
        y,
        K=3,
        prior_L_kwargs={"penalty": alpha_l, "scale": scale},
        prior_F_kwargs={"penalty": 4.0, "scale": scale},
        allow_backfitting=False,
        device="cpu",
    )
    model.initialise_factors()
    model.fit(maxit=2)
    assert len({id(state) for state in model.model_state_L + model.model_state_F}) == 6

    def check_objective():
        penalty = -sum(((alpha_l - 1) * state.pi.log()).sum().item() for state in model.model_state_L)
        penalty -= 3 * sum(state.pi[0].log().item() for state in model.model_state_F)
        assert model.obj[-1] == pytest.approx(model.negative_elbo[-1] + penalty, abs=1e-10)

    check_objective()
    model._prune_indices([1])
    assert model.obj == model.negative_elbo == []
    model._cal_obj()
    check_objective()
    model.iter_once()
    check_objective()
