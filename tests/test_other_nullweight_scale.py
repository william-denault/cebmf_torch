"""Full-count null penalties for CGB, sharp CGB, and spiked EMDN."""

import math

import numpy as np
import pytest
import torch

from cebmf_torch.cebnm import cov_gb_prior, cov_sharp_gb_prior, spiked_emdn

CASES = [
    pytest.param(cov_gb_prior, "cgb_posterior_means", id="cgb"),
    pytest.param(cov_sharp_gb_prior, "sharp_cgb_posterior_means", id="sharp_cgb"),
    pytest.param(spiked_emdn, "spiked_emdn_posterior_means", id="spiked_emdn"),
]


def _reference(args, penalty, n_obs):
    """Independent two-component density, including the existing floors."""
    pi1, pi2, mu, variance, y, se = args
    pi = torch.stack((pi1, pi2), dim=1)
    location = torch.stack((torch.zeros_like(y), torch.zeros_like(y) + mu), dim=1)
    scale = torch.stack((torch.zeros_like(y), torch.sqrt(torch.zeros_like(y) + variance)), dim=1)
    density = torch.distributions.Normal(location, (scale.square() + se[:, None].square()).sqrt())
    log_pi = pi.clamp_min(1e-8).log()
    result = -torch.logsumexp(log_pi + density.log_prob(y[:, None]), dim=1).mean()
    coefficient = (penalty - 1) / n_obs if penalty > 1 else 0
    return result - coefficient * log_pi[:, 0].mean()


def _args(pi0, y):
    return pi0, 1 - pi0, 0.0, 9.0, y, torch.ones_like(y)


@pytest.mark.parametrize("penalty", [0.5, 1.0, 1.5, 10.0])
def test_loss_values_and_gradients(penalty):
    pi0 = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    args = _args(pi0, torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64))
    loss = cov_gb_prior.cgb_loss
    for n_obs in (1, 7):
        actual = loss(*args, penalty=penalty, n_obs=n_obs)
        expected = _reference(args, penalty, n_obs)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, pi0, retain_graph=True)[0],
            torch.autograd.grad(expected, pi0, retain_graph=True)[0],
        )


@pytest.mark.parametrize("n_obs", [0, -1])
def test_count_must_be_positive(n_obs):
    args = _args(torch.full((2,), 0.5), torch.zeros(2))
    with pytest.raises(ValueError):
        cov_gb_prior.cgb_loss(*args, penalty=10.0, n_obs=n_obs)


@pytest.mark.parametrize("module,solver_name", CASES)
def test_public_default_full_count_and_unpenalized_loss(module, solver_name, monkeypatch):
    loss_name = "mdn_spike_loss_with_varying_noise" if module is spiked_emdn else "cgb_loss"
    real_loss = getattr(module, loss_name)
    batch_sizes = []

    def check_loss(*args, **kwargs):
        batch_sizes.append(len(args[0]))
        assert kwargs["n_obs"] == 7
        assert kwargs["penalty"] == 10
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(module, loss_name, check_loss)
    torch.manual_seed(41)
    y = torch.tensor([-2.0, 0.5, 0.0, 1.0, 3.0, -0.2, 0.7])
    result = getattr(module, solver_name)(
        torch.arange(len(y), dtype=y.dtype)[:, None],
        y,
        torch.ones_like(y),
        n_epochs=1,
        batch_size=3,
        hidden_dim=4,
        n_layers=0,
        device=torch.device("cpu"),
    )
    assert 3 in batch_sizes and any(size < 3 for size in batch_sizes)
    if module is spiked_emdn:
        pi, location, scale = result.pi_np, result.location, result.scale
    else:
        pi = torch.stack((result.pi, 1 - result.pi), dim=1)
        location = torch.tensor([0.0, result.mu_2])
        scale = torch.tensor([0.0, result.sigma_2])
    log_lik = torch.distributions.Normal(location, (scale.square() + 1).sqrt()).log_prob(y[:, None])
    expected = -torch.logsumexp(pi.log() + log_lik, dim=1).sum()
    assert result.loss == pytest.approx(expected.item(), abs=2e-6)


def test_fixed_total_penalty_recovers_known_null_weight():
    """Fit only the null weight; fix both component shapes at their true values."""
    n = 5000
    rng = np.random.default_rng(11)
    is_null = rng.random(n) < 0.6
    theta = np.where(is_null, 0.0, rng.normal(0, 3, n))
    y = torch.from_numpy(theta + rng.normal(size=n))
    fitted = []
    for n_obs in (1, n):
        logit = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [logit], max_iter=100, tolerance_grad=1e-12, tolerance_change=1e-20, line_search_fn="strong_wolfe"
        )

        def closure(optimizer=optimizer, logit=logit, n_obs=n_obs):
            optimizer.zero_grad()
            args = _args(logit.sigmoid().expand(n), y)
            loss = cov_gb_prior.cgb_loss(*args, penalty=1.5, n_obs=n_obs)
            loss.backward()
            return loss

        optimizer.step(closure)
        q = logit.detach().sigmoid().requires_grad_()
        reference = _reference(_args(q.expand(n), y), 1.5, n_obs)
        # The loss is strictly convex in q. A stationary interior point is its minimum.
        assert abs(float(torch.autograd.grad(reference, q)[0])) < 1e-8
        fitted.append(float(q.detach()))
    per_observation, fixed_total = fitted
    assert abs(fixed_total - 0.6) < 0.03
    assert per_observation > fixed_total + 0.15
    print(f"Per-observation null weight={per_observation:.6f}; fixed-total={fixed_total:.6f}")


@pytest.mark.parametrize("penalty", [0.5, 1.0, 10.0])
def test_spiked_emdn_count_scaling_and_separate_beta_prior(penalty):
    pi0 = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    pi = torch.stack((pi0, 1 - pi0), dim=1)
    y = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    args = (pi, torch.zeros(3, 1), torch.full((3, 1), math.log(3)), y, torch.ones_like(y))
    loss = spiked_emdn.mdn_spike_loss_with_varying_noise
    for n_obs in (1, 7):
        actual = loss(*args, penalty=penalty, n_obs=n_obs, beta_prior=(2, 3))
        # Unlike CGB, spiked EMDN keeps negative null coefficients below one.
        expected = _reference(_args(pi0, y), 1.0, n_obs)
        expected -= ((penalty - 1) / n_obs) * pi0.log().mean()
        expected -= (pi0.log() + 2 * (1 - pi0).log()).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, pi0, retain_graph=True)[0],
            torch.autograd.grad(expected, pi0, retain_graph=True)[0],
        )
    with pytest.raises(ValueError):
        loss(*args, penalty=penalty, n_obs=0)
