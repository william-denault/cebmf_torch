"""CASH distributes the null penalty over the full training set."""

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebnm import cash_solver
from cebmf_torch.priors.learned import LearnedPriorType
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch


def _fit(penalty=None):
    torch.manual_seed(41)
    return cash_solver.cash_posterior_means(
        X=torch.arange(7, dtype=torch.float32).reshape(-1, 1),
        betahat=torch.tensor([-2.0, 0.5, 0.0, 1.0, 3.0, -0.2, 0.7]),
        sebetahat=torch.ones(7),
        n_epochs=2,
        batch_size=3,
        hidden_dim=4,
        n_layers=0,
        num_classes=3,
        device=torch.device("cpu"),
        **({} if penalty is None else {"penalty": penalty}),
    )


@pytest.mark.parametrize("penalty", [None, 1.0, 1.5])
def test_training_uses_full_count_in_every_batch(penalty, monkeypatch):
    real_loss = cash_solver.pen_loglik_loss
    batch_sizes = []
    expected_penalty = 10.0 if penalty is None else penalty

    def check_loss(pred_pi, marginal_log_lik, penalty, n_obs):
        batch_sizes.append(len(pred_pi))
        assert n_obs == 7
        actual = real_loss(pred_pi, marginal_log_lik, penalty=penalty, n_obs=n_obs)
        log_pi = pred_pi.clamp_min(1e-10).log()
        expected = -torch.logsumexp(log_pi + marginal_log_lik, dim=1).sum()
        expected = expected - ((expected_penalty - 1) / 7) * log_pi[:, 0].sum()
        actual_gradient = torch.autograd.grad(actual, pred_pi, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, pred_pi, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        return actual

    monkeypatch.setattr(cash_solver, "pen_loglik_loss", check_loss)
    actual = _fit(penalty)
    assert 3 in batch_sizes and any(size < 3 for size in batch_sizes)
    if penalty is None:
        monkeypatch.setattr(cash_solver, "pen_loglik_loss", real_loss)
        explicit = _fit(10.0)
        for name in ("post_mean", "post_mean2", "post_sd", "pi_np"):
            torch.testing.assert_close(getattr(actual, name), getattr(explicit, name))

    log_lik = get_data_loglik_normal_torch(
        betahat=torch.tensor([-2.0, 0.5, 0.0, 1.0, 3.0, -0.2, 0.7]),
        sebetahat=torch.ones(7),
        location=torch.zeros_like(actual.scale),
        scale=actual.scale,
    )
    expected = -torch.logsumexp(actual.pi_np.log() + log_lik, dim=1).sum()
    assert actual.loss == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)


def test_cebmf_uses_each_matrix_side_count(monkeypatch):
    torch.manual_seed(42)
    model = cEBMF(
        torch.randn(7, 11),
        K=1,
        prior_L=LearnedPriorType.CASH,
        prior_F=LearnedPriorType.CASH,
        X_l=torch.randn(7, 1),
        X_f=torch.randn(11, 1),
        prior_L_kwargs={"n_epochs": 1, "penalty": 10.0, "batch_size": 3, "hidden_dim": 4, "n_layers": 0},
        prior_F_kwargs={"n_epochs": 1, "penalty": 10.0, "batch_size": 3, "hidden_dim": 4, "n_layers": 0},
        S=1.0,
        device=torch.device("cpu"),
    )
    model.initialise_factors()
    real_loss = cash_solver.pen_loglik_loss
    coefficients = []

    def record_loss(pred_pi, marginal_log_lik, penalty, n_obs):
        coefficients.append((penalty - 1) / n_obs)
        return real_loss(pred_pi, marginal_log_lik, penalty=penalty, n_obs=n_obs)

    monkeypatch.setattr(cash_solver, "pen_loglik_loss", record_loss)
    model.fit(maxit=1)
    assert set(coefficients) == {9 / 7, 9 / 11}


@pytest.mark.parametrize("n_obs", [0, -1])
def test_count_must_be_positive(n_obs):
    with pytest.raises(ValueError):
        cash_solver.pen_loglik_loss(torch.full((2, 2), 0.5), torch.zeros(2, 2), penalty=10.0, n_obs=n_obs)
