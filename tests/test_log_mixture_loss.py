"""Contracts for the shared log-space mixture loss."""

import pytest
import torch

from cebmf_torch.cebnm.cash_solver import pen_loglik_loss_from_log_pi


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_weights_preserve_small_components_and_gradients(dtype):
    log_pi = torch.tensor([[-1000.0, 0.0]], dtype=dtype, requires_grad=True)
    log_lik = torch.tensor([[1000.0, 0.0]], dtype=dtype)
    loss = pen_loglik_loss_from_log_pi(log_pi, log_lik, null_coefficient=0.0)
    gradient = torch.autograd.grad(loss, log_pi)[0]
    torch.testing.assert_close(loss, -torch.tensor(2.0, dtype=dtype).log())
    torch.testing.assert_close(gradient, torch.tensor([[-0.5, -0.5]], dtype=dtype))


def test_zero_null_coefficient_accepts_negative_infinite_null_log_weight():
    log_pi = torch.tensor([[-torch.inf, 0.0], [-torch.inf, 0.0]], dtype=torch.float64, requires_grad=True)
    log_lik = torch.tensor([[4.0, -2.0], [10.0, 3.0]], dtype=torch.float64)
    loss = pen_loglik_loss_from_log_pi(log_pi, log_lik)
    gradient = torch.autograd.grad(loss, log_pi)[0]
    torch.testing.assert_close(loss, -log_lik[:, 1].sum(), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=gradient.dtype))


@pytest.mark.parametrize("penalty", [1.0, 1.5, 10.0])
def test_constant_weights_match_ash_penalty(penalty):
    pi = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64, requires_grad=True)
    log_lik = torch.linspace(-2.0, 2.0, 21, dtype=pi.dtype).reshape(7, 3)
    n = len(log_lik)
    actual = pen_loglik_loss_from_log_pi(pi.log().expand(n, -1), log_lik, (penalty - 1) / n)
    expected = -(pi * log_lik.exp()).sum(dim=1).log().sum() - (penalty - 1) * pi[0].log()
    actual_gradient = torch.autograd.grad(actual, pi)[0]
    expected_gradient = torch.autograd.grad(expected, pi)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-14, atol=1e-14)
