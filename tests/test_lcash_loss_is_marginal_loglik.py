"""The returned loss is the summed, unpenalized negative marginal log likelihood.

cEBMF uses this value to compute its KL term. The training penalty must not
be included in the reported value.
"""

import math

import pytest
import torch

from cebmf_torch.cebnm.cash_solver import cash_posterior_means
from cebmf_torch.cebnm.lcash import lcash_posterior_means, po_lcash_posterior_means


def _proper_marginal_loglik(betahat, sebetahat, scale, pi):
    """Compute the proper full-data marginal log-likelihood.

    For each observation g and component k:
        log p(beta_g | 0, sqrt(se_g^2 + scale_k^2))
    Then sum_g logsumexp_k (log pi_g,k + log_density_g,k).
    """
    se = sebetahat.to(torch.float64)
    sd = scale.to(torch.float64)
    bh = betahat.to(torch.float64)
    pi_d = pi.to(torch.float64)

    total_var = se.unsqueeze(1) ** 2 + sd.unsqueeze(0) ** 2  # (G, K)
    log_norm_const = -0.5 * torch.log(2 * math.pi * total_var)
    log_density = log_norm_const - 0.5 * (bh.unsqueeze(1) ** 2) / total_var
    log_pi = torch.log(pi_d.clamp_min(1e-300))
    return float(torch.logsumexp(log_pi + log_density, dim=1).sum().item())


def _make_data(n=2000, seed=2026):
    g = torch.Generator().manual_seed(seed)
    # 30% slab signal at sd 1.0, 70% spike at 0
    is_slab = (torch.rand(n, generator=g) < 0.3).float()
    theta = is_slab * torch.randn(n, generator=g)
    s = torch.full((n,), 0.5)
    x = theta + s * torch.randn(n, generator=g)
    X = torch.randn(n, 1, generator=g)
    return X, x, s


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
@pytest.mark.parametrize("penalty", [1.0, 10.0])
def test_conditional_loss_is_unpenalized_marginal_loglik(solver, penalty):
    X, x, s = _make_data()
    result = solver(X, x, s, max_iter=50, penalty=penalty, device="cpu")
    expected = -_proper_marginal_loglik(x, s, result.scale, result.pi_np)
    assert result.loss == pytest.approx(expected, abs=1e-3)


def test_cash_loss_equals_negative_marginal_loglik_penalty1():
    """Same regression test for the CASH solver."""
    X, x, s = _make_data()
    res = cash_posterior_means(
        X=X,
        betahat=x,
        sebetahat=s,
        n_epochs=20,
        batch_size=512,
        penalty=1.0,
        device=torch.device("cpu"),
    )
    # CASH does not return scale or pi_np in the same shape as LC-ASH;
    # rebuild the proper marginal from res.pi_np and res.scale.
    proper_ll = _proper_marginal_loglik(x, s, res.scale, res.pi_np)
    assert abs(res.loss - (-proper_ll)) < 1e-3
