"""Regression test for cebmf_torch.cebnm.lcash and cebnm.cash_solver:

Asserts that ``result.loss == -log p(y | fitted prior)`` (i.e. the negative
full-data marginal log-likelihood, no penalty), as required by the cebmf
consumer at ``cebmf/cebmf.py:299``: ``self.kl_l[k] = (-resL.loss) - nm_ll_L``.

Before the fix the returned ``loss`` was ``final_epoch_loss`` from training
(a sum of per-batch ``pen_loglik_loss`` values on the last epoch), which is
not on the marginal-log-lik scale once ``penalty > 1`` and clamps the inner
density at ``1e-10``. After the fix it is computed via a single full-batch
``logsumexp`` over the saved per-observation pi values.
"""
import math

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


def test_lcash_loss_equals_negative_marginal_loglik_penalty1():
    """With penalty=1.0 (no Dirichlet penalty) the returned loss must be
    the proper negative marginal log-lik, not the training-loss surrogate."""
    X, x, s = _make_data()
    res = lcash_posterior_means(
        X=X, betahat=x, sebetahat=s,
        n_epochs=50, batch_size=512,
        lr=1e-3, weight_decay=1e-3,
        penalty=1.0, ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
    )
    proper_ll = _proper_marginal_loglik(x, s, res.scale, res.pi_np)
    expected_loss = -proper_ll
    # Tight tolerance: the post-fix loss is computed from exactly the same
    # tensor that the test recomputes here.
    assert abs(res.loss - expected_loss) < 1e-3, (
        f"LC-ASH loss = {res.loss}, expected -marginal_ll = {expected_loss}"
    )


def test_po_lcash_loss_equals_negative_marginal_loglik_penalty1():
    X, x, s = _make_data()
    res = po_lcash_posterior_means(
        X=X, betahat=x, sebetahat=s,
        n_epochs=50, batch_size=512,
        lr=1e-3, weight_decay=1e-3,
        penalty=1.0, ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
    )
    proper_ll = _proper_marginal_loglik(x, s, res.scale, res.pi_np)
    assert abs(res.loss - (-proper_ll)) < 1e-3


def test_lcash_loss_independent_of_penalty():
    """Two LC-ASH fits whose only difference is `penalty` should produce
    `loss` values that, after the fix, both equal the proper marginal log-lik
    (computed from the respective fitted pi). Pre-fix the two values would
    differ by exactly (penalty-1) * sum_g log(pi_g,0)."""
    X, x, s = _make_data()
    common = {
        "X": X,
        "betahat": x,
        "sebetahat": s,
        "n_epochs": 50,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "ash_init": True,
        "verbose": False,
        "device": torch.device("cpu"),
        "seed": 42,
    }
    res1 = lcash_posterior_means(penalty=1.0, **common)
    res2 = lcash_posterior_means(penalty=1.5, **common)
    ll1 = _proper_marginal_loglik(x, s, res1.scale, res1.pi_np)
    ll2 = _proper_marginal_loglik(x, s, res2.scale, res2.pi_np)
    assert abs(res1.loss - (-ll1)) < 1e-3
    assert abs(res2.loss - (-ll2)) < 1e-3


def test_cash_loss_equals_negative_marginal_loglik_penalty1():
    """Same regression test for the CASH solver."""
    X, x, s = _make_data()
    res = cash_posterior_means(
        X=X, betahat=x, sebetahat=s,
        n_epochs=20, batch_size=512,
        penalty=1.0,
        device=torch.device("cpu"),
    )
    # CASH does not return scale or pi_np in the same shape as LC-ASH;
    # rebuild the proper marginal from res.pi_np and res.scale.
    proper_ll = _proper_marginal_loglik(x, s, res.scale, res.pi_np)
    assert abs(res.loss - (-proper_ll)) < 1e-3
