"""Tests for the mixture weight optimizer (L-BFGS + softmax)."""

import math

import torch

from cebmf_torch.utils.mixture import autoselect_scales_mix_norm, optimize_pi_logL, optimize_pi_logL_lbfgs
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch


def _penalised_ll(pi, logL, penalty=10.0):
    """Compute penalised log-likelihood."""
    K = logL.shape[1]
    vec_pen = torch.ones(K, dtype=pi.dtype)
    vec_pen[0] = penalty
    active = pi > 0
    pi_a = pi[active]
    log_pi = torch.log(pi_a)
    log_mix = logL[:, active] + log_pi.unsqueeze(0)
    ll = torch.logsumexp(log_mix, dim=1).sum()
    ll += ((vec_pen[active] - 1.0) * log_pi).sum()
    return ll.item()


def test_lbfgs_produces_sparse_solution():
    """L-BFGS should produce fewer active components than pure EM."""
    torch.manual_seed(42)
    betahat = torch.randn(5000)
    se = torch.ones(5000)
    scale = autoselect_scales_mix_norm(betahat, se, mult=math.sqrt(2))
    logL = get_data_loglik_normal_torch(
        betahat, se, torch.zeros_like(scale), scale,
    )

    pi_em = optimize_pi_logL(logL, penalty=10.0, verbose=False)
    pi_lbfgs = optimize_pi_logL_lbfgs(logL, penalty=10.0)

    # Compare at the same threshold: components > 1e-6
    n_em = (pi_em > 1e-6).sum().item()
    n_lbfgs = (pi_lbfgs > 1e-6).sum().item()
    assert n_lbfgs < n_em, (
        f"L-BFGS ({n_lbfgs}) should have fewer active than EM ({n_em})"
    )


def test_lbfgs_simplex_constraint():
    """L-BFGS result should be on the simplex."""
    torch.manual_seed(7)
    betahat = torch.randn(500)
    se = torch.ones(500) * 0.5
    scale = autoselect_scales_mix_norm(betahat, se, mult=math.sqrt(2))
    logL = get_data_loglik_normal_torch(
        betahat, se, torch.zeros_like(scale), scale,
    )

    pi = optimize_pi_logL_lbfgs(logL, penalty=10.0)
    assert (pi >= 0).all(), "All weights must be >= 0"
    assert abs(pi.sum().item() - 1.0) < 1e-6


def test_optimizer_em_is_deterministic():
    """With EM, results are deterministic and have no exact zeros."""
    torch.manual_seed(1)
    betahat = torch.randn(500)
    se = torch.ones(500)
    scale = autoselect_scales_mix_norm(betahat, se)
    logL = get_data_loglik_normal_torch(
        betahat, se, torch.zeros_like(scale), scale,
    )

    pi_a = optimize_pi_logL(logL, penalty=10.0, verbose=False)
    pi_b = optimize_pi_logL(logL, penalty=10.0, verbose=False)
    assert torch.allclose(pi_a, pi_b, atol=1e-10)
    assert (pi_a > 0).all(), "Pure EM should not produce exact zeros"


def test_lbfgs_spike_dominated():
    """On pure-noise data, the spike should dominate."""
    torch.manual_seed(123)
    betahat = torch.randn(1000)
    se = torch.ones(1000)
    scale = autoselect_scales_mix_norm(betahat, se, mult=math.sqrt(2))
    logL = get_data_loglik_normal_torch(
        betahat, se, torch.zeros_like(scale), scale,
    )

    pi = optimize_pi_logL_lbfgs(logL, penalty=10.0)
    assert pi[0] > 0.5, f"Spike should dominate, got {pi[0]:.4f}"
    assert pi[0] == pi.max()


def test_lbfgs_vs_em_spike_dominated():
    """On spike-dominated data, L-BFGS finds the sparse solution that EM misses.

    EM spreads mass across aliased near-spike components because it
    structurally cannot produce exact zeros.  L-BFGS with softmax
    drives aliases to zero via the BFGS Hessian approximation,
    producing a sparser solution with higher penalised log-likelihood.
    """
    torch.manual_seed(42)
    n = 10000
    se = 0.1 * torch.ones(n)

    # 95% null, 5% signal from N(0, 0.04)
    n_null = int(0.95 * n)
    beta_null = torch.randn(n_null) * se[:n_null]
    theta = torch.randn(n - n_null) * 0.2
    beta_signal = theta + torch.randn(n - n_null) * se[n_null:]
    betahat = torch.cat([beta_null, beta_signal])
    betahat = betahat[torch.randperm(n)]

    scale = autoselect_scales_mix_norm(betahat, se, mult=math.sqrt(2))
    logL = get_data_loglik_normal_torch(
        betahat, se, torch.zeros_like(scale), scale,
    )

    # Pure EM
    pi_em = optimize_pi_logL(logL, penalty=10.0, verbose=False)
    # L-BFGS
    pi_lbfgs = optimize_pi_logL_lbfgs(logL, penalty=10.0)

    # L-BFGS should be sparser
    n_active_em = (pi_em > 1e-6).sum().item()
    n_active_lbfgs = (pi_lbfgs > 0).sum().item()
    assert n_active_lbfgs < n_active_em, (
        f"L-BFGS ({n_active_lbfgs}) should have fewer active than "
        f"EM ({n_active_em})"
    )

    # L-BFGS should have higher spike weight
    assert pi_lbfgs[0] > pi_em[0] + 0.01, (
        f"L-BFGS spike ({pi_lbfgs[0]:.4f}) should exceed "
        f"EM spike ({pi_em[0]:.4f})"
    )

    # L-BFGS should achieve higher penalised log-likelihood
    obj_em = _penalised_ll(pi_em, logL)
    obj_lbfgs = _penalised_ll(pi_lbfgs, logL)
    assert obj_lbfgs > obj_em, (
        f"L-BFGS objective ({obj_lbfgs:.2f}) should exceed "
        f"EM objective ({obj_em:.2f})"
    )
