# tests/test_ebnm_gb.py
import math
import torch
import pytest

# import your implementation
from cebmf_torch.ebnm.generalized_binary import ebnm_gb # adjust path if needed

torch.set_num_threads(1)

def simulate_gb(n=2000, pi=0.3, mu=2.0, omega=0.2, s_val=0.5, seed=1, device="cpu"):
    """
    Simulate (theta, x, s) from the generalized-binary prior:
      theta ~ (1-pi)*delta0 + pi * TruncNormal(μ, (ωμ)^2) on [0, +inf)
      x = theta + Normal(0, s^2)
    """
    g = torch.Generator(device=device).manual_seed(seed)

    # Bernoulli indicators
    probs = torch.full((n,), float(pi), dtype=torch.float32, device=device)
    z = torch.bernoulli(probs, generator=g)

    # Slab: N(μ, (ω μ)^2) then truncate at 0
    sigma = float(omega) * float(mu)
    slab = torch.normal(mean=float(mu), std=float(sigma), size=(n,), generator=g, device=device)
    slab = torch.clamp(slab, min=0.0)

    theta = z * slab
    s = torch.full((n,), float(s_val), dtype=torch.float32, device=device)

    # Observations
    x = theta + torch.normal(mean=0.0, std=1.0, size=(n,), generator=g, device=device) * s
    return theta, x, s


@pytest.mark.parametrize("device", ["cpu"])
def test_basic_parameter_recovery(device):
    # larger n keeps this stable but still quick
    true_pi, true_mu, true_omega, s_val = 0.3, 2.0, 0.2, 0.5
    _, x, s = simulate_gb(n=3000, pi=true_pi, mu=true_mu, omega=true_omega, s_val=s_val, seed=123)
    x, s = x.to(device), s.to(device)

    res = ebnm_gb(x, s, omega=true_omega)

    # finite outputs
    assert math.isfinite(res.log_lik)
    assert 0.0 < res.pi0 < 1.0
    assert res.mode >= 0.0
    assert torch.isfinite(res.post_mean).all()
    assert torch.isfinite(res.post_sd).all()
    assert torch.isfinite(res.post_mean2).all()

    # parameter recovery (tolerances generous to allow stochastic variation)
    assert abs(res.pi0 - true_pi) < 0.06
    assert abs(res.mode - true_mu) < 0.35

    # moment consistency: Var >= 0, E[X^2] >= E[X]^2
    pm, pm2 = res.post_mean, res.post_mean2
    assert torch.all(pm2 >= pm**2 - 1e-8)


def test_nonnegativity_and_spike_behavior():
    true_pi, true_mu, true_omega, s_val = 0.25, 1.8, 0.2, 0.6
    _, x, s = simulate_gb(n=1500, pi=true_pi, mu=true_mu, omega=true_omega, s_val=s_val, seed=7)
    res = ebnm_gb(x, s, omega=true_omega)

    # support: posterior mean must be nonnegative (mixture of 0 and N_+(·))
    assert torch.all(res.post_mean >= -1e-8)

    # for clearly negative observations, mean should be near zero
    neg_mask = x < (-2.0 * s)  # "clearly negative relative to noise"
    if neg_mask.any():
        assert res.post_mean[neg_mask].mean().item() < 0.05


def test_variance_is_consistent():
    # random x to stress numeric stability
    g = torch.Generator().manual_seed(2024)
    x = torch.randn(2000, generator=g)
    s = torch.full_like(x, 0.7)
    res = ebnm_gb(x, s, omega=0.25)

    # sd^2 = E[X^2] - E[X]^2, and must be >= 0
    v = res.post_mean2 - res.post_mean ** 2
    assert torch.all(v >= -1e-8)
    assert torch.all(res.post_sd >= 0.0)
    # post_sd should agree with sqrt(var) numerically
    assert torch.allclose(res.post_sd, torch.clamp_min(v, 0).sqrt(), atol=1e-5, rtol=1e-4)


def test_shrinkage_and_omega_effect():
    # Same data, different ω. Larger ω => wider slab => less shrinkage (bigger posterior means).
    true_pi, true_mu, s_val = 0.35, 2.2, 0.5
    theta, x, s = simulate_gb(n=2500, pi=true_pi, mu=true_mu, omega=0.2, s_val=s_val, seed=99)

    res_narrow = ebnm_gb(x, s, omega=0.15)
    res_wide   = ebnm_gb(x, s, omega=0.50)

    # Look only at positive observations (where shrinkage is most interpretable)
    pos = x > 0
    pm_narrow = res_narrow.post_mean[pos]
    pm_wide   = res_wide.post_mean[pos]
    x_pos     = x[pos]

    # Both should be positively associated with x
    corr_narrow = torch.corrcoef(torch.stack([x_pos, pm_narrow]))[0,1].item()
    corr_wide   = torch.corrcoef(torch.stack([x_pos, pm_wide]))[0,1].item()
    assert corr_narrow > 0.7 and corr_wide > 0.7

    # Wider slab shrinks less on average
    assert pm_wide.mean().item() >= pm_narrow.mean().item() - 1e-6

    # Posterior means should not explode: they’re bounded above by (roughly) x for large signals.
    # Allow a small numerical slack.
    overshoot = (pm_wide - x_pos).max().item()
    assert overshoot < 0.25


def test_extreme_cases_pi_near_zero_and_one():
    # Almost all spikes
    _, x0, s0 = simulate_gb(n=2000, pi=0.02, mu=2.0, omega=0.2, s_val=0.5, seed=555)
    r0 = ebnm_gb(x0, s0, omega=0.2)
    assert r0.pi0 < 0.15
    assert r0.post_mean.mean().item() < 0.15

    # Almost all slabs
    _, x1, s1 = simulate_gb(n=2000, pi=0.98, mu=2.0, omega=0.2, s_val=0.5, seed=556)
    r1 = ebnm_gb(x1, s1, omega=0.2)
    assert r1.pi0 > 0.8
    assert r1.post_mean.mean().item() > 0.5
