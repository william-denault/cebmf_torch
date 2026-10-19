"""Independent Laplace likelihood, posterior moment, and spike-boundary checks."""

import math

import numpy as np
import pytest
import torch

from cebmf_torch.ebnm.point_laplace import ebnm_point_laplace, logg_laplace_convolved_with_normal


def quadrature(x, s, weight, rate, location):
    """Integrate the spike and continuous Laplace prior without CDF formulas."""
    nodes, weights = np.polynomial.laguerre.laggauss(96)
    theta = np.concatenate((location + nodes / rate, location - nodes / rate, [location]))
    prior_weights = np.concatenate((weight * weights / 2, weight * weights / 2, [1 - weight]))
    density = np.exp(-0.5 * ((x[:, None] - theta) / s[:, None]) ** 2) / (s[:, None] * np.sqrt(2 * np.pi))
    weighted = density * prior_weights
    marginal = weighted.sum(axis=1)
    return np.log(marginal).sum(), weighted @ theta / marginal, weighted @ theta**2 / marginal


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weight", [0.0001, 0.6, 1.0])
def test_fixed_prior_matches_independent_quadrature(dtype, weight):
    x = torch.tensor([-1.0, 0.2, 1.5, 0.0, 0.0], dtype=dtype)
    s = torch.tensor([0.7, 2.0, 1.0, 500000.0, 1e12], dtype=dtype)
    rate, location = 2.0, 0.3
    logit = math.log(weight / (1 - weight)) if weight < 1 else math.inf
    result = ebnm_point_laplace(x, s, par_init=(logit, math.log(rate), location), fix_par=(True, True, True))
    log_lik, mean, mean2 = quadrature(x.double().numpy(), s.double().numpy(), weight, rate, location)
    tolerance = 3e-6 if dtype == torch.float32 else 2e-10
    assert result.log_lik.item() == pytest.approx(log_lik, rel=tolerance)
    torch.testing.assert_close(result.post_mean, torch.tensor(mean, dtype=dtype), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(result.post_mean2, torch.tensor(mean2, dtype=dtype), rtol=tolerance, atol=tolerance)
    assert result.pi_slab.item() == pytest.approx(weight, rel=tolerance)
    assert (result.post_mean2 >= result.post_mean.square()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_noise_laplace_density_matches_erfcx_identity(dtype):
    s = torch.tensor([500000.0, 1e12], dtype=dtype)
    rate = torch.tensor(2.0, dtype=dtype)
    actual = logg_laplace_convolved_with_normal(torch.zeros_like(s), s, rate)
    # At x=0 the convolution is (rate/2) * erfcx(s*rate/sqrt(2)).
    expected = (rate / 2).log() + torch.special.erfcx(s * rate / math.sqrt(2)).log()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spike_boundary_has_consistent_weight_likelihood_and_moments(dtype):
    x = torch.zeros(4, dtype=dtype)
    s = torch.ones_like(x)
    # At zero the all-spike model has strictly higher density than the
    # initial 50/50 mixture. Keep max_iter=0 to isolate the boundary decision.
    result = ebnm_point_laplace(x, s, par_init=(0.0, 0.0, 0.0), max_iter=0)
    assert result.pi_slab == 0
    assert torch.equal(result.post_mean, x)
    assert torch.equal(result.post_mean2, x)
    assert torch.equal(result.post_sd, x)
    torch.testing.assert_close(result.log_lik, torch.distributions.Normal(0, s).log_prob(x).sum())


def test_fixed_spike_preserves_exact_point_mass():
    x = torch.tensor([-0.5, 0.3, 1.0], dtype=torch.float64)
    result = ebnm_point_laplace(x, torch.ones_like(x), par_init=(-math.inf, 0.0, 0.3), fix_par=(True, True, True))
    assert result.pi_slab == 0
    torch.testing.assert_close(result.post_mean, torch.full_like(x, 0.3))
    torch.testing.assert_close(result.post_mean2, torch.full_like(x, 0.09))
    assert torch.equal(result.post_sd, torch.zeros_like(x))
