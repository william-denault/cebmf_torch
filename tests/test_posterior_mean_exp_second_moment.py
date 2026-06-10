"""Regression tests for the spike+exponential posterior second moment.

The guard ``post_mean2 = maximum(post_mean2, post_mean)`` inflated the second
moment whenever ``post_mean < 1`` (the common case): since E[theta^2] >=
E[theta]^2 always holds, the correct numerical floor is ``post_mean**2``.
"""

import math

import numpy as np
import torch

from cebmf_torch.utils.posterior import posterior_mean_exp

CPU = torch.device("cpu")


def _brute_exp_posterior(x, s, pi, scale, n=400_001):
    """Brute-force E[theta], E[theta^2] under pi0*delta_0 + sum pi_k Exp(1/scale_k)."""
    theta = np.linspace(0.0, max(scale) * 40 + 20 * s, n)
    rates = 1.0 / np.asarray(scale[1:])
    slab_pdf = np.zeros_like(theta)
    for w, a in zip(pi[1:], rates):
        slab_pdf += w * a * np.exp(-a * theta)
    lik = np.exp(-0.5 * ((x - theta) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    spike = pi[0] * (np.exp(-0.5 * (x / s) ** 2) / (s * math.sqrt(2 * math.pi)))
    unnorm = slab_pdf * lik
    z = np.trapz(unnorm, theta) + spike
    m1 = np.trapz(theta * unnorm, theta) / z
    m2 = np.trapz(theta * theta * unnorm, theta) / z
    return m1, m2


def test_second_moment_not_inflated_matches_brute_force():
    pi = [0.2, 0.4, 0.3, 0.1]
    scale = [0.0, 0.5, 1.0, 2.0]
    xs = np.linspace(0.1, 0.9, 8)  # small effects -> post_mean < 1
    bh = torch.tensor(xs, dtype=torch.float64, device=CPU)
    se = torch.ones(8, dtype=torch.float64, device=CPU)
    out = posterior_mean_exp(
        bh, se, torch.log(torch.tensor(pi, dtype=torch.float64, device=CPU)),
        torch.tensor(scale, dtype=torch.float64, device=CPU),
    )
    for j, x in enumerate(xs):
        _, m2_ref = _brute_exp_posterior(float(x), 1.0, pi, scale)
        assert abs(float(out.post_mean2[j]) - m2_ref) < 5e-3, (
            f"x={x}: got {float(out.post_mean2[j])}, expected {m2_ref}"
        )


def test_second_moment_at_least_first_moment_squared():
    pi = [0.5, 0.3, 0.2]
    scale = [0.0, 0.5, 1.5]
    bh = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64, device=CPU)
    se = torch.ones(11, dtype=torch.float64, device=CPU)
    out = posterior_mean_exp(
        bh, se, torch.log(torch.tensor(pi, dtype=torch.float64, device=CPU)),
        torch.tensor(scale, dtype=torch.float64, device=CPU),
    )
    assert torch.all(out.post_mean2 >= out.post_mean.pow(2) - 1e-9)
