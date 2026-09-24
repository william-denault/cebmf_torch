import torch

from cebmf_torch.utils.posterior import posterior_point_mass_normal


def test_point_mass_normal_posterior_stable_in_low_density_tail():
    """Posterior weights should preserve likelihood ratios in low-density tails."""
    x = torch.tensor([10.0], dtype=torch.float32)
    se = torch.tensor([1.0], dtype=torch.float32)
    pi = torch.tensor([0.5], dtype=torch.float32)
    mu0 = 0.0
    mu1 = 20.0
    sigma0 = 0.1

    post_mean, post_var = posterior_point_mass_normal(x, se, pi, mu0, mu1, sigma0)

    # Compute the same posterior independently in log space.
    slab_sd = torch.sqrt(se.square() + sigma0**2)
    normal = torch.distributions.Normal
    log_spike = torch.log(pi) + normal(torch.tensor(mu0), se).log_prob(x)
    log_slab = torch.log1p(-pi) + normal(torch.tensor(mu1), slab_sd).log_prob(x)
    log_norm = torch.logaddexp(log_spike, log_slab)
    w0 = torch.exp(log_spike - log_norm)
    w1 = torch.exp(log_slab - log_norm)

    slab_var = 1.0 / (1.0 / sigma0**2 + 1.0 / se.square())
    slab_mean = slab_var * (mu1 / sigma0**2 + x / se.square())
    expected_mean = w0 * mu0 + w1 * slab_mean
    expected_var = w0 * (mu0 - expected_mean).square() + w1 * (slab_var + (slab_mean - expected_mean).square())

    torch.testing.assert_close(post_mean, expected_mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(post_var, expected_var, rtol=1e-5, atol=1e-6)
