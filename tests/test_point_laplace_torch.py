import numpy as np
import torch

from cebmf_torch.torch_ebnm.torch_ebnm_point_laplace import ebnm_point_laplace


def test_ebnm_point_laplace_solver_loglik_and_postmean():
    x = torch.tensor([0.0, 1.0, -0.5])
    s = torch.tensor([1.0, 0.2, 1.0])
    res = ebnm_point_laplace(x, s, pen_pi0=0)
    expected_log_lik = -4.161880337595547
    expected_post_mean = np.array([0.0, 0.9326135, -0.15496329])
    assert np.isclose(res.log_lik, expected_log_lik, atol=2e-2)
    np.testing.assert_allclose(
        res.post_mean.cpu().numpy(), expected_post_mean, atol=2e-2
    )
