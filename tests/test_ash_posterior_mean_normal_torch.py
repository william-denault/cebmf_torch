import numpy as np
import torch

# Updated to use clean imports from new API
from cebmf_torch import ash
from cebmf_torch.utils import autoselect_scales_mix_norm, optimize_pi_logL, posterior_mean_norm
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch


def test_ash_loglik_and_scale():
    betahat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sebetahat = torch.tensor([1.0, 0.4, 5.0, 1.0, 1.0])
    res = ash(betahat, sebetahat, mult=torch.sqrt(torch.tensor(2.0)).item(), prior="norm")
    expected_log_lik = -16.91767637608251
    expected_scale = np.array(
        [
            0.0,
            0.03827328,
            0.05412659,
            0.07654655,
            0.10825318,
            0.15309311,
            0.21650635,
            0.30618622,
            0.4330127,
            0.61237244,
            0.8660254,
            1.22474487,
            1.73205081,
            2.44948974,
            3.46410162,
            4.89897949,
            6.92820323,
            9.79795897,
        ]
    )
    np.testing.assert_allclose(res.scale.cpu().numpy(), expected_scale, rtol=1e-5)
    np.testing.assert_allclose(
        res.post_mean.cpu().numpy(),
        np.array([0.11126873, 1.97346787, 0.20802628, 3.6663574, 4.61542534]),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        res.post_mean2.cpu().numpy(),
        np.array([0.21398654, 4.05293203, 1.93632865, 14.45535405, 22.22774277]),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(res.log_lik, expected_log_lik, atol=1e-3)


def test_optimize_pi_and_posterior_mean_norm_shape():
    betahat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sebetahat = torch.tensor([1.0, 0.4, 5.0, 1.0, 1.0])
    scale = autoselect_scales_mix_norm(betahat, sebetahat, mult=2.0)
    L = get_data_loglik_normal_torch(betahat, sebetahat, torch.zeros_like(scale), scale)
    pi = optimize_pi_logL(L, penalty=10)
    out = posterior_mean_norm(
        betahat,
        sebetahat,
        torch.log(pi + 1e-32),
        L,
        scale,
        location=torch.zeros_like(scale),
    )
    result = torch.exp(L) * torch.exp(pi)

    excepted_postmean = np.array([0.0902, 1.9818, 0.2705, 3.7831, 4.7769])

    excepted_postmean2 = np.array([0.1756, 4.0868, 2.6417, 15.3774, 23.7875])
    np.testing.assert_allclose(out.post_mean.cpu().numpy(), excepted_postmean, atol=1e-3)

    np.testing.assert_allclose(out.post_mean2.cpu().numpy(), excepted_postmean2, atol=1e-3)

    assert result.shape == (5, scale.numel())
