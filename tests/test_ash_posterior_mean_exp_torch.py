import numpy as np
import torch

from cebmf_torch.ebnm.ash import ash
from cebmf_torch.utils.maths import my_etruncnorm


def truncated_moment():
    np.testing.assert_allclose(my_etruncnorm(0, 2, 3, 1).cpu().numpy(), 1.48995049)
    np.testing.assert_allclose(my_etruncnorm(0, 2, 3, 1).cpu().numpy(), 2.39340536)


def test_ash_loglik_and_scale():
    betahat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sebetahat = torch.tensor([1.0, 0.4, 5.0, 1.0, 1.0])
    mult = torch.sqrt(torch.tensor(2.0)).item()
    res = ash(betahat, sebetahat, prior="exp", mult=mult, batch_size=5)
    expected_log_lik = -15.244064765169643
    expected_scale = np.array(
        [
            0.0,
            0.02929687,
            0.04143204,
            0.05859375,
            0.08286408,
            0.1171875,
            0.16572815,
            0.234375,
            0.3314563,
            0.46875,
            0.66291261,
            0.9375,
            1.32582521,
            1.875,
            2.65165043,
            3.75,
            5.30330086,
            7.5,
        ]
    )
    np.testing.assert_allclose(res.scale.cpu().numpy(), expected_scale, rtol=1e-5)
    np.testing.assert_allclose(
        res.post_mean.cpu().numpy(),
        np.array([0.21361966, 1.95003336, 0.68125659, 3.67773689, 4.69599116]),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        res.post_mean2.cpu().numpy(),
        np.array([0.34021317, 3.96295414, 3.17114472, 14.58222929, 23.05642398]),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(res.log_lik, expected_log_lik, atol=1e-3)
