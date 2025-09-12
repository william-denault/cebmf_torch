import numpy as np
import torch

from cebmf_torch.ebnm.ash import ash
from cebmf_torch.utils.maths import (
    my_e2truncnorm,
    my_etruncnorm,
)


def test_trucnorm():
    assert np.isclose(my_etruncnorm(0, 2, 3, 1).cpu().numpy(), 1.48995049, atol=1e-7)
    assert np.isclose(my_e2truncnorm(0, 2, 3, 1), 2.39340536, atol=1e-7)


def test_ash_exp():
    betahat = torch.tensor([1, 2, 3, 4, 5], dtype=float)
    sebetahat = torch.tensor([1, 0.4, 5, 1, 1], dtype=float)
    mult = torch.sqrt(torch.tensor(2.0)).item()
    res = ash(betahat, sebetahat, prior="exp", mult=mult)
    expected_posteriror = np.array([0.2136, 1.9500, 0.6813, 3.6777, 4.6960])
    expected_posteriror2 = np.array([0.3402, 3.9630, 3.1711, 14.5822, 23.0564])
    np.testing.assert_allclose(
        res.post_mean.cpu().numpy(), expected_posteriror, atol=1e-3
    )
    np.testing.assert_allclose(
        res.post_mean2.cpu().numpy(), expected_posteriror2, atol=1e-3
    )
