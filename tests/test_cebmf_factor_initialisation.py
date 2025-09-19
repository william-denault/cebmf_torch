import torch
import pytest

from cebmf_torch.cebmf._initialisation import INIT_STRATEGIES, user_provided_factors


def test_user_provided_factors():
    N, P, K = 5, 4, 3
    device = torch.device("cpu")
    L_user = torch.randn(N, K, device=device)
    F_user = torch.randn(P, K, device=device)

    L_out, F_out = user_provided_factors(L_user, F_user, N, P, K, device)

    assert torch.allclose(L_out, L_user), "L matrix not correctly returned"
    assert torch.allclose(F_out, F_user), "F matrix not correctly returned"


def test_user_provided_factors_shape_mismatch():
    N, P, K = 5, 4, 3
    device = torch.device("cpu")
    L_user = torch.randn(N + 1, K, device=device)  # Incorrect shape
    F_user = torch.randn(P, K, device=device)

    with pytest.raises(ValueError, match=r"Provided L has shape torch\.Size\(\[6, 3\]\), expected \(5, 3\)"):
        user_provided_factors(L_user, F_user, N, P, K, device)

    L_user = torch.randn(N, K, device=device)
    F_user = torch.randn(P + 1, K, device=device)  # Incorrect shape

    with pytest.raises(ValueError, match=r"Provided F has shape torch\.Size\(\[5, 3\]\), expected \(4, 3\)"):
        user_provided_factors(L_user, F_user, N, P, K, device)
