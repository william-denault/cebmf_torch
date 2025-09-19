import pytest
import torch

from cebmf_torch.cebmf._initialisation import random_initialise, svd_initialise, user_provided_factors, zero_initialise


@pytest.fixture
def N() -> int:
    return 5


@pytest.fixture
def P() -> int:
    return 4


@pytest.fixture
def K() -> int:
    return 3


@pytest.fixture
def Y(N: int, P: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(N, P)


def test_user_provided_factors(N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    L_user = torch.randn(N, K, device=device)
    F_user = torch.randn(P, K, device=device)

    L_out, F_out = user_provided_factors(L_user, F_user, N, P, K, device)

    assert torch.allclose(L_out, L_user), "L matrix not correctly returned"
    assert torch.allclose(F_out, F_user), "F matrix not correctly returned"


def test_user_provided_factors_shape_mismatch(N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    L_user = torch.randn(N + 1, K, device=device)  # Incorrect shape
    F_user = torch.randn(P, K, device=device)

    with pytest.raises(ValueError, match=r"Provided L has shape torch\.Size\(\[6, 3\]\), expected \(5, 3\)"):
        user_provided_factors(L_user, F_user, N, P, K, device)

    L_user = torch.randn(N, K, device=device)
    F_user = torch.randn(P + 1, K, device=device)  # Incorrect shape

    with pytest.raises(ValueError, match=r"Provided F has shape torch\.Size\(\[5, 3\]\), expected \(4, 3\)"):
        user_provided_factors(L_user, F_user, N, P, K, device)


def test_zero_initialise(Y: torch.Tensor, N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    L, F = zero_initialise(Y, N, P, K, device)
    assert L.shape == (N, K), "L shape incorrect"
    assert F.shape == (P, K), "F shape incorrect"
    assert torch.all(L == 0), "L not all zeros"
    assert torch.all(F == 0), "F not all zeros"


def test_random_initialise(Y: torch.Tensor, N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    L, F = random_initialise(Y, N, P, K, device)
    assert L.shape == (N, K), "L shape incorrect"
    assert F.shape == (P, K), "F shape incorrect"
    assert not torch.all(L == 0), "L should not be all zeros"
    assert not torch.all(F == 0), "F should not be all zeros"


def test_svd_initialise(Y: torch.Tensor, N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    L, F = svd_initialise(Y, N, P, K, device)
    assert L.shape == (N, K), "L shape incorrect"
    assert F.shape == (P, K), "F shape incorrect"
