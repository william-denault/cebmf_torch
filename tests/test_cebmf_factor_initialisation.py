import pytest
import torch

from cebmf_torch import cEBMF
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


def test_cEBMF_initialise_factors_with_default_strategy(Y: torch.Tensor, K: int) -> None:
    cebmf = cEBMF(Y, K, device=torch.device("cpu"))
    cebmf.initialise_factors()
    assert cebmf.L is not None, "L should be initialized"
    assert cebmf.F is not None, "F should be initialized"
    expected_L, expected_F = svd_initialise(Y, cebmf.N, cebmf.P, K, cebmf.device)
    assert torch.allclose(cebmf.L, expected_L), "L not initialized as expected with svd"
    assert torch.allclose(cebmf.F, expected_F), "F not initialized as expected with svd"


def test_cEBMF_initialise_with_user_factors(Y: torch.Tensor, N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    L_user = torch.randn(N, K, device=device)
    F_user = torch.randn(P, K, device=device)

    cebmf = cEBMF(Y, K, device=device)
    cebmf.initialise_factors(L=L_user, F=F_user)

    assert cebmf.L is not None, "L should be initialized"
    assert cebmf.F is not None, "F should be initialized"
    assert torch.allclose(cebmf.L, L_user), "L matrix not correctly set from user input"
    assert torch.allclose(cebmf.F, F_user), "F matrix not correctly set from user input"


def test_cEBMF_initialise_with_only_L(Y: torch.Tensor, N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    L_user = torch.randn(N, K, device=device)

    cebmf = cEBMF(Y, K, device=device)
    with pytest.warns(UserWarning, match="Provided L without F"):
        cebmf.initialise_factors(L=L_user)

    # test fallback to svd
    expected_L, expected_F = svd_initialise(Y, N, P, K, device)
    assert torch.allclose(cebmf.L, expected_L), "L not initialized as expected with svd"
    assert torch.allclose(cebmf.F, expected_F), "F not initialized as expected with svd"


def test_cEBMF_initialise_with_only_F(Y: torch.Tensor, N: int, P: int, K: int) -> None:
    device = torch.device("cpu")
    F_user = torch.randn(P, K, device=device)

    cebmf = cEBMF(Y, K, device=device)
    with pytest.warns(UserWarning, match="Provided F without L"):
        cebmf.initialise_factors(F=F_user)

    # test fallback to svd
    expected_L, expected_F = svd_initialise(Y, N, P, K, device)
    assert torch.allclose(cebmf.L, expected_L), "L not initialized as expected with svd"
    assert torch.allclose(cebmf.F, expected_F), "F not initialized as expected with svd"


def test_cEBMF_initialise_with_unknown_strategy(Y: torch.Tensor, K: int) -> None:
    cebmf = cEBMF(Y, K, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="Unknown initialization method 'unknown_strategy'"):
        cebmf.initialise_factors(method="unknown_strategy")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["svd", "random", "zero", "provided"])
def test_initialisation_and_first_update_match_data_dtype(dtype, method):
    # Use the *opposite* global dtype: matching defaults hid this regression.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32 if dtype == torch.float64 else torch.float64)
    try:
        y = torch.arange(1, 25, dtype=dtype).reshape(6, 4) / 24
        model = cEBMF(y, K=5, prior_F="hmm", prior_F_kwargs={"maxiter": 1}, allow_backfitting=False, device="cpu")
        if method == "provided":
            model.initialise_factors(L=torch.ones(6, 5), F=torch.ones(4, 5))
        else:
            model.initialise_factors(method=method)
        assert model.L.dtype == model.F.dtype == model.L2.dtype == model.F2.dtype == dtype
        model.fit(1)
        assert torch.isfinite(model.L).all() and torch.isfinite(model.F).all()
        assert torch.isfinite(torch.tensor(model.obj)).all()
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_input_retains_supported_float32_promotion(dtype):
    y = torch.arange(1, 25, dtype=dtype).reshape(6, 4)
    model = cEBMF(y, K=1, device="cpu", allow_backfitting=False)
    model.initialise_factors()
    assert model.Y.dtype == model.L.dtype == model.F.dtype == torch.float32
    model.fit(1)
    assert torch.isfinite(torch.tensor(model.obj)).all()
