"""
Tests for the user-supplied (known) standard-error path of cEBMF.

Covers:
- scalar S (z-score case);
- (N, P) matrix S;
- broadcasting of (P,) row-vector S;
- NaN / non-positive entries in S being folded into the missing-data mask;
- update_tau being a no-op once S is supplied;
- recovery of a known low-rank structure with provided variance;
- precedence of S over an explicitly requested noise_type;
- shape-mismatch validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebmf import NoiseType


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _make_lowrank(n: int = 60, p: int = 50, rank: int = 2, sigma: float = 0.1, seed: int = 0):
    """Generate an (n, p) matrix of rank ``rank`` plus i.i.d. Gaussian noise."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, rank))
    F = rng.normal(size=(p, rank))
    truth = L @ F.T
    Y = truth + rng.normal(scale=sigma, size=(n, p))
    return torch.tensor(Y, dtype=torch.float32), torch.tensor(truth, dtype=torch.float32), sigma


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


# ----------------------------------------------------------------------
# constructor / dispatch
# ----------------------------------------------------------------------

def test_scalar_S_sets_known_noise_type_and_unit_precision():
    """Passing S=1.0 (z-score case) -> tau_map is 1 everywhere."""
    Y = torch.randn(20, 15)

    model = cEBMF(data=Y, K=3, S=1.0)

    assert model.noise.type == NoiseType.KNOWN
    assert model.S is not None and model.S.shape == (20, 15)
    assert torch.allclose(model.S, torch.ones_like(model.S))
    assert torch.allclose(model.tau_map, torch.ones_like(model.tau_map))


def test_scalar_S_arbitrary_value():
    Y = torch.randn(8, 6)
    sigma = 0.25
    model = cEBMF(data=Y, K=2, S=sigma)
    expected_tau = 1.0 / (sigma * sigma)
    assert torch.allclose(model.tau_map, torch.full_like(model.tau_map, expected_tau))


def test_matrix_S_full_shape():
    """Passing a full (N, P) S tensor sets tau_map = 1/S^2 elementwise."""
    torch.manual_seed(0)
    Y = torch.randn(12, 9)
    S = torch.rand(12, 9) * 0.5 + 0.1  # all strictly positive

    model = cEBMF(data=Y, K=2, S=S)

    assert model.noise.type == NoiseType.KNOWN
    assert model.S.shape == (12, 9)
    assert torch.allclose(model.tau_map, 1.0 / (S * S), atol=1e-6)


def test_S_broadcasts_over_columns():
    """A (P,) S vector should broadcast to (N, P) (e.g., per-tissue SE)."""
    torch.manual_seed(0)
    n, p = 10, 7
    Y = torch.randn(n, p)
    S_col = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    model = cEBMF(data=Y, K=2, S=S_col)

    assert model.S.shape == (n, p)
    # every row should equal the broadcast vector
    for i in range(n):
        assert torch.allclose(model.S[i], S_col)
    expected_tau = (1.0 / (S_col * S_col)).expand(n, p)
    assert torch.allclose(model.tau_map, expected_tau)


def test_S_broadcasts_over_rows():
    """A (N, 1) S column vector should broadcast to (N, P) (e.g., per-gene SE)."""
    torch.manual_seed(0)
    n, p = 8, 5
    Y = torch.randn(n, p)
    S_row = torch.linspace(0.1, 1.0, n).reshape(n, 1)

    model = cEBMF(data=Y, K=2, S=S_row)

    assert model.S.shape == (n, p)
    expected_tau = (1.0 / (S_row * S_row)).expand(n, p)
    assert torch.allclose(model.tau_map, expected_tau)


def test_S_shape_mismatch_raises():
    Y = torch.randn(6, 4)
    bad_S = torch.rand(7, 4)  # not broadcastable

    with pytest.raises(ValueError, match="not broadcastable"):
        cEBMF(data=Y, K=2, S=bad_S)


def test_scalar_S_must_be_positive():
    Y = torch.randn(5, 4)
    with pytest.raises(ValueError, match="positive"):
        cEBMF(data=Y, K=2, S=0.0)
    with pytest.raises(ValueError, match="positive"):
        cEBMF(data=Y, K=2, S=-1.0)
    with pytest.raises(ValueError, match="positive"):
        cEBMF(data=Y, K=2, S=float("nan"))


def test_S_overrides_noise_type_with_warning():
    Y = torch.randn(8, 6)
    with pytest.warns(UserWarning, match="treated as known"):
        model = cEBMF(data=Y, K=2, S=1.0, noise_type=NoiseType.ROW_WISE)
    assert model.noise.type == NoiseType.KNOWN


def test_no_S_means_no_S_attribute_set():
    """With learned noise, model.S should be None."""
    Y = torch.randn(8, 6)
    model = cEBMF(data=Y, K=2)
    assert model.S is None


# ----------------------------------------------------------------------
# missingness merging
# ----------------------------------------------------------------------

def test_nan_in_S_is_folded_into_mask():
    torch.manual_seed(0)
    n, p = 6, 5
    Y = torch.randn(n, p)
    S = torch.full((n, p), 0.5)
    S[1, 2] = float("nan")
    S[3, 4] = 0.0  # non-positive
    S[5, 0] = -0.1  # negative

    with pytest.warns(UserWarning, match="non-finite or non-positive"):
        model = cEBMF(data=Y, K=2, S=S)

    # those 3 entries should be masked
    assert model.mask[1, 2].item() == 0.0
    assert model.mask[3, 4].item() == 0.0
    assert model.mask[5, 0].item() == 0.0
    # the rest should remain observed
    n_observed = int(model.mask.sum().item())
    assert n_observed == n * p - 3
    # internally the bad S values should have been replaced; tau_map should be finite everywhere
    assert torch.isfinite(model.tau_map).all()


def test_nan_in_Y_and_S_combine():
    """NaN in Y and bad S at *different* positions should both be masked."""
    torch.manual_seed(0)
    n, p = 5, 4
    Y = torch.randn(n, p)
    Y[0, 0] = float("nan")
    S = torch.full((n, p), 0.3)
    S[2, 3] = float("nan")

    with pytest.warns(UserWarning):
        model = cEBMF(data=Y, K=2, S=S)

    assert model.mask[0, 0].item() == 0.0  # missing in Y
    assert model.mask[2, 3].item() == 0.0  # bad S
    assert int(model.mask.sum().item()) == n * p - 2


def test_bad_S_at_already_missing_Y_does_not_warn():
    """If S is bad only where Y is also NaN, no warning should be raised."""
    n, p = 4, 4
    Y = torch.randn(n, p)
    Y[0, 0] = float("nan")
    S = torch.full((n, p), 0.5)
    S[0, 0] = float("nan")  # bad S, but Y is also missing here

    # Should not warn (the pytest.warns context would fail if we did)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into an error
        model = cEBMF(data=Y, K=2, S=S)

    assert model.mask[0, 0].item() == 0.0


# ----------------------------------------------------------------------
# behavioural: variance is not learned
# ----------------------------------------------------------------------

def test_tau_map_is_invariant_under_iter_once():
    """With known S, tau_map must NOT change as we iterate."""
    torch.manual_seed(0)
    Y = torch.randn(40, 30)
    S = torch.rand(40, 30) * 0.4 + 0.1
    model = cEBMF(data=Y, K=3, S=S)
    model.initialise_factors("svd")
    tau_before = model.tau_map.clone()
    for _ in range(5):
        model.iter_once()
    assert torch.equal(model.tau_map, tau_before), "tau_map should be frozen when S is supplied"


def test_update_tau_is_noop_for_known():
    """Calling update_tau() directly must leave tau_map untouched."""
    torch.manual_seed(0)
    Y = torch.randn(20, 15)
    model = cEBMF(data=Y, K=2, S=2.0)
    model.initialise_factors("svd")
    tau_before = model.tau_map.clone()
    model.update_tau()
    assert torch.equal(model.tau_map, tau_before)


# ----------------------------------------------------------------------
# behavioural: recovery
# ----------------------------------------------------------------------

@pytest.mark.parametrize("S_kind", ["scalar", "matrix"])
def test_recovers_lowrank_with_known_variance(S_kind: str):
    """cEBMF with provided S should recover the underlying low-rank truth."""
    Y, truth, sigma = _make_lowrank(n=80, p=60, rank=2, sigma=0.1, seed=42)

    if S_kind == "scalar":
        S: torch.Tensor | float = float(sigma)
    else:
        S = torch.full_like(Y, sigma)

    model = cEBMF(data=Y, K=4, prior_L="norm", prior_F="norm", S=S)
    model.initialise_factors("svd")
    for _ in range(60):
        model.iter_once()

    fitted = model.L @ model.F.T
    err = _rmse(fitted, truth)
    # noise floor on a single observation is sigma; the fit averages many obs,
    # so the rank-2 estimate should comfortably beat sigma.
    assert err < sigma, f"RMSE {err:.4f} should be below noise level {sigma:.4f}"


def test_zscore_case_runs_end_to_end():
    """Smoke-test: passing S=1.0 (z-score convention) should produce a usable fit."""
    torch.manual_seed(0)
    Y = torch.randn(30, 20)
    model = cEBMF(data=Y, K=3, S=1.0)
    model.initialise_factors("svd")
    result = model.fit(maxit=5)
    assert math.isfinite(result.history_obj[-1])
    # tau_map is the per-entry precision; should still be ones
    assert torch.allclose(model.tau_map, torch.ones_like(model.tau_map))
