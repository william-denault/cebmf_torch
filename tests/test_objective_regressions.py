"""Regressions for the reported noisy rank-one fits with five retained factors.

These deliberately use float64, disable pruning, and run past the first few
improving iterations. The old low-noise RMSE test did not exercise these cases.
Plateaus are valid; a positive step above 1e-9 or any non-finite value is not.
The default seed 0, unpenalized seed 1, Laplace seed 7, and default penalty
accounting checks were verified to fail on the pre-fix revision 3d87c39.
"""

import math

import pytest
import torch

from cebmf_torch import cEBMF

TOL = 1e-9


@pytest.fixture(scope="module", autouse=True)
def small_matrix_threads():
    """Avoid thread-pool overhead without changing the rest of the test suite."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def noisy_rank_one(seed, noise_std):
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(50, generator=generator, dtype=torch.float64)
    v = torch.rand(40, generator=generator, dtype=torch.float64)
    return torch.outer(u, v) + noise_std * torch.randn(50, 40, generator=generator, dtype=torch.float64)


def assert_valid_posterior(model):
    for side in ("L", "F"):
        mean = getattr(model, side)
        second = getattr(model, side + "2")
        kl = getattr(model, "kl_" + side.lower())
        assert torch.isfinite(mean).all() and torch.isfinite(second).all()
        assert (second - mean.square() >= -TOL).all(), f"Negative posterior variance on {side}"
        assert torch.isfinite(kl).all() and (kl >= -TOL).all(), f"Invalid KL on {side}: {kl}"


@pytest.mark.parametrize(
    "scenario,seed",
    [
        # These use omitted prior options, exactly as in the reported example.
        ("default", 0),
        ("default", 1),
        ("default", 13),
        # Disabling the penalty alone did not fix discarded ASH warm starts.
        ("unpenalized", 1),
        ("unpenalized", 2),
        ("unpenalized", 4),
        # Previously observed jumps included about +7 and +15 in these seeds.
        ("laplace", 0),
        ("laplace", 7),
        ("laplace", 14),
    ],
    ids=lambda value: str(value),
)
def test_reported_fits_have_valid_posteriors_and_nonincreasing_objectives(scenario, seed, monkeypatch):
    options = {}
    noise_std, iterations = 1.0, 50
    if scenario == "unpenalized":
        options = {"prior_L_kwargs": {"penalty": 1}, "prior_F_kwargs": {"penalty": 1}}
    elif scenario == "laplace":
        options = {"prior_L": "laplace", "prior_F": "laplace"}
        noise_std, iterations = 0.51, 20
    model = cEBMF(noisy_rank_one(seed, noise_std), K=5, allow_backfitting=False, device="cpu", **options)
    model.initialise_factors()

    # Inspect every sweep, while exercising the same public fit(maxit=...)
    # call as the notebook. A bad intermediate KL must not escape detection.
    calculate_objective = model._cal_obj

    def checked_objective():
        assert_valid_posterior(model)
        calculate_objective()

    monkeypatch.setattr(model, "_cal_obj", checked_objective)
    result = model.fit(maxit=iterations)
    history = torch.tensor(result.history_obj, dtype=torch.float64)
    assert model.model.K == 5
    assert history.numel() == iterations  # no dropped or reset history
    assert result.history_obj == model.obj
    assert torch.isfinite(history).all(), f"Non-finite objective: {history}"
    increments = history.diff()
    worst = int(increments.argmax())
    assert increments[worst] <= TOL, (
        f"{scenario}, seed={seed}, iteration {worst + 1}->{worst + 2}: "
        f"objective increased by {increments[worst].item():.12g}"
    )


@pytest.mark.parametrize("penalties", [None, (1.0, 1.0), (3.0, 7.0)], ids=["defaults", "none", "different-sides"])
def test_reported_objective_matches_independent_penalized_elbo(penalties):
    """Catch missing/double penalties, wrong signs, and clipped loss histories."""
    options = (
        {}
        if penalties is None
        else {
            "prior_L_kwargs": {"penalty": penalties[0]},
            "prior_F_kwargs": {"penalty": penalties[1]},
        }
    )
    alpha_l, alpha_f = (10.0, 10.0) if penalties is None else penalties
    y = noisy_rank_one(0, 1.0)
    model = cEBMF(y, K=5, allow_backfitting=False, device="cpu", **options)
    model.initialise_factors()
    for iteration in range(3):
        result = model.fit(maxit=1)
        # Reconstruct E[||Y-LF'||^2] without the model's likelihood/residual
        # helpers. The existing Gaussian-evidence tests independently check KL.
        expected_sse = (y - model.L @ model.F.T).square().sum()
        for k in range(5):
            expected_sse += (
                torch.outer(model.L2[:, k], model.F2[:, k])
                - torch.outer(model.L[:, k].square(), model.F[:, k].square())
            ).sum()
        negative_loglik = 0.5 * (y.numel() * (math.log(2 * math.pi) - model.tau.log()) + model.tau * expected_sse)
        negative_elbo = (negative_loglik + model.kl_l.sum() + model.kl_f.sum()).item()
        penalty = -(alpha_l - 1) * sum(weight.log().item() for weight in model.pi0_L)
        penalty -= (alpha_f - 1) * sum(weight.log().item() for weight in model.pi0_F)
        expected = negative_elbo + penalty
        assert model.obj[-1] == pytest.approx(expected, rel=0, abs=TOL)
        assert model.negative_elbo[-1] == pytest.approx(negative_elbo, rel=0, abs=TOL)
        assert len(result.history_obj) == len(model.negative_elbo) == iteration + 1
