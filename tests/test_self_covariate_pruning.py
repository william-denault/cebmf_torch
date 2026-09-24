"""Self-covariate priors must remain usable when backfitting removes factors."""

import pytest
import torch

from cebmf_torch import cEBMF


@pytest.mark.parametrize(
    "prior", ["spiked_emdn", "emdn", "cash", "cgb", "cgb_sharp", "cgb_sharp_2", "lcash", "po_lcash"]
)
@pytest.mark.parametrize("side", ["L", "F"])
@pytest.mark.parametrize("external", [False, True])
def test_learned_self_covariate_prior_can_refit_after_pruning(prior, side, external):
    torch.manual_seed(12)
    data = torch.randn(16, 12)
    n = data.shape[0 if side == "L" else 1]
    prior_kwargs = {"n_epochs": 1}
    if prior in {"lcash", "po_lcash"}:
        prior_kwargs["ash_init"] = False
    else:
        prior_kwargs.update(hidden_dim=8, n_layers=1)
    model = cEBMF(
        data,
        K=4,
        allow_backfitting=False,
        device="cpu",
        **{
            f"prior_{side}": prior,
            f"prior_{side}_kwargs": prior_kwargs,
            "self_row_cov" if side == "L" else "self_col_cov": True,
            f"X_{side.lower()}": torch.randn(n, 2) if external else None,
        },
    )
    model.initialise_factors()
    model.fit(1)

    # Force pruning at a sweep boundary, independently of stochastic training.
    # Remove a middle factor first, then reduce to the intercept-only case.
    for dropped in ([1], [0, 2]):
        old_rank = model.model.K
        model.pi0_L = [1.0 if k in dropped else 0.0 for k in range(old_rank)]
        model.pi0_F = [0.0] * old_rank
        model.model.allow_backfitting = True
        model._backfit()
        assert model.model.K == old_rank - len(dropped)
        model.model.allow_backfitting = False

        # Two sweeps exercise both the changed design and subsequent warm starts.
        model.fit(2)
        assert torch.isfinite(model.L).all()
        assert torch.isfinite(model.F).all()
        assert torch.isfinite(model.L2).all()
        assert torch.isfinite(model.F2).all()
        assert torch.isfinite(torch.tensor(model.obj)).all()


@pytest.mark.parametrize("self_row_cov,self_col_cov", [(True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize("dropped", [0, 1, 2])
def test_pruning_preserves_cached_priors_only_for_unchanged_covariates(self_row_cov, self_col_cov, dropped):
    model = cEBMF(torch.randn(6, 5), K=3, self_row_cov=self_row_cov, self_col_cov=self_col_cov, device="cpu")
    model.initialise_factors()
    states_l = [{"factor": k} for k in range(3)]
    states_f = [{"factor": k} for k in range(3)]
    model.model_state_L = states_l.copy()
    model.model_state_F = states_f.copy()

    # A sweep without pruning must preserve all warm starts.
    model._backfit()
    assert all(a is b for a, b in zip(model.model_state_L, states_l, strict=True))
    assert all(a is b for a, b in zip(model.model_state_F, states_f, strict=True))

    model.pi0_L[dropped] = 1.0
    model._backfit()
    for states, original, self_cov in (
        (model.model_state_L, states_l, self_row_cov),
        (model.model_state_F, states_f, self_col_cov),
    ):
        assert len(states) == 2
        keep = [i for i in range(3) if i != dropped]
        for state, i in zip(states, keep, strict=True):
            if self_cov and i > dropped:
                assert state is None
            else:
                assert state is original[i]


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("external", [False, True])
def test_self_covariates_include_exactly_the_earlier_factors(rank, external):
    model = cEBMF(torch.randn(6, 5), K=rank, device="cpu")
    factors = torch.arange(6 * rank, dtype=torch.float64).reshape(6, rank)
    covariates = torch.full((6, 2), -1.0, dtype=factors.dtype) if external else None
    for k in range(rank):
        actual = model._build_covariate_matrix(covariates, True, factors, k, 6)
        if k == 0:
            expected = covariates if external else factors.new_ones(6, 1)
        elif external:
            expected = torch.cat((covariates, factors[:, :k]), dim=1)
        else:
            expected = factors[:, :k]
        torch.testing.assert_close(actual, expected)

        # Disabling self-covariates leaves the supplied design unchanged.
        assert model._build_covariate_matrix(covariates, False, factors, k, 6) is covariates
