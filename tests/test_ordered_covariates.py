"""Self-covariates contain parents only, never the current/future factors."""

import pytest
import torch

from cebmf_torch import cEBMF


@pytest.mark.parametrize("side", ["L", "F"])
@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("external", [False, True])
def test_ordered_parents(side, k, external):
    model = cEBMF(torch.ones(6, 5), K=4, device="cpu")
    n = model.N if side == "L" else model.P
    factors = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
    cov = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) if external else None
    result = model._build_covariate_matrix(cov, True, factors, k, n)
    expected = torch.cat((cov, factors[:, :k]), 1) if external else factors[:, :k] if k else torch.ones(n, 1)
    torch.testing.assert_close(result, expected)

    # A perturbation to the current factor or any descendant cannot change
    # the predictors used to estimate this factor's conditional prior.
    saved = result.clone()
    factors[:, k:] += 1000
    torch.testing.assert_close(model._build_covariate_matrix(cov, True, factors, k, n), saved)


def test_vector_external_covariate_and_unconditional_first_factor():
    model = cEBMF(torch.ones(6, 5), K=1, device="cpu")
    cov = torch.arange(6, dtype=torch.float32)
    parents = model._build_covariate_matrix(cov, True, model.L, 0, model.N)
    torch.testing.assert_close(parents, cov[:, None])
    intercept = model._build_covariate_matrix(None, True, model.L, 0, model.N)
    torch.testing.assert_close(intercept, torch.ones(6, 1))
    assert model._build_covariate_matrix(cov, False, model.L, 0, model.N) is cov


@pytest.mark.parametrize("self_row_cov,self_col_cov", [(True, False), (False, True), (True, True)])
def test_pruning_discards_networks_with_changed_parent_columns(self_row_cov, self_col_cov):
    model = cEBMF(
        torch.ones(6, 5), K=3, self_row_cov=self_row_cov, self_col_cov=self_col_cov, device="cpu"
    )
    model.model_state_L = [{"factor": k} for k in range(3)]
    model.model_state_F = [{"factor": k} for k in range(3)]
    model._prune_indices([1])
    retained = [{"factor": 0}, {"factor": 2}]
    assert model.model_state_L == ([None, None] if self_row_cov else retained)
    assert model.model_state_F == ([None, None] if self_col_cov else retained)


def test_atac_then_rna_conditioning_runs_with_factor_specific_network_widths():
    torch.manual_seed(24)
    options = dict(
        prior_L="cgb",
        prior_F="hmm_pos",
        self_row_cov=True,
        allow_backfitting=False,
        device="cpu",
        prior_L_kwargs={"n_epochs": 2, "n_layers": 1, "hidden_dim": 4},
        prior_F_kwargs={"mu": [0, 0.5, 1], "prior_sd": [0, 0.2], "maxiter": 2, "learn_state_means": False},
    )
    atac = cEBMF(torch.rand(12, 16), K=2, **options)
    rna = cEBMF(torch.rand(12, 16), K=3, X_l=atac.L.clone(), **options)
    atac.initialise_factors()
    rna.initialise_factors()
    for _ in range(2):
        atac.iter_once()
        rna.covariate.X_l = atac.L.detach().clone()
        rna.iter_once()
    assert [s["input_layer.weight"].shape[1] for s in atac.model_state_L] == [1, 1]
    assert [s["input_layer.weight"].shape[1] for s in rna.model_state_L] == [2, 3, 4]
    assert atac.covariate.X_l is None
    assert torch.isfinite(atac.L).all() and torch.isfinite(rna.L).all()
