import pytest
import torch

from cebmf_torch.cebnm.lcash import lcash_posterior_means, po_lcash_posterior_means


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
@pytest.mark.parametrize("ash_init", [False, True])
def test_conditional_posterior_shrinks_covariate_dependent_signals(solver, ash_init):
    generator = torch.Generator().manual_seed(1)
    n = 1000
    covariate = torch.randn(n, generator=generator)
    truth = torch.randn(n, generator=generator) * covariate.clamp(min=0)
    observed = truth + torch.randn(n, generator=generator)
    result = solver(
        covariate[:, None], observed, torch.ones_like(observed), max_iter=100, penalty=1.5, ash_init=ash_init
    )
    assert torch.mean((result.post_mean - truth).square()) < 0.7
    assert result.model_param is not None
    assert result.pi_np.shape == (n, len(result.scale))
    if ash_init:
        assert len(result.scale) < 20


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
def test_further_training_preserves_posterior_accuracy(solver):
    generator = torch.Generator().manual_seed(42)
    truth = torch.randn(5000, generator=generator)
    observed = truth + 0.5 * torch.randn(5000, generator=generator)
    inputs = (truth[:, None], observed, torch.full_like(observed, 0.5))
    first = solver(*inputs, max_iter=20, ash_init=False)
    updated = solver(*inputs, max_iter=20, ash_init=False, model_param=first.model_param)
    first_mse = torch.mean((first.post_mean - truth).square())
    updated_mse = torch.mean((updated.post_mean - truth).square())
    assert updated.model_param is not None
    assert updated_mse < first_mse + 0.05


def test_lcash_ash_result_pi_field():
    """ASHResult exposes the full weight vector in its pi field."""
    from cebmf_torch.ebnm.ash import PriorType, ash

    torch.manual_seed(1)
    x = torch.randn(1000)
    s = torch.ones(1000)

    result = ash(x, s, prior=PriorType.NORM, verbose=False)
    assert hasattr(result, "pi"), "ASHResult should have a 'pi' field"
    assert result.pi.shape == result.scale.shape, "pi should have same shape as scale"
    assert abs(result.pi.sum().item() - 1.0) < 1e-5, "pi should sum to 1"
    assert abs(result.pi[0].item() - result.pi0) < 1e-5, "pi[0] should equal pi0"


def test_po_lcash_k2_edge_case():
    """PO-LC-ASH should work with K=2 (single cut-point, no delta_gaps)."""
    from cebmf_torch.cebnm.lcash import PropOddsLcashNet

    torch.manual_seed(1)
    F, K = 3, 2
    model = PropOddsLcashNet(F, K)

    # delta_gaps should be None for K=2
    assert model.delta_gaps is None, "K=2 should have no delta_gaps"

    # Forward pass should produce valid probabilities
    x = torch.randn(100, F)
    pi = model(x)
    assert pi.shape == (100, 2)
    assert torch.allclose(pi.sum(dim=1), torch.ones(100), atol=1e-5)
    assert (pi > 0).all(), "All probabilities should be positive"


def test_po_lcash_fewer_params_than_softmax():
    """PO-LC-ASH should have fewer parameters than softmax LC-ASH."""
    from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet

    F, K = 10, 20
    softmax = LcashNet(F, K)
    propodds = PropOddsLcashNet(F, K)

    n_softmax = sum(p.numel() for p in softmax.parameters())
    n_propodds = sum(p.numel() for p in propodds.parameters())

    print(f"Softmax params: {n_softmax}, PropOdds params: {n_propodds}")
    # Softmax: F*K + K = 10*20 + 20 = 220
    # PropOdds: F + 1 + (K-2) = 10 + 1 + 18 = 29
    assert n_propodds < n_softmax, f"PropOdds ({n_propodds}) should have fewer params than Softmax ({n_softmax})"
