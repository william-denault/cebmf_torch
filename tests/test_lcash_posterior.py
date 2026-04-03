import pytest
import torch

from cebmf_torch.cebnm.lcash import lcash_posterior_means, po_lcash_posterior_means


@pytest.mark.parametrize("n_samples", [1000])
def test_lcash_posterior_means_mse_autoselect(n_samples):
    """LC-ASH with auto-selected grid should shrink on linearly-separable data."""
    torch.manual_seed(1)

    # Linearly-separable signal: high covariate -> large effect, low -> near zero
    y = torch.randn(n_samples)
    X = y.view(-1, 1)

    # Signal strength scales linearly with covariate
    signal_sd = torch.clamp(y, min=0.0)  # only positive y has signal
    xtrue = torch.randn(n_samples) * signal_sd

    x = xtrue + torch.randn_like(xtrue)  # betahat = true + noise
    s = torch.ones_like(x)  # se = 1

    res = lcash_posterior_means(
        X=X,
        betahat=x,
        sebetahat=s,
        n_epochs=100,
        batch_size=256,
        lr=1e-3,
        penalty=1.5,
        ash_init=False,
    )

    mse = torch.mean((res.post_mean - xtrue).pow(2)).item()
    # Baseline: no shrinkage MSE ~ 1.0 (noise variance)
    # Good shrinkage should reduce this below 0.7
    print(f"LC-ASH (auto-select) MSE: {mse:.4f}")
    assert mse < 0.7, f"MSE too large: {mse}"
    assert res.model_param is not None
    assert res.pi_np.shape == (n_samples, res.scale.shape[0])


@pytest.mark.parametrize("n_samples", [1000])
def test_lcash_posterior_means_mse_inherit(n_samples):
    """LC-ASH with inherited grid should shrink and produce fewer components."""
    torch.manual_seed(1)

    y = torch.randn(n_samples)
    X = y.view(-1, 1)
    signal_sd = torch.clamp(y, min=0.0)
    xtrue = torch.randn(n_samples) * signal_sd

    x = xtrue + torch.randn_like(xtrue)
    s = torch.ones_like(x)

    res = lcash_posterior_means(
        X=X,
        betahat=x,
        sebetahat=s,
        n_epochs=100,
        batch_size=256,
        lr=1e-3,
        penalty=1.5,
        ash_init=True,
        ash_threshold=0.0,
    )

    mse = torch.mean((res.post_mean - xtrue).pow(2)).item()
    print(f"LC-ASH (inherit grid) MSE: {mse:.4f}")
    assert mse < 0.7, f"MSE too large: {mse}"
    K = res.scale.shape[0]
    print(f"LC-ASH (inherit grid) K={K}")
    assert K < 20, f"Expected pruned grid with K < 20, got K={K}"


def test_lcash_warm_start():
    """Warm-starting from a previous model_param should work."""
    torch.manual_seed(42)
    n = 5000
    y = torch.randn(n)
    X = y.view(-1, 1)
    x = y + 0.5 * torch.randn(n)
    s = 0.5 * torch.ones(n)

    res1 = lcash_posterior_means(
        X=X, betahat=x, sebetahat=s,
        n_epochs=20, ash_init=False,
    )

    res2 = lcash_posterior_means(
        X=X, betahat=x, sebetahat=s,
        n_epochs=20, ash_init=False,
        model_param=res1.model_param,
    )

    assert res2.model_param is not None
    mse1 = torch.mean((res1.post_mean - y).pow(2)).item()
    mse2 = torch.mean((res2.post_mean - y).pow(2)).item()
    print(f"Cold-start MSE: {mse1:.4f}, Warm-start MSE: {mse2:.4f}")
    assert mse2 < mse1 + 0.05, "Warm-start should not degrade substantially"


def test_lcash_ash_result_pi_field():
    """ASHResult should now have a pi field with the full weight vector."""
    from cebmf_torch.ebnm.ash import PriorType, ash

    torch.manual_seed(1)
    x = torch.randn(1000)
    s = torch.ones(1000)

    result = ash(x, s, prior=PriorType.NORM, verbose=False)
    assert hasattr(result, "pi"), "ASHResult should have a 'pi' field"
    assert result.pi.shape == result.scale.shape, "pi should have same shape as scale"
    assert abs(result.pi.sum().item() - 1.0) < 1e-5, "pi should sum to 1"
    assert abs(result.pi[0].item() - result.pi0) < 1e-5, "pi[0] should equal pi0"


# ---- Proportional Odds LC-ASH tests ----


@pytest.mark.parametrize("n_samples", [1000])
def test_po_lcash_posterior_means_mse_autoselect(n_samples):
    """PO-LC-ASH with auto-selected grid should shrink on linearly-separable data."""
    torch.manual_seed(1)

    y = torch.randn(n_samples)
    X = y.view(-1, 1)
    signal_sd = torch.clamp(y, min=0.0)
    xtrue = torch.randn(n_samples) * signal_sd

    x = xtrue + torch.randn_like(xtrue)
    s = torch.ones_like(x)

    res = po_lcash_posterior_means(
        X=X,
        betahat=x,
        sebetahat=s,
        n_epochs=100,
        batch_size=256,
        lr=1e-3,
        penalty=1.5,
        ash_init=False,
    )

    mse = torch.mean((res.post_mean - xtrue).pow(2)).item()
    print(f"PO-LC-ASH (auto-select) MSE: {mse:.4f}")
    assert mse < 0.7, f"MSE too large: {mse}"
    assert res.model_param is not None
    assert res.pi_np.shape == (n_samples, res.scale.shape[0])


@pytest.mark.parametrize("n_samples", [1000])
def test_po_lcash_posterior_means_mse_inherit(n_samples):
    """PO-LC-ASH with inherited grid should shrink and produce fewer components."""
    torch.manual_seed(1)

    y = torch.randn(n_samples)
    X = y.view(-1, 1)
    signal_sd = torch.clamp(y, min=0.0)
    xtrue = torch.randn(n_samples) * signal_sd

    x = xtrue + torch.randn_like(xtrue)
    s = torch.ones_like(x)

    res = po_lcash_posterior_means(
        X=X,
        betahat=x,
        sebetahat=s,
        n_epochs=100,
        batch_size=256,
        lr=1e-3,
        penalty=1.5,
        ash_init=True,
        ash_threshold=0.0,
    )

    mse = torch.mean((res.post_mean - xtrue).pow(2)).item()
    print(f"PO-LC-ASH (inherit grid) MSE: {mse:.4f}")
    assert mse < 0.7, f"MSE too large: {mse}"
    K = res.scale.shape[0]
    print(f"PO-LC-ASH (inherit grid) K={K}")
    assert K < 20, f"Expected pruned grid with K < 20, got K={K}"


def test_po_lcash_warm_start():
    """Warm-starting PO-LC-ASH from a previous model_param should work."""
    torch.manual_seed(42)
    n = 5000
    y = torch.randn(n)
    X = y.view(-1, 1)
    x = y + 0.5 * torch.randn(n)
    s = 0.5 * torch.ones(n)

    res1 = po_lcash_posterior_means(
        X=X, betahat=x, sebetahat=s,
        n_epochs=20, ash_init=False,
    )

    res2 = po_lcash_posterior_means(
        X=X, betahat=x, sebetahat=s,
        n_epochs=20, ash_init=False,
        model_param=res1.model_param,
    )

    assert res2.model_param is not None
    mse1 = torch.mean((res1.post_mean - y).pow(2)).item()
    mse2 = torch.mean((res2.post_mean - y).pow(2)).item()
    print(f"PO cold-start MSE: {mse1:.4f}, PO warm-start MSE: {mse2:.4f}")
    assert mse2 < mse1 + 0.05, "Warm-start should not degrade substantially"


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
    assert n_propodds < n_softmax, (
        f"PropOdds ({n_propodds}) should have fewer params than Softmax ({n_softmax})"
    )
