import math

import pytest
import torch

from cebmf_torch.cebnm.cash_solver import cash_posterior_means


@pytest.mark.parametrize("n_samples", [20000])  # keep test fast
def test_cash_posterior_means_mse(n_samples):
    torch.manual_seed(1)

    # Generate synthetic data
    y = torch.empty(n_samples).uniform_(-0.5, 2.5)  # U(-0.5, 2.5)
    X = y.view(-1, 1)

    mask_zero = ((y > 0) & (y < 0.5)) | ((y > 1.5) & (y < 2.0))
    noise_std = 0.5 + torch.abs(torch.sin(math.pi * y))
    xtrue = torch.where(mask_zero, torch.zeros_like(y), torch.randn_like(y) * noise_std)

    # Observed noisy data
    x = xtrue + torch.randn_like(xtrue)  # betahat
    s = torch.ones_like(x)  # sebetahat

    # Run CASH posterior means
    res = cash_posterior_means(
        X=X,
        betahat=x,
        sebetahat=s,
        n_epochs=50,  #
        n_layers=2,
        num_classes=10,
        hidden_dim=32,
        batch_size=128,
        lr=1e-3,
        penalty=1.5,
    )

    # Compute mean squared error between posterior mean and truth
    mse = torch.mean((res.post_mean - xtrue).pow(2)).item()
    print("Test MSE:", mse)

    # Check threshold
    assert mse < 0.4, f"MSE too large: {mse}"
