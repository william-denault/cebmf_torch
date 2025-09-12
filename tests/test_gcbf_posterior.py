import pytest
import torch

from cebmf_torch.cebnm.cov_gb_prior import cgb_posterior_means


@pytest.mark.parametrize("n_epochs, lr", [(500, 1e-2)])
def test_cov_gb_prior_quality(n_epochs, lr):
    # ---- deterministic CPU run ----
    torch.manual_seed(12)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu")

    # ---- data gen (matches your script) ----
    N = 500
    a, b = 5.0, -1.0
    mu1_true, sig1_true = 0.0, 0.2
    mu2_true, sig2_true = 2.0, 0.3
    sd_noise = 0.4

    x = torch.empty(N, device=device).uniform_(-2.0, 2.0)
    pi2_true = torch.sigmoid(a * x + b)
    comp = torch.bernoulli(pi2_true).to(dtype=torch.int64)

    y = torch.empty(N, device=device)
    n0 = int((comp == 0).sum())
    n1 = int((comp == 1).sum())
    y[comp == 0] = torch.normal(mean=mu1_true, std=sig1_true, size=(n0,), device=device)
    y[comp == 1] = torch.normal(mean=mu2_true, std=sig2_true, size=(n1,), device=device)

    betahat = y + torch.normal(mean=0.0, std=sd_noise, size=(N,), device=device)
    sebetahat = torch.full((N,), sd_noise, dtype=torch.float32, device=device)

    # ---- fit ----
    res = cgb_posterior_means(
        X=x,
        betahat=betahat,
        sebetahat=sebetahat,
        n_epochs=n_epochs,
        lr=lr,
    )

    # ---- checks ----
    # mse of posterior means vs truth
    mse = torch.mean((res.post_mean - y).pow(2)).item()
    assert mse < 0.058, f"MSE too high: {mse:.6f} (threshold 0.058)"

    # learned mu_2 close to ground truth
    mu2_err = abs(res.mu_2 - float(mu2_true))
    assert mu2_err < 0.07, f"|mu2_hat - mu2_true| = {mu2_err:.6f} (threshold 0.07)"

    # learned sigma_2 (std) close to ground truth
    sig2_err = abs(res.sigma_2 - float(sig2_true))
    assert sig2_err < 0.033, (
        f"|sigma2_hat - sigma2_true| = {sig2_err:.6f} (threshold 0.03)"
    )
