"""Reproduce CASH nullweight and posterior behavior on a known mixture."""

import io
from contextlib import redirect_stdout

import numpy as np
import torch

from cebmf_torch.cebnm import cash_solver
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.posterior import posterior_mean_norm


def _sample(n, seed):
    rng = np.random.default_rng(seed)
    is_null = rng.random(n) < 0.6
    theta = np.where(is_null, 0.0, rng.normal(0, 3, n))
    y = theta + rng.normal(size=n)
    return torch.from_numpy(y), torch.from_numpy(theta), torch.from_numpy(is_null)


def test_cash_nullweight_recovers_known_mixture(monkeypatch):
    """Check the default and compare penalty coefficients on a fixed grid."""
    n = 5000
    y, _, _ = _sample(n, seed=11)
    test_y, test_theta, test_null = _sample(20000, seed=100011)
    scale = torch.tensor([0.0, 3.0], dtype=torch.float64)
    log_lik = get_data_loglik_normal_torch(y, torch.ones_like(y), scale * 0, scale)
    test_log_lik = get_data_loglik_normal_torch(test_y, torch.ones_like(test_y), scale * 0, scale)
    # Known component scales isolate the null penalty from grid selection.
    monkeypatch.setattr(cash_solver, "autoselect_scales_mix_norm", lambda **kwargs: scale.float())
    # This case uses a fixed full batch. Other tests cover DataLoader batching.
    monkeypatch.setattr(
        cash_solver, "DataLoader", lambda dataset, **kwargs: [(dataset.X, dataset.betahat, dataset.sebetahat)]
    )
    rows = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            for label, coefficient in (
                ("default_10", 9 / n),
                ("per_observation", 0.5),
                ("fixed_total", 0.5 / n),
            ):
                torch.manual_seed(12345)
                with redirect_stdout(io.StringIO()):
                    fit = cash_solver.cash_posterior_means(
                        torch.zeros(n, 1),
                        y,
                        torch.ones_like(y),
                        n_epochs=300,
                        hidden_dim=4,
                        n_layers=0,
                        num_classes=2,
                        batch_size=n,
                        lr=0.03,
                        device=torch.device("cpu"),
                        **({} if label == "default_10" else {"penalty": 1 + n * coefficient}),
                    )
                q = float(fit.pi_np[:, 0].mean())
                if label == "default_10":
                    assert abs(q - 0.6) < 0.03, f"Default fitted null weight {q:.6f} differs from truth 0.6."
                # The mixture objective is strictly concave in q. An interior
                # stationary point is its optimum, independently of Adam's path.
                probability = torch.tensor(q, dtype=torch.float64, requires_grad=True)
                log_pi = torch.stack((probability.log(), torch.log1p(-probability)))
                objective = torch.logsumexp(log_lik + log_pi, dim=1).mean() + coefficient * log_pi[0]
                gradient = float(torch.autograd.grad(objective, probability)[0])
                assert 0.01 < q < 0.99
                assert abs(gradient) < 1e-4
                log_pi = log_pi.detach()
                posterior = posterior_mean_norm(
                    test_y, torch.ones_like(test_y), log_pi, test_log_lik, scale, scale * 0
                )
                mean_lfdr = float(torch.softmax(test_log_lik + log_pi, dim=1)[:, 0].mean())
                mse = float((posterior.post_mean - test_theta).square().mean())
                rows.append((label, q, mean_lfdr, mse, gradient))
    finally:
        torch.set_num_threads(previous_threads)

    default, strong, scaled = rows
    assert abs(scaled[1] - 0.6) < 0.03
    assert strong[1] > scaled[1] + 0.15
    assert abs(scaled[2] - float(test_null.double().mean())) < 0.02
    assert abs(default[2] - float(test_null.double().mean())) < 0.02
    assert strong[2] > 0.70
    assert scaled[3] < 0.95 * strong[3]
    assert default[3] < 0.95 * strong[3]
    print("\ncoefficient        fitted_pi0  mean_test_lfdr  posterior_mean_mse  objective_gradient")
    for label, q, mean_lfdr, mse, gradient in rows:
        print(f"{label:18s} {q:.6f}    {mean_lfdr:.6f}        {mse:.6f}            {gradient:.2e}")
