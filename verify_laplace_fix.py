"""
Verification script for the point-Laplace ELBO monotonicity fix.

Reproduces the user's stress-test conditions from
`test_monoticity_laplace.ipynb` and `test_monoticity_point_exp.ipynb`,
and compares:
  - non-monotonic ELBO trajectories (count of runs that decreased)
  - mean RMSE  (relative to flashier-style reference: point-exp)

Run after the fix lands and confirm:
  * laplace's non-monotonic-run count drops to ~ exp's level
  * laplace's RMSE is close to exp's (and to the flashier reference)
"""

import statistics

import torch

from cebmf_torch import cEBMF
from cebmf_torch.ebnm.point_laplace import ebnm_point_laplace

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Sanity: the R-port unit test should still pass with the bounded-`a` change.
# -----------------------------------------------------------------------------
def smoke_test_rport_compat():
    x = torch.tensor([0.0, 1.0, -0.5])
    s = torch.tensor([1.0, 0.2, 1.0])
    res = ebnm_point_laplace(x, s)
    print(f"  R-port smoke: log_lik = {float(res.log_lik):.6f} (expected ≈ -4.1619)")
    print(f"                post_mean = {res.post_mean.tolist()}")
    assert abs(float(res.log_lik) - (-4.161880337595547)) < 2e-2


# -----------------------------------------------------------------------------
# Stress test: K=5 factors fitted on rank-1 + heavy noise (mostly null factors).
# -----------------------------------------------------------------------------
def is_decreasing_at_any_step(xs, tol=1e-3):
    """Return True if obj[t+1] > obj[t] + tol at any t (since obj = -ELBO,
    monotonic optimisation means obj must be non-increasing)."""
    return any(xs[t + 1] > xs[t] + tol for t in range(len(xs) - 1))


def stress(prior_name, n_runs=20, N=50, P=40, noise_std=0.51, K=5, maxit=20):
    bad = 0
    rmses = []
    for seed in range(n_runs):
        torch.manual_seed(seed)
        u = torch.rand(N, device=device)
        v = torch.rand(P, device=device)
        true = torch.outer(u, v)
        Y = true + noise_std * torch.randn(N, P, device=device)

        model = cEBMF(
            data=Y, K=K, prior_F=prior_name, prior_L=prior_name,
            allow_backfitting=False,
        )
        model.initialise_factors()
        model.fit(maxit=maxit)
        if is_decreasing_at_any_step(model.obj):
            bad += 1
        model._update_fitted_value()
        rmses.append(torch.sqrt(torch.mean((model.Y_fit - true) ** 2)).item())

    return {
        "non_monotonic_runs": bad,
        "rmse_mean": statistics.mean(rmses),
        "rmse_sd": statistics.stdev(rmses) if len(rmses) > 1 else 0.0,
    }


if __name__ == "__main__":
    print("== R-port compatibility check ==")
    smoke_test_rport_compat()
    print("OK")

    print("\n== Stress test (noise_std=0.51, K=5, true rank=1) ==")
    for prior in ("exp", "laplace"):
        out = stress(prior)
        print(
            f"  {prior:>8}: non-monotonic runs = {out['non_monotonic_runs']}/20, "
            f"RMSE = {out['rmse_mean']:.4f} ± {out['rmse_sd']:.4f}"
        )

    print("\n== Easier stress test (noise_std=0.1, K=5, true rank=1) ==")
    for prior in ("exp", "laplace"):
        out = stress(prior, noise_std=0.1, maxit=50)
        print(
            f"  {prior:>8}: non-monotonic runs = {out['non_monotonic_runs']}/20, "
            f"RMSE = {out['rmse_mean']:.4f} ± {out['rmse_sd']:.4f}"
        )
