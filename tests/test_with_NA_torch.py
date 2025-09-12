import numpy as np
import torch

from cebmf_torch import cEBMF


def test_cebmf_handles_nan_end_to_end():
    rng = np.random.default_rng(42)
    n, p = 50, 40
    u = rng.random(n)
    v = rng.random(p)
    X = np.outer(u, v) + rng.normal(0, 0.1, (n, p))
    X[1, 1] = np.nan
    X[10, 5] = np.nan
    X[30:33, 12:14] = np.nan
    Y = torch.tensor(X, dtype=torch.float32)
    m = cEBMF(Y, K=5, prior_L="norm", prior_F="norm")
    m.initialize("svd")
    for _ in range(10):
        m.iter_once()
    # Ensure no nans in learned factors and tau positive
    assert torch.isfinite(m.L).all() and torch.isfinite(m.F).all()
    assert float(m.tau) > 0
