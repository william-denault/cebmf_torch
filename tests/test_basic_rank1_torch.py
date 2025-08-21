
import numpy as np
import torch
from cebmf_torch import cEBMF

def rmse(A,B):
    return np.sqrt(((A-B)**2).mean())

def test_rank1_iter_and_obj():
    rng = np.random.default_rng(0)
    n,p=50,40
    u = rng.random(n); v=rng.random(p)
    X = np.outer(u,v) + rng.normal(0,0.1,(n,p))

    Y = torch.tensor(X, dtype=torch.float32)
    m = cEBMF(Y, K=5, prior_L="norm", prior_F="norm")
    m.initialize("svd")
    base = rmse((m.L@m.F.T).cpu().numpy(), np.outer(u,v))
    for _ in range(20):
        m.iter_once()
    improved = rmse((m.L@m.F.T).cpu().numpy(), np.outer(u,v))
    assert improved <= base + 1e-8
    assert len(m.obj) >= 1
