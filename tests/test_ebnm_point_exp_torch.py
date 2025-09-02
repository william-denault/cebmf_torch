
import numpy as np
import torch
from cebmf_torch.torch_ebnm.torch_ebnm_point_exp import ebnm_point_exp

def test_optimize_pe_like():
    x = torch.tensor([1.0, 1.0, -0.5])
    s = torch.tensor([1.0, 1.0, 1.0])
    res = ebnm_point_exp(x, s )
    # Expected numbers from original tests (allow small tolerance)
    assert np.isclose(res.log_lik, -3.636553632132083, atol=1e-2)
    assert np.isclose(float(res.pi0), 0.9999563044116645, atol=1e-3)
    assert np.isclose(float(res.scale), 3.047337093696241, atol=1e-2)
    assert np.isclose(float(res.mode), 0.0, atol=1e-4)

