
import numpy as np
import torch
from cebmf_torch import ash
from cebmf_torch.torch_utils_mix import autoselect_scales_mix_exp
from cebmf_torch.torch_distribution_operation import get_data_loglik_exp
from cebmf_torch.torch_mix_opt import optimize_pi_logL_torch
from cebmf_torch.torch_posterior import posterior_mean_exp

def test_ash_loglik_and_scale():
    betahat = torch.tensor([1.,2.,3.,4.,5.])
    sebetahat = torch.tensor([1.,0.4,5.,1.,1.])
    mult = torch.sqrt(torch.tensor(2.0)).item()
    res = ash(betahat, sebetahat, prior="exp", mult=mult, method="em", steps=3000, batch_size=5)
    expected_log_lik = -15.244064765169643
    expected_scale = np.array([ 0.        , 0.02929687, 0.04143204, 0.05859375, 0.08286408,
                                0.1171875 , 0.16572815, 0.234375  , 0.3314563 , 0.46875   ,
                                0.66291261, 0.9375    , 1.32582521, 1.875     , 2.65165043,
                                3.75      , 5.30330086, 7.5       ])
    np.testing.assert_allclose(res.scale.cpu().numpy(), expected_scale, rtol=1e-5)
    np.testing.assert_allclose(res.post_mean.cpu().numpy(), np.array([0.21361966, 1.95003336, 0.68125659, 3.67773689, 4.69599116]) , rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res.post_mean2.cpu().numpy(), np.array([ 0.34021317,  3.96295414,  3.17114472, 14.58222929, 23.05642398]) , rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res.log_lik, expected_log_lik, atol=1e-3)

def test_optimize_pi_and_posterior_mean_norm_shape():
    betahat = torch.tensor([1.,2.,3.,4.,5.])
    sebetahat = torch.tensor([1.,0.4,5.,1.,1.]) 
    scale = autoselect_scales_mix_exp(betahat, sebetahat)
    L = get_data_loglik_exp(betahat, sebetahat,   scale)
    pi = optimize_pi_logL_torch(L, penalty=10, method="em", steps=5000, batch_size=5)
    out = posterior_mean_exp(betahat, sebetahat, torch.log(pi+1e-32), scale)
    result = torch.exp(L) * torch.exp(pi)
    assert result.shape[0] == 5
