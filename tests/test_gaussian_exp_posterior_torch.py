
import numpy as np
import torch
from cebmf_torch.torch_utils_mix import autoselect_scales_mix_exp
from cebmf_torch.torch_distribution_operation import get_data_loglik_exp
from cebmf_torch.torch_utils import logsumexp
from cebmf_torch.torch_posterior import truncated_normal_moments as my_trunc_mom

def my_etruncnorm(a,b,mu,s):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64) if np.isfinite(b) else torch.tensor(float('inf'), dtype=torch.float64)
    mu = torch.tensor(mu, dtype=torch.float64)
    s = torch.tensor(s, dtype=torch.float64)
    EX, _ = my_trunc_mom(a,b,mu,s)
    return float(EX)

def my_e2truncnorm(a,b,mu,s):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64) if np.isfinite(b) else torch.tensor(float('inf'), dtype=torch.float64)
    mu = torch.tensor(mu, dtype=torch.float64)
    s = torch.tensor(s, dtype=torch.float64)
    _, EX2 = my_trunc_mom(a,b,mu,s)
    return float(EX2)

def wpost_exp(x, s, w, scale):
    # responsibilities given uniform prior weights w over components
    device = torch.device('cpu')
    betahat = torch.tensor([x], dtype=torch.float64, device=device)
    se = torch.tensor([s], dtype=torch.float64, device=device)
    sc = torch.tensor(scale, dtype=torch.float64, device=device)
    L = get_data_loglik_exp(betahat, se, sc)  # (1,K)
    logw = torch.log(torch.tensor(w, dtype=torch.float64, device=device))
    log_post = L + logw.view(1,-1)
    post = torch.softmax(log_post, dim=1)[0].cpu().numpy()
    return post

def test_trucnorm():
    assert np.isclose(my_etruncnorm(0,2,3,1), 1.48995049, atol=1e-7)
    assert np.isclose(my_e2truncnorm(0,2,3,1), 2.39340536, atol=1e-7)

def test_convolved_loglik_postmean():
    betahat=  np.array([1,2,3,4,5], dtype=float)
    sebetahat=np.array([1,0.4,5,1,1], dtype=float)
    device = torch.device('cpu')
    scale = autoselect_scales_mix_exp ( torch.tensor(betahat),  torch.tensor(sebetahat)).cpu().numpy()

    non_informativ = np.full( scale.shape[0], 1/ scale.shape[0])
    n=betahat.shape[0]
    log_pi =  np.log( np.tile(non_informativ, (n, 1)))
    assignment = np.exp(log_pi)[0]
    assignment = assignment /   sum(assignment)
    w=assignment
    x=betahat[1]
    s=sebetahat[1]

    obs_wpost= wpost_exp ( x, s, w, scale)
    expected_wpost=np.array([3.53987758e-06, 6.61493885e-06, 1.06391083e-05, 2.86976960e-05,
       1.69165759e-04, 1.40776600e-03, 8.91255655e-03, 3.45101678e-02,
       8.34205613e-02, 1.38160703e-01, 1.72844260e-01, 1.77093219e-01,
       1.57939387e-01, 1.28091688e-01, 9.74010338e-02])

    # Compute posteriors and truncated-moment means
    # Note: first column is spike at 0; skip in expectation
    post_assign =   np.zeros ( (betahat.shape[0], scale.shape[0]))
    for i in range(betahat.shape[0]):
        post_assign[i,] = wpost_exp ( x=betahat[i], s=sebetahat[i], w=np.exp(log_pi)[i,], scale=scale) 

    post_mean = np.zeros(betahat.shape[0])
    post_mean2 = np.zeros(betahat.shape[0])
    for i in range(post_mean.shape[0]):
        mu_i = betahat[i] - sebetahat[i]**2 * (1/scale[1:])
        ex = np.array([my_etruncnorm(0, np.inf, m, sebetahat[i]) for m in mu_i])
        ex2 = np.array([my_e2truncnorm(0, 99999, m, sebetahat[i]) for m in mu_i])
        post_mean[i]=  np.sum( post_assign[i,1:] * ex )
        post_mean2[i] = np.sum( post_assign[i,1:] * ex2 )

    expected_post_mean= np.array([0.4836384 , 1.89328341, 1.05670973, 3.57009763, 4.66379651])
    expected_post_mean2= np.array([ 0.59674218,  3.75372025,  4.34337321, 13.89299102, 22.81107216])

    # Use our get_data_loglik + softmax responsibilities in the same way
    bet = torch.tensor(betahat, dtype=torch.float64)
    se = torch.tensor(sebetahat, dtype=torch.float64)
    sc = torch.tensor(scale, dtype=torch.float64)
    L = get_data_loglik_exp(bet, se, sc)  # (n,K)
    post = torch.softmax(L + torch.log(torch.tensor(non_informativ, dtype=torch.float64)).view(1,-1), dim=1).cpu().numpy()

    np.testing.assert_allclose(obs_wpost, expected_wpost, atol=1e-7)
    np.testing.assert_allclose(post_mean, expected_post_mean, atol=1e-7)
    np.testing.assert_allclose(post_mean2, expected_post_mean2, atol=1e-7)
