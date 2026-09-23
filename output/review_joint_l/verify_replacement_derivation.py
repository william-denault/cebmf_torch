"""Independent numerical checks of the proposed variational replacement.

These validate the derivation, not an implemented replacement solver.
The production implementation is deliberately unchanged during the audit.
"""

import itertools
import json
import math
from pathlib import Path

import torch

from cebmf_torch.experimental.conditional import GaussianMixture

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def gaussian_no_parent_limit():
    generator = torch.Generator().manual_seed(311)
    n, h = 23, 4
    log_weights = torch.randn(n, h, generator=generator).log_softmax(1)
    means = torch.randn(n, h, generator=generator)
    variances = torch.rand(n, h, generator=generator) + .02
    means[:, 0], variances[:, 0] = 0., 0.
    a, b = torch.rand(n, generator=generator)*3, torch.randn(n, generator=generator)
    a[0], b[0] = 0., 0.
    reference, evidence = GaussianMixture(log_weights, means, variances).posterior(a, b)
    # Equation from expected log Gaussian component + quadratic likelihood.
    A = a[:, None] + variances[:, 1:].reciprocal()
    B = b[:, None] + means[:, 1:] / variances[:, 1:]
    C = log_weights[:, 1:] - .5*(2*math.pi*variances[:, 1:]).log() - .5*means[:, 1:].square()/variances[:, 1:]
    mass = torch.cat((log_weights[:, :1], C+B.square()/(2*A)+.5*(2*math.pi/A).log()), 1)
    weights = mass.softmax(1)
    mean = torch.cat((torch.zeros(n, 1), B/A), 1)
    variance = torch.cat((torch.zeros(n, 1), 1/A), 1)
    error = max(float((torch.logsumexp(mass, 1)-evidence).abs().max()),
                float((weights-reference.log_weight.exp()).abs().max()),
                float((mean-reference.mean).abs().max()),
                float((variance-reference.variance).abs().max()))
    assert error < 1e-12
    return {"max_absolute_error": error, "includes_spike_and_missing_likelihood": True}


def profile_objective_check():
    states = torch.tensor(list(itertools.product([0., 1.], repeat=3)))
    F = torch.tensor([[1., .7, -.2], [.4, 1., .6]])
    y, tau = torch.tensor([.8, -.3]), torch.tensor([1.3, .7])
    slopes = torch.tensor([[0.,0.,0.],[1.4,0.,0.],[-1.,1.7,0.]])
    p = torch.tensor([.27,.61,.44])
    k = 1
    a = (tau*F[:, k].square()).sum()
    b = (tau*(y-F@p+F[:, k]*p[k])*F[:, k]).sum()
    q_other = torch.prod(states[:, [0, 2]]*p[[0, 2]]+(1-states[:, [0, 2]])*(1-p[[0, 2]]),1)
    likelihood = -.5*((y-states@F.T).square()*tau).sum(1)
    constants = []
    for theta in [-2., -.3, .8, 2.]:
        logits = torch.tensor([-.4, theta, -.6])+states@slopes.T
        log_prior = (states*torch.nn.functional.logsigmoid(logits)+(1-states)*torch.nn.functional.logsigmoid(-logits)).sum(1)
        psi = torch.stack([(q_other[states[:, k]==v]*log_prior[states[:, k]==v]).sum()+b*v-.5*a*v*v for v in (0.,1.)])
        updated = p.clone()
        updated[k] = torch.sigmoid(psi[1]-psi[0])
        q = (states*updated+(1-states)*(1-updated)).prod(1)
        full_elbo = (q*(likelihood+log_prior-q.log())).sum()
        constants.append(float(full_elbo-torch.logsumexp(psi,0)))
    spread = max(constants)-min(constants)
    assert spread < 1e-12
    return {"max_variation_in_theta_independent_constant":spread,
            "parameter_settings_checked":4}


def exact_binary_variational_updates(dependence):
    states = torch.tensor(list(itertools.product([0., 1.], repeat=3)))
    F = torch.tensor([[1., .7, -.2], [.4, 1., .6]])
    y, tau = torch.tensor([.8, -.3]), torch.tensor([1.3, .7])
    intercept = torch.tensor([-.4, .1, -.6])
    slopes = torch.tensor([[0.,0.,0.],[1.4,0.,0.],[-1.,1.7,0.]]) * dependence

    def prior_terms(v):
        logits = intercept + v @ slopes.T
        return v*torch.nn.functional.logsigmoid(logits)+(1-v)*torch.nn.functional.logsigmoid(-logits)

    likelihood = -.5*((y-states@F.T).square()*tau).sum(1)
    log_joint = likelihood + prior_terms(states).sum(1)
    probability = torch.tensor([.27,.61,.44])

    def mass(p):
        return (states*p+(1-states)*(1-p)).prod(1)

    def elbo(p):
        q = mass(p)
        return float((q*(log_joint-q.log())).sum())

    errors, gains = [], []
    for k in range(3):
        old = elbo(probability)
        mean_residual = y-F@probability+F[:,k]*probability[k]
        a = (tau*F[:,k].square()).sum()
        b = (tau*mean_residual*F[:,k]).sum()
        # Average only over the other coordinates, accounting for both the
        # node's own prior and every child's prior under its candidate value.
        other_mass = torch.ones(len(states))
        for j in range(3):
            if j != k:
                other_mass *= states[:,j]*probability[j]+(1-states[:,j])*(1-probability[j])
        v0, v1 = states.clone(), states.clone()
        v0[:,k], v1[:,k] = 0.,1.
        prior_difference = .5*(other_mass*(prior_terms(v1)-prior_terms(v0)).sum(1)).sum()
        proposed = torch.sigmoid(b-.5*a+prior_difference)
        conditional_scores=[]
        for value in (0.,1.):
            mask=states[:,k]==value
            conditional_scores.append((other_mass[mask]*log_joint[mask]).sum())
        exact = torch.sigmoid(conditional_scores[1]-conditional_scores[0])
        errors.append(float(abs(proposed-exact)))
        if dependence == 0:
            assert abs(proposed-torch.sigmoid(b-.5*a+intercept[k])) < 1e-12
        probability[k]=proposed
        gains.append(elbo(probability)-old)
    assert max(errors)<1e-12 and min(gains)>-1e-12
    return {"max_coordinate_probability_error":max(errors), "exact_elbo_gains":gains}


def structured_loading_cross_moment():
    # One row has q(00)=q(11)=1/2; y=1 and current F_2=1.
    # E[L1]=E[L2]=1/2, E[L1 L2]=1/2. Correct F1 numerator is
    # 1*(1/2)-1*(1/2)=0; multiplying means incorrectly gives 1/4.
    states=torch.tensor([[0.,0.],[1.,1.]])
    weights=torch.tensor([.5,.5])
    mu=(weights[:,None]*states).sum(0)
    cross=(weights*states[:,0]*states[:,1]).sum()
    exact_b=mu[0]-cross
    naive_b=mu[0]-mu[0]*mu[1]
    v=torch.tensor(0.,requires_grad=True)
    loglik=-.5*(weights*(1-states[:,0]*v-states[:,1]).square()).sum()
    direct_b=torch.autograd.grad(loglik,v)[0]
    assert abs(exact_b-direct_b)<1e-12
    return {"correct_numerator":float(exact_b), "mean_product_numerator":float(naive_b)}


def main():
    result={"scope":"Independent derivation checks; implementation regressions live in tests/test_conditional_*.py.",
            "gaussian_mixture_no_self_reference":gaussian_no_parent_limit(),
            "profile_log_normalizer_matches_optimized_ELBO":profile_objective_check(),
            "binary_no_self_reference":exact_binary_variational_updates(0.),
            "binary_with_self_reference":exact_binary_variational_updates(1.),
            "structured_loading_F_update":structured_loading_cross_moment(),
            "no_self_one_M_step_is_not_collapsed_EBNM":{
                "observation":2.,"noise_variance":1.,"fixed_prior_variance":1.,
                "old_prior_mean":0.,"collapsed_EBNM_fitted_prior_mean":2.,
                "one_exact_EM_fitted_prior_mean":1.,
                "collapsed_EBNM_posterior_mean":2.,"posterior_mean_after_one_EM_step":1.5}}
    path=Path(__file__).with_name("derivation_checks.json")
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
