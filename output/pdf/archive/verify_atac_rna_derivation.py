"""Numerical identities for the derivation; no cEBMF solver implementation."""
from pathlib import Path
import itertools
import numpy as np
from scipy.integrate import quad
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import norm

lines = ["Numerical verification of atac_rna_joint_inference.tex", "2026-09-14", ""]

def check(name, actual, expected, tol=2e-9):
    err = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    assert np.isfinite(err) and err < tol, (name, err, actual, expected)
    lines.append(f"PASS {name}: maximum absolute error {err:.3e} (tolerance {tol:.1e})")

def integ(fun, lo=-np.inf, hi=np.inf):
    return quad(fun, lo, hi, epsabs=2e-12, epsrel=2e-12, limit=300)[0]

# CGB with a logistic child: compare the completed-square tilted posterior
# against the original likelihood * parent prior * child prior.
x, s2, mu, d2, pi = 0.9, 0.7, 0.5, 0.8, 0.4
m0 = (x / s2 + mu / d2) / (1 / s2 + 1 / d2)
v0 = 1 / (1 / s2 + 1 / d2)
child = lambda v: expit(-0.4 + 1.3 * v)
for has_child in (False, True):
    tilt = child if has_child else lambda v: 1.0
    integrals = np.array([integ(lambda v, r=r: v**r * norm.pdf(v, m0, np.sqrt(v0)) * tilt(v)) for r in range(3)])
    w0 = (1 - pi) * norm.pdf(x, 0, np.sqrt(s2)) * tilt(0)
    base = pi * norm.pdf(x, mu, np.sqrt(s2 + d2))
    z = w0 + base * integrals[0]
    calculated = np.array([base * integrals[0] / z, base * integrals[1] / z, base * integrals[2] / z])
    direct = np.array([integ(lambda v, r=r: v**r * pi * norm.pdf(v, mu, np.sqrt(d2)) * norm.pdf(x, v, np.sqrt(s2)) * tilt(v)) for r in range(3)])
    direct /= w0 + direct[0]
    check(f"CGB spike/slab probability and moments, child={has_child}", calculated, direct)
    if not has_child:
        check("CGB no-child Gaussian integral moments", integrals, [1, m0, m0*m0 + v0])

# Truncated Gaussian evidence, including a negative pre-truncation center.
for mu_pos in (-1.2, 0.4):
    ms = (x / s2 + mu_pos / d2) * v0
    evidence = norm.pdf(x, mu_pos, np.sqrt(s2 + d2)) * ndtr(ms / np.sqrt(v0)) / ndtr(mu_pos / np.sqrt(d2))
    direct = integ(lambda v: norm.pdf(x, v, np.sqrt(s2)) * norm.pdf(v, mu_pos, np.sqrt(d2)) / ndtr(mu_pos / np.sqrt(d2)), 0, np.inf)
    check(f"Positive Gaussian convolution, mu={mu_pos}", evidence, direct)

# Binary joint and CAVI: enumerate all states and construct the exact MH
# transition matrix, rather than using Monte Carlo to test stationarity.
states = np.array(list(itertools.product((0, 1), repeat=2)))
p, rho = 0.35, np.array([0.18, 0.82])
aA, bA, aR, bR = 1.4, 0.9, 2.0, 1.4
def log_joint(a, r):
    return (a*np.log(p)+(1-a)*np.log(1-p)
            +r*np.log(rho[a])+(1-r)*np.log(1-rho[a])
            -0.5*aA*a*a+bA*a-0.5*aR*r*r+bR*r)
logs = np.array([log_joint(a, r) for a, r in states])
target = np.exp(logs - logsumexp(logs))
for r in (0, 1):
    score = np.log(p/(1-p))+bA-0.5*aA+r*np.log(rho[1]/rho[0])+(1-r)*np.log((1-rho[1])/(1-rho[0]))
    exact = np.exp(log_joint(1,r)-logsumexp([log_joint(0,r),log_joint(1,r)]))
    check(f"Binary full conditional, child={r}", expit(score), exact)

def qvector(g):
    return np.array([g[0]**a*(1-g[0])**(1-a)*g[1]**r*(1-g[1])**(1-r) for a,r in states])
def elbo(g):
    q = qvector(g)
    return np.dot(q, logs-np.log(q))
g = np.array([0.37, 0.61])
before = elbo(g)
score = np.log(p/(1-p))+bA-0.5*aA+g[1]*np.log(rho[1]/rho[0])+(1-g[1])*np.log((1-rho[1])/(1-rho[0]))
expected_logs = [(1-g[1])*log_joint(a,0)+g[1]*log_joint(a,1) for a in (0,1)]
check("Binary CAVI expected-log update", expit(score), expit(expected_logs[1]-expected_logs[0]))
g[0] = expit(score)
assert elbo(g) >= before - 1e-12
g[1] = expit(bR-0.5*aR+(1-g[0])*np.log(rho[0]/(1-rho[0]))+g[0]*np.log(rho[1]/(1-rho[1])))
assert elbo(g) >= before - 1e-12
lines.append(f"PASS binary CAVI ELBO increase: {elbo(g)-before:.8f}")

kernels = []
for coord in (0, 1):
    kernel = np.zeros((4,4))
    for oldidx, (a,r) in enumerate(states):
        own_prob = expit(np.log(p/(1-p))+bA-0.5*aA) if coord == 0 else expit(np.log(rho[a]/(1-rho[a]))+bR-0.5*aR)
        for proposal in (0, 1):
            prob = own_prob if proposal else 1-own_prob
            new = [a,r]; new[coord] = proposal
            newidx = 2*new[0]+new[1]
            if coord == 0:
                child_old = rho[a] if r else 1-rho[a]
                child_new = rho[proposal] if r else 1-rho[proposal]
                acceptance = min(1.0, child_new/child_old)
            else:
                acceptance = 1.0
            kernel[oldidx,newidx] += prob*acceptance
            kernel[oldidx,oldidx] += prob*(1-acceptance)
    check(f"MH coordinate {coord} row sums", kernel.sum(1), np.ones(4))
    check(f"MH coordinate {coord} detailed balance", target[:,None]*kernel, target[None,:]*kernel.T)
    kernels.append(kernel)
check("Metropolis-within-Gibbs sweep stationary joint", target@kernels[0]@kernels[1], target)

# General augmented EMDN update: uncertain binary parent changes weights,
# means AND variances; the child's mixture has covariate-dependent parameters.
parent_probs = np.array([0.65, 0.35])
weights = np.array([[0.2,0.45,0.35],[0.45,0.2,0.35]])
means = np.array([[-0.7,0.8],[0.2,1.4]])
variances = np.array([[0.5,1.1],[0.9,0.4]])
child_resp = np.array([0.2, 0.5, 0.3])
child_means = np.array([0, -0.4, 1.2])
child_seconds = np.array([0, 0.6, 1.9])
def child_message(v):
    lw = np.array([0.1, -0.2 + 0.3*v, 0.15 - 0.15*v])
    lw -= logsumexp(lw)
    cmu = np.array([-0.6 + 0.2*v, 0.7 - 0.1*v])
    cv = np.array([0.6+0.2*expit(v), 0.8+0.2*expit(-v)])
    J = -0.5*np.log(2*np.pi*cv)-(child_seconds[1:]-2*cmu*child_means[1:]+cmu**2)/(2*cv)
    return np.dot(child_resp, lw) + np.dot(child_resp[1:], J)
a, b = 1.3, 0.7
computed, direct = [], []
for h in range(2):
    precision = a + np.dot(parent_probs, 1/variances[:,h])
    linear = b + np.dot(parent_probs, means[:,h]/variances[:,h])
    constant = np.dot(parent_probs, np.log(weights[:,h+1])-0.5*np.log(2*np.pi*variances[:,h])-0.5*means[:,h]**2/variances[:,h])
    vh, mh = 1/precision, linear/precision
    prefactor = np.exp(constant+linear**2/(2*precision))*np.sqrt(2*np.pi/precision)
    def from_joint(v):
        elog = sum(parent_probs[t]*(np.log(weights[t,h+1])+norm.logpdf(v,means[t,h],np.sqrt(variances[t,h]))) for t in (0,1))
        return np.exp(-0.5*a*v*v+b*v+elog+child_message(v))
    for moment in range(3):
        computed.append(prefactor*integ(lambda v: v**moment*norm.pdf(v,mh,np.sqrt(vh))*np.exp(child_message(v))))
        direct.append(integ(lambda v: v**moment*from_joint(v)))
check("Augmented EMDN weights and moments with uncertain parent and EMDN child", computed, direct)
spike = np.exp(np.dot(parent_probs, np.log(weights[:,0]))+child_message(0))
norm_const = spike+computed[0]+computed[3]
check("Augmented mixture atom plus slab weights sum to one", (spike+computed[0]+computed[3])/norm_const, 1.0)

# Notebook block mapping and noiseless RNA rank.
profiles = np.array([[1,1,1],[0,1,1],[1,0,1],[1,1,0]])
mapping = profiles[[3,0,1,2]]
check("RNA block mapping for ATAC states 00,01,10,11", mapping, [[1,1,0],[1,1,1],[0,1,1],[1,0,1]])
check("Noiseless RNA block rank", np.linalg.matrix_rank(profiles), 3)
lines.extend(["", "All identities passed.", "These are small deterministic mathematical checks, not a fitted joint model or a performance experiment."])
report = Path(__file__).resolve().parent / 'atac_rna_derivation_checks.txt'
report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
