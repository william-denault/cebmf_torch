"""Independent numerical checks for the manuscript, not a package solver.

Run with Python, NumPy and SciPy. Direct integration evaluates the original
prior times likelihood separately from the completed-square formulas.
"""

from pathlib import Path
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, expit, log_ndtr, logsumexp
from scipy.stats import norm


REPORT = ["Mixture-family manuscript verification", "2026-09-15", ""]


def check(name, actual, expected, atol=2e-9, rtol=2e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
    REPORT.append(f"PASS {name}: max absolute error {np.max(np.abs(actual-expected)):.3e}")


def integrate(fn, low, high):
    return quad(fn, low, high, epsabs=1e-11, epsrel=1e-11, limit=250)[0]


def gaussian(a, b, mu, variance):
    t = 1 + a * variance
    v = variance / t
    m = (mu + b * variance) / t
    log_i = -0.5 * np.log(t) + (-0.5*a*mu**2+b*mu+0.5*b*b*variance)/t
    return log_i, m, m*m+v


def exponential(a, b, rate, sign=1):
    if a == 0:
        assert b == 0
        return 0., sign/rate, 2/rate**2
    z = (sign*b-rate)/np.sqrt(a)
    log_i = np.log(rate) + 0.5*np.log(2*np.pi/a) + 0.5*z*z + log_ndtr(z)
    if z < 0:
        log_i = np.log(rate) + 0.5*np.log(np.pi/(2*a)) + np.log(erfcx(-z/np.sqrt(2)))
    if z < -35:
        # Expand exp(-t^2/2) after rescaling the tail integral by x=-z.
        # Ratios of these asymptotic integrals avoid subtracting O(z^2)
        # quantities to recover an O(1/z^2) second moment.
        x = -z
        series = [
            sum((-1)**k * math.factorial(n+2*k)
                / (2**k * math.factorial(k) * x**(2*k)) for k in range(9))
            for n in range(3)
        ]
        return (log_i, sign*series[1]/(x*series[0]*np.sqrt(a)),
                series[2]/(x*x*series[0]*a))
    r = np.exp(norm.logpdf(z)-log_ndtr(z))
    return log_i, sign*(z+r)/np.sqrt(a), (1+z*z+z*r)/a


def original_gaussian_integrals(a, b, mu, variance):
    # Integrate in prior standard-deviation units to resolve narrow slabs.
    sd = np.sqrt(variance)
    def f(t, power):
        value = mu + sd*t
        return value**power * np.exp(-0.5*a*value**2+b*value) * norm.pdf(t)
    return np.array([integrate(lambda t: f(t, p), -12., 12.) for p in range(3)])


def original_exp_integrals(a, b, rate, sign=1):
    # This change of units resolves mass near zero in negative-likelihood tails.
    scale = 1/max(rate-sign*b, np.sqrt(a), 0.2)
    def f(t, power):
        value = sign*scale*t
        return (value**power * rate*scale
                * np.exp(-0.5*a*value**2+b*value-rate*scale*t))
    return np.array([integrate(lambda t: f(t, p), 0., np.inf) for p in range(3)])


for a, b in [(0., 0.), (0.3, -1.2), (2., 1.1), (8., -2.)]:
    for mu, variance in [(-0.6, .7), (1.2, .04), (.5, 1e-10)]:
        log_i, m, m2 = gaussian(a, b, mu, variance)
        direct = original_gaussian_integrals(a, b, mu, variance)
        check(f"Gaussian evidence and moments a={a}, mu={mu}, var={variance}",
              [log_i, m, m2], [np.log(direct[0]), direct[1]/direct[0], direct[2]/direct[0]])

for a, b in [(0., 0.), (.3, -.7), (2., 2.), (1., -19.), (1., -79.)]:
    for rate in [.4, 1.7]:
        for sign in [-1, 1]:
            # Keep positive-tail evidence within ordinary floating-point range.
            if b < -10 and sign == -1:
                continue
            log_i, m, m2 = exponential(a, b, rate, sign)
            direct = original_exp_integrals(a, b, rate, sign)
            check(f"Exponential evidence and moments a={a}, b={b}, rate={rate}, sign={sign}",
                  [log_i, m, m2],
                  [np.log(direct[0]), direct[1]/direct[0], direct[2]/direct[0]],
                  atol=5e-9, rtol=2e-7)

# The very far-tail normalizer is checked independently of unstable moment
# subtraction. The manuscript explicitly requires stable moment routines there.
log_i = exponential(1., -999., 1.)[0]
direct = original_exp_integrals(1., -999., 1.)
check("Exponential log evidence at z=-1000", log_i, np.log(direct[0]))
check("Exponential zero-precision evidence limit",
      exponential(1e-10, 0., 1.3)[0], 0., atol=1e-9)


# Gaussian + exponential + nonzero and zero atoms, with and without children.
# Each item's stored moments are obtained analytically; the separate integrals
# below evaluate the original component measures.
components = [
    ("atom", 0., 0.), ("atom", 1., 0.),
    ("normal", -.6, .7), ("normal", .9, .15),
    ("exp", 1.2, 1), ("exp", .8, -1),
]
weights = np.array([.13, .07, .21, .19, .25, .15])
a, b = 1.4, .8


def component_integrals(item, log_tilt, power=0, prior_only=False):
    kind, x, y = item
    def likelihood(v):
        return 0. if prior_only else -0.5*a*v*v+b*v
    if kind == "atom":
        return x**power*np.exp(likelihood(x)+log_tilt(x))
    if kind == "normal":
        def f(t):
            v = x+np.sqrt(y)*t
            return v**power*np.exp(likelihood(v)+log_tilt(v))*norm.pdf(t)
        return integrate(f, -12., 12.)
    def f(t):
        v = y*t
        return v**power*x*np.exp(-x*t+likelihood(v)+log_tilt(v))
    return integrate(f, 0., np.inf)


def child_log(v):
    # Child has a parent-dependent weight, rate AND Gaussian parameters.
    gate = expit(.2+.5*np.tanh(v))
    rate = np.exp(.1+.3*np.tanh(v))
    sd = np.exp(-.3+.1*np.tanh(v))
    mu = .4+.2*v*v
    return np.log(gate)+np.log(rate)-rate*.8 + norm.logpdf(.3, mu, sd)


log_i = []
for kind, x, y in components:
    if kind == "atom":
        log_i.append(-.5*a*x*x+b*x)
    elif kind == "normal":
        log_i.append(gaussian(a,b,x,y)[0])
    else:
        log_i.append(exponential(a,b,x,y)[0])
log_i = np.array(log_i)

for name, labels in [
    ("Gaussian mixture", [2,3]), ("Gaussian mixture plus spike", [0,2,3]),
    ("one-slab CGB", [0,3]), ("two-slab CGB", [0,2,3]),
    ("two-slab without spike", [2,3]), ("exponential mixture", [4,5]),
    ("exponential mixture plus spike", [0,4,5]),
]:
    w=weights[labels]/weights[labels].sum()
    evidence=np.exp(log_i[labels])
    original=np.array([component_integrals(components[j],lambda v: 0.)
                       for j in labels])
    check(f"{name} normalized normal-means weights",
          w*evidence/(w@evidence),w*original/(w@original))

for has_children in [False, True]:
    tilt = child_log if has_children else lambda v: 0.
    direct = np.array([[component_integrals(item, tilt, p) for p in range(3)]
                       for item in components])
    # Independently integrate completed-square component posteriors.
    tilted = []
    for item, evidence in zip(components, log_i):
        kind, x, y = item
        if kind == "atom":
            js = np.array([x**p*np.exp(tilt(x)) for p in range(3)])
        elif kind == "normal":
            _, m, m2 = gaussian(a,b,x,y)
            sd = np.sqrt(m2-m*m)
            js = np.array([integrate(lambda z: (m+sd*z)**p * np.exp(tilt(m+sd*z))
                                    * norm.pdf(z), -12, 12) for p in range(3)])
        else:
            mean = (y*b-x)/a
            sd = 1/np.sqrt(a)
            lower = -mean/sd
            js = np.array([integrate(
                lambda t: (y*t)**p * np.exp(tilt(y*t))
                * norm.pdf((t-mean)/sd) / (sd*norm.sf(lower)),
                0, np.inf) for p in range(3)])
        tilted.append(np.exp(evidence)*js)
    tilted = np.array(tilted)
    check(f"Mixed-family tilted component integrals, children={has_children}", tilted, direct)
    z = weights@direct[:,0]
    check(f"Mixed-family normalized masses and moments, children={has_children}",
          np.r_[weights*tilted[:,0]/(weights@tilted[:,0]),
                weights@tilted[:,1:]/(weights@tilted[:,0])],
          np.r_[weights*direct[:,0]/z, weights@direct[:,1:]/z])


# Check the Metropolis cancellation against independently evaluated complete
# densities on every target label pair, including exact atoms.
def prior_component_log(v, label):
    kind, x, y = components[label]
    if kind == "atom":
        assert v == x
        return np.log(weights[label])
    if kind == "normal":
        return np.log(weights[label])+norm.logpdf(v,x,np.sqrt(y))
    assert y*v > 0
    return np.log(weights[label])+np.log(x)-x*y*v


values = [0., 1., -.3, .6, .5, -.8]
qnorm = logsumexp(np.log(weights)+log_i)
for j, old in enumerate(values):
    for k, new in enumerate(values):
        old_q = -.5*a*old**2+b*old+prior_component_log(old,j)-qnorm
        new_q = -.5*a*new**2+b*new+prior_component_log(new,k)-qnorm
        old_joint = -.5*a*old**2+b*old+prior_component_log(old,j)+child_log(old)
        new_joint = -.5*a*new**2+b*new+prior_component_log(new,k)+child_log(new)
        full_ratio = new_joint-old_joint+old_q-new_q
        check(f"Augmented MH ratio labels {j}->{k}", child_log(new)-child_log(old), full_ratio)
        check(f"Augmented detailed balance labels {j}->{k}",
              old_joint+new_q+min(0,full_ratio),
              new_joint+old_q+min(0,-full_ratio))


# Exponential augmented CAVI with uncertain parents: direct expectation of the
# complete log prior versus the derived expected-rate quadratic coefficients.
parent_values = np.array([-.9,.25,1.4])
parent_prob = np.array([.2,.35,.45])
rates = np.exp(.1+.6*parent_values)
gates = expit(-.2+.5*parent_values)
mean_rate = parent_prob@rates
c = parent_prob@(np.log(gates)+np.log(rates))
for precision, linear in [(0.,0.), (1.3,.9)]:
    B = linear-mean_rate
    def direct_exp_vi(v):
        return (-.5*precision*v*v+linear*v
                +parent_prob@(np.log(gates)+np.log(rates)-rates*v))
    for v in [.01,.4,1.8]:
        check(f"Exponential VI expected log prior a={precision}, v={v}",
              -.5*precision*v*v+B*v+c, direct_exp_vi(v))
    if precision == 0:
        H = 1/mean_rate
        qdensity = lambda v: mean_rate*np.exp(-mean_rate*v)
    else:
        mean, sd = B/precision, 1/np.sqrt(precision)
        H = np.sqrt(2*np.pi/precision)*np.exp(B*B/(2*precision))*norm.cdf(B/np.sqrt(precision))
        qdensity = lambda v: norm.pdf((v-mean)/sd)/(sd*norm.cdf(mean/sd))
    direct = np.array([integrate(lambda v: v**p*np.exp(direct_exp_vi(v)+child_log(v)),
                                0, np.inf) for p in range(3)])
    derived = np.exp(c)*H*np.array([integrate(lambda v: v**p*qdensity(v)*np.exp(child_log(v)),
                                            0,np.inf) for p in range(3)])
    check(f"Exponential VI normalized tilted integrals a={precision}", derived, direct)

# Exponential child expectation and parent score include its rate normalizer.
v=.35
rate=lambda x: np.exp(.1+.3*np.tanh(x))
gate=lambda x: expit(.2+.5*np.tanh(x))
child_values=np.array([.2,.7,1.5])
child_prob=np.array([.25,.5,.25])
actual=sum(p*(np.log(gate(v))+np.log(rate(v))-rate(v)*x)
           for p,x in zip(child_prob,child_values))
check("Exponential child VI expected log density", actual,
      np.log(gate(v))+np.log(rate(v))-rate(v)*(child_prob@child_values))
eps=1e-5
fn=lambda x: np.log(gate(x))+np.log(rate(x))-rate(x)*.8
expected=(1-gate(v))*.5/np.cosh(v)**2 + (1-.8*rate(v))*.3/np.cosh(v)**2
check("Exponential child parent derivative", (fn(v+eps)-fn(v-eps))/(2*eps),expected)

# A complete spike-plus-two-exponential variational coordinate, with all
# parameters depending on uncertain parents; compare normalized masses.
logits=np.stack([np.full(3,-.2),.3*parent_values,-.4*parent_values],axis=1)
logw=logits-logsumexp(logits,axis=1,keepdims=True)
rates2=np.stack([np.exp(.1+.6*parent_values),np.exp(-.2-.3*parent_values)],axis=1)
for precision,linear in [(0.,0.),(1.3,.9)]:
    expected_logs=parent_prob@logw
    analytic=[np.exp(expected_logs[0]+child_log(0.))]
    direct=[np.exp(sum(parent_prob*logw[:,0])+child_log(0.))]
    for j in range(2):
        mean_rate=parent_prob@rates2[:,j]
        const=expected_logs[j+1]+parent_prob@np.log(rates2[:,j])
        def density(v):
            return np.exp(-.5*precision*v*v+linear*v+
                          sum(parent_prob*(logw[:,j+1]+np.log(rates2[:,j])-rates2[:,j]*v))
                          +child_log(v))
        direct.append(integrate(density,0,np.inf))
        if precision==0:
            mass=np.exp(const)/mean_rate
            q=lambda v: mean_rate*np.exp(-mean_rate*v)
        else:
            mean=(linear-mean_rate)/precision
            sd=1/np.sqrt(precision)
            mass=np.exp(const)*np.sqrt(2*np.pi/precision)*np.exp(.5*precision*mean**2)*norm.cdf(mean/sd)
            q=lambda v: norm.pdf((v-mean)/sd)/(sd*norm.cdf(mean/sd))
        analytic.append(mass*integrate(lambda v: q(v)*np.exp(child_log(v)),0,np.inf))
    check(f"Spike and two-exponential VI probabilities a={precision}",
          np.array(analytic)/sum(analytic),np.array(direct)/sum(direct))


# Global M-step: compare closed forms to numerical maximization of original
# weighted complete-data likelihoods, including constrained CGB mean.
from scipy.optimize import minimize, minimize_scalar

draws = np.array([.1,.4,.9,1.6,2.1])
resp = np.array([.2,.8,.6,.5,.3])
n = resp.sum()
mu = resp@draws/n
variance = resp@(draws-mu)**2/n
gauss_loss=lambda par: -sum(resp*norm.logpdf(draws,par[0],np.exp(par[1])))
fit=minimize(gauss_loss,[.8,-.5],method="BFGS",tol=1e-10)
check("Gaussian M-step objective at numerical optimum",
      gauss_loss([mu,.5*np.log(variance)]),fit.fun,atol=1e-8)
rate_hat=n/(resp@draws)
exp_loss=lambda logr: -sum(resp*(logr-np.exp(logr)*draws))
fit_exp=minimize_scalar(exp_loss,bounds=(-5,5),method="bounded",options={"xatol":1e-12})
check("Exponential M-step rate",rate_hat,np.exp(fit_exp.x),atol=1e-7)
negative_mean=-.05
negative_var=resp@(draws-negative_mean)**2/n
fit_neg=minimize(gauss_loss,[-.2,.2],bounds=[(None,-.05),(None,None)],method="L-BFGS-B",
                 options={"ftol":1e-13,"gtol":1e-9})
check("Sign-constrained CGB Gaussian objective",
      gauss_loss([negative_mean,.5*np.log(negative_var)]),fit_neg.fun,atol=1e-8)


# TwinEB vector mixture vs its exact chain-rule gates: direct marginal
# evaluation is independent of the sequential conditional calculation.
rng=np.random.default_rng(7)
means=rng.normal(size=(3,4))
scales=np.exp(rng.normal(scale=.3,size=(3,4)))
log_weights=np.log([.2,.3,.5])
for idx, z in enumerate(rng.normal(size=(10,4))):
    component_logs=norm.logpdf(z[None,:],means,scales)
    direct=logsumexp(log_weights+component_logs.sum(axis=1))
    gate_log=log_weights.copy()
    sequential=0.
    for k in range(4):
        conditional=logsumexp(gate_log+component_logs[:,k])
        sequential+=conditional
        gate_log=gate_log+component_logs[:,k]-conditional
    check(f"TwinEB autoregressive identity vector {idx}",sequential,direct)

count=sum(line.startswith("PASS") for line in REPORT)
REPORT += ["", f"{count} numerical checks passed.",
           "These checks validate identities, not MCMC mixing or empirical superiority."]
path=Path(__file__).with_name("mixture_family_derivation_checks.txt")
path.write_text("\n".join(REPORT)+"\n",encoding="utf-8",newline="\n")
print("\n".join(REPORT))
