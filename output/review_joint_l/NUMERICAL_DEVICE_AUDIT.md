# Adversarial review of the conditional-loading update

16 September 2026. Current implementation review, following the original
sampler audit and the controlled tree benchmark.

## Conclusion

The model remains **p(L) = product_k p(L_k | L_<k)** with ash features. The
profiled variational coordinate is mathematically correct for the specified
factorized, component-augmented posterior. Removing latent-parent and child
terms gives the ordinary cEBNM marginal-likelihood subproblem, consistent
with the joint prior/posterior update in [cEBMF, Appendix A.1/A.3](https://arxiv.org/html/2505.11639v2).
The public no-edge route reuses the ordinary implementation. A dependent
generative prior does not make this a correlated posterior approximation.

**The implementation did contain numerical bugs.** Fixing them is necessary;
it is not evidence that joint learning always predicts better.

## Counterexamples and corrections

1. **Narrow off-center Gaussian:** spike probability .2, slab N(10, 1e-8),
   likelihood `a=2, b=3`. Float32 natural-parameter completion returned a
   loading mean about `.0125`, while the analytic answer is `1.59e-29`.
   Large quadratic terms canceled. The revised implementation evaluates
   centered potentials, masses and entropy. Tests cover float32/float64,
   zero likelihood information and very large likelihood precision.
2. **Child variance:** a coordinate supported near `10 ± .001` has positive
   variance, but float32 `E[V²]-E[V]²` returns zero. Child densities and the
   prior/entropy objective now use directly centered component variances.
3. **Ordinary spike-normal posterior:** both likelihood densities can
   underflow for an extreme observation. Flooring their sum then assigns
   the wrong component. Component odds now use log-sum-exp.
4. **Normal-mixture tails:** silently clipping log densities at -100000
   can make very different components equally likely. The default no longer
   clips; the low-level explicit clipping option remains available.
5. **Saved models:** model IDs change after unpickling. Graph lookup caches
   are rebuilt on load, including row and column graphs; earlier benchmark
   checkpoints acquire the new cached indices automatically.

## Stable derivation used in code

For each Gaussian component, let

```
t = E[1/s(P)^2]
c = E[mu(P)/s(P)^2] / t
h = E[log pi(P)] - .5 E[log(2*pi*s(P)^2) + (mu(P)-c)^2/s(P)^2]
A(v) = h - .5*t*(v-c)^2
alpha = a+t; r = t/alpha
m = r*c + b/alpha; V = 1/alpha
B = h + r*(-.5*a*c^2+b*c) + b^2/(2*alpha) + .5*log(2*pi/alpha)
```

The slab mass is `exp(B) E_N(m,V)[exp D(v)]`. The zero atom contributes
`exp(E log pi_0 + D(0))`. The parent-dependent height `h` is retained:
normalizing `exp(E log prior)` and discarding that constant would be wrong.
Entropy uses the completed Gaussian density at standardized quadrature
nodes, not the unstable expanded quadratic. Child Gaussian log densities
use `Var(V|Z) + (E[V|Z]-mu(P))²`.

The child's sparsity penalty enters its parent's tilt. Its own penalty is
constant in its own loading and remains outside its log normalizer. For
partially observed modalities, the implemented regularizer is restricted
to active nodes after terminal missing branches are collapsed; it is not
a globally applied penalty on those eliminated branches.

## Attempts to falsify the derivation

- Exhaustively enumerate binary states and compare coordinate probabilities
  and full ELBO gains, with/without latent edges.
- Compare the optimized profile objective with the full enumerated ELBO
  at different prior parameters; the omitted constant varies by at most
  `6.7e-16` in the independent check.
- Compare Gaussian-mixture evidence, moments and continuous/label entropy
  with a separate analytic normal-means posterior.
- Compare a nonlinear logistic child tilt and its gradient with dense
  trapezoidal integration, using a different integration scheme.
- Compare implemented child feedback with direct integration over child
  values/labels, including covariate-dependent means, variances, penalties,
  and an entirely missing branch.
- Preserve exact no-edge dispatch and finite-update equality, and test
  save/reload/continuation for row, column and combined conditional graphs.

Tests: `tests/test_conditional_numerics.py`,
`tests/test_conditional_variational.py`, and
`output/review_joint_l/verify_replacement_derivation.py`.

## What the review does not prove

Finite quadrature and parent integration need sensitivity checks. Fixed
Sobol points reduce optimization noise but are still an approximation.
Different numerical expectations need not be exact derivatives of one
globally consistent discretized ELBO. Accepting a better local approximate
profile does **not** establish global exact-ELBO monotonicity. Original
scalar neural solvers also use finite, sometimes heuristic updates; the
no-edge compatibility requirement preserves that route, rather than
claiming it is an exact marginal-likelihood optimizer.

**Correction after inspecting saved checkpoints:** the tree benchmark used
float32 for observations and fitted loadings. The earlier statement that it
used float64 was incorrect; float64 was used for some scoring calculations.
The narrow-slab bug could therefore affect those fits. The counterexample
alone does not establish its contribution to the observed performance gap;
that requires matched reruns with and without the numerical fixes. Prior-family,
initialization and sign-orientation findings remain observations of the
recorded source version. The 60 saved main fits have not been rerun or
relabeled. Their source hashes remain in `output/tree_prior_benchmark/provenance.json`.

## Device cleanup and evidence

The conditional loop now uses device tensors for objectives/checkpoints,
cached indices and prediction samples, and CUDA-capturable Adam. Network
parameters allocate on the requested device. Neural scalar solvers retain
losses/parameters on-device and use device minibatching on CUDA. ASH EM
freezes its first converged iterate with a tensor flag. Histories remain
device tensors until explicit reporting. Initialization respects data
dtype and handles entirely missing columns without CPU index extraction.

Adaptive ASH grids still read one scalar array length on the host; scale
amplitudes stay on-device. Passing an explicit `scales` grid avoids that
decision and still relearns weights. Setup validation, CPU-generated Sobol
tables, pruning, optional L-BFGS, the HMM parameter optimizer, legacy
sampling, and reporting are not claimed to be synchronization-free.

`tests/test_device_contract.py` contains real CUDA-only sweep, optimizer,
cross-modality, profiler, memory and CPU/GPU agreement checks. This machine
has **PyTorch 2.11.0+cpu**, `torch.version.cuda=None`, and no available CUDA
device. Those tests are **skipped, not passed**. See
`docs/source/device_contract.rst` for the executable hardware checks.

The ordinary CPU suite and example executions are recorded in
`validation.json`. Ruff was unavailable; its attempted local installation
could not reach PyPI under the network restrictions. No lint pass is claimed.
