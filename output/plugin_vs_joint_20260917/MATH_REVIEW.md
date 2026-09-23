# Reconsidering the conditional-loading update

## What the user wants to model

The intended prior is a directed factorization

\[
p(L)=\prod_i\prod_{k=1}^{K}g_k(L_{ik}\mid L_{i,<k};\theta_k).
\]

The inputs are earlier factor coordinates of the same row (cell), not
loadings from other rows.

That is a valid model. Fitting each conditional prior using the latest
estimated earlier loadings is a reasonable empirical-Bayes **plug-in
approximation**. It retains the intended conditional dependence. It is
not necessary to integrate the parents or send feedback through every
descendant to define or use those conditional priors.

Conversely, a claim to optimize the full latent-variable model's joint
ELBO requires terms that the plug-in algorithm omits. This distinguishes
two fitting procedures; it does not establish which reconstructs better.
The previous implementation conflated a more elaborate objective with a
more useful estimator. That was not justified.

## The two coordinate updates

Write the expected residual likelihood for one loading as

\[
\ell_k(v)=-\tfrac12 a_kv^2+b_kv,\qquad
a_k=\sum_j\tau_{ij}E[F_{jk}^2],\quad
b_k=\sum_j\tau_{ij}R^k_{ij}E[F_{jk}].
\]

Here \(R^k_{ij}=Y_{ij}-\sum_{h\ne k}E[L_{ih}]E[F_{jh}]\), and the
likelihood precision \(\tau_{ij}\) is held fixed during this update.

For the plug-in procedure, let \(x_i=m_{i,<k}\), the latest earlier
posterior means. At this update, freeze \(x_i\). Then fit \(\theta_k\)
by ordinary cEBNM marginal likelihood and compute

\[
q_k(v)\propto\exp\{\ell_k(v)\}g_k(v\mid x_i;\theta_k).
\]

For Gaussian-mixture priors this is analytic after the prior fit. It still
uses posterior variances in \(a_k\), feature updates and noise updates.
It does **not** discard all uncertainty: it omits uncertainty in the
covariates supplied to the learned prior. Repeating this update is not,
in general, coordinate ascent for one fixed joint ELBO, since the frozen
covariates change between updates.

The original cEBMF result uses externally fixed covariates and reduces
its coordinate update to cEBNM (Appendix A.2.1, Proposition A.2).
With no latent covariate edges, the package dispatches to this existing
scalar update. The new conditional fitter is not needed for that case.
[Original cEBMF derivation](https://arxiv.org/html/2505.11639v1).

The implemented joint procedure introduces mixture labels \(Z_k\) and
factorizes \(q=\prod_kq_k(L_k,Z_k)\), independently across factors.
Its ideal coordinate update, before adding the explicit penalty below, is

\[
\log q_k(v,z)=\ell_k(v)
 +E_{q_{<k}}\log g_k(v,z\mid L_{<k})
 +\sum_{j:k\in\mathrm{pa}(j)}
 E_{q_{-k}}\log g_j(L_j,Z_j\mid L_{\mathrm{pa}(j)\setminus k},v)
 +C.
\]

The second term integrates uncertain parents. The third is feedback from
children. They are separate changes. One can remove either to diagnose
performance, but those ablations are not coordinate ascent for the same
joint ELBO. Setting parent variances to zero alone does not generally
remove child feedback. Removing latent edges does.

## Where uncertainty can actively hurt this approximation

Consider just \(V\mid U\sim N(wU,t^2)\), with fixed marginal moments
\(m_U,s_U^2\). Under a factorized variational distribution,

\[
E_{q_U}\log p(V\mid U)
=\log N(V;wm_U,t^2)-\frac{w^2s_U^2}{2t^2}.
\]

The extra term penalizes slopes when the parent is uncertain. It is a
correct expectation under this approximation, not an algebra mistake.
It also differs from **posterior predictive averaging**. In this Gaussian
example, if \(q(U)=N(m_U,s_U^2)\), forward averaging gives

\[
\int p(V\mid U)q(U)\,dU=N(V;wm_U,t^2+w^2s_U^2).
\]

The mean-field coordinate instead uses the unnormalized geometric average
\(\exp E\log p(V\mid U)\), whose Gaussian part still has variance
\(t^2\). Thus the current update does not simply add parent uncertainty
to the child's predictive variance. With fixed w and one Gaussian, its
extra constant cancels when normalizing q(V); it matters for learning w
and, in mixtures, for comparing components. These are different operations
and should not both be described vaguely as 'uncertainty propagation'.

But the full second moment is

\[
E[(V-wU)^2]=(m_V-wm_U)^2+s_V^2+w^2s_U^2
             -2w\operatorname{Cov}(U,V).
\]

The current family forces the covariance to zero. Consequently the
variance penalty can be retained while its covariance compensation is
lost. For a learned slope,

\[
w_{\rm full}=\frac{\sum_i(m_{Ui}m_{Vi}+\operatorname{Cov}_i(U,V))}
                   {\sum_i(m_{Ui}^2+s_{Ui}^2)},\qquad
w_{\rm MF}=\frac{\sum_i m_{Ui}m_{Vi}}
                 {\sum_i(m_{Ui}^2+s_{Ui}^2)}.
\]

Here the moments in the two formulas belong to their respective posterior
approximations; the variances need not match. Marginal means and variances
alone do not recover the omitted covariance. This is one reason that
calling this scheme 'full uncertainty propagation' would be misleading.
The independence limitation is a standard property of mean-field VI;
the concrete slope calculation above is an independent derivation.
[VI review, Section 2.4](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf).

An independently checked example uses \(U\sim N(0,1)\),
\(V\mid U\sim N(U,0.1)\), observation noise variance 1 for both
variables, and observed values (1,1). The exact posterior covariance
off-diagonal is 0.3125. One slope M-step gives 0.9861 with the exact
posterior, 0.8778 with mean-field, and 1.0476 with a point plug-in.
This illustrates a mechanism, not proof that plug-in always wins. In this
fixed-parameter Gaussian example the optimal mean-field posterior means
are actually exact; its variances and subsequent parameter update differ.

For neural mixture priors there is another restriction: the augmented
factorized family cannot let a child's component label depend on its
parent's posterior realization. Averaging component log densities across
parent values is a geometric averaging operation, not averaging the
conditional predictive distributions. It can favor stable, less
input-sensitive components. More Sobol points do not remove this
variational restriction.

## What the code does correctly, and what is not guaranteed

The residual sufficient statistics use second moments in the denominator.
The expected Gaussian prior coefficients retain the parent-uncertainty
constant. The augmented child score correctly uses child component masses,
means and variances. The Gaussian algebra for a genuinely quadratic child
term is analytic and independently tested. A point-parent/no-feedback
benchmark control reproduces ordinary mixture evidence for its current
parameters to double-precision tolerance.

However, the production default uses a **local**, component-specific Taylor
approximation to child feedback, at the previous component means. It clips
positive curvature to zero and freezes this approximation while fitting
the current prior. This ensures a proper Gaussian update when base
precisions are positive; it does not make the quadratic a global lower
bound. ReLU-region crossings, changing mixture weights and large steps can
invalidate its accuracy. Selecting the best inner surrogate objective does
not guarantee improvement in the actual nonlinear ELBO. In seed 1's cold
fit, the recorded negative objective increased in two sweeps, once by
1,905.25. Therefore a monotonic-ELBO claim would be false for this run.

More quadrature points only approximate the selected mean-field objective
more closely. They cannot repair its missing covariance, altered sharp
regularization, initialization damage or redundant factors. Frozen-fit
numerical checks assess local integration sensitivity; they cannot prove
that early optimization would follow the same trajectory.

### A captured failure, not only a theoretical concern

The primary seed-7 cold fit reached RMSE 0.08044 at sweep 25 and ended at
0.53122. A diagnostic replay, with small GPU numerical differences, captured
a failure at sweep 26, factor 5. The inner surrogate improved by 292.33,
while the evaluated negative objective rose by 1,191,078.34.

Starting each alternative from the **same saved pre-update state** gives:

| Coordinate update | Change in negative objective, evaluated with 256 parent points |
|---|---:|
| Quadratic, 32 parent points | +1,522,450.03 |
| Quadratic, 256 parent points | +1,976,254.78 |
| Quadrature, 24 nodes and 32 parent points | -2.09 |
| Quadrature, 96 nodes and 32 parent points | -2.09 |
| Feedback disabled, 32 parent points | +56.59 |

Positive changes are deterioration. These remain numerical objective
evaluations, not exact integrals. Their small integration discrepancies
cannot account for the million-unit deterioration. Increasing sampling
alone did not fix it; quadrature avoided this particular failure. Disabling
feedback does not optimize the same joint objective, as its positive
change also illustrates.

For one affected row, a component with only 0.083% mass had its Taylor
expansion at 2.87. The update moved its mean to -4.87, variance to 4.53,
and mass to 83.6%. The quadratic feedback at that new mean was +2.98;
evaluating the child model gave -315.35, with still worse values in the
new Gaussian's tails. Its local curvature was already negative (-0.148):
no positive-curvature clipping occurred for this component. The feature
column norm was only 0.00024, so the data weakly constrained this loading.

This directly refutes the suggestion that positive resulting precision
is a sufficient safeguard. It only makes the Gaussian integral finite.
An unconstrained Taylor extrapolation around a nearly unused component
can invent a favorable region of the objective. Any repaired quadratic
algorithm needs a justified bound or a trust-region/actual-objective
acceptance check, including changes of component mass and variance.
The current implementation has neither safeguard.

See `failure_trace_seed07/branches.json`, `failure_geometry.json`, and
`failure_geometry.png` for the saved-state comparison and visualization.

The spike penalty also deserves precise labeling. The joint code includes
\((\lambda-1)E\log\pi_0(L_{<k})\), including its effect on parents.
This defines a penalized latent-variable objective. A regularizer applied
to network predictions at fixed inputs does not automatically justify
using that regularizer as evidence about latent parent values.

## Differences that invalidate an 'only uncertainty changed' comparison

1. **Rank:** scalar backfitting can prune factors; the conditional graph
   fixes the initial rank. The seven-column tree truth has rank four.
2. **Sharpness:** scalar `cgb_sharp_2` repeatedly multiplies its variance
   estimate by `omega`. Conditional fitting defaults to freely learned
   effective slab variances; omega only initializes them. The scalar rule
   is a regularized moment update, not an exact variance M-step. Replacing
   it changes the estimator and can remove its useful clustering bias.
3. **Warm initialization:** graph construction replaces all supplied
   loading means by posteriors under newly initialized priors before any
   optimization. Narrow default sharp slabs initially centered at
   +/-softplus(0) can badly distort an otherwise good warm start.
4. **Training:** scalar spiked EMDN defaults to batch 512; the graph
   defaults to 128. Initial component locations/scales differ. The graph
   profiles a different objective, clips gradients and selects an epoch
   by its surrogate score. It also uses fixed consecutive batches, whereas
   scalar neural fitting shuffles rows each epoch. Here consecutive rows
   belong to the same leaf group, so batch order is a consequential
   optimizer difference. Scalar updates use their existing optimizers.
5. **Covariate normalization:** the graph freezes the initial standardizer;
   the scalar solvers recompute scaling at each fit. Earlier loading
   scales can move substantially during factor fitting.

The rank-four signal does not identify seven unique biological/tree
components. This benchmark scores reconstruction, not recovery of the
intended hierarchy or calibration of the full loading posterior.

## Consequence for the implementation decision

The controls do not support the opposite blanket claim that uncertainty
integration is intrinsically harmful. The current CGB precursor performs
well, and removing parent integration while retaining quadratic feedback
makes the diagnostic cold fits markedly unstable. That hybrid is not a
coherent coordinate-ascent algorithm for the original joint objective.
Removing both modifications still leaves the modern sharp-prior gap:
matching a fixed-parameter evidence identity does not match the two
fitters' regularization, initialization or finite optimization schedules.

The shared-precursor controls show that the current CGB precursor is usable
by the scalar sharp-prior solver. The separate plug-in restart after
modern mean replacement also recovers, but this restart reinitializes
noise and posterior moments. It does not establish that modern mean
replacement is harmless. Likewise, freezing the modern initial slab
scales does not reproduce the legacy repeated-omega variance update.
These controls separate some explanations without identifying one sole
cause for the sharp-prior gap.

Treat the simple plug-in algorithm as a serious estimator and reference
implementation. Do not describe it as failing merely because it is not
full joint variational inference. Do not retain the new fitter as the
preferred practical algorithm solely because its derivation is more
elaborate. Use the paired benchmark and controlled ablations to decide.
If retaining both, expose the distinction plainly: plug-in conditional
cEBMF versus approximate joint variational fitting. Preserve the original
prior semantics, warm starts and pruning before claiming a controlled
upgrade. No production behavior was changed during this audit.

## Implementation references

The benchmark manifest hashes these files and retains copies under
`primary/source_snapshot/`. The relevant entry points are:

| Finding | Code |
|---|---|
| Plug-in covariates are strictly earlier loading columns | [`cEBMF._build_covariate_matrix`](../../src/cebmf_torch/cebmf/cebmf.py) |
| Graph initialization replaces supplied loading means; fixed batches and surrogate-based optimization | [`LoadingGraph.__init__` and `LoadingGraph.update`](../../src/cebmf_torch/cebmf/_conditional.py) |
| Conditional wrapper defaults to learning effective sharp slab scales | [`LoadingGraph._make_prior`](../../src/cebmf_torch/cebmf/_conditional.py) |
| Legacy sharp solver repeatedly multiplies its variance estimate by omega | [Scalar sharp-prior training loop](../../src/cebmf_torch/cebnm/cov_sharp_2gb_prior.py) |
| Quadratic curvature clipping and analytic component update | [`local_feedback` and `quadratic_coordinate`](../../src/cebmf_torch/cebmf/_quadratic.py) |
| Scalar solvers shuffle rows on the existing device | [`density_batches`](../../src/cebmf_torch/utils/batching.py) |
