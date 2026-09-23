# First versus first-and-second posterior-moment covariates

## Question and comparison

Compare only the original scalar plug-in cEBMF algorithm, with two choices
of neural prior inputs for loading column k:

* `mean`: earlier posterior means, `L[:, :k]`.
* `mean_second`: concatenate `L[:, :k]` and `L2[:, :k]`, where
  `L2 = E[L^2]` is the raw posterior second moment. It is not `E[L]^2`.

The first factor has the same constant intercept input in both methods.
No current or later loading enters its own covariates. No truth, leaf label,
sampling, parent integration, or child-prior feedback is added to fitting.
The extra features include information about variance via
`Var(L) = E[L^2] - E[L]^2`; they also provide a squared-mean feature.
Thus this experiment cannot attribute a gain exclusively to variance
information rather than the expanded neural feature set.

## Fixed simulation and fitting protocol

Thirty paired seeds (1–30), the corrected four-leaf example: N=1000,
P=200, initial K=6, seven independent sparsity masks, contiguous groups,
noise SD 1.25. Truth is used for scoring only. The true signal rank is four.

Each seed shares the same SVD factors across methods. Each stage resets the
neural RNG to `100000 + 10*seed + stage_index`, with indices 0/1/2.
The added inputs increase first-layer parameter count and change neural
initialization draws; the networks cannot have identical input weights
when their input dimensions differ. Method order reverses on even seeds.

* Cold: `spiked_emdn`, penalty 1.1, 30 sweeps.
* Precursor: `cgb`, penalty 1.051, 10 sweeps.
* Warm: `cgb_sharp_2`, penalty 1.0511, omega 0.01, 20 sweeps, starting
  from that method's own precursor means and retained rank.
* Feature priors: normal-mixture ASH, its existing default penalty 10.
* Existing scalar network widths, batches and ten epochs per update;
  native sharp variance updates and factor pruning are preserved.
* Warm initialization carries L and F, as in the supplied example; both
  methods initially set L2=L^2 and F2=F^2, then update posterior moments.

Every primary endpoint uses the full requested sweep budget. No truth-based
early stopping or choice of the best sweep. A failed fit is retained in its
JSON with its exception; independent stages continue. Completed checkpoints
are resumable and a source/settings mismatch is rejected.

## What the ELBO means here

Freeze the fitted per-row prior densities and all neural input matrices X
at their fitted values. For the factorized posterior q, evaluate

\[
\mathcal E_X = E_q\log p(Y\mid L,F,\tau)
 -\sum_k KL(q(L_k)\Vert g_k(\cdot\mid X_k))
 -\sum_k KL(q(F_k)\Vert g^F_k).
\]

**Higher is better.** The package records its negative in `model.obj`.
The scalar cEBNM losses are unpenalized marginal negative log likelihoods;
the cached KLs follow the normal-means evidence identity. The Gaussian
data term is independently recomputed in float64 on the model device from
posterior first and second moments. The difference from `-model.obj[-1]`
is saved to expose numerical discrepancies.

We separately report the penalized score obtained by adding the learned
row-prior term `(penalty-1) * sum(log(pi0_i))` and the existing ASH prior
regularization. This matches their penalty forms; the repeated omega
variance-shrinkage rule is not claimed to maximize that score exactly.

These are **fitted frozen-covariate ELBOs**, not the joint autoregressive
ELBO involving `E_q log g_k(L_k | L_<k)`. Moment covariates are estimated
from the same data and are refreshed during fitting. The scores are useful
conditional-fit diagnostics; a higher score alone does not establish
better generalization, posterior calibration, or joint-model evidence.
Compare paired score differences together with noiseless reconstruction.
No penalty for all additional neural weights is included.

Because updates proceed from earlier to later factors, fitted input
snapshots match current moments after an unpruned sweep. Pruning can make
these snapshots stale until the next sweep; every score records this flag.
Such scores describe the retained priors from their last fits, not newly
evaluated neural inputs after pruning. Final endpoints must be checked
for this distinction. ELBO monotonicity is not assumed.
Training includes penalties that the unpenalized ELBO excludes, and uses
finite neural optimization. A downward step in that reported score alone
does not demonstrate an implementation error.

## Execution and checks

GPU: user-provided `cebmf-rocm` Python, PyTorch 2.12.0+rocm7.14.1,
AMD Radeon 8060S. Production files and benchmark sources are hashed and
copied under `primary/source_snapshot`. The production API is unchanged.

Seed 1 runs as a single benchmark worker and supplies a timing reference.
Later independent workers share the GPU; their elapsed times are recorded
but are not interpreted as isolated speed measurements. GPU deterministic
algorithms are not enforced, matching package usage. Paired bootstrap
intervals summarize these simulation seeds, not training-repeat variability.

Five CPU checks passed, including exact agreement of the mean-only fitter
with the original scalar path, and an independent Gaussian entropy/ELBO
calculation. Two actual-GPU checks passed. The additional covariate building
step has no host reads or device transfers; full-fit parameters and moments
remain on the GPU. Native pruning and adaptive ASH retain their existing
synchronization behavior. Reporting and checkpointing are explicit boundaries.

An additional independent audit integrates the marginal mixture KL, including
the exact spike contribution, for all three row priors and ASH. Across eight
small checks (both input modes), the maximum discrepancy from the evidence
identity is 4.5e-6, and refinement from 128 to 256 quadrature nodes changes
the integrated KL by less than 4e-15. This checks the score's algebra and
sign separately from whether a higher fitted score improves reconstruction.
