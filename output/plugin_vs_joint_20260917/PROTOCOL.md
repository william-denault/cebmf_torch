# Plug-in versus conditional fitting: protocol

## Primary comparison

Thirty paired simulations, seeds 1–30, using the user's corrected tree:
1,000 rows, 200 features, independent masks for all seven profiles, four
contiguous leaf groups, Gaussian noise SD 1.25, and initial K=6.
Feature priors are the package's default normal-mixture ASH (`prior_F="norm"`).
The two pipelines are:

* Cold: `spiked_emdn`, penalty 1.1, 30 sweeps.
* Warm: `cgb`, penalty 1.051, 10 sweeps, followed by `cgb_sharp_2`,
  penalty 1.0511 and omega 0.01, for 20 sweeps. Each method uses its own
  fitted CGB precursor, including its resulting rank.

Current means the production quadratic conditional fitter, with 32 parent
points and 24 Gaussian quadrature points. Plug-in uses the existing scalar
cEBNM update with the latest earlier-factor means as covariates; it is
enabled only in a benchmark subclass. No production algorithm is changed.

The same GPU SVD initialization is supplied to both methods for each seed.
Neural RNG seeds are reset independently for each stage, so the cold fit's
RNG consumption cannot change the warm pipeline's initialization. This is
a deliberate matching step beyond the literal user loop. The methods still
have different neural initializers and optimizers. Truth is used only for
evaluation, never fitting, orientation, stopping, or parameter selection.
The primary endpoint is the final noiseless-signal RMSE, not the best sweep.
All trajectories, initialization changes, ranks and noise estimates are saved.

The original keyword arguments retain unequal internal defaults: scalar
spiked EMDN uses batch 512, conditional fitting uses batch 128. Both use
10 epochs per coordinate. Therefore this is an as-configured comparison,
not a claim that every optimizer step is matched.

## Predetermined diagnostic subset

Seeds 1–5, same data and full 30/10/20 sweeps, batch 128 and fixed rank 6:

| Method | Parent inputs | Child feedback | Fitter |
|---|---|---|---|
| current | integrated | quadratic | modern conditional |
| parent_only | integrated | disabled | modern conditional |
| mean_feedback | posterior means | quadratic | modern conditional |
| mean_only | posterior means | disabled | modern conditional |
| plugin | posterior means | absent | original scalar cEBNM |

The modern controls retain the same graph setup, network initialization,
optimizer and prior families. Current primary runs on these five seeds
already use batch 128 and fixed rank 6 internally and can serve as the
current reference; the controls deliberately do not duplicate them.
Comparing modern controls isolates parent integration and child feedback.
Comparing `mean_only` with `plugin` exposes remaining implementation changes.
Controls begin after the common modern graph setup, so they do not isolate
its initial factor replacement. That issue is measured separately.

## Additional diagnostics

* Seeds 1–5: warm-stage fits from the same current CGB precursor, keeping
  rank six. Compare scalar plug-in directly, scalar plug-in after the
  modern setup's mean replacement, and modern fitting with slab scales
  fixed at their initial values. Fixing scales is not equivalent to the
  legacy repeated-omega moment update.
  The plug-in restart after mean replacement also initializes its noise
  estimate and posterior moments afresh. It tests recovery under the
  scalar solver's normal initialization, not the isolated causal effect
  of replacing means inside an otherwise identical modern fit.
* Seeds 1–5: frozen-fit coordinate checks at eight prespecified rows and
  coordinates 0, 3 and 5, increasing quadrature nodes 24→64→96 and parent
  samples 32→256. These test local numerical sensitivity, not complete
  alternative optimization trajectories.
* After observing the seed-7 instability: replay with coordinate-level
  objective checks, save the first large deterioration, and compare
  alternative updates from that identical saved state. This is a targeted
  debugging experiment, not an additional independent simulation replicate.
* Post-hoc seeds 1 and 7: repeat the full pipelines with fixed K=4 and
  batch 128 for both methods to examine the role of redundant factors.
  This uses knowledge of the simulation rank and is not part of the primary
  as-configured comparison or its confidence intervals.

## Execution and provenance

Python: the user-supplied `cebmf-rocm` environment, PyTorch
2.12.0+rocm7.14.1, AMD Radeon 8060S. Source hashes and frozen source copies
are recorded in each manifest. Completed stages are resumable. Incomplete
stages are restarted rather than counted as completed fits.

Seed 1 of the primary comparison ran alone and supplies uncontended timing.
Remaining seeds and diagnostic jobs run in independent processes sharing
the GPU. Their elapsed times are retained for provenance but must not be
interpreted as uncontended algorithm speed comparisons. Independent RNG
states and matched data make their accuracy comparisons valid.

GPU deterministic-algorithm enforcement is not enabled, matching normal
package usage. A diagnostic replay of seed 7 began with identical initial
RMSE but developed small numerical differences and failed at a different
sweep. Saved checkpoints, rather than an assertion of bitwise replay,
anchor the numerical failure comparisons. Each primary seed contributes
one training run per method; training-repeat variability is not separately
estimated by the seed-bootstrap interval.

Fitting runs on the GPU; metric reporting and checkpoint serialization are
explicit transfer boundaries. The default adaptive ASH scale grid still
has synchronization during grid sizing, as documented by the package;
this benchmark does not claim a completely synchronization-free loop.

Report paired differences (current minus plug-in), sample mean/SD, median,
win counts, and paired bootstrap confidence intervals over simulation seeds.
These intervals describe this simulation, not general superiority of one
inference method. Do not compare the two fitters' objective values directly:
they represent different objectives and regularization conventions.

Nonfinite diagnostic fits are recorded in `failures.json` and are not rerun
until they happen to succeed. Their partial trajectories are retained, but
their last successful sweep is not treated as a final endpoint. Independent
remaining stages are launched separately if an earlier stage's exception
stops its worker. Tables show completed endpoints and failures separately.
