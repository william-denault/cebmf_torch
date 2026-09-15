# Automatic self-covariate joint inference validation

The existing neural conditional mixtures and child-prior Metropolis correction
are now shared by the coupled ATAC/RNA sampler and the cEBMF matrix backend.
No linear-logistic CGB, VampPrior, exponential-mixture or VI solver was added.

## Checks run

- Full package suite: 353 passed, 1 skipped before the final integration guards.
- Final focused suite: 51 passed (30 joint-matrix tests and 21 ordered-covariate tests).
- Independent full-joint density ratios checked for all eight scalar learned
  priors on each axis, with missing entries and unequal known variances.
- Both axes enabled, MCEM fixed-draw objective acceptance, fixed parameter
  sampling, per-draw reconstruction moments, HMM on either opposite axis,
  fixed side-information enforcement, retained rank, and precision checked.
- Both new notebooks executed cell by cell with their default quick settings;
  plots were inspected. No simulated loadings/factors were used for fitting.

## Runnable examples

- examples/check_tree_consistency_joint.ipynb
- examples/ATAC_RNA_self_cov_joint.ipynb
- examples/ATAC_RNA_hmm_joint.ipynb (the existing fully coupled model)

The first two use the automatic cEBMF interface. Separate cEBMF calls treat
cross-modality side information as fixed; only JointATACRNA jointly samples
both modalities' latent loadings and propagates feedback between them.

## Tree diagnostics (one seed, short chains)

The corrected tree has seven generating programs but four leaf profiles;
its noiseless rank is at most four, so reconstruction does not identify the
original named tree. The baseline uses independent Gaussian mixtures and
therefore differs in prior family as well as inference method.

| Configuration | Signal RMSE | Held-out noisy RMSE |
|---|---:|---:|
| Independent baseline | 0.28166 | 1.26644 |
| Neural CGB loading hierarchy; independent feature mixtures | 0.22693 | 1.25905 |
| Two-slab sharp CGB loadings; spiked-EMDN feature hierarchy | 0.29646 | 1.29049 |

Settings and timings are in the corresponding tree_joint_*_diagnostic.json
files. These runs establish executability, not convergence, superiority or
identification. More conditioning did not improve this short-run comparison.

## Current limits

CPU; Gaussian observation likelihood; rank, noise, fixed covariates,
non-neural priors and preprocessing frozen after initialization. The matrix
backend supports the eight learned scalar families, norm, and HMM priors.
Unsupported prior families raise an error instead of falling back to an
uncorrected update. fit(maxit) learns priors for maxit rounds then samples;
iter_once() performs one sweep at fixed parameters. See the joint guide for
all settings and for how these semantics differ from variational fitting.
