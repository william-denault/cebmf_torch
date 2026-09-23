# Conditional-loading implementation and cleanup

Completed 16 September 2026. This supersedes the implementation status in the
earlier audit; it does not rewrite the audit's observations about the old code.

Subsequent numerical and CUDA cleanup: see [NUMERICAL_DEVICE_AUDIT.md](./NUMERICAL_DEVICE_AUDIT.md)
and `validation.json` for the latest checks. The validation counts below
describe the first implementation pass.

## Model and learning

`self_row_cov=True` now fits the directed prior `p(L) = product_k p(L_k | L_<k)`
with scalar variational coordinates. Each update integrates uncertain parents,
includes feedback from child prior terms, and profiles the loading posterior
while learning its prior parameters. Feature priors and noise are refitted by
the ordinary cEBMF routines. The main example uses ash features (`prior_F="norm"`).

If there are no latent covariate edges, the public API calls the existing cEBMF
update path. This preserves its finite optimizer behavior, initialization,
schedule and prior options, in addition to the mathematical normal-means limit.
The explicit legacy `joint_kwargs` route remains for reference experiments and
emits a deprecation warning. The flag alone never selects that sampler.

`fit_joint(atac, rna)` couples two aligned cEBMF objects through
`p(L_ATAC) p(L_RNA | L_ATAC)`. A union of cell IDs supports partially paired data.
Missing terminal loading branches integrate out of the fitting objective;
missing upstream loadings remain latent when observed descendants need them.
Prediction of collapsed branches is a separate ancestral integration step.

## Maintained entry points

- `examples/ATAC_RNA_joint.ipynb` and the equivalent `run_atac_rna_joint.py`.
- `examples/tree_joint_simple.ipynb` for a single matrix.
- `docs/source/joint_inference.rst` for the API and limitations.
- `output/pdf/atac_rna_joint_inference.pdf` for the full draft and positioning.
- `output/pdf/conditional_loading_prior_derivation.pdf` for the short derivation.

Older notebooks are preserved in `examples/archive/`; earlier manuscript
supplements are in `output/pdf/archive/`. A snapshot of the pre-cleanup files
is retained at `tmp/before_conditional_cleanup.zip`.

## Validation performed

- Full suite: **406 passed, 1 skipped** (62 warnings), about 158 seconds.
- Analytic checks compare mixture evidence, moments and entropy with exact
  normal-means results, and a quadratic child tilt and its parameter gradient
  with a closed-form Gaussian result.
- Regressions cover the exact single-factor cEBMF limit, ordinary fixed
  covariates, both axes, noise updates, missing modalities, child feedback,
  stored variational states and option validation.
- All eight supported conditional prior families completed a finite one-sweep
  smoke check on CPU.
- Both maintained notebooks executed all code cells successfully. The ATAC-RNA
  notebook uses 120 cells (80 paired, 20 ATAC only, 20 RNA only) and took about
  35 seconds for eight sweeps with one CPU thread. Its saved figures and metrics
  come from that execution; simulation truth never enters fitting.
- Both rebuilt PDFs were rendered and all pages visually inspected.

## Limits and interpretation

The variational posterior factorizes across loading coordinates, although the
generative prior is dependent. Quadrature handles each target loading; fixed
Sobol integration handles multiple uncertain parents. Finite integration and
neural optimization do not guarantee a monotone exact ELBO. Connected graphs
retain fixed ranks. Effective slab variances in sharp CGB families are optimized
in the profile objective; omega sets their initialization. No performance
superiority or resolution of the original tree benchmark has been established.

The draft retains conditional program priors as its proposed contribution. It
removes the incorrect claim that the ATAC-RNA toy is impossible for every shared
linear representation: four shared state indicators can reconstruct both views.
Matched predictive benchmarks and integration-sensitivity checks remain needed.
