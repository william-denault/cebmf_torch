# Conditional ATAC-RNA factorization

The maintained example is [ATAC_RNA_joint.ipynb](examples/ATAC_RNA_joint.ipynb).
It constructs two cEBMF objects and calls `fit_joint(atac, rna)`.

Each observation model has its own loading and feature matrices. The prior is

`p(L_ATAC, L_RNA | X) = p(L_ATAC | X) p(L_RNA | L_ATAC, X)`.

Within either modality, `self_row_cov=True` adds dependence on earlier loading
coordinates. Feature columns use ordinary learned ash priors in the example.
Missing modalities are represented by observation masks after aligning cell IDs.
RNA-only cells retain uncertain ATAC parents; ATAC-only cells integrate out
the absent RNA branch during fitting, then predict it afterward.

## What the toy establishes

Two binary ATAC programs define four joint states, each activating a different
RNA program. This is a useful example of a nonlinear *conditional relationship*
among a chosen set of modality-specific programs.

It does **not** prove that MOFA or a shared linear factorization cannot reconstruct
the signals. Four shared one-hot state indicators span the RNA signal, and both
ATAC indicators are linear combinations of those states. The distinction is the
chosen prior, program parameterization, inference and statistical efficiency at
matched capacity, not an unconditional representational impossibility.

No claim of causal identification, unique tree recovery or predictive superiority
follows from this toy. Compare matched ranks and initialization, evaluate complete
held-out modalities separately from observed-entry denoising, and check sensitivity
to the posterior approximation and numerical integration.

## Derivation and positioning

- [Conditional inference guide](docs/source/joint_inference.rst)
- [Updated manuscript](output/pdf/atac_rna_joint_inference.pdf)
- [Short mathematical derivation](output/pdf/conditional_loading_prior_derivation.pdf)
- [Historical example archive](examples/archive/README.md)

The manuscript retains the comparisons with TwinEB, VampPrior, autoregressive
priors, iVAE and multi-omics methods. Prior dependence is distinguished from
posterior dependence, and exact reduction to ordinary cEBMF is explicit.
