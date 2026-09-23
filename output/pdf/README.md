# Current derivation and draft

- **`atac_rna_joint_inference.pdf`** is the maintained manuscript: conditional
  loading priors, the cEBMF reduction, partially paired modalities, positioning,
  the current implementation, and the executed ATAC-RNA example.
- **`conditional_loading_prior_derivation.pdf`** is the shorter mathematical
  derivation of learning `p(L_k | L_<k)` with ordinary ash feature updates.

The matching `.tex` files are the authoring sources. The manuscript includes
`missing_modality_derivation.tex`, `twineb_positioning.tex`, and
`vae_prior_positioning.tex`, and uses the figure in `../atac_rna_joint/`.
Both PDFs include `numerical_stability_audit.tex`: the centered numerical
formulas, adversarial checks, CUDA contract, and hardware-validation limit.

## Historical material

`archive/` preserves earlier sampler implementation sections, alternative
mixture derivations, capacity assessments, and their original checks. These
files are research history, not documentation of the current fitting algorithm.
Use the maintained manuscript for the revised positioning and supported claims.
No historical result has been relabeled as a result of the new solver.

The implementation and validation record is in
`../review_joint_l/IMPLEMENTATION.md`. The earlier audit in that directory
describes the previous sampler and should be read as historical context.
