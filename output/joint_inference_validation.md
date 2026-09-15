# Joint-inference validation — 15 September 2026

The experimental implementation lives in `src/cebmf_torch/experimental/`.
The production `cEBMF.iter_once()` and learned-prior fit adapters are unchanged.

## Checks performed

- Full package suite: **328 passed, 1 skipped**, in 51.16 seconds on the development CPU environment. Three TorchScript deprecation warnings were emitted.
- **126 new tests** cover the conditional mixture interface and the joint sampler, including all 64 scalar parent–child family pairings.
- References include direct numerical integration, native network outputs, detailed balance, independent parallel-chain moments, missing-modality marginals, finite-difference scores, exhaustive HMM path enumeration, fixed-parameter HMM posterior moments, and truncated-normal moments.
- Integration tests run every scalar learned family with partial overlap, reordered sample IDs, fixed side information and missing entries. They check frozen parameters, residual consistency and joint reconstruction moments.
- The original derivation verification script also passes all 18 checks, including its augmented variational EMDN calculation with uncertain parents.
- Ruff formatting and lint checks pass for the added Python files.
- All seven code cells of `examples/ATAC_RNA_hmm_joint.ipynb` executed successfully. Its three figures were inspected.
- The 35-page manuscript PDF was compiled and visually checked; its TeX log contains no overfull boxes, undefined references or LaTeX warnings.

Reproduce the package checks from the repository root:

```sh
python -m pytest tests -q
python output/pdf/verify_atac_rna_derivation.py
```

The new tests require only the existing development dependencies, including NumPy, PyTorch and pytest. The separate historical derivation script uses SciPy.

## Full-size simulation diagnostic

Command: `python examples/run_atac_rna_joint.py --output output/joint_atac_rna_full_diagnostic.json`

Settings: N=2,000, P=1,000 per view, seed 1, Gaussian noise SD 1.5, CGB loading priors, positive HMM factors, 1,200 paired rows, 400 ATAC-only rows and 400 RNA-only rows. Two percent of otherwise observed entries were withheld. The initialized ranks were 2 for ATAC and 3 for RNA. Eight prior-learning rounds preceded 100 burn-in sweeps and 100 retained draws with thinning 2. Total runtime was approximately 181 seconds.

| Modality | Sample group | Independent signal MSE | Joint signal MSE |
|---|---|---:|---:|
| ATAC | Paired | 0.0008912 | 0.0002582 |
| RNA | Paired | 0.0012652 | 0.0003949 |
| ATAC | Observed, unpaired | 0.0008440 | 0.0014189 |
| RNA | Observed, unpaired | 0.0017448 | 0.0022524 |
| ATAC | Modality missing | Not available | 0.0030456 |
| RNA | Modality missing | Not available | 0.0024468 |

Held-out noisy-entry MSE was 2.27073 versus 2.27105 for ATAC and 2.24366 versus 2.24330 for RNA (joint versus independent). Those small differences do not establish a predictive advantage.

The independent reference is the initializing `gbinary` model. The joint fit changes the loading family and inference method, so this comparison does not isolate the effect of the child correction. Paired-row signal MSE improved for both views; the observed unpaired groups worsened. The quick notebook uses a smaller, lower-information simulation and saves a separate report. No result is a multiseed benchmark or a convergence certificate.

## Scope of the correctness claim

The loading kernels and HMM block sampler preserve the specified augmented posterior at fixed parameters, subject to ordinary numerical precision and adequate support. This does not establish mixing of a particular finite chain. The optional generalized Monte Carlo EM phase approximately learns loading-prior parameters; it does not jointly optimize all HMM/noise parameters or integrate parameter uncertainty. Sharp scales are fixed by default. The experimental target has normalized priors without the legacy sparsity penalty. Exact within-component scores are implemented; learned-score proposals and variational optimization are derived/discussed but not implemented.
