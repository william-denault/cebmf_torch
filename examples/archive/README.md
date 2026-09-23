# Historical experiments

These files were consolidated on 15 September 2026. They are kept for provenance;
their saved outputs describe earlier algorithms and have not been relabeled as
results from the current variational solver.

| Earlier example | What it explored | Current replacement |
| --- | --- | --- |
| `atac_rna/model_RNA_ATAC.ipynb` | Repeated fits with estimated loadings as fixed covariates | `../ATAC_RNA_joint.ipynb` |
| `atac_rna/ATAC_RNA_hmm_ordered.ipynb` | Ordered factors and HMM smoothing | HMM prior guide |
| `atac_rna/ATAC_RNA_self_cov_joint.ipynb` | Automatic sampler dispatch; sequential modality fits | `../ATAC_RNA_joint.ipynb` |
| `atac_rna/ATAC_RNA_hmm_joint.ipynb` | Experimental joint MCMC with HMM features | `../ATAC_RNA_joint.ipynb` |
| `atac_rna/run_atac_rna_sampler.py` | Command-line driver for that sampler | `../run_atac_rna_joint.py` |
| `tree/*` | Sampler diagnostics, controls and repeated walkthroughs | `../tree_joint_simple.ipynb` |

The old sampling backend remains under `cebmf_torch.experimental` for reference
checks. Explicit legacy `joint_kwargs` emits a deprecation warning. New examples
use the conditional variational backend with no sampling configuration. Paths
inside archived notebooks reflect their original location and are not maintained.
