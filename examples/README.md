# Start here

- **[First versus first-and-second moment covariates](benchmarks/tree/benchmark_moment_covariates.py)**:
  the corrected tree simulation using only the original plug-in fitter,
  with ash columns, 30 paired seeds, and fitted ELBO comparisons.
  See the [results and score definition](../output/moment_covariates_20260917/README.md).
- **[Current versus plug-in: corrected tree benchmark](benchmarks/tree/benchmark_plugin_vs_joint.py)**:
  the 30-seed, N=1000/P=200 example with full cold and warm pipelines,
  paired results, and controls separating parent uncertainty from child
  feedback. See the [17 September audit and results](../output/plugin_vs_joint_20260917/README.md).
- **[Joint ATAC-RNA](ATAC_RNA_joint.ipynb)**: two `cEBMF` models, a learned
  `p(L_RNA | L_ATAC)`, ash feature priors, and partially paired cells.
  Run from the first cell to the last. The equivalent script is
  [run_atac_rna_joint.py](run_atac_rna_joint.py).
- **[Conditional loadings in one matrix](tree_joint_simple.ipynb)**:
  `self_row_cov=True` learns `p(L_k | L_<k)` using variational updates.
- **[Tree-prior benchmark](benchmark_tree_priors.ipynb)**: matched GB, CGB,
  plug-in and spiked-EMDN comparisons, the source-notebook audit, and sensitivity
  checks. Its [supporting scripts](benchmarks/tree/README.md) reproduce the results.
- **[CUDA tree timing](benchmarks/tree/run_tree_cuda.py)**: the three-stage
  cold/CGB/warm example with explicit approximation, device checks, consistent
  warm starts and synchronized per-sweep timings. Start with one seed.
- **[Fixed versus self-covariate timing](benchmarks/tree/benchmark_covariate_update.py)**:
  one loading update with identical covariate dimension and matched neural
  training budgets, measured on CPU or CUDA/ROCm.
- HMM feature priors remain available separately; see the
  [HMM guide](../docs/source/hmm_priors.rst).

Install the package with its example dependencies in the notebook kernel's
environment (`pip install -e '.[examples]'`). The earlier examples use CPU and set
one torch thread because the small neural networks otherwise incur substantial
threading overhead. Larger problems may benefit from different settings.
For CUDA configuration and the fixed-grid ASH option, see the
[device guide](../docs/source/device_contract.rst). Saved tree benchmark
measurements retain their original source-version provenance.

## Earlier experiments

The [archive](archive/README.md) preserves the previous exploratory notebooks
and their outputs. They are not the current fitting API or a benchmark of the
new algorithm. Simulation truths are used only to generate observations and
evaluate predictions, never as inputs to fitting.
