# Why the conditional update is slower

16 September 2026. Measured on the current CPU implementation, one torch
thread. This is a small operation-count diagnostic, not a predictive benchmark.

The flag now selects a different inference algorithm. The former plug-in
update treated earlier posterior means as fixed covariates. It fit a scalar
normal-means model and used analytic Gaussian-mixture moments. The current
update integrates parent uncertainty and includes child-prior feedback.

For every candidate loading, each child prior is evaluated over possible
values of its other parents. With C components, Q quadrature points and S
parent draws, this uses roughly C*Q*S neural inputs per data row per child,
instead of one input per row. With a single remaining uncertain parent the
code uses that parent's entire C_parent*Q quadrature representation instead
of S draws. All earlier factors are parents: rank K creates K*(K-1)/2 edges.

At package defaults, a CGB coordinate has 2*24=48 candidate entries. A
child with several other uncertain parents uses 32 contexts for each,
giving 1,536 network inputs per original row. This repeats for every child,
optimization epoch and checkpoint score. The current code also repeats the
zero atom across quadrature nodes and recomputes fixed child moments.

## Fresh matched diagnostic

One measured sweep after setup/warm-up: N=128, P=40, K=6, CGB, hidden=12,
zero additional hidden layers, two neural epochs, batch=128, ash features,
float32. Same simulated observations and initial factors. Counts include
training, inference and objective checks. Timings include Python profiler
overhead and are single measurements, not stable hardware benchmarks.

| Update | Q / parent draws | Seconds | Neural input rows processed |
|---|---:|---:|---:|
| Old plug-in | none | 0.075 | 4,608 |
| Conditional | 8 / 8 | 0.630 | 1,629,056 |
| Conditional, default integration | 24 / 32 | 4.844 | 17,890,176 |

At default integration, `child_term` accounted for 4.28 of 4.84 seconds
(88%, inclusive). Full-profile rescoring before/after epochs accounted for
2.18 seconds (45%, overlapping with child evaluation). Ash feature updates
accounted for 0.062 seconds. Raw measurements: `runtime_diagnosis.json`.

Some extra cost is required by uncertainty integration and child feedback.
The observed factor is specific to this implementation: duplicate atoms,
recomputed child statistics/context, dense dependencies and repeated full
scores create optimization opportunities. Modeling p(L_k|L_<k) does not
require this particular quadrature implementation or this runtime.

## Simulation entry points and expensive defaults

- `examples/benchmarks/tree/benchmark_tree_priors.py`: tree simulation and
  comparison; `cgb_plugin` retains the previous update as a benchmark control.
- `examples/benchmark_tree_priors.ipynb`: displays the completed results.
- `examples/tree_joint_simple.ipynb`: one conditional cEBMF simulation.
- `examples/run_atac_rna_joint.py` and `examples/ATAC_RNA_joint.ipynb`: paired
  and unpaired modalities, with two cEBMF objects and `fit_joint`.

The benchmark script's default command runs **50 fits**: five seeds, two
scenarios, five methods, each with 30 sweeps and five neural epochs. The
historical main benchmark used lighter two-epoch, Q=8, S=8 settings. Both
maintained simulation scripts explicitly select CPU. They do not automatically
switch to CUDA if hardware becomes available.

For a short three-method check, run from the repository root:

```powershell
python examples/benchmarks/tree/benchmark_tree_priors.py --seeds 1 --scenarios corrected --methods cgb_plugin cgb_self spiked_self --steps 5 --n 200 --p 60 --rank 4 --epochs 2 --quadrature 8 --parents 8 --output output/tree_speed_check
```

This is a smoke comparison, not an integration-accuracy assessment. The
ATAC-RNA script also accepts `--quick`.
