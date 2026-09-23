# Plug-in versus the current conditional cEBMF fitter

The completed benchmark supports your concern. The original plug-in solver
is substantially better for the warm sharp-prior pipeline and avoids the
late cold-fit collapses observed in the current quadratic fitter.

All 180 primary fitting stages completed on the supplied GPU environment.
Across the full audit, 265 fitting stages completed and two additional
diagnostic stages failed with nonfinite objectives. Every planned outcome
is accounted for in [validation.json](validation.json); failures were
retained rather than retried until success.

## Main results

Thirty paired simulations using your corrected example, ash columns,
matched SVD starts, original penalties, and the full 30/10/20 sweeps.
These are final noiseless-signal RMSEs; lower is better.

| Pipeline | Plug-in mean | Current quadratic mean | Paired winner |
|---|---:|---:|---|
| Cold spiked EMDN, 30 sweeps | 0.08472 | 0.14753 | Plug-in, 18/30 |
| CGB precursor, 10 sweeps | 0.17479 | 0.16618 | Current, 30/30 |
| CGB followed by sharp CGB, 10 + 20 sweeps | 0.07941 | 0.13010 | Plug-in, 30/30 |

The warm difference is 0.05069 RMSE, with a paired bootstrap 95% interval
of [0.04905, 0.05233]. The cold mean difference is driven largely by four
late collapses: current cold fits have median 0.08652 versus plug-in
0.08412, but maximum 0.73342 versus 0.09744. Thus the result does not say
that every current cold fit is worse, or that uncertainty is always harmful.

In the uncontended seed-1 run, cold fitting took 191.5 versus 16.2 seconds,
and the complete warm pipeline took 108.4 versus 41.3 seconds, current
versus plug-in. These as-configured timings include different batch and
pruning behavior. Other jobs shared the GPU and are not serial timing evidence.

## What went wrong

1. **The quadratic approximation can accept a disastrous update.** In a
   saved seed-7 state, its surrogate improved while the evaluated negative
   objective worsened by over one million. Increasing parent points from
   32 to 256 did not fix it; quadrature avoided that particular failure.
   Positive resulting precision ensures a finite Gaussian integral, not
   an accurate approximation away from the expansion point.
2. **The new fitter changes more than uncertainty handling.** Its default
   freely learned sharp slab scales make `omega` an initialization setting;
   the original solver repeatedly applies omega to its variance estimate.
   It also changes warm-start means, pruning, covariate normalization,
   initialization and neural optimization.
3. **Removing uncertainty alone is not the repair.** On five seeds with
   rank six and batch 128, removing both parent integration and feedback
   from the modern fitter still gives warm RMSE 0.13345, versus 0.07836
   for the scalar plug-in solver. The latter also reaches 0.07825 from
   the exact same modern CGB precursors. Keeping feedback but removing
   parent integration produces severe cold instability, including two
   nonfinite failures.

## Mathematical conclusion and recommendation

Modeling `p(L_k | L_<k)` does not require this particular joint variational
algorithm. Fitting the conditional prior at the latest earlier loading
means is a legitimate plug-in approximation. It still uses posterior
variances in ordinary cEBMF likelihood, feature and noise updates.

It is generally not coordinate ascent for a fixed joint latent-variable
ELBO, because its covariates move. But that limitation does not establish
inferior reconstruction. The current mean-field method also omits
cross-factor posterior covariances, so it should not be called full
uncertainty propagation. With externally fixed covariates and no latent
edges, the existing original cEBMF solver remains the update path.

**Use the original plug-in procedure as the practical reference for this
example.** The joint fitter needs its approximation safeguards and prior
semantics repaired before being presented as an upgrade. This audit adds
benchmarks and documentation; it does not change production behavior.

## Read the evidence

* [Numerical results and completion counts](RESULTS.md)
* [Protocol, controls and limitations](PROTOCOL.md)
* [Mathematical and implementation review](MATH_REVIEW.md)
* [Paired reconstruction errors](paired_rmse.png)
* [Convergence trajectories](trajectories.png)
* [Factor contribution patterns, seed 1](factor_contributions_seed01.png)
* [Factor contribution patterns, unstable seed 7](factor_contributions_seed07.png)
* [Captured failure geometry](failure_trace_seed07/failure_geometry.png)
* [Exact saved-state alternative updates](failure_trace_seed07/branches.json)

Primary fitting uses the supplied `cebmf-rocm` environment on the AMD
Radeon 8060S. The benchmark freezes the current source and does not retune
the production algorithm. Per-sweep JSON, fitted checkpoints and source
hashes are under `primary/` and `ablations/`.

## Reproduce from the repository root

```powershell
$cebmfPython = 'C:\Users\willi\miniconda3\envs\cebmf-rocm\python.exe'
& $cebmfPython examples/benchmarks/tree/benchmark_plugin_vs_joint.py --methods plugin current --output output/plugin_vs_joint_fresh/primary
& $cebmfPython examples/benchmarks/tree/benchmark_plugin_vs_joint.py --seeds 1 2 3 4 5 --methods plugin parent_only mean_feedback mean_only --batch 128 --fixed-rank --output output/plugin_vs_joint_fresh/ablations
& $cebmfPython examples/benchmarks/tree/benchmark_warm_diagnostics.py --input output/plugin_vs_joint_fresh/primary --output output/plugin_vs_joint_fresh/warm_diagnostics
& $cebmfPython examples/benchmarks/tree/summarize_plugin_vs_joint.py --input output/plugin_vs_joint_fresh --require-complete
```

The first command defaults to all 30 seeds and the full requested
30/10/20 sweeps. Reusing an output directory resumes completed stages;
incomplete stages restart. A source/settings mismatch is rejected.

The driver stops on a nonfinite metric. In this audit two diagnostic stages
failed and were recorded in `failures.json`; their independent remaining
stages were run separately with `--stages warm`. A failed stage was not
repeated until it succeeded. The summarizer's `--require-complete` checks
the 180 primary stages; the separate output verifier also accounts for
every control and recorded failure.

`plugin` is a benchmark subclass routing self-row covariates through the
existing scalar cEBNM fitter. It is not a new accepted value for
`conditional_kwargs['approximation']` in the public API.

## Script guide

All scripts are in `examples/benchmarks/tree/`:

| Script | Purpose |
|---|---|
| `benchmark_plugin_vs_joint.py` | Primary benchmark and controlled changes to parent integration/feedback |
| `tree_update_ablation.py` | Benchmark-only controls; no additional public API |
| `summarize_plugin_vs_joint.py` | Completed-run tables, paired bootstrap intervals, CSV and figures |
| `benchmark_warm_diagnostics.py` | Same-precursor sharp-prior fits and initialization controls |
| `audit_plugin_joint_math.py` | Independent Gaussian algebra, exact simulation check and control identity |
| `audit_tree_quadrature.py` | Frozen-fit numerical sensitivity |
| `trace_quadratic_failure.py` | Targeted replay and saved pre-failure state |
| `compare_failure_updates.py` | Alternate updates from that state; `--inspect-only` plots the failure |
| `verify_plugin_joint_outputs.py` | Verify this audit's source snapshots, matched starts, complete trajectories and all planned outcomes |

## Validation

* 71 conditional/numerical/quadratic regression tests passed on CPU.
* 12 actual-GPU sweep/device tests passed, including tests that reject host
  tensor reads and device transfers in warmed fitting with a fixed ASH grid.
* The separate mathematical audit verified the literal simulation, rank
  four, the Gaussian parent-variance identity and the point-parent,
  no-feedback evidence identity (maximum error 3.6e-15).

Passing the algebra tests does not validate an unconstrained local Taylor
approximation away from its expansion point. The captured failure shows
why these tests and end-to-end reconstruction benchmarks are both needed.
