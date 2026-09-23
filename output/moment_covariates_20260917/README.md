# Plug-in tree fits with first and second posterior moments

Completed on the Radeon 8060S with the supplied `cebmf-rocm` environment:
**30 paired seeds, 180 fitted stages, no failures.** Both methods use the
original scalar plug-in updates and ASH columns. The only input change is
adding earlier raw posterior second moments, `E[L²]`, alongside `E[L]`.

## Conclusion

**This tree experiment does not show a clear reconstruction benefit from
adding second moments.** I would keep means alone for this example.

| Fit | Means only RMSE | Means + second RMSE | Paired difference [95% CI] |
|---|---:|---:|---:|
| Cold spiked EMDN, 30 sweeps | 0.08474 | 0.08448 | -0.00026 [-0.00073, +0.00016] |
| CGB precursor, 10 sweeps | 0.17480 | 0.17792 | +0.00313 [+0.00240, +0.00384] |
| Warm sharp CGB, 20 sweeps after CGB | 0.07944 | 0.07906 | -0.00038 [-0.00117, +0.00013] |

Differences are second-moment minus mean-only; negative favors the extra
inputs. These are final-sweep endpoints with paired bootstrap intervals.
Second moments improve RMSE in 17/30 cold fits and 17/30 warm fits, but
worsen the precursor in 29/30 seeds. Both final fits retain rank four in
every seed. The precursor's mean retained rank rises from 5.00 to 5.67.

The apparent average warm gain is mostly driven by seed 19, where its
RMSE difference is -0.00997. The median paired difference is only -0.0000044.
As a sensitivity check, omitting seed 19 gives -0.000051; all 30 seeds
remain in the primary results above. This is not evidence that second
moments can never help, only that they do not give a robust improvement
under this tree simulation and fitting budget.

## What the ELBO says

| Fit | Mean ELBO difference [95% CI] | Extra inputs have higher ELBO |
|---|---:|---:|
| Cold spiked EMDN | +4.65 [-1.59, +10.58] | 24/30 |
| CGB precursor | +35.96 [+21.55, +50.97] | 24/30 |
| Warm sharp CGB | +0.36 [-5.15, +6.00] | 14/30 |

Higher is better. These are **ELBOs for the fitted priors with their
covariates held fixed**, excluding training penalties. They are not the
joint autoregressive ELBO or a held-out predictive score. More neural
inputs add parameters, and these scores do not penalize that full increase
in network capacity. The [full report](RESULTS.md) gives absolute scores,
likelihood and KL terms, and separately adds the row and ASH penalties.

For CGB, the extra inputs increase expected log likelihood by 116.77,
loading KL by 22.63, and feature KL by 58.19, leaving an ELBO gain of
35.96. Its penalized score also increases (+32.90), despite worse signal
RMSE. This makes ELBO alone a poor selector between these two fits here.
The experiment does not isolate whether the change comes from added
variance information, squared-mean features, neural initialization, or
different pruning paths: `E[L²] = E[L]² + Var(L)`.

All cold and warm endpoints have current fitted inputs. CGB with second
moments prunes a factor on the last sweep in seed 26, so that endpoint's
score uses retained priors from their last fits. Excluding this pair
leaves an ELBO difference of +35.12 [20.28, 50.56] over 29 pairs, preserving
the conclusion. Both methods can have downward ELBO steps; finite neural
optimization, training penalties, refreshed covariates and the native
sharpening rule do not guarantee ascent of this unpenalized score.

## Checks and runtime

Five CPU tests and two actual-GPU tests passed. The mean-only implementation
matches the original scalar fitter exactly in the CPU regression test.
Eight independent marginal-mixture KL checks agree with the evidence
identity within 4.5e-6. The recomputed full-fit ELBO differs from the package
score by at most 0.05445; tiny negative cached KLs (minimum -0.000488) are
consistent with float32 cancellation. All 49 frozen fitting-source hashes,
180 checkpoints, sweep histories, finite scores and paired starts were verified.

The extra covariates are built on the GPU; moments and network parameters
remain there. Native pruning/ASH synchronization and explicit reporting
are unchanged. There is no integration or quadrature during fitting.
On the single-worker seed-1 reference, cold fits took 15.71 versus 17.17 s,
and the CGB-plus-warm pipelines 50.30 versus 48.67 s. Other seeds shared
the GPU, so their elapsed times are not isolated speed measurements.

## Files

* [Results and ELBO decomposition](RESULTS.md)
* [Protocol and the precise ELBO definition](PROTOCOL.md)
* [Paired reconstruction plot](paired_rmse.png)
* [Paired ELBO-difference trajectories](elbo_difference_trajectories.png)
* [ELBO versus reconstruction differences](elbo_vs_rmse_differences.png)
* [Per-seed final metrics](final_metrics.csv)
* [Independent mixture-KL audit](elbo_identity_audit.json)

## Run the simulation

From the repository root:

```powershell
& 'C:/Users/willi/miniconda3/envs/cebmf-rocm/python.exe' examples/benchmarks/tree/benchmark_moment_covariates.py
& 'C:/Users/willi/miniconda3/envs/cebmf/python.exe' examples/benchmarks/tree/summarize_moment_covariates.py --require-complete
```

The first command runs 30 seeds with cold spiked EMDN (30 sweeps) and a
CGB precursor (10) followed by sharp CGB (20), all with ASH columns.
Add `--seeds 1` for one example. Completed fitting stages resume; changing
source or settings requires a new `--output` directory. The two methods
are `mean` and `mean_second`, with no quadratic joint fitting in either.

The [benchmark class](../../examples/benchmarks/tree/plugin_moments.py)
accepts `row_cov_moments="mean"` or `row_cov_moments="mean_second"`.
Both route through the existing scalar cEBNM solvers. The latter appends
`L2[:, :k]` to `L[:, :k]` on the current device. This argument belongs to
the benchmark class, not the public `cEBMF` constructor.

Scripts:

* [Simulation driver](../../examples/benchmarks/tree/benchmark_moment_covariates.py)
* [Summary and figures](../../examples/benchmarks/tree/summarize_moment_covariates.py)
* [Independent ELBO audit](../../examples/benchmarks/tree/audit_moment_elbo.py)

Production fitting defaults are unchanged. All simulation records,
checkpoints, and frozen source copies are under `primary/`.
