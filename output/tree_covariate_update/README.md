# GPU check: fixed versus self covariates

**Default update after this benchmark:** quadratic feedback is now the default.
The fit emits a warning suggesting `conditional_kwargs={"approximation": "quadrature"}`
to select the other method. The measurements and original run instructions
below document the earlier default; the saved timings remain unchanged.

16 September 2026. Actual device: **AMD Radeon 8060S, gfx1151**.
Python: `C:/Users/willi/miniconda3/envs/cebmf-rocm/python.exe`.
PyTorch: `2.12.0+rocm7.14.1`; HIP: `7.14.60850`; device: `cuda:0`.

## Change one argument

Keep the existing model, prior settings, and `self_row_cov=True`. Add:

```python
conditional_kwargs={"approximation": "quadratic"}
```

Use `device="cuda"` in the ROCm notebook kernel. This still learns
`p(L_k | L_<k)`, integrates uncertain parents, and includes locally approximated
child feedback. The default without this argument remains quadrature.
The approximation does not guarantee identical fitted results or ELBO ascent.

## Matched loading-update measurements

N=1000, P=200, K=6; update `L[:, 3]`, with **three inputs and two children**.
The fixed input is an actual frozen copy of the same `L[:, :3]`, not random
covariates. Earlier columns refer to factors for each cell, not other cells.

Both methods use the same data, starting factors/moments/noise, own neural
weights, widths, four extra hidden layers, ten training epochs, learning rate
0.001 and batch size 128. CGB width=32, penalty=1.0511; spiked EMDN width=64,
five total components, penalty=1.1. Integration uses 24 quadrature points and
32 parent draws. These settings retain the joint example's original budget.
The ordinary spiked solver's batch size is explicitly matched to 128.

Median of three synchronized repeats, after an untimed warm-up. Every repeat
resets the starting state; setup and reset are excluded. Models run sequentially.

| Prior | Fixed side information | Self, default quadrature | Self, quadratic | Speedup from the one argument |
|---|---:|---:|---:|---:|
| CGB | 0.298 s | 2.141 s | 0.926 s | 2.31x |
| Spiked EMDN | 0.374 s | 9.261 s | 0.968 s | 9.56x |

Extra peak allocated GPU memory fell from **293 to 35 MiB** for CGB and
**1541 to 197 MiB** for spiked EMDN. These are increments above the case's
starting allocation, not total process memory or allocator reservation.

**These are single loading updates, not complete sweeps.** ASH column updates,
noise updates and full-model objective evaluation are excluded. This experiment
measures cost; it does not compare reconstruction accuracy. The ordinary and
joint optimizers also differ in their scale updates and objective checks.

## Why the remaining cost?

The argument removes most repeated child-network evaluation. For this one
spiked-EMDN update, the measured number of input rows processed by child networks
falls from **168,960,000 to 320,000**. Quadratic feedback evaluates derivatives
once per minibatch and reuses them throughout the ten training epochs. Those
evaluations use float64 on the GPU; the own network trains in float32.

The own network still processes 704,000 input rows because it integrates
parent uncertainty and the conditional fitter evaluates its objective between
epochs. A diagnostic using the same conditional fitter but frozen inputs and
no children takes 0.904 s (CGB) and 0.871 s (spiked EMDN), versus 0.926 s and
0.968 s for quadratic self covariates. This suggests much of the remaining
overhead here is the fitting machinery and numerous small GPU operations;
the float64 child calculation is not the dominant added cost in this case.
That diagnosis is hardware/configuration specific, not a general speed claim.

## The three-stage example, with original settings

The supplied cold/precursor/warm sequence also ran on this GPU with the quadratic
argument and the original neural/integration budgets. Two sweeps per stage:

| Stage | First sweep | Second sweep |
|---|---:|---:|
| Cold spiked EMDN | 8.34 s | 6.44 s |
| CGB precursor | 4.66 s | 3.39 s |
| Warm sharp CGB, two slabs | 3.36 s | 3.35 s |

These full sweeps **include ASH F updates**, noise updates and objective
evaluation. Warm initialization uses `initialise_factors(L=..., F=...)` to keep
second moments and residuals consistent. Only two sweeps were run per stage;
these times and RMSEs are not convergence or long-run performance evidence.
The user's full ten-seed 30/10/20 schedule contains 600 sweeps, so even the
quadratic path can take substantial time.

## Hardware validation and fixes

- [Device suite](device_tests.xml): **22 passed, zero skipped**, including
  **14 actual GPU cases** covering ordinary and joint fits, both approximations,
  partially paired views, fixed-grid ASH agreement, optimizer residency,
  scalar reads/copies, profiler events and retained memory.
- [CPU regression suite](cpu_tests.xml): **70 passed**, 14 hardware tests
  skipped in the separate CPU-only environment.
- Removed CPU-scalar assignment paths for regularization and the ASH penalty.
  Replaced tensor-valued CDF fills with literal fills, avoiding a scalar
  extraction hidden inside ATen. Mathematical updates are unchanged.
- The strict no-host-read sweep tests use a **fixed ASH scale grid**. The
  default adaptive grid still reads a dynamic grid length on the host. Setup,
  validation, reporting and plotting can synchronize by design.
- The execution sandbox initially caused `hipErrorInvalidImage` even for
  `torch.ones(2, device='cuda')`. All reported GPU runs succeeded outside that
  sandbox; no driver/environment changes were needed.

The test runner imported the existing CPU environment's pure-Python pytest
installation after loading ROCm torch; it did not install packages or replace
torch. The profiler's Kineto metadata warning did not prevent its ATen event
checks. This validates the tested PyTorch-level operations, not every internal
driver transfer.

## Reproduce / inspect

- [Matched benchmark script](../../examples/benchmarks/tree/benchmark_covariate_update.py)
- [Raw matched timings and source hashes](rocm_original/timings.json)
- [All four diagnostic methods](rocm_original/REPORT.md)
- [Three-stage script](../../examples/benchmarks/tree/run_tree_cuda.py)
- [Full-sweep raw results](../tree_cuda_example/rocm_original_quadratic/timings.json)

```powershell
& 'C:/Users/willi/miniconda3/envs/cebmf-rocm/python.exe' examples/benchmarks/tree/benchmark_covariate_update.py --device cuda --preset original --methods fixed fixed_profile quadratic quadrature --output output/tree_covariate_update/recheck
```

`cpu_smoke/` is only script validation. `host_reads.json` and
`host_reads_trace.json` preserve the **pre-fix** diagnostic showing the hidden
CDF fill extraction; the passing device-suite XML records the corrected code.
