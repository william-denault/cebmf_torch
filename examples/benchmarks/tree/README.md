# Tree-prior benchmark

## Plug-in first versus first-and-second moments

`benchmark_moment_covariates.py` compares just two scalar plug-in fits:
earlier `L` columns as inputs, or earlier `L` together with `L2=E[L²]`.
Both preserve the original prior fitting, sharpening, pruning and ASH columns.
It runs the corrected 30-seed cold/CGB/warm protocol and records reconstruction,
fitted frozen-covariate ELBOs, their decomposition, and separate penalty terms.

```powershell
& 'C:/Users/willi/miniconda3/envs/cebmf-rocm/python.exe' examples/benchmarks/tree/benchmark_moment_covariates.py
```

See the [moment comparison report](../../../output/moment_covariates_20260917/README.md).
The extra input option belongs to the benchmark's `MomentPluginCEBMF` class;
it does not change the public `cEBMF` default or introduce a joint update.

## Current versus plug-in: full corrected example

`benchmark_plugin_vs_joint.py` runs the requested 30 seeds at N=1000, P=200,
initial K=6, with ash columns: cold spiked EMDN for 30 sweeps, and CGB for
10 sweeps followed by sharp CGB for 20. It compares the current quadratic
fitter against the scalar cEBNM update using earlier loading means as inputs.
Initial SVD factors and per-stage neural seeds are paired.

```powershell
& 'C:/Users/willi/miniconda3/envs/cebmf-rocm/python.exe' examples/benchmarks/tree/benchmark_plugin_vs_joint.py --methods plugin current --output output/tree_plugin_comparison/primary
```

The [17 September audit](../../../output/plugin_vs_joint_20260917/README.md)
contains completion counts, raw results, figures and the mathematical review.
It identifies a real failure of the unconstrained quadratic update: positive
posterior precision does not prevent a large deterioration of the evaluated
objective. The report separates this instability from changes in sharpening,
initialization, pruning and neural optimization. Earlier timing measurements
below are not evidence of equivalent estimation accuracy.

## Default approximation and alternative

With `self_row_cov=True`, quadratic feedback is now the default. An explicit
`conditional_kwargs={"approximation": "quadratic"}` selects the same behavior.
Use the `cebmf-rocm` notebook kernel and `device="cuda"` for the AMD GPU.
The fitting loop and prior penalties do not need to change. This uses the
local quadratic approximation to child feedback; parent uncertainty is still
integrated. A warning at graph creation explains the approximation and suggests
`conditional_kwargs={"approximation": "quadrature"}` for the quadrature method.
The warning does not repeat on each sweep.

## Fixed versus self-covariate update timing

`benchmark_covariate_update.py` compares a single loading update using
`X_l=L[:, :k].clone(), self_row_cov=False` against the joint update using
`self_row_cov=True`. The fixed matrix has exactly the same input dimension and
values as the earlier loading means at the start of the update. Here “self
covariates” means earlier **loading columns for the same row**, not other cells.

```powershell
& 'C:/Users/willi/miniconda3/envs/cebmf-rocm/python.exe' examples/benchmarks/tree/benchmark_covariate_update.py --device cuda
```

The default uses N=1000, P=200, K=6, updates zero-based column 3 (three parent
inputs, two children), and compares fixed, quadratic and quadrature updates.
Each case has one untimed warm-up and three timed repeats reset to the same
starting factors, moments, noise precision and own network weights. Timings
synchronize the GPU and exclude model construction, copying and uploads.

The original preset matches network widths/layers, ten training epochs,
learning rates and minibatch sizes across methods. It explicitly uses batch
size 128 for both families, including the ordinary spiked-EMDN control whose
standalone default differs. `--preset compact` changes training and integration
budgets separately. `--coordinates 1 3 5` tests different numbers of parents and
children. The optional `--methods fixed fixed_profile quadratic quadrature`
adds a control using the conditional optimizer with fixed inputs and no child
feedback, to separate optimizer overhead from latent-covariate work.

Results include raw update times, extra peak allocated GPU memory, network
input-row counts, source hashes and a Markdown report. A coordinate update
does not include ASH column fitting or a full-model objective calculation, and
its timing is not a full-sweep time. The ordinary and joint solvers have
different scale/optimizer rules; these timings do not compare fitting accuracy.

The [executed Radeon 8060S report](../../../output/tree_covariate_update/README.md)
contains matched GPU timings and the hardware test results. With the original
budget, the one-argument change reduced the spiked-EMDN coordinate update from
9.26 s to 0.97 s and CGB from 2.14 s to 0.93 s in that experiment.

## CUDA timing for the three-stage notebook example

`run_tree_cuda.py` reproduces the cold spiked-EMDN, CGB precursor, and warm
two-sided sharp-CGB sequence. It prints the imported package path, actual GPU,
selected inference method, and synchronized setup/sweep timings. Start with one
seed and a few sweeps:

```powershell
python examples/benchmarks/tree/run_tree_cuda.py --device cuda
```

The compact preset explicitly selects quadratic feedback, two training epochs,
16 hidden units, no extra hidden layers, a full-row batch, and 8/8 parent
integration settings. Spiked EMDN keeps five total components. This changes
capacity/training/integration budgets; compare accuracy before using it for a
final analysis. The pure approximation change, retaining the original network
and integration defaults, is:

```powershell
python examples/benchmarks/tree/run_tree_cuda.py --device cuda --preset original --stage-steps 1 1 1 --output output/tree_cuda_original_budget
```

Add `--approximation quadrature` to time the alternative quadrature
method. The user's full experiment requires `--seeds 10 --stage-steps
30 10 20`, which is 600 sweeps. Do the short timing first. CUDA timings include
explicit synchronization at reporting boundaries. `--device cuda` raises an
error if CUDA is unavailable; it never silently falls back to CPU.

The script fixes warm starts through `initialise_factors(L=..., F=...)`, computes
RMSE on the fitted device, and saves optional heatmaps with `--plot` after fitting.
It preserves the notebook's simulation by default; `--scenario corrected` fixes
the shared leaf mask and skipped rows. The default cold/warm runs use different
prior families, so their errors do not isolate the effect of warm starting.
Use `--cold-prior cgb_sharp_2` to compare the same final prior family (training
budgets and initial loading states still differ).

Even quadratic feedback uses cached float64 child evaluations on the model's
device for numerical accuracy. Its performance depends on the GPU's float64
capability as well as network size; CPU speedups are not CUDA measurements.

Start with [the executed notebook](../../benchmark_tree_priors.ipynb) or the
[full report](../../../output/tree_prior_benchmark/REPORT.md).
These display the historical runs with a shared neural spike penalty of 1.05;
they are not results for the updated settings below.

## What "no covariates" means

With `X_l=None` and `self_row_cov=False`, the learned-prior builder supplies
an all-ones input column. Standardization maps it to zero. Network biases still
learn, but every row receives the same prior parameters **within each factor**.
Different factors have separately fitted priors. Each row's posterior still
depends on its own normal-means estimate and standard error.

- `cgb`: a zero spike plus one Gaussian slab, with a shared learned weight,
  slab mean and slab variance. Only the weight depends on inputs when actual
  covariates are present.
- `spiked_emdn`: a zero spike plus learned Gaussian slabs (two slabs in this
  benchmark). Weights, means and scales depend on inputs when present; without
  covariates they are shared across rows.

The neural network is therefore an overparameterized way to fit a global mixture
in the no-covariate controls. It does not get row IDs, truth labels, or observation
values as covariate inputs. The observations train it through the likelihood.

## Requested penalties (16 September 2026)

The driver now defaults to `--cgb-penalty 1.0511` and
`--spiked-penalty 1.1`, on both independent and conditional fits.
These add `(penalty - 1) * sum(log(pi_spike))` to the maximized objective;
values above one favor more spike mass. Plain `emdn` has no zero spike:
the requested EMDN configuration here is `spiked_emdn`.

Plain `cgb` has **no omega parameter**. To use the requested dictionary
`prior_L_kwargs={"penalty": 1.0511, "omega": 0.01}`, choose `prior_L="cgb_sharp"`.
The optional methods `cgb_sharp`, `cgb_sharp_plugin`, and `cgb_sharp_self` use
`--cgb-omega 0.01`. The existing `--omega 0.1` applies only to ordinary `gbinary`.

**Sharp-prior interpretation:** the ordinary/plug-in sharp solver multiplies its
slab-variance estimate by omega during fitting. In the joint variational solver,
omega scales the initial slab variance; the effective variance is then learned.
These are different scale-update rules, so a sharp comparison does not isolate
the effect of integrating latent covariates. Fixing a variance would be another
explicit model choice; it is not done silently here.

New runs default to `output/tree_prior_benchmark/penalty_20260916/main`, and their
JSON records include resolved prior names and kwargs. The legacy `--penalty`
option overrides both family-specific penalties when supplied. To reproduce the
historical hyperparameters on current code, pass `--penalty 1.05` explicitly;
this does not reproduce the earlier source version.

A short verification run (not a performance benchmark):

```powershell
python examples/benchmarks/tree/benchmark_tree_priors.py --seeds 1 --scenarios corrected --methods cgb cgb_plugin cgb_self spiked spiked_self cgb_sharp cgb_sharp_plugin cgb_sharp_self --steps 2 --n 64 --p 24 --rank 3 --epochs 1 --hidden 6 --layers 0 --quadrature 6 --parents 6 --output output/tree_prior_benchmark/penalty_20260916/smoke
```

The historical six-method comparison, using the new penalties:

```powershell
python examples/benchmarks/tree/benchmark_tree_priors.py --methods gbinary cgb cgb_plugin cgb_self spiked spiked_self --epochs 2 --hidden 12 --layers 0 --quadrature 8 --parents 8
```

Sharp methods are opt-in. The existing summary script and notebook continue to
read the historical six-method results; the commands above do not replace those
figures automatically.

## Scripts

`compare_quadratic_feedback.py` runs a separate, sequential comparison of the
quadrature and default quadratic-feedback methods. By default it uses three
seeds of the corrected tree, N=200, P=60, rank 4, 12 sweeps, two training epochs,
and 16 integration points. It saves a `COMPARISON.md` and JSON alongside the
individual records. Timing excludes the first sweep. Run:

```powershell
python examples/benchmarks/tree/compare_quadratic_feedback.py
```

The tree driver and public API now default to quadratic feedback. To select
quadrature in the driver, add `--approximation quadrature`; in the public API:

```python
conditional_kwargs={"approximation": "quadrature"}
```

The default quadratic method retains averaging over uncertain parents but replaces extensive
candidate-loading integration with cached derivatives and analytic component
updates. It clips positive feedback curvature to zero and records how often.
It is a local approximation, not a guarantee of unchanged accuracy or ELBO
ascent. See the [derivation](../../../docs/source/quadratic_feedback.rst).

Run scripts from the repository root:

- `benchmark_tree_priors.py`: simulate the exact notebook or corrected tree and
  fit the requested prior/update configurations. See `--help` for controls.
- `summarize_tree_benchmark.py`: aggregate the 60 main fits and regenerate figures.
- `check_tree_integration.py`: compare local integration settings at a saved
  conditional-model checkpoint, without refitting the model.
- `continue_tree_benchmark.py`: run the additional optimizer-effort diagnostic.

All observed data and initial factors are matched within a seed. Simulation truth
is used only to generate observations and score results. The plug-in control is
a benchmark-only subclass; it does not change the package's public fitting API.

Completed configurations are skipped on rerun. Use a distinct output directory
for a different protocol. Do not run the same unfinished configuration concurrently.
The supplied source notebook outside this repository is left untouched.
