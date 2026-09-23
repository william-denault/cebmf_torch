Device residency and numerical validation
========================================

Conditional loading inference uses the device and floating dtype of the
model observations. Networks, residuals, moments, integration states and
optimizer state remain there. Objective histories contain scalar tensors;
convert them explicitly when reporting:

.. code-block:: python

   history_for_plotting = torch.stack(result.history_obj).cpu().tolist()

The ordinary neural cEBNM solvers retain their existing float32 computation;
their returned moments and scalar losses stay on the chosen device. CUDA
minibatch permutations are generated on CUDA. Inner-prior training progress is
off by default for CGB, CASH and EMDN; their own ``verbose=True`` enables it.
At the cEBMF level, ``verbose=True`` is the default and prints a short message
after each sweep. These messages use only a Python counter: they do not read
GPU tensors or add explicit CUDA synchronization. Use ``cEBMF(..., verbose=False)``
to silence sweep messages. Approximation warnings remain independent of this flag.

Fitting with a fixed ASH grid
----------------------------

.. code-block:: python

   import torch
   from cebmf_torch import cEBMF

   device = torch.device("cuda")
   Y_device = Y.to(device=device, dtype=torch.float32)
   # Choose a range appropriate for the feature-factor scale in this fit.
   scales = torch.cat((torch.zeros(1, device=device),
                       torch.logspace(-3, 1, 24, device=device)))
   model = cEBMF(
       Y_device, K=4, device=device, self_row_cov=True,
       prior_L="spiked_emdn", prior_F="norm", allow_backfitting=False,
       prior_L_kwargs=dict(hidden_dim=16, n_layers=1, penalty=1),
       prior_F_kwargs=dict(scales=scales, penalty=1),
   )
   model.initialise_factors()
   result = model.fit(20)

ASH still refits its mixture weights on every update. Supplying ``scales``
changes the grid selection policy; it is not automatically equivalent to
an adaptive-grid fit. Check the grid range against your factor scaling.
The default adaptive grid remains supported: only its dynamic array length
is read by Python, while scale amplitudes remain on-device.

The conditional coordinate loop caches observed/missing row indices and
Sobol prediction tables, makes checkpoint decisions with scalar tensors,
and uses CUDA-capturable Adam state. ASH EM retains the first converged
iterate with a device flag and completes its fixed iteration budget; it
does not read convergence flags on the host. This can perform more EM
iterations than CPU early stopping, in exchange for avoiding synchronization.

Boundaries and scope
--------------------

Input validation, graph construction, rank pruning, serialization, plotting
and explicit reporting can synchronize. PyTorch's Sobol generator is CPU
based: its fixed tables transfer once at graph setup. Boundary validation
in ``fit_joint`` can read scalars; the established engine's sweeps and
missing-modality predictions use cached device tensors. CUDA finite-value
checks use asynchronous device assertions.

The default quadratic-feedback path computes local first/second derivatives
on the same device and freezes them during each coordinate's inner training.
Its analytic mixture moments, entropy, curvature-clipping counts and cached
feedback stay on-device. No derivative graph is retained between coordinates.
Cached child-network evaluations/derivatives and analytic normalizers use
temporary float64 tensors on that same device; model training parameters and
stored moments retain the observation dtype. This avoids narrow-feedback
cancellation, and is a dtype conversion rather than a host transfer.
CUDA residency and retained-memory tests cover both feedback options.

The strict warmed-loop contract covers conditional CGB/EMDN-family loadings
with ASH EM and a supplied grid, and the tested ordinary CGB/spiked-EMDN
routes. It does not claim that the deprecated experimental sampler, HMM
parameter optimization, optional L-BFGS, or structural pruning is free of
host synchronization. No tensor-backed model can promise that user code
calling ``.cpu()``, ``.item()`` or plotting incurs no transfer.

Tests and current evidence
--------------------------

``tests/test_device_contract.py`` includes CPU-runnable tests of branchless
EM and the profile math, plus actual CUDA-only checks for:

* host scalar reads, dynamic-size indexing and device copies during a
  complete warmed sweep;
* parameter, quadrature, history and Adam-state residency;
* partially paired modalities and row/column conditional graphs;
* CPU/CUDA numerical agreement for a fixed ASH grid;
* profiler device-to-host events and retained memory across sweeps.

On a CUDA machine, run:

.. code-block:: console

   python -c "import torch; assert torch.cuda.is_available()"
   python -m pytest tests/test_device_contract.py -q -rs

On 16 September 2026, all 22 tests passed in the user's ``cebmf-rocm``
environment: PyTorch ``2.12.0+rocm7.14.1``, HIP ``7.14.60850``, AMD Radeon
8060S (``gfx1151``). ROCm uses PyTorch's ``cuda`` device API. These runs used
actual GPU kernels outside the restricted execution sandbox; inside the
sandbox even a float32 fill failed with ``hipErrorInvalidImage``.

The first hardware run exposed scalar assignment paths that created CPU
scalar tensors, and a tensor-valued fill that hid an ATen scalar extraction.
Literal fills and device-tensor copies now avoid those paths. The dispatch
guard also rejects tensor-valued fills; profiler checks independently cover
operations inside ATen. Test results are saved in
``output/tree_covariate_update/device_tests.xml``. The separate CPU regression
run passed 70 tests with 14 GPU-only skips.

The profiler and dispatch tests cover the fixed-grid fitting loop described
above. They do not establish the absence of every driver-internal transfer,
nor remove the adaptive ASH grid's documented size-selection synchronization.
