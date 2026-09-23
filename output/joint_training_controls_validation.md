# Joint training controls and ELBO diagnostic

Validated on 15 September 2026 with the `cebmf` Python 3.11 environment.

## Tests

- `python -m pytest tests -q`: 392 passed, 1 skipped (84.20 seconds).
- After adding three more whole-model HMM-support checks:
  `python -m pytest tests/test_joint_learning_elbo.py -q`: 37 passed.
- Remaining warnings were PyTorch's existing `torch.jit.script` deprecation.

The focused tests include direct full-joint Metropolis ratios with penalties
for all seven spiked learned families on both axes; the analytic spike-gate
optimum; epoch/batch step counts and normalization; invalid controls;
mixed discrete/continuous entropy; HMM normalizers checked using quadrature
and path enumeration; finite-state and Gaussian ELBO calculations; and
preservation of the sampling state/RNG during ELBO evaluation.

## Executed examples

- `examples/tree_joint_training_controls.ipynb`: four code cells, quick
  configuration (300 samples, 80 features, one seed). Both final priors use
  `cgb_sharp_2`, penalty 1.1, four training epochs and batch size 256.
  The warm start uses explicit `initialise_factors(L=..., F=...)` arguments.
  Cold signal RMSE: 0.165423; warm signal RMSE: 0.166421; retained ranks 5/5.
  Final regularized ELBO estimates: -41033.10 and -41019.89; respective
  Monte Carlo standard errors approximately 3.15 and 4.35.
- `examples/tree_joint_simple.ipynb`: all ten code cells executed.
- `examples/tree_joint_walkthrough.ipynb`: all twelve code cells executed.
  The unpenalized example retained its previous signal and held-out RMSEs;
  ELBO evaluation uses a separate RNG and does not change the chain.

The full ten-seed configuration was not run. These checks establish numerical
and API behavior, not a predictive advantage or posterior mixing guarantee.

## Objective interpretation

The automatic single-matrix interface records negative regularized Monte
Carlo ELBO estimates in `history_obj`. These use an explicit normalized
posterior approximation, with labelled scalar mixtures and whole HMM
blocks. They are diagnostics for the joint sampler, not a directly optimized
variational E-step. Finite estimates need not be monotone or lie below the
evidence. Standard errors quantify evaluation noise conditional on the
approximation, not its bias or the chain's mixing.

For penalty >= 1, the extra log-spike term is included in both neural training
and child-prior corrections, with an explicit auxiliary-observation target.
The ordinary data ELBO and the regularized ELBO are both exposed. Unspiked
EMDN requires penalty=1. The separate coupled `JointATACRNA` API retains its
existing controls and log-joint output.
