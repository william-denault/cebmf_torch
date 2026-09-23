# Quadratic conditional update: implementation and validation

Date: 16 September 2026. This record describes the final implementation with the float64 feedback safeguard.

## Public API

```python
conditional_kwargs={"approximation": "quadratic"}
```

Use this in cEBMF, or identically in both cEBMF objects passed to fit_joint. Quadrature remains the default.

## Mathematical scope

- The model remains p(L_k | L_<k); each coordinate still receives child feedback.
- Feedback is expanded at the previous posterior mean of each Gaussian component. Spike feedback at zero is exact.
- Positive curvature is capped at zero; posterior slab precision is therefore at least its own-prior-plus-likelihood precision.
- Frozen derivative caches are reused throughout the current prior fit. Component masses, moments and augmented entropy are analytic.
- Own-prior expectations and other uncertain parents still use the existing quadrature/Sobol integration. This is not a completely moment-only method.
- The local optimizer score is a surrogate. Reported objectives reevaluate the original regularized prior under the fitted posterior; neither global ELBO ascent nor equal accuracy is guaranteed.
- With no latent edges, the original cEBMF solver and its finite training schedule are used exactly.

Derivation: [quadratic_feedback.rst](../../docs/source/quadratic_feedback.rst).

## Numerical safeguard

An adversarial broad Gaussian own prior with child feedback -0.5*(v-10)^2/1e-8 returned log integral -7.71034 in the preliminary float32 calculation, versus -129.21034 analytically, despite an apparently correct posterior mean near 10.
The final implementation evaluates cached child networks/derivatives and analytic normalizers in float64 on the existing model device. It stores posterior moments in the observation dtype. A regression also checks tiny component probabilities against independent Gaussian convolution, rather than checking only the mean.

## Completed CPU comparison

Corrected tree, N=200, P=60, K=4, three seeds, 12 sweeps, two neural epochs per sweep, 12 hidden units, zero extra hidden layers, 16 Hermite and 16 Sobol points. CGB penalty 1.0511; spiked EMDN penalty 1.1; ash columns, penalty 10. Runs were sequential; sweep timings exclude the first sweep.

| Prior | Quadrature seconds/sweep | Quadratic seconds/sweep | Speed ratio | Quadrature mean RMSE | Quadratic mean RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| CGB | 1.370 | 0.225 | 6.10x | 0.3342 | 0.3341 |
| Spiked EMDN | 3.364 | 0.321 | 10.48x | 0.2532 | 0.2545 |

No curvature clipping was used in these runs. This small finite-budget diagnostic does not establish equal accuracy across datasets or GPU performance.

Final raw records, checkpoints, protocol and source hashes: [validated comparison](../tree_prior_benchmark/quadratic_20260916/validated/comparison.json). The earlier comparison directory is preliminary and explicitly superseded.

## Validation

- Final focused suite: 70 passed, 14 CUDA-only skips (6.13 seconds).
- Earlier broad regression suite before the final precision safeguard: 454 passed, 15 skipped. The subsequent focused suite covers the changed math, all eight conditional families, original quadrature regressions, device math, no-edge equivalence, missing modalities, both axes, and checkpoint continuation.
- Source hashes from the completed benchmark match the final implementation.
- git diff --check passed.
- PyTorch 2.11.0+cpu; CUDA hardware/runtime unavailable. CUDA tests check warmed-loop transfers, optimizer/state residency and retained memory for both approximation options, but were not executed here.

## Reproduce

```powershell
python examples/benchmarks/tree/compare_quadratic_feedback.py
python -m pytest tests/test_quadratic_conditional.py tests/test_conditional_variational.py tests/test_conditional_numerics.py tests/test_device_contract.py -q
```
