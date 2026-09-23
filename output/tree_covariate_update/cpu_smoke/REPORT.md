# Fixed versus self covariates: loading-update timings

Device: CPU; torch 2.11.0+cpu; preset `compact`.

One coordinate update, not one full sweep. Setup, state reset and warm-up excluded.
All repeats start from identical factors, moments, precision and own neural weights.
Fixed inputs are a frozen copy of the earlier loading means, with the same dimension.

| Prior | k (zero based) / input dimension | Method | Median seconds | Extra peak MiB |
|---|---:|---|---:|---:|
| cgb | 3 | fixed | 0.0076 | 0.0 |
| cgb | 3 | fixed_profile | 0.0117 | 0.0 |
| cgb | 3 | quadratic | 0.0237 | 0.0 |
| cgb | 3 | quadrature | 0.0656 | 0.0 |
| spiked_emdn | 3 | fixed | 0.0080 | 0.0 |
| spiked_emdn | 3 | fixed_profile | 0.0166 | 0.0 |
| spiked_emdn | 3 | quadratic | 0.0393 | 0.0 |
| spiked_emdn | 3 | quadrature | 0.1715 | 0.0 |

`fixed` is the ordinary cEBMF row update with X_l and self_row_cov=False.
`fixed_profile` is a diagnostic using the conditional optimizer with frozen inputs and no children.
`quadratic` and `quadrature` integrate uncertain parents and include child feedback.
The ordinary solver differs in optimizer/scaling details; in particular sharp-family omega
shrinks the ordinary EM variance repeatedly but only initializes the joint learned variance.
Thus this is a compute comparison, not evidence about estimator accuracy.
Peak memory is extra allocated memory above each case's initial state, excluding allocator reserve.
No column-factor updates, noise updates or full-model objective evaluations are timed.
See timings.json for raw repeats, resolved settings, input-row counts and source hashes.
