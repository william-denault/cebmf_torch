# Fixed versus self covariates: loading-update timings

Device: AMD Radeon(TM) 8060S Graphics; torch 2.12.0+rocm7.14.1; preset `original`.

One coordinate update, not one full sweep. Setup, state reset and warm-up excluded.
All repeats start from identical factors, moments, precision and own neural weights.
Fixed inputs are a frozen copy of the earlier loading means, with the same dimension.

| Prior | k (zero based) / input dimension | Method | Median seconds | Extra peak MiB |
|---|---:|---|---:|---:|
| cgb | 3 | fixed | 0.2982 | 3.1 |
| cgb | 3 | fixed_profile | 0.9041 | 2.3 |
| cgb | 3 | quadratic | 0.9260 | 35.4 |
| cgb | 3 | quadrature | 2.1414 | 293.1 |
| spiked_emdn | 3 | fixed | 0.3740 | 3.1 |
| spiked_emdn | 3 | fixed_profile | 0.8714 | 2.7 |
| spiked_emdn | 3 | quadratic | 0.9684 | 196.7 |
| spiked_emdn | 3 | quadrature | 9.2614 | 1541.3 |

`fixed` is the ordinary cEBMF row update with X_l and self_row_cov=False.
`fixed_profile` is a diagnostic using the conditional optimizer with frozen inputs and no children.
`quadratic` and `quadrature` integrate uncertain parents and include child feedback.
The ordinary solver differs in optimizer/scaling details; in particular sharp-family omega
shrinks the ordinary EM variance repeatedly but only initializes the joint learned variance.
Thus this is a compute comparison, not evidence about estimator accuracy.
Peak memory is extra allocated memory above each case's initial state, excluding allocator reserve.
No column-factor updates, noise updates or full-model objective evaluations are timed.
See timings.json for raw repeats, resolved settings, input-row counts and source hashes.
