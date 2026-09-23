# Quadratic feedback comparison

Preliminary implementation, before the float64 feedback-normalizer safeguard.
See `../validated/COMPARISON.md` for the final implementation comparison.

Sequential runs; timing excludes the first sweep. Small finite-budget diagnostic.

| Simulation | Prior/update | Feedback | Signal RMSE (mean ± SD) | Seconds/sweep (median) | Clipped curvature |
| --- | --- | --- | ---: | ---: | ---: |
| corrected | cgb_self | quadrature | 0.3342 ± 0.0330 | 1.881 | - |
| corrected | cgb_self | quadratic | 0.3341 ± 0.0337 | 0.142 | 0.00% |
| corrected | spiked_self | quadrature | 0.2532 ± 0.0183 | 5.441 | - |
| corrected | spiked_self | quadratic | 0.2545 ± 0.0175 | 0.204 | 0.00% |
