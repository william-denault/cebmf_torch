# Quadratic feedback comparison

Sequential runs; timing excludes the first sweep. Small finite-budget diagnostic.

| Simulation | Prior/update | Feedback | Signal RMSE (mean ± SD) | Seconds/sweep (median) | Clipped curvature |
| --- | --- | --- | ---: | ---: | ---: |
| corrected | cgb_self | quadrature | 0.3342 ± 0.0330 | 1.370 | - |
| corrected | cgb_self | quadratic | 0.3341 ± 0.0337 | 0.225 | 0.00% |
| corrected | spiked_self | quadrature | 0.2532 ± 0.0183 | 3.364 | - |
| corrected | spiked_self | quadratic | 0.2545 ± 0.0175 | 0.321 | 0.00% |
