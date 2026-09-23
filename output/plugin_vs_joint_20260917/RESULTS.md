# Numerical results

Primary comparison complete: 30 paired seeds.

Signal reconstruction RMSE; lower is better. SD is across simulation seeds.

| Pipeline | Method | Paired seeds | Mean | SD | Median | Maximum |
|---|---|---:|---:|---:|---:|---:|
| cold | plugin | 30 | 0.08472 | 0.00375 | 0.08412 | 0.09744 |
| cold | current | 30 | 0.14753 | 0.16321 | 0.08652 | 0.73342 |
| cgb | plugin | 30 | 0.17479 | 0.00732 | 0.17184 | 0.19416 |
| cgb | current | 30 | 0.16618 | 0.00872 | 0.16429 | 0.18626 |
| warm | plugin | 30 | 0.07941 | 0.00350 | 0.07844 | 0.08962 |
| warm | current | 30 | 0.13010 | 0.00378 | 0.12940 | 0.14006 |

## Paired differences

Current minus plug-in. Positive differences favor plug-in. Confidence intervals use
20,000 paired bootstrap resamples of the same simulation seeds.

| Pipeline | Mean difference | 95% interval | Current wins | Plug-in wins |
|---|---:|---:|---:|---:|
| cold | 0.06281 | [0.01312, 0.12625] | 12/30 | 18/30 |
| cgb | -0.00861 | [-0.01008, -0.00730] | 30/30 | 0/30 |
| warm | 0.05069 | [0.04905, 0.05233] | 0/30 | 30/30 |

## Large late deterioration

Descriptive list of fits ending more than 0.05 RMSE above an earlier fitted sweep.
The final value remains the primary endpoint; earlier truth-based minima are diagnostic only.

| Method | Stage | Seed | Earlier sweep | Earlier RMSE | Final RMSE |
|---|---|---:|---:|---:|---:|
| current | cold | 7 | 25 | 0.08044 | 0.53122 |
| current | cold | 13 | 23 | 0.08865 | 0.51438 |
| current | cold | 21 | 8 | 0.08321 | 0.37470 |
| current | cold | 28 | 14 | 0.08705 | 0.73342 |

## Rank-6, batch-128 diagnostic subset

Each cell shows mean RMSE and completed seeds out of the five planned seeds.
See PROTOCOL.md for what each control changes.

| Method | Cold | CGB precursor | Warm |
|---|---:|---:|---:|
| plugin | 0.09186 (5/5) | 0.17395 (5/5) | 0.07836 (5/5) |
| current | 0.08597 (5/5) | 0.16263 (5/5) | 0.13194 (5/5) |
| parent_only | 0.08317 (5/5) | 0.16302 (5/5) | 0.13323 (5/5) |
| mean_feedback | 0.81058 (3/5); 2 failed | 0.16301 (5/5) | 0.13242 (5/5) |
| mean_only | 0.08605 (5/5) | 0.16309 (5/5) | 0.13345 (5/5) |

### Numerical failures

Failed fits are not silently retried or included as successful endpoints.
Their last available sweep is not substituted for the requested final sweep.

| Group | Method | Stage | Seed | Failed sweep | Reason |
|---|---|---|---:|---:|---|
| ablations | mean_feedback | cold | 3 | 13 | FloatingPointError: Nonfinite metric; objective was NaN |
| ablations | mean_feedback | cold | 4 | 12 | FloatingPointError: Nonfinite metric; objective was NaN |

## Shared-precursor warm controls

These all start from the current CGB precursor and keep rank 6.

| Sharp-stage control | Mean RMSE | Completed seeds |
|---|---:|---:|
| plugin_common | 0.07825 | [1, 2, 3, 4, 5] |
| plugin_after_reset | 0.07838 | [1, 2, 3, 4, 5] |
| current_fixed_scales | 0.14505 | [1, 2, 3, 4, 5] |

`plugin_after_reset` starts a fresh scalar fit from the graph-replaced means,
including fresh noise and posterior-moment initialization. It tests recoverability,
not the isolated causal effect of replacing means in an otherwise identical modern fit.
`current_fixed_scales` freezes initial slab scales; it does not restore the legacy
repeated-omega variance update.

## Frozen-fit numerical sensitivity

Maximum absolute coordinate-mean change across the sampled rows and coordinates.
These are local checks at fitted states, not full refits or guarantees about earlier sweeps.

| Change | Cold | Warm |
|---|---:|---:|
| quadratic_vs_quadrature96 | 0.00165844 | 0.00942683 |
| quadrature24_vs_96 | 8.58307e-05 | 1.95503e-05 |
| parent_points32_vs_256 | 0.00173098 | 0.000179835 |
| parent_scramble0_vs_7_at256 | 0.000463784 | 9.0925e-06 |

## Post-hoc rank-four diagnostic

Seeds 1 and 7 selected after observing the instability; fixed rank and batch 128.

| Seed | Method | Stage | RMSE |
|---:|---|---|---:|
| 1 | current | cgb | 0.17048 |
| 7 | current | cgb | 0.16330 |
| 1 | current | cold | 0.08792 |
| 7 | current | cold | 0.07821 |
| 1 | current | warm | 0.14018 |
| 7 | current | warm | 0.12876 |
| 1 | plugin | cgb | 0.17756 |
| 7 | plugin | cgb | 0.16925 |
| 1 | plugin | cold | 0.08776 |
| 7 | plugin | cold | 0.10235 |
| 1 | plugin | warm | 0.08194 |
| 7 | plugin | warm | 0.07790 |

## Uncontended GPU timing

Seed 1 only. Other jobs share the GPU, so their timings are not used as serial speed comparisons.

| Method | Cold, seconds | CGB + warm, seconds |
|---|---:|---:|
| current | 191.5 | 108.4 |
| plugin | 16.2 | 41.3 |

Known-leaf-group sample-mean reference, 30 seeds: 0.07893 RMSE.
This uses true group labels for comparison only; it is not a formal lower bound.

Final endpoints use all requested sweeps. Early best RMSEs are not substituted for failed late fits.
The two fitters have different objectives; their objective values are not compared as model scores.
