# Posterior-moment covariate benchmark

Complete: 30 paired seeds, 180 fitted stages.

Only original scalar plug-in fitting is compared. Second means raw E[L²], not E[L]².

## Reconstruction

| Stage | Means only RMSE | Means + second RMSE | Difference [95% CI] | Second wins |
|---|---:|---:|---:|---:|
| cold | 0.08474 | 0.08448 | -0.00026 [-0.00073, +0.00016] | 17/30 |
| cgb | 0.17480 | 0.17792 | +0.00313 [+0.00240, +0.00384] | 1/30 |
| warm | 0.07944 | 0.07906 | -0.00038 [-0.00117, +0.00013] | 17/30 |

Differences are second-moment minus mean-only. Negative RMSE differences favor second moments.
Intervals use 20,000 paired bootstrap resamples of the simulation seeds.

## Fitted frozen-covariate ELBO

Higher is better; these exclude training penalties.

| Stage | Means only | Means + second | Difference [95% CI] | Second higher |
|---|---:|---:|---:|---:|
| cold | -332118.39 | -332113.73 | +4.65 [-1.59, +10.58] | 24/30 |
| cgb | -336798.06 | -336762.10 | +35.96 [+21.55, +50.97] | 24/30 |
| warm | -332011.99 | -332011.62 | +0.36 [-5.15, +6.00] | 14/30 |

## Where the ELBO difference comes from

| Stage | Δ expected log likelihood | Δ loading KL | Δ feature KL | Δ penalized score [95% CI] |
|---|---:|---:|---:|---:|
| cold | +2.09 | -2.03 | -0.54 | +4.02 [-7.04, +13.08] |
| cgb | +116.77 | +22.63 | +58.19 | +32.90 [+17.41, +49.25] |
| warm | +0.07 | +0.92 | -1.22 | +1.38 [-4.00, +7.66] |

Δ ELBO = Δ expected log likelihood − Δ loading KL − Δ feature KL.
The penalized score additionally includes the row spike penalty and ASH penalty.

## Agreement, rank and score checks

| Stage | Both ELBO and RMSE favor second | Seeds where ELBO and RMSE disagree | Mean rank: first / second |
|---|---:|---|---:|
| cold | 12/30 | [2, 3, 4, 5, 11, 12, 13, 15, 17, 18, 19, 23, 25, 26, 28, 29, 30] | 4.00 / 4.00 |
| cgb | 0/30 | [1, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] | 5.00 / 5.67 |
| warm | 10/30 | [5, 6, 8, 9, 11, 14, 17, 20, 22, 24, 28] | 4.00 / 4.00 |

Maximum discrepancy between recomputed ELBO and negative package objective: 0.054441.
Minimum cached individual loading/feature KL: -0.000488.

| Stage | Method | Current fitted inputs at endpoint | Fits with an ELBO decrease > 1 unit |
|---|---|---:|---:|
| cold | mean | 30/30 | 30/30 |
| cold | mean_second | 30/30 | 30/30 |
| cgb | mean | 30/30 | 11/30 |
| cgb | mean_second | 29/30 | 2/30 |
| warm | mean | 30/30 | 30/30 |
| warm | mean_second | 30/30 | 30/30 |

Decreases count consecutive fitted sweeps with unchanged rank and current fitted inputs.
These moving-covariate algorithms are not guaranteed to ascend a fixed joint ELBO.

The score holds the fitted covariates fixed. It is not the autoregressive joint ELBO,
and it is not a held-out score or a complexity-adjusted comparison of neural networks.
See [PROTOCOL.md](PROTOCOL.md) for the exact definition and limitations.

**Final pruning flag (cgb):** mean_second, seeds [26]. These scores use retained priors from their last fits; inputs changed through pruning. Excluding the flagged pairs leaves 29 pairs and an ELBO difference of +35.12 [+20.28, +50.56].

## Seed-1 timing reference

Only this seed ran without another benchmark worker.

| Method | Cold seconds | CGB + warm seconds |
|---|---:|---:|
| mean | 15.71 | 50.30 |
| mean_second | 17.17 | 48.67 |
