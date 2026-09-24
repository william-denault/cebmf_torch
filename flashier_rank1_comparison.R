# R counterpart of the cEBMF rank-1 simulation, using flashier.
#
# Matches the Python loop:
#   - n = 50, p = 40, noise_std = 0.1
#   - rank-1 truth: u %o% v with u, v ~ Uniform(0, 1)
#   - K = 1, normal-scale-mixture priors on L and F (cEBMF's default
#     `prior="norm"` calls into ASH with a scale-mixture-of-normals prior,
#     i.e. `ebnm_normal_scale_mixture` here)
#   - constant noise variance (var_type = 0)
#   - SVD init, up to 50 inner iterations on the single factor
#   - 100 simulation replicates
#
# Compare against the Python output:
#   results : RMSE of Y_fit vs. true rank-1 matrix
#   tau_est : 1 / tau, i.e. estimated noise variance (truth = noise_std^2 = 0.01)

suppressPackageStartupMessages({
  library(flashier)
})

n        <- 50
p        <- 40
noise_sd <- 0.1
maxit    <- 50
n_sims   <- 100

rmse_vec   <- numeric(n_sims)
sigma2_vec <- numeric(n_sims)

for (i in seq_len(n_sims)) {
  set.seed(i)                                     # change to match your Python seeding scheme

  u      <- runif(n)
  v      <- runif(p)
  rank_1 <- u %o% v                               # n x p
  Y      <- rank_1 + matrix(rnorm(n * p, sd = noise_sd), n, p)

  fit <- flash_init(Y, var_type = 0) |>           # 0 = single constant variance
    flash_greedy(
      Kmax      = 1,
      ebnm_fn   = ebnm_normal_scale_mixture,      # closest analog to cEBMF prior = "norm"
      maxiter   = maxit,
      verbose   = 0
    ) |>
    flash_backfit(maxiter = maxit, verbose = 0)   # redundant for K=1 but mirrors the Python `fit(50)`

  Y_fit <- fitted(fit)                            # L F^T

  rmse_vec[i]   <- sqrt(mean((Y_fit - rank_1)^2))
  # var_type = 0: flashier stores a single scalar/constant precision.
  tau_val       <- as.numeric(fit$flash_fit$tau)[1]
  sigma2_vec[i] <- 1 / tau_val
}

cat("True noise variance:        ", noise_sd^2,        "\n")
cat("Mean RMSE  (Y_fit vs truth): ", mean(rmse_vec),   " (sd ", sd(rmse_vec),   ")\n", sep = "")
cat("Mean est.  noise variance  : ", mean(sigma2_vec), " (sd ", sd(sigma2_vec), ")\n", sep = "")

# Quick sanity plot, if you want:
# par(mfrow = c(1, 2))
# hist(rmse_vec,   main = "RMSE",                  xlab = "sqrt(mean((Y_fit - truth)^2))")
# hist(sigma2_vec, main = "Estimated noise var.",  xlab = "1 / tau"); abline(v = noise_sd^2, col = "red")
