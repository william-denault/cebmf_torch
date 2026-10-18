# Run from the repository root with ebnm and jsonlite installed.
# Reference: stephenslab/ebnm 825edf1e308af9f3e594a5482300b9fb99793129
# (version 1.1-42). No R installation is needed to run the Python tests.
library(ebnm)
library(jsonlite)
stopifnot(packageVersion("ebnm") == package_version("1.1.42"))

fixed_case <- function(name, x, s, weight, rate, mode) {
  prior <- gammamix(pi = c(1 - weight, weight), shape = c(1, 1),
                   scale = c(0, 1 / rate), shift = rep(mode, 2))
  result <- ebnm_point_exponential(
    x, s, g_init = prior, fix_g = TRUE,
    output = c("posterior_mean", "posterior_sd", "posterior_second_moment",
               "fitted_g", "log_likelihood")
  )
  list(name = name, x = x, s = s, weight = weight, rate = rate, mode = mode,
       mean = result$posterior$mean, mean2 = result$posterior$second_moment,
       sd = result$posterior$sd, log_lik = as.numeric(result$log_likelihood))
}

fixed <- list(
  fixed_case("heteroskedastic", c(-4, -1, 0, .5, 3, 8),
             c(.5, .8, 1, 2, .7, 1.5), .6, .7, 0),
  fixed_case("shifted", c(-4, -1, 0, 2), c(.8, .7, 1, 2), .75, 1.3, -1.2),
  fixed_case("spike", c(-2, 0, 3), c(.5, 1, 2), 0, 1, .3),
  fixed_case("exponential", c(-2, 0, 3), c(.5, 1, 2), 1, .8, 0)
)

x <- c(1, 1, -.5)
s <- c(1, 1, 1)
fitted <- ebnm_point_exponential(
  x, s, output = c("posterior_mean", "posterior_sd", "posterior_second_moment",
                   "fitted_g", "log_likelihood")
)
fit <- list(x = x, s = s, weight = tail(fitted$fitted_g$pi, 1),
            rate = 1 / tail(fitted$fitted_g$scale, 1),
            mean = fitted$posterior$mean,
            mean2 = fitted$posterior$second_moment,
            log_lik = as.numeric(fitted$log_likelihood))

write_json(
  list(ebnm_version = as.character(packageVersion("ebnm")),
       ashr_version = as.character(packageVersion("ashr")),
       ebnm_revision = "825edf1e308af9f3e594a5482300b9fb99793129",
       fixed = fixed, fit = fit),
  "tests/fixtures/point_exp_r_reference.json", auto_unbox = TRUE,
  pretty = TRUE, digits = NA
)
