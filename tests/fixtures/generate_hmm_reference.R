# Regenerate with base R only, from the reference fSuSiE checkout:
# Rscript --vanilla tests/fixtures/generate_hmm_reference.R <hmm_routines.R> <output.json>
# Reference: fsusieR 9f0fd55, ash_hmm_version() == "3.0.0-variational".
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
source(args[1L])

# Small JSON writer to avoid depending on jsonlite in the reference environment.
json <- function(x) {
  if (is.matrix(x)) return(paste0("[", paste(vapply(seq_len(nrow(x)), function(i)
    paste0("[", paste(sprintf("%.17g", x[i, ]), collapse = ","), "]"), ""), collapse = ","), "]"))
  if (is.list(x)) {
    values <- vapply(x, json, "")
    if (!is.null(names(x))) return(paste0("{", paste(paste0('"', names(x), '":', values), collapse = ","), "}"))
    return(paste0("[", paste(values, collapse = ","), "]"))
  }
  if (is.logical(x)) return(ifelse(x, "true", "false"))
  if (is.character(x)) return(paste0('"', x, '"'))
  paste0("[", paste(sprintf("%.17g", x), collapse = ","), "]")
}

short_y <- c(0.1, -0.3, 1.8, 1.3, 2.1, -0.1, -1.7, -1.1, -0.9, 0.2, 0.4, 0)
short_se <- rep(c(0.3, 0.5, 0.8), 4)
long_y <- c(rep(0, 20), rep(2.3, 30), rep(-1.6, 30), rep(0, 20)) + 0.15 * cos(seq_len(100))
positive_y <- c(rep(0, 20), rep(1.3, 30), rep(3.2, 30), rep(0, 20)) + 0.15 * cos(seq_len(100))
cases <- list(
  fixed_signed = list(y = short_y, se = short_se, mu = c(0, 1.4, -1.2), prior_sd = c(0, 0.3, 1),
                      learn_state_means = FALSE, prune_states = FALSE, maxiter = 12L, tolerance = 1e-12),
  fixed_positive = list(y = short_y, se = short_se, mu = c(0, 1.4, 2.8), prior_sd = c(0, 0.3, 1),
                        nonnegative_state_means = TRUE, learn_state_means = FALSE, prune_states = FALSE,
                        maxiter = 12L, tolerance = 1e-12),
  learned_signed = list(y = long_y, se = rep(0.3, 100), mu = c(0, 1.8, 3, -1.2, -2.5),
                        prior_sd = c(0, 0.1, 0.5), prune_states = FALSE, maxiter = 8L,
                        mean_min_self_transition = 0.7, tolerance = 1e-12),
  learned_positive = list(y = positive_y, se = rep(0.3, 100), mu = c(0, 1, 2.8, 4),
                          prior_sd = c(0, 0.1, 0.5), nonnegative_state_means = TRUE,
                          prune_states = FALSE, maxiter = 8L, mean_min_self_transition = 0.7, tolerance = 1e-12),
  automatic_signed = list(y = long_y, se = rep(c(0.2, 0.5), 50)),
  automatic_positive = list(y = positive_y, se = rep(c(0.2, 0.5), 50), nonnegative_state_means = TRUE),
  adaptive_hub = list(y = short_y, se = short_se, mu = c(0, 1.4, -1.2), prior_sd = c(0, 0.3, 1),
                      null_state = "adaptive", topology = "hub", shared_mixture = TRUE, estimate_init = TRUE,
                      learn_state_means = FALSE, prune_states = FALSE, maxiter = 12L, tolerance = 1e-12)
)
cases$penalized_signed <- c(cases$fixed_signed, list(penalty = 1.5))
cases$penalized_positive <- c(cases$fixed_positive, list(penalty = 1.5, estimate_init = TRUE))
cases$penalized_adaptive_hub <- c(cases$adaptive_hub, list(penalty = 1.5))
output <- lapply(cases, function(input) {
  r_input <- input
  if (!is.null(input$penalty)) {
    # Map Python's scalar zero-favoring penalty to fSuSiE's Dirichlet priors.
    # Penalized reference cases supply explicit state/scale grids.
    r_input$penalty <- NULL
    state_prior <- c(input$penalty, rep(1, length(input$mu) - 1L))
    r_input$transition_prior <- state_prior
    if (isTRUE(input$estimate_init)) r_input$init_prior <- state_prior
    if (identical(input$null_state, "adaptive")) {
      r_input$mixture_prior <- matrix(1, length(input$mu), length(input$prior_sd))
      r_input$mixture_prior[1L, 1L] <- input$penalty
    }
  }
  fit <- do.call(fit_ash_hmm, r_input)
  list(input = input, post_mean = fit$posterior$mean,
       post_mean2 = fit$posterior$mean^2 + fit$posterior$sd^2,
       pi0_null = fit$posterior$probability_zero, lfsr = fit$posterior$lfsr,
       log_likelihood = fit$log_likelihood,
       state_probability = fit$state_probability, mu = fit$fitted$mu, prior_sd = fit$fitted$prior_sd,
       transition = fit$fitted$transition, init_prob = fit$fitted$init_prob,
       mixture_weight = fit$fitted$mixture_weight, history = fit$history$log_likelihood,
       objective_history = fit$history$objective,
       state_counts = fit$history$states)
})
writeLines(json(output), args[2L])
