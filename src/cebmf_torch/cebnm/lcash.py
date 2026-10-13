"""LC-ASH: Linear Covariate Adaptive Shrinkage.

For F features and K components, two parameterisations:
  - Softmax (multinomial logistic): K*F coefficients and K intercepts.
  - Proportional odds (ordered logistic): F shared coefficients and K-1 cut-points.

Both map gene features to mixture weights.  A linear alternative to the
MLP-based CASH solver, with ash-based bias/cut-point initialisation and
grid pruning.
"""

import math
import operator
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from cebmf_torch.cebnm.cash_solver import (
    cash_PosteriorMeanNorm,
    pen_loglik_loss_from_log_pi,
)
from cebmf_torch.ebnm.ash import PriorType, ash
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.mixture import autoselect_scales_mix_norm
from cebmf_torch.utils.posterior import posterior_mean_norm

_MODEL_PARAM_VERSION = 1


def _require_finite(value: torch.Tensor, name: str) -> None:
    """Raise FloatingPointError if a tensor contains non-finite values."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} must be finite.")


def _log_gap_mass(log_gap: torch.Tensor) -> torch.Tensor:
    """Compute log(1-exp(-exp(log_gap))) with a finite small-gap derivative."""
    finfo = torch.finfo(log_gap.dtype)
    small = log_gap <= math.log(finfo.eps)
    large = log_gap >= math.log(-math.log(finfo.tiny))
    # Mask before exponentiation: an invalid unused branch can spoil gradients.
    regular_input = torch.where(small | large, torch.zeros_like(log_gap), log_gap)
    gap = regular_input.exp()
    regular = torch.log(-torch.expm1(-gap))
    return torch.where(small, log_gap, torch.where(large, torch.zeros_like(log_gap), regular))


# ============================================================
# Model classes
# ============================================================


class LcashNet(nn.Module):
    """Multinomial logistic regression: features -> mixture weights.

    A single nn.Linear(F, K) followed by softmax. Equivalent to
    multinomial logistic regression with K classes and F features.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of mixture components (output classes).
    log_pi_init : torch.Tensor or None
        If provided, (K,) tensor of centred log-weights from a global ash
        fit. Used to initialise the bias and zero slopes so that the
        initial mixture matches the retained weights from ``ash``.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        log_pi_init: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if log_pi_init is not None:
            device = log_pi_init.device if device is None else device
            dtype = log_pi_init.dtype if dtype is None else dtype
        # Allocate without random initialization, then set the mixture weights.
        self.linear = nn.utils.skip_init(
            nn.Linear, input_dim, num_classes, device=torch.empty(0).device if device is None else device, dtype=dtype
        )
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            if log_pi_init is not None:
                self.linear.bias.copy_(log_pi_init)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized log mixture weights."""
        return F.log_softmax(self.linear(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_probs(x).exp()


class PropOddsLcashNet(nn.Module):
    """Proportional odds (ordered logistic) mapping: features -> mixture weights.

    A single shared weight vector maps features to a scalar signal
    strength s_i = x_i^T w.  K-1 ordered cut-points convert s_i to
    mixture weights via cumulative logistic probabilities.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of mixture components (K).
    log_pi_init : torch.Tensor or None
        If provided, (K,) tensor of centred log-weights from a global ash
        fit.  Used to initialise ordered cut-points so that the model
        recovers the global ash pi when all feature coefficients are zero.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        log_pi_init: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        K = num_classes

        if log_pi_init is not None:
            device = log_pi_init.device if device is None else device
            dtype = log_pi_init.dtype if dtype is None else dtype
        self.w = nn.Parameter(torch.zeros(input_dim, device=device, dtype=dtype))

        # Cut-point parameterisation: delta_1 (free), delta_2..K-1 (gaps)
        if log_pi_init is not None and K > 1:
            # Use CPU float64 for initialization, including when fitting on MPS.
            log_pi = _as_float64(log_pi_init, torch.device("cpu"))
            log_pi = log_pi - torch.logsumexp(log_pi, dim=0)
            init_cuts = self._init_cutpoints_from_pi(log_pi, K)
            if K > 2:
                # logit(C+p)-logit(C) = log1p(p/C)+log1p(p/(1-C-p)).
                gaps = F.softplus(log_pi[1:-1] - F.logsigmoid(init_cuts[:-1]))
                gaps = gaps + F.softplus(log_pi[1:-1] - F.logsigmoid(-init_cuts[1:]))
                log_gaps = gaps.log()
        else:
            init_cuts = torch.linspace(-2.0, 2.0, K - 1, device=device, dtype=dtype)
            if K > 2:
                log_gaps = (init_cuts[1:] - init_cuts[:-1]).log()

        self.delta_1 = nn.Parameter(init_cuts[0:1].to(self.w))  # (1,), empty for K=1
        if K > 2:
            self.delta_gaps = nn.Parameter(log_gaps.to(self.w))  # (K-2,)
        else:
            self.delta_gaps = None

        self._K = K

    @staticmethod
    def _init_cutpoints_from_pi(log_pi_init: torch.Tensor, K: int) -> torch.Tensor:
        """Compute log cumulative mass minus log survival mass at zero slopes."""
        log_cdf = torch.logcumsumexp(log_pi_init, dim=0)[: K - 1]
        log_survival = torch.logcumsumexp(log_pi_init.flip(0), dim=0).flip(0)[1:K]
        return log_cdf - log_survival

    def _get_cutpoints(self) -> torch.Tensor:
        """Reconstruct ordered cut-points from unconstrained parameters."""
        # K=1: degenerate case, all weight on the single component.
        if self._K == 1:
            return self.delta_1.new_empty(0)
        if self.delta_gaps is not None:
            # Leave room for the cumulative sum of saturated, positive gaps.
            max_log_gap = math.log(torch.finfo(self.delta_gaps.dtype).max) - math.log(self._K) - 1
            gaps = self.delta_gaps.clamp(max=max_log_gap).exp()
            cuts = torch.cat([self.delta_1, self.delta_1 + torch.cumsum(gaps, dim=0)])
        else:
            cuts = self.delta_1  # K = 2: single cut-point
        return cuts

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return log category masses from cumulative logits and log gaps."""
        if self._K == 1:
            return x.new_zeros((x.shape[0], 1))
        z = self._get_cutpoints()[None, :] - (x @ self.w)[:, None]
        parts = [F.logsigmoid(z[:, :1])]
        if self._K > 2:
            parts.append(F.logsigmoid(z[:, 1:]) + F.logsigmoid(-z[:, :-1]) + _log_gap_mass(self.delta_gaps)[None, :])
        parts.append(F.logsigmoid(-z[:, -1:]))
        result = torch.cat(parts, dim=1)
        result = result - torch.logsumexp(result, dim=1, keepdim=True)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_probs(x).exp()


# ============================================================
# Shared helpers
# ============================================================


def _as_float64(value, device: torch.device) -> torch.Tensor:
    """Move tensors before casting; MPS cannot create a float64 intermediate."""
    if isinstance(value, torch.Tensor):
        value = value.to(device=device)
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _prepare_inputs(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert inputs to float64 tensors on the fitting device."""
    X = _as_float64(X, device)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    betahat = _as_float64(betahat, device)
    sebetahat = _as_float64(sebetahat, device)
    return X, betahat, sebetahat


def _feature_statistics(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute column means and SDs; return zero for all-missing columns."""
    mask = ~torch.isnan(X)
    counts = mask.sum(dim=0)  # (F,)

    # Replace NaN with 0 for safe summation
    X_filled = torch.where(mask, X, torch.zeros_like(X))

    # Mean on observed values
    safe_counts = counts.clamp(min=1)
    mu = X_filled.sum(dim=0) / safe_counts  # (F,)

    # Divide by the observed count, using one for all-missing columns.
    diff = torch.where(mask, X - mu, torch.zeros_like(X))
    var = (diff**2).sum(dim=0) / safe_counts  # (F,)
    sd = var.sqrt()
    return mu, sd


def _nanstandardise(
    X: torch.Tensor,
    feature_mean: torch.Tensor | None = None,
    feature_sd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply saved feature statistics, or estimate them for a fresh fit.

    Missing entries and columns with zero training SD contribute zero.
    Retaining zero SDs keeps constant and all-missing training columns
    disabled when the saved model is applied to new observations.
    """
    if feature_mean is None:
        feature_mean, feature_sd = _feature_statistics(X)
    safe_sd = torch.where(feature_sd > 0, feature_sd, torch.ones_like(feature_sd))
    return torch.where(
        ~torch.isnan(X) & (feature_sd > 0),
        (X - feature_mean) / safe_sd,
        torch.zeros_like(X),
    )


def _validate_grid(scale: torch.Tensor) -> None:
    """Require an ordered spike-and-slab grid for conditional mixture fitting."""
    if scale.ndim != 1 or scale.numel() == 0:
        raise ValueError("LC-ASH and PO-LC-ASH require at least one mixture component.")
    if not torch.isfinite(scale).all() or not torch.all(scale[1:] > scale[:-1]) or scale[0] != 0:
        raise ValueError("Mixture scales must be finite and strictly increasing, starting with the spike at zero.")


def _restore_model_param(
    model_param: dict, label: str, X: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate fitted-state metadata before loading model coefficients."""
    required = {"version", "solver", "scale", "feature_mean", "feature_sd", "state_dict"}
    if not isinstance(model_param, dict) or not required.issubset(model_param):
        raise ValueError(
            "model_param must be the complete fitted-state dictionary returned by the solver, "
            "including the component grid, model coefficients and feature statistics. "
            "Refit with model_param=None to obtain complete state."
        )
    if model_param["version"] != _MODEL_PARAM_VERSION or model_param["solver"] != label:
        raise ValueError(f"model_param must be {label} fitted state with version {_MODEL_PARAM_VERSION}.")
    scale, feature_mean, feature_sd = (
        _as_float64(model_param[key], X.device).detach().clone() for key in ("scale", "feature_mean", "feature_sd")
    )
    _validate_grid(scale)
    if feature_mean.shape != (X.shape[1],) or feature_sd.shape != feature_mean.shape:
        raise ValueError("model_param feature statistics do not match the number of covariate columns in X.")
    if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_sd).all() or (feature_sd < 0).any():
        raise ValueError("model_param requires finite feature means and finite, nonnegative feature SDs.")
    return scale, feature_mean, feature_sd


def _select_grid(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    mult: float,
    ash_init: bool,
    ash_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select a component grid and optional ``ash`` initialization for a fresh fit.

    With ``ash_init=True``, retain components whose ``ash`` weights exceed
    ``ash_threshold`` and compute centred log-weights for initialization.
    Otherwise retain the full automatically selected grid.

    Parameters
    ----------
    mult : float
        Multiplicative step between grid SDs.  Smaller values give a
        finer grid with more components.
    ash_init : bool
        If True, run ``ash`` internally with ``optimizer="lbfgs"`` to
        prune the grid to active components and initialise bias from
        the ``ash`` weights.
    ash_threshold : float
        Pruning threshold: components with ``pi <= ash_threshold``
        are dropped. Only used when ash_init=True.

    Returns
    -------
    scale : tensor (K,)
        Mixture component standard deviations.
    log_pi_init : tensor (K,) or None
        Centred log-weights for bias/cut-point initialisation, or
        None when ``ash_init=False``.
    """
    if ash_init:
        if not math.isfinite(ash_threshold) or ash_threshold < 0:
            raise ValueError("ash_threshold must be finite and nonnegative.")
        ash_result = ash(
            betahat,
            sebetahat,
            prior=PriorType.NORM,
            penalty=10.0,
            verbose=False,
            optimizer="lbfgs",
            mult=mult,
            zero_threshold=0.0,
        )
        pi_full = ash_result.pi
        active = pi_full > ash_threshold
        scale = ash_result.scale[active].to(device=device, dtype=betahat.dtype)
        try:
            _validate_grid(scale)
        except ValueError as error:
            discarded = pi_full[~active]
            largest_discarded = discarded.max().item() if discarded.numel() else 0.0
            raise ValueError(
                f"{error} ASH initialization retained {scale.numel()} components "
                f"with ash_threshold={ash_threshold:g} (retained scales: {scale.tolist()}; "
                f"largest discarded weight: {largest_discarded:.6g}; "
                f"spike weight: {pi_full[0].item():.6g}). "
                "Inspect the ASH fit and consider lowering ash_threshold."
            ) from error
        pi_active = pi_full[active]
        log_pi_init = torch.log(pi_active)
        log_pi_init = log_pi_init - log_pi_init.mean()
        log_pi_init = log_pi_init.to(device=device, dtype=betahat.dtype)
        return scale, log_pi_init

    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, mult=mult)
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=betahat.dtype, device=device)
    else:
        scale = scale.to(device=device, dtype=betahat.dtype)
    _validate_grid(scale)
    return scale, None


def _conditional_loss(
    model: nn.Module,
    X_scaled: torch.Tensor,
    data_loglik: torch.Tensor,
    penalty: float,
    coefficient_prior: torch.distributions.StudentT | None,
) -> torch.Tensor:
    """Mean negative log likelihood with nullweight and optional slope priors."""
    n = X_scaled.shape[0]
    loss = pen_loglik_loss_from_log_pi(model.log_probs(X_scaled), data_loglik, null_coefficient=(penalty - 1) / n)
    if coefficient_prior is not None:
        # Multinomial slopes are log-odds contrasts against the null component.
        slopes = model.linear.weight[1:] - model.linear.weight[:1] if isinstance(model, LcashNet) else model.w
        loss = loss - coefficient_prior.log_prob(slopes).sum()
    return loss / n


def _train_model(
    model: nn.Module,
    X_scaled: torch.Tensor,
    data_loglik: torch.Tensor,
    max_iter: int,
    penalty: float,
    verbose: bool,
    label: str,
    coefficient_prior_scale: float | None = 1.0,
    coefficient_prior_df: float = 5.0,
) -> None:
    """Fit the mean objective with nullweight and optional Student-t slope priors.

    A non-finite objective or gradient raises instead of returning a fit.
    Each call uses fresh curvature history for its normal-means problem.
    """
    model.train()
    params = list(model.parameters())
    coefficient_prior = None
    if coefficient_prior_scale is not None:
        # The public entry point validates these scalars once. Avoid repeated
        # distribution support checks inside the L-BFGS closure.
        coefficient_prior = torch.distributions.StudentT(
            X_scaled.new_tensor(coefficient_prior_df), scale=coefficient_prior_scale, validate_args=False
        )
    optimizer = torch.optim.LBFGS(
        params,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = _conditional_loss(model, X_scaled, data_loglik, penalty, coefficient_prior)
        _require_finite(loss, "Conditional optimization objective")
        loss.backward()
        _require_finite(torch.cat([p.grad.reshape(-1) for p in params]), "Conditional optimization gradient")
        return loss

    optimizer.step(closure)
    # LBFGS.step returns its initial loss; evaluate the actual endpoint.
    final_loss = closure().detach().item()
    grad_max = max(p.grad.abs().max().item() for p in params if p.numel())
    if grad_max > optimizer.param_groups[0]["tolerance_grad"]:
        warnings.warn(
            f"{label} L-BFGS returned a finite fit above the gradient tolerance. Convergence is not established.",
            RuntimeWarning,
            stacklevel=4,
        )
    if verbose:
        state = optimizer.state[params[0]]
        print(
            f"[{label}] L-BFGS iterations: {state['n_iter']} | "
            f"Evaluations: {state['func_evals']} | Mean loss: {final_loss:.8g} | "
            f"Max gradient: {grad_max:.3g}"
        )


def _compute_posteriors(
    model: nn.Module,
    X_scaled: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
    data_loglik: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Vectorised posterior computation with per-observation pi.

    Assumes location = 0 for all mixture components (spike mean is 0).
    This matches the zero-centred normal mixture prior used by LC-ASH.

    Returns
    -------
    post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik
        ``marginal_loglik`` is the full-data marginal log-likelihood
        ``sum_g logsumexp_k (log pi_g,k + log p(beta_g | 0, sqrt(se_g^2 + scale_k^2)))``,
        i.e. ``log p(y | fitted prior)`` without fitting penalties.
        The outer cEBMF KL calculation requires this unpenalized sum.
    """
    model.eval()
    loc = torch.zeros_like(scale)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            _require_finite(parameter, f"Conditional parameter {name}")
        log_pi_all = model.log_probs(X_scaled)
        _require_finite(log_pi_all, "Conditional log probabilities")
        all_pi_values = log_pi_all.exp()  # (G, K)

        loglik = torch.logsumexp(data_loglik + log_pi_all, dim=1).sum()
        _require_finite(loglik, "Conditional marginal log likelihood")
        posterior = posterior_mean_norm(betahat, sebetahat, log_pi_all, data_loglik, scale, location=loc)
        for name in ("post_mean", "post_mean2", "post_sd"):
            _require_finite(getattr(posterior, name), f"Conditional {name}")

    return posterior.post_mean, posterior.post_mean2, posterior.post_sd, all_pi_values, float(loglik.item())


def _fit_lcash(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    model_class: type,
    label: str,
    *,
    max_iter: int | None = None,
    penalty: float = 10.0,
    coefficient_prior_scale: float | None = 1.0,
    coefficient_prior_df: float = 5.0,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> cash_PosteriorMeanNorm:
    """Shared implementation for both softmax and proportional odds LC-ASH.

    Parameters
    ----------
    model_class : type
        Either ``LcashNet`` or ``PropOddsLcashNet``.
    label : str
        Label for verbose logging (e.g. "LC-ASH" or "PO-LC-ASH").

    See ``lcash_posterior_means`` for other parameter descriptions.
    """
    # Inherit from input tensor when available; avoids silent device hops if
    # the caller (e.g. cEBMF) is on CPU/MPS but CUDA is also visible.
    if device is None:
        device = (
            betahat.device
            if isinstance(betahat, torch.Tensor)
            else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )
    output_device = torch.device(device)
    device = torch.device("cpu") if output_device.type == "mps" else output_device
    if max_iter is None:
        max_iter = 500

    X, betahat, sebetahat = _prepare_inputs(X, betahat, sebetahat, device)
    if X.ndim != 2 or betahat.ndim != 1 or sebetahat.shape != betahat.shape or X.shape[0] != betahat.shape[0]:
        raise ValueError("X, betahat, and sebetahat must have matching observation counts.")
    if betahat.numel() == 0:
        raise ValueError("Conditional fitting requires a positive observation count.")
    try:
        max_iter = operator.index(max_iter)
    except TypeError:
        raise ValueError("max_iter must be a nonnegative integer or None.") from None
    if max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer or None.")
    if max_iter > 0 and (not math.isfinite(penalty) or penalty < 1):
        raise ValueError("penalty must be finite and at least one.")
    if max_iter > 0 and coefficient_prior_scale is not None:
        for name, value in (
            ("coefficient_prior_scale", coefficient_prior_scale),
            ("coefficient_prior_df", coefficient_prior_df),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
    if model_param is None:
        feature_mean, feature_sd = _feature_statistics(X)
        scale, log_pi_init = _select_grid(betahat, sebetahat, mult, ash_init, ash_threshold, device)
    else:
        scale, feature_mean, feature_sd = _restore_model_param(model_param, label, X)
        log_pi_init = None
    X_scaled = _nanstandardise(X, feature_mean, feature_sd)

    K = scale.shape[0]
    model = model_class(X_scaled.shape[1], K, log_pi_init=log_pi_init, device=device, dtype=X.dtype)
    if model_param is not None:
        model.load_state_dict({name: value.to(device=device) for name, value in model_param["state_dict"].items()})

    with torch.no_grad():
        data_loglik = get_data_loglik_normal_torch(betahat, sebetahat, torch.zeros_like(scale), scale)
        _require_finite(data_loglik, "Conditional component log likelihoods")
    if K > 1 and max_iter > 0:
        _train_model(
            model,
            X_scaled,
            data_loglik,
            max_iter,
            penalty,
            verbose,
            label,
            coefficient_prior_scale,
            coefficient_prior_df,
        )

    post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik = _compute_posteriors(
        model,
        X_scaled,
        betahat,
        sebetahat,
        scale,
        data_loglik,
    )

    fitted_state = {
        "version": _MODEL_PARAM_VERSION,
        "solver": label,
        "scale": scale.detach().clone(),
        "feature_mean": feature_mean.detach().clone(),
        "feature_sd": feature_sd.detach().clone(),
        "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
    }
    output = {
        name: value.to(device=output_device, dtype=torch.float32)
        for name, value in (
            ("post_mean", post_mean),
            ("post_mean2", post_mean2),
            ("post_sd", post_sd),
            ("pi_np", all_pi_values),
            ("scale", scale),
        )
    }
    # Separate float32 rounding must preserve E[theta^2] >= E[theta]^2.
    output["post_mean2"] = torch.maximum(output["post_mean2"], output["post_mean"].square())
    for name, value in output.items():
        _require_finite(value, f"Returned conditional {name}")
    _validate_grid(output["scale"])
    # cEBMF's per-factor KL calculation uses the unpenalized sum.
    return cash_PosteriorMeanNorm(**output, loss=-marginal_loglik, model_param=fitted_state)


# ============================================================
# Public entry points
# ============================================================


def lcash_posterior_means(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    *,
    max_iter: int | None = None,
    penalty: float = 10.0,
    coefficient_prior_scale: float | None = 1.0,
    coefficient_prior_df: float = 5.0,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> cash_PosteriorMeanNorm:
    """LC-ASH: linear covariate-modulated mixture weights.

    Multinomial logistic regression models the weights of zero-centred
    normal components. The component grid is their ordered standard
    deviations, with zero denoting a point mass at zero (the spike).

    Parameters
    ----------
    X : tensor (G, F)
        Feature matrix with G observations and F covariates. Columns are
        standardized using covariate means and SDs from non-NaN values
        on the fresh fit. For columns with observed values, variance uses
        their count as its divisor. Warm starts use the saved statistics.
        Missing entries and columns with zero training SD contribute
        zero. Covariate columns must keep their meaning, order and units.
    betahat : tensor (G,)
        Effect estimates.
    sebetahat : tensor (G,)
        Standard errors.
    max_iter : int or None
        Maximum full-batch L-BFGS iterations. None uses 500 standalone or
        twice internal_epoch inside cEBMF. An explicit integer, including
        zero, overrides this budget. The mean objective includes the null
        penalty and the coefficient prior. Each iteration can evaluate
        the objective more than once. A budget limit does not imply convergence.
    penalty : float
        Nullweight penalty as in ``ash``, finite and at least one. The coefficient in
        conditional training is (penalty - 1) / n for the full training count.
        Default 10.0; 1.0 omits the null penalty. The ``ash`` initializer always
        uses 10.0.
    coefficient_prior_scale : float or None
        Scale of a zero-centred Student-t prior on each component's
        standardized covariate slopes relative to the null component.
        Intercepts are not penalized. Default 1.0; None disables the prior.
        A supplied scale must be finite and positive. Scale is not the
        prior's standard deviation. The summed negative log density from
        ``torch.distributions.StudentT`` is divided by the training count
        in the mean objective.
    coefficient_prior_df : float
        Degrees of freedom of the coefficient prior. Default 5.0.
        Must be finite and positive when the prior is enabled.
    mult : float
        Multiplicative step between mixture grid SDs for a fresh fit.
        Smaller values give a finer grid. Default sqrt(2).
        Ignored when ``model_param`` is supplied.
    ash_init : bool
        If True (default), fit ASH with L-BFGS, select components using
        ``ash_threshold``, and initialise the bias from their retained
        weights. At zero feature coefficients, the initial mixture
        weights equal the retained ASH weights after renormalization
        within floating-point accuracy. The initial coefficients are zero.
        If False, use the full grid with uniform bias initialisation.
        Ignored when ``model_param`` is supplied.
    ash_threshold : float
        For fresh ``ash`` initialization, remove components with fitted
        weight at or below this value. Default 1e-6.
        Explicit zero retains every positive weight returned by ``ash``, including
        weights supplied by its numerical floor. Initialization
        raises if no component remains or the spike is removed. A single
        retained spike returns its exact posterior without optimization.
        Ignored when ``ash_init=False`` or ``model_param`` is supplied.
    model_param : dict or None
        Pass the complete returned state unchanged to the same solver.
        It preserves the fitted prior, component grid and covariate
        standardization without repeating ``ash`` or grid selection.
        Effect estimates, standard errors and observation count may change; feature
        columns must retain their meaning, order and units.
        Each training call creates a new L-BFGS optimizer unless only the
        spike remains. Use ``max_iter=0`` to calculate posterior summaries
        with the saved prior, without evaluating either penalty.
        Training settings are not stored; pass the same nullweight and
        coefficient-prior settings to continue the same objective.
    device : torch.device or None
        Compute device. Inherited from ``betahat`` when it is a tensor;
        otherwise defaults to CUDA if available, then CPU. Fitting and saved
        state use float64. For MPS, fitting runs on CPU, including ``ash``.
        Posterior summaries, probabilities and scales return as float32 on
        the requested device; saved state remains on the fitting device.
    verbose : bool
        If True, print the final objective, gradient and iteration counts.
        Default False. Convergence warnings do not depend on this option.

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with post_mean, post_mean2, post_sd, pi_np (G, K),
        scale (K,), loss, and model_param (fitted state for warm-starting).
        ``loss`` is the summed negative marginal log likelihood without
        either penalty.

    Raises
    ------
    ValueError
        If component selection leaves no components or removes the spike,
        or supplied state has incompatible metadata,
        scales or feature statistics.
    FloatingPointError
        If model parameters, probabilities, likelihoods, or gradients are
        non-finite. Invalid updates are not returned as fitted models.
    RuntimeError
        If supplied model parameter names or shapes are incompatible.
    """
    return _fit_lcash(
        X,
        betahat,
        sebetahat,
        model_class=LcashNet,
        label="LC-ASH",
        max_iter=max_iter,
        penalty=penalty,
        coefficient_prior_scale=coefficient_prior_scale,
        coefficient_prior_df=coefficient_prior_df,
        mult=mult,
        ash_init=ash_init,
        ash_threshold=ash_threshold,
        model_param=model_param,
        device=device,
        verbose=verbose,
    )


def po_lcash_posterior_means(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    *,
    max_iter: int | None = None,
    penalty: float = 10.0,
    coefficient_prior_scale: float | None = 1.0,
    coefficient_prior_df: float = 5.0,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> cash_PosteriorMeanNorm:
    """Proportional odds LC-ASH: ordered logistic covariate-modulated weights.

    For F features and K components, a shared weight vector maps
    features to a scalar signal strength
    s_i = x_i^T w.  K-1 ordered cut-points convert s_i to mixture
    weights via cumulative logistic probabilities. The model stores
    F + K - 1 parameters. Softmax LC-ASH stores K * F coefficients
    and K intercepts.

    The coefficient prior is Student-t with zero location, scale
    ``coefficient_prior_scale`` (default 1.0), and degrees of freedom
    ``coefficient_prior_df`` (default 5.0). It applies to the shared
    standardized slope vector w. Cut-points are not penalized.
    Set ``coefficient_prior_scale=None`` to disable it.

    On a fresh fit with ``ash_init=True``, select components from ASH
    at penalty 10 and initialise cut-points from retained, normalized
    weights. Zero initial feature coefficients preserve this mixture
    within floating-point accuracy.
    With ``ash_init=False``, retain the full grid and initialise
    cut-points evenly between -2 and 2.

    Supplied ``model_param`` restores the selected component grid, shared
    feature coefficients, cut-point parameters and covariate standardization.
    The cut-point parameters are the first cut-point and log increments
    between successive cut-points. ASH and component selection are skipped.
    Each training call creates a new L-BFGS optimizer unless only the spike
    remains. Other parameters have the same meaning as in
    :func:`lcash_posterior_means`.

    Parameters
    ----------
    X : tensor (G, F)
        Feature matrix, standardized as in :func:`lcash_posterior_means`.
        Columns must retain their meaning, order and units.

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with post_mean, post_mean2, post_sd, pi_np (G, K),
        scale (K,), loss, and model_param (fitted state for warm-starting).

    Raises
    ------
    ValueError
        If component selection or supplied fitted state is incompatible;
        see :func:`lcash_posterior_means`.
    RuntimeError
        If supplied model parameter names or shapes are incompatible.
    """
    return _fit_lcash(
        X,
        betahat,
        sebetahat,
        model_class=PropOddsLcashNet,
        label="PO-LC-ASH",
        max_iter=max_iter,
        penalty=penalty,
        coefficient_prior_scale=coefficient_prior_scale,
        coefficient_prior_df=coefficient_prior_df,
        mult=mult,
        ash_init=ash_init,
        ash_threshold=ash_threshold,
        model_param=model_param,
        device=device,
        verbose=verbose,
    )
