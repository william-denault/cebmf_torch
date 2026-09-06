"""LC-ASH: Linear Covariate Adaptive Shrinkage.

Two parameterisations:
  - Softmax (multinomial logistic): K independent logit vectors, K*F params.
  - Proportional odds (ordered logistic): shared weight vector, F+K-1 params.

Both map gene features to mixture weights.  A linear alternative to the
MLP-based CASH solver, with ash-based bias/cut-point initialisation and
grid pruning.
"""

import torch
import torch.nn as nn

from cebmf_torch.cebnm.cash_solver import (
    cash_PosteriorMeanNorm,
    pen_loglik_loss,
)
from cebmf_torch.ebnm.ash import PriorType, ash
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.mixture import autoselect_scales_mix_norm

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
        fit. Used to initialise the bias so that softmax(bias) approximates
        the global ash pi when all feature coefficients are zero.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        log_pi_init: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        # Small random perturbation breaks symmetry across features.
        # Starting from exact zeros leads Adam to different local
        # optima on high-dimensional feature sets (F > 100).
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01, generator=generator)
        if log_pi_init is not None:
            with torch.no_grad():
                self.linear.bias.copy_(log_pi_init)
        else:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=1)


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
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        K = num_classes

        # Shared feature weights: initialised near zero so the model
        # starts close to the exchangeable prior.
        self.w = nn.Parameter(torch.empty(input_dim))
        nn.init.normal_(self.w, mean=0.0, std=0.01, generator=generator)

        # Cut-point parameterisation: delta_1 (free), delta_2..K-1 (gaps)
        if log_pi_init is not None and K > 1:
            init_cuts = self._init_cutpoints_from_pi(log_pi_init, K)
        else:
            init_cuts = torch.linspace(-2.0, 2.0, K - 1)

        self.delta_1 = nn.Parameter(init_cuts[0:1])  # (1,)
        if K > 2:
            gaps = torch.log(torch.clamp(init_cuts[1:] - init_cuts[:-1], min=1e-6))
            self.delta_gaps = nn.Parameter(gaps)  # (K-2,)
        else:
            self.delta_gaps = None

        self._K = K

    @staticmethod
    def _init_cutpoints_from_pi(log_pi_init: torch.Tensor, K: int) -> torch.Tensor:
        """Initialise cut-points so that sigma(theta_k) approx cumprob_k.

        At initialisation w ~ 0, so s_i ~ 0 for all genes.  Then
        pi_k = sigma(theta_{k+1}) - sigma(theta_k), so we need
        sigma(theta_k) = sum_{j<k} pi_j, i.e. theta_k = logit(cumprob_k).
        """
        pi = torch.exp(log_pi_init - log_pi_init.max())
        pi = pi / pi.sum()
        cumprob = torch.cumsum(pi, dim=0)[:-1]  # K-1 values
        cumprob = torch.clamp(cumprob, 1e-6, 1 - 1e-6)
        cuts = torch.log(cumprob / (1 - cumprob))
        return cuts

    def _get_cutpoints(self) -> torch.Tensor:
        """Reconstruct ordered cut-points from unconstrained parameters."""
        # K=1: degenerate case, all weight on the single component.
        if self._K == 1:
            return torch.empty(0, device=self.delta_1.device)
        if self.delta_gaps is not None:
            gaps = torch.exp(self.delta_gaps)
            return torch.cat([self.delta_1, self.delta_1 + torch.cumsum(gaps, dim=0)])
        return self.delta_1  # K = 2: single cut-point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mixture weights pi_k for each gene.

        Parameters
        ----------
        x : tensor (G, F)
            Feature matrix.

        Returns
        -------
        tensor (G, K)
            Per-gene mixture weights.
        """
        s = x @ self.w  # (G,)
        theta = self._get_cutpoints()  # (K-1,)

        # Cumulative probabilities: P(category <= k) = sigma(theta_k - s)
        cum_probs = torch.sigmoid(theta.unsqueeze(0) - s.unsqueeze(1))  # (G, K-1)

        # Convert cumulative to category probabilities
        ones = torch.ones(s.shape[0], 1, device=x.device)
        zeros = torch.zeros(s.shape[0], 1, device=x.device)
        cum_ext = torch.cat([zeros, cum_probs, ones], dim=1)  # (G, K+1)
        pi = cum_ext[:, 1:] - cum_ext[:, :-1]  # (G, K)

        # Numerical safety: clamp small negatives from floating-point
        pi = torch.clamp(pi, min=1e-10)
        pi = pi / pi.sum(dim=1, keepdim=True)

        return pi


# ============================================================
# Shared helpers
# ============================================================


def _prepare_inputs(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert inputs to float32 tensors on device and standardise X.

    Uses NaN-aware standardisation: mean and std are computed on
    non-NaN values only, then NaN positions are zero-filled.  This
    ensures that missing features contribute nothing to the logits
    (falling back to the intercept/global prior) and that the
    statistics are not biased by the zero-fill.
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    betahat = torch.as_tensor(betahat, dtype=torch.float32, device=device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32, device=device)
    X_scaled = _nanstandardise(X)
    return X_scaled, betahat, sebetahat


def _nanstandardise(X: torch.Tensor) -> torch.Tensor:
    """Standardise columns using non-NaN values, then zero-fill NaN.

    Vectorised implementation. For each column, compute mean and
    population std on observed (non-NaN) entries, standardise observed
    values, and set NaN positions to 0. Columns with zero std
    (constant or all-NaN) are set to 0.
    """
    mask = ~torch.isnan(X)
    counts = mask.sum(dim=0)  # (F,)

    # Replace NaN with 0 for safe summation
    X_filled = torch.where(mask, X, torch.zeros_like(X))

    # Mean on observed values
    safe_counts = counts.clamp(min=1)
    mu = X_filled.sum(dim=0) / safe_counts  # (F,)

    # Population std on observed values
    diff = torch.where(mask, X - mu, torch.zeros_like(X))
    var = (diff**2).sum(dim=0) / safe_counts  # (F,)
    sd = var.sqrt()

    # Standardise observed, zero-fill missing
    safe_sd = torch.where((sd > 0) & (counts > 1), sd, torch.ones_like(sd))
    X_out = torch.where(
        mask & (sd > 0).unsqueeze(0) & (counts > 1).unsqueeze(0),
        diff / safe_sd,
        torch.zeros_like(X),
    )
    return X_out


def _select_grid(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    mult: float,
    ash_init: bool,
    ash_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select mixture grid and (optionally) initialise from ash.

    Always builds the grid via ``autoselect_scales_mix_norm(mult=mult)``.
    When ``ash_init=True``, additionally runs a full ash fit (L-BFGS
    optimizer) to determine which components are active and initialise
    the bias/cut-points from the ash mixture weights.

    Parameters
    ----------
    mult : float
        Multiplicative step between grid SDs.  Smaller values give a
        finer grid with more components (sqrt(2) ≈ 27 components,
        2.0 ≈ 15 components for typical data).
    ash_init : bool
        If True, run ash internally with ``optimizer="lbfgs"`` to
        prune the grid to active components and initialise bias from
        the ash weights.
    ash_threshold : float
        Pruning threshold: components with ``pi <= ash_threshold``
        are dropped.  Only used when ``ash_init=True``.

    Returns
    -------
    scale : tensor (K,)
        Mixture component standard deviations.
    log_pi_init : tensor (K,) or None
        Centred log-weights for bias/cut-point initialisation, or
        None when ``ash_init=False``.
    """
    if ash_init:
        ash_result = ash(betahat, sebetahat, prior=PriorType.NORM, verbose=False, optimizer="lbfgs", mult=mult)
        pi_full = ash_result.pi
        active = pi_full > ash_threshold
        # Fallback: ensure at least K=2 (spike + one slab)
        if active.sum() < 2:
            active = torch.zeros_like(pi_full, dtype=torch.bool)
            active[0] = True
            non_spike = pi_full.clone()
            non_spike[0] = -1.0
            active[non_spike.argmax()] = True
        scale = ash_result.scale[active].to(device=device, dtype=torch.float32)
        pi_active = pi_full[active]
        log_pi_init = torch.log(pi_active.clamp(min=1e-30))
        log_pi_init = log_pi_init - log_pi_init.mean()
        log_pi_init = log_pi_init.to(device=device, dtype=torch.float32)
        return scale, log_pi_init

    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, mult=mult)
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    else:
        scale = scale.to(device=device, dtype=torch.float32)
    return scale, None


def _train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_scaled: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    penalty: float,
    verbose: bool,
    label: str,
    seed: int = 42,
) -> float:
    """Run the training loop. Returns the final-epoch total loss.

    Pre-computes the (G, K) log-likelihood matrix once rather than
    recomputing per mini-batch (logL is constant during training).
    Batch ordering is seeded for reproducibility.
    """
    model.train()
    device = X_scaled.device

    # Pre-compute log-likelihood matrix (constant during training).
    loc = torch.zeros_like(scale)
    with torch.no_grad():
        logL_all = get_data_loglik_normal_torch(
            betahat=betahat,
            sebetahat=sebetahat,
            location=loc,
            scale=scale,
        )

    # Seeded manual batching (5x faster than DataLoader due to
    # avoiding per-sample __getitem__ and collation overhead).
    # Generator and permutation are created on the same device as the
    # data to avoid CPU/GPU device mismatches.
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    n = X_scaled.shape[0]
    n_batches = max(1, (n + batch_size - 1) // batch_size)

    final_epoch_loss = 0.0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        perm = torch.randperm(n, generator=g, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pi_pred = model(X_scaled[idx])
            loss = pen_loglik_loss(pi_pred, logL_all[idx], penalty=penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_epoch_loss = epoch_loss
        if verbose and (epoch + 1) % 50 == 0:
            print(f"[{label}] Epoch {epoch + 1}/{n_epochs} | Loss: {epoch_loss / n_batches:.4f}")

    return final_epoch_loss


def _compute_posteriors(
    model: nn.Module,
    X_scaled: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Vectorised posterior computation with per-observation pi.

    Assumes location = 0 for all mixture components (spike mean is 0).
    This matches the zero-centred normal mixture prior used by LC-ASH.

    Returns
    -------
    post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik
        ``marginal_loglik`` is the full-data marginal log-likelihood
        ``sum_g logsumexp_k (log pi_g,k + log p(beta_g | 0, sqrt(se_g^2 + scale_k^2)))``,
        i.e. ``log p(y | fitted prior)`` without any spike Dirichlet penalty.
        It is what the cebmf consumer at ``cebmf.py:299``
        (``self.kl_l[k] = (-resL.loss) - nm_ll_L``) requires of the loss
        field on this object.
    """
    model.eval()
    loc = torch.zeros_like(scale)
    with torch.no_grad():
        all_pi_values = model(X_scaled)  # (G, K)

        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat, sebetahat=sebetahat, location=loc, scale=scale
        )  # (G, K)

        # Use dtype-appropriate eps to avoid log(0) in float32.
        eps = torch.finfo(all_pi_values.dtype).tiny
        log_pi_all = torch.log(torch.clamp(all_pi_values, min=eps))  # (G, K)
        combined = data_loglik + log_pi_all  # (G, K)
        log_norm = torch.logsumexp(combined, dim=1, keepdim=True)  # (G, 1)
        # log_norm[g] is the per-gene marginal log-likelihood of the fitted
        # mixture; summing gives the full-data marginal log-lik (no penalty).
        marginal_loglik = float(log_norm.sum().item())
        resp = torch.exp(combined - log_norm)  # (G, K) responsibilities

        s2 = sebetahat.pow(2).unsqueeze(1)  # (G, 1)
        t2 = scale.pow(2).unsqueeze(0)  # (1, K)

        denom = (1.0 / s2) + torch.where(t2 > 0, 1.0 / t2, torch.zeros_like(t2))
        post_var_comp = torch.where(t2 > 0, 1.0 / denom, torch.zeros_like(denom))  # (G, K)

        m_comp = torch.where(
            t2 > 0,
            post_var_comp * (betahat.unsqueeze(1) / s2),
            torch.zeros(1, device=device),
        )  # (G, K)

        post_mean = torch.sum(resp * m_comp, dim=1)
        post_mean2 = torch.sum(resp * (post_var_comp + m_comp.pow(2)), dim=1)
        post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean.pow(2), min=0.0))

    return post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik


def _fit_lcash(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    model_class: type,
    label: str,
    n_epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    penalty: float = 1.5,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
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
    if n_epochs is None:
        n_epochs = 200

    X_scaled, betahat, sebetahat = _prepare_inputs(X, betahat, sebetahat, device)
    scale, log_pi_init = _select_grid(betahat, sebetahat, mult, ash_init, ash_threshold, device)

    # Local RNG for reproducible weight init and batch ordering.
    # Does not mutate global torch RNG state.
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    K = scale.shape[0]
    model = model_class(X_scaled.shape[1], K, log_pi_init=log_pi_init, generator=rng).to(device)
    if model_param is not None:
        model.load_state_dict(model_param)

    # Build optimizer: weight_decay on feature weights only.
    if model_class is LcashNet:
        param_groups = [
            {"params": [model.linear.weight], "weight_decay": weight_decay},
            {"params": [model.linear.bias], "weight_decay": 0.0},
        ]
    else:  # PropOddsLcashNet
        cutpoint_params = [model.delta_1]
        if model.delta_gaps is not None:
            cutpoint_params.append(model.delta_gaps)
        param_groups = [
            {"params": [model.w], "weight_decay": weight_decay},
            {"params": cutpoint_params, "weight_decay": 0.0},
        ]
    optimizer = torch.optim.Adam(param_groups, lr=lr)

    _train_model(
        model,
        optimizer,
        X_scaled,
        betahat,
        sebetahat,
        scale,
        n_epochs,
        batch_size,
        penalty,
        verbose,
        label,
        seed=seed,
    )

    post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik = _compute_posteriors(
        model,
        X_scaled,
        betahat,
        sebetahat,
        scale,
        device,
    )

    # `loss` is the negative full-data marginal log-likelihood under the
    # fitted prior, *without* the spike Dirichlet penalty. This matches
    # the convention used by `cebnm/emdn.py` and is the meaning required
    # by `cebmf.py`'s per-factor `kl_l[k] = (-loss) - nm_ll_L` formula.
    # The previous training-loss-on-final-epoch return value was an
    # unfinished refactor (cf. the `# compute proper full negative
    # marginal log-likelihood (no penalty)` TODO comments that used to
    # live in `cash_solver.py`).
    return cash_PosteriorMeanNorm(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi_np=all_pi_values,
        loss=-marginal_loglik,
        scale=scale,
        model_param=model.state_dict(),
    )


# ============================================================
# Public entry points
# ============================================================


def lcash_posterior_means(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    n_epochs: int | None = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    penalty: float = 1.5,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
) -> cash_PosteriorMeanNorm:
    """LC-ASH: linear covariate-modulated mixture weights.

    Parameters
    ----------
    X : tensor (G, F)
        Feature matrix. Standardised internally with NaN-aware
        statistics (mean/std computed on non-NaN values, NaN positions
        zero-filled). Pre-standardisation is not required.
    betahat : tensor (G,)
        Effect estimates.
    sebetahat : tensor (G,)
        Standard errors.
    n_epochs : int or None
        Training epochs. Inside cEBMF, overridden by internal_epoch.
    batch_size : int
        Mini-batch size for Adam.
    lr : float
        Learning rate.
    weight_decay : float
        L2 penalty on feature coefficients only (not bias).
    penalty : float
        Dirichlet spike penalty (lambda_pen). 1.0 = no penalty.
    mult : float
        Multiplicative step between mixture grid SDs.  Smaller values
        give a finer grid with more components.  Default sqrt(2)
        matches R ashr and gives ~27 components before pruning.
    ash_init : bool
        If True (default), run ash internally (L-BFGS optimizer) to
        prune the grid to active components and initialise the bias
        from the ash weights, so the model starts at the exchangeable
        ash solution when all feature coefficients are zero.
        If False, use the full grid with uniform bias initialisation.
    ash_threshold : float
        Pruning threshold: components with pi <= threshold are dropped.
        Only used when ``ash_init=True``.
    model_param : dict or None
        State dict from a previous call, for warm-starting.
        Incompatible parameter names or shapes raise ``RuntimeError``.
    device : torch.device or None
        Compute device. Defaults to CUDA if available.
    verbose : bool
        If True (default), print training progress every 50 epochs.
    seed : int
        Random seed for weight initialisation and batch ordering.

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with post_mean, post_mean2, post_sd, pi_np (G, K),
        scale (K,), loss, model_param (state dict for warm-starting).
    """
    return _fit_lcash(
        X,
        betahat,
        sebetahat,
        model_class=LcashNet,
        label="LC-ASH",
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        penalty=penalty,
        mult=mult,
        ash_init=ash_init,
        ash_threshold=ash_threshold,
        model_param=model_param,
        device=device,
        verbose=verbose,
        seed=seed,
    )


def po_lcash_posterior_means(
    X: torch.Tensor,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    n_epochs: int | None = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    penalty: float = 1.5,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
) -> cash_PosteriorMeanNorm:
    """Proportional odds LC-ASH: ordered logistic covariate-modulated weights.

    A shared weight vector maps features to a scalar signal strength
    s_i = x_i^T w.  K-1 ordered cut-points convert s_i to mixture
    weights via cumulative logistic probabilities.  This has F + K - 1
    parameters (vs K * F for softmax LC-ASH), making it more parsimonious
    when K is large relative to F.

    When ``ash_init=True``, the grid is pruned to ash's active components
    and the cut-points are initialised from the ash weights, so the model
    starts at the exchangeable ash solution.

    Parameters
    ----------
    X : tensor (G, F)
        Feature matrix. Standardised internally with NaN-aware statistics.
    betahat, sebetahat, n_epochs, batch_size, lr, weight_decay, penalty,
    mult, ash_init, ash_threshold, model_param, device, verbose, seed :
        See ``lcash_posterior_means``.

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with post_mean, post_mean2, post_sd, pi_np (G, K),
        scale (K,), loss, model_param (state dict for warm-starting).
    """
    return _fit_lcash(
        X,
        betahat,
        sebetahat,
        model_class=PropOddsLcashNet,
        label="PO-LC-ASH",
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        penalty=penalty,
        mult=mult,
        ash_init=ash_init,
        ash_threshold=ash_threshold,
        model_param=model_param,
        device=device,
        verbose=verbose,
        seed=seed,
    )
