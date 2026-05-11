import math

import torch


@torch.no_grad()
def optimize_pi_logL(
    logL: torch.Tensor,
    penalty: float | torch.Tensor,
    max_iters: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
    batch_size: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    check_every: int = 10,
) -> torch.Tensor:
    """
    EM algorithm for optimizing mixture weights on the simplex given a log-likelihood matrix.

    Parameters
    ----------
    logL : torch.Tensor
        (n, K) tensor with entries logL[j, k] = log l_{jk}.
    penalty : float or torch.Tensor
        Dirichlet pseudo-count alpha_1 on component 0 (or a length-K vector).
        In the original code, vec_pen[0] = penalty and others = 1.
    max_iters : int, optional
        Number of EM epochs. Default is 100.
    tol : float, optional
        L2 tolerance on pi change for convergence. Default is 1e-6.
    verbose : bool, optional
        Print convergence message if True. Default is False (cEBMF calls this
        in its inner loop and the convergence prints would spam stdout).
    batch_size : int or None, optional
        If None, do full-batch; else iterate over mini-batches each epoch. Default is None.
    shuffle : bool, optional
        Whether to shuffle rows each epoch when using batches. Default is False.
    seed : int or None, optional
        RNG seed used when shuffle=True. Default is None.
    check_every : int, optional
        How many EM steps to run between convergence checks. Each check forces a
        host sync, so checking every step (the previous behaviour) cost up to
        ``max_iters`` syncs per call — bad inside cEBMF's hot loop. Default is 10.

    Returns
    -------
    torch.Tensor
        (K,) tensor of optimized mixture weights on the simplex.
    """
    assert logL.ndim == 2, "logL must be (n, K)"
    n, K = logL.shape
    device = logL.device
    dtype = logL.dtype

    # Initialize pi ∝ exp(-k)
    k = torch.arange(K, device=device, dtype=dtype)
    pi = torch.exp(-k)
    pi = pi / pi.sum()

    # Penalty vector (Dirichlet α): default α = [penalty, 1, 1, ..., 1]
    if isinstance(penalty, torch.Tensor):
        vec_pen = penalty.to(device=device, dtype=dtype)
        assert vec_pen.shape == (K,), "penalty tensor must have shape (K,)"
    else:
        vec_pen = torch.ones(K, device=device, dtype=dtype)
        vec_pen[0] = penalty

    eps = torch.tensor(1e-12, device=device, dtype=dtype)

    # batching helper
    if batch_size is None or batch_size >= n:
        batch_size = n
    indices = torch.arange(n, device=device)

    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    # Track convergence with a 0-d tensor so most EM iterations stay sync-free.
    # We only force a host sync every `check_every` iterations.
    eps_floor = 1e-12  # was eps.item(); a Python literal avoids the per-iter sync
    converged_iter = -1

    for it in range(max_iters):
        pi_old = pi.clone()

        # accumulate expected counts across all mini-batches
        n_k = torch.zeros(K, device=device, dtype=dtype)

        if shuffle and batch_size < n:
            idx_all = torch.randperm(n, generator=g, device=device)
        else:
            idx_all = indices

        # compute log_pi once per EM iteration (constant across batches)
        log_pi = torch.log(pi + eps)  # (K,)

        for start in range(0, n, batch_size):
            idx = idx_all[start : start + batch_size]
            Lb = logL[idx]  # (B, K)

            # E-step: responsibilities r_{jk} ∝ pi_k * exp(logL_{jk})
            log_r = Lb + log_pi.unsqueeze(0)  # (B, K)
            log_norm = torch.logsumexp(log_r, dim=1, keepdim=True)  # (B,1)
            r = torch.exp(log_r - log_norm)  # (B, K)

            # accumulate expected counts
            n_k += r.sum(dim=0)  # (K,)

        # M-step with Dirichlet prior α (as pseudo-counts)
        n_k = torch.clamp(n_k + (vec_pen - 1.0), min=eps_floor)
        pi = n_k / n_k.sum()

        # Sync-light convergence check: only force host comparison every
        # `check_every` iterations so the inner EM loop stays GPU-resident.
        if (it + 1) % check_every == 0:
            if torch.linalg.norm(pi - pi_old).item() < tol:
                converged_iter = it
                break

    if verbose and converged_iter >= 0:
        print(f"Converged after {converged_iter} iterations.")

    return pi


def _calculate_scales(
    sigmaamax: float,
    sigmaamin: float,
    mult: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Calculate a sequence of scales for mixture components.

    Parameters
    ----------
    sigmaamax : float
        Maximum scale value.
    sigmaamin : float
        Minimum scale value.
    mult : float
        Multiplicative step between scales.
    device : torch.device
        Device for the output tensor.

    Returns
    -------
    torch.Tensor
        1D tensor of scales, including 0 as the first element.
    """
    npoint = int(math.ceil(float(math.log2(sigmaamax / sigmaamin)) / math.log2(mult)))
    seq = torch.arange(-npoint, 1, device=device, dtype=torch.int64)
    return torch.cat(
        [
            torch.tensor([0.0], device=device, dtype=dtype),
            (1.0 / mult) ** (-seq.to(dtype=torch.float64)).to(dtype)
            * torch.tensor(sigmaamax, device=device, dtype=dtype),
        ]
    )


def autoselect_scales_mix_norm(betahat: torch.Tensor, sebetahat: torch.Tensor, max_class=None, mult: float = 2.0):
    """
    Automatically select scales for a normal mixture prior.

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates.
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates.
    max_class : int or None, optional
        If provided, number of mixture components (scales) to return.
    mult : float, optional
        Multiplicative step between scales. Default is 2.0.

    Returns
    -------
    torch.Tensor
        1D tensor of selected scales.
    """
    device = betahat.device
    dtype = betahat.dtype

    sigmaamin = torch.min(sebetahat) / 10.0
    # Branchless max-scale selection: sigmaamax is `2 * sqrt(max(0, max(b^2 - s^2)))`
    # when the data shows signal, else `8 * sigmaamin`. Computing both arms with
    # `torch.where` avoids the host sync from `if torch.all(...)`.
    diff_sq = (betahat**2 - sebetahat**2).clamp_min(0.0)
    sig_sigmaamax = 2.0 * torch.sqrt(torch.max(diff_sq))
    no_signal = torch.max(diff_sq) <= 0
    sigmaamax = torch.where(no_signal, 8.0 * sigmaamin, sig_sigmaamax)

    if mult == 0:
        return torch.stack([torch.tensor(0.0, device=device, dtype=dtype), (sigmaamax / 2.0).to(dtype)], dim=0)

    scales = _calculate_scales(float(sigmaamax), float(sigmaamin), mult, device, dtype)
    if max_class is not None:
        if scales.numel() != max_class:
            scales = torch.linspace(scales.min(), scales.max(), steps=max_class, device=device, dtype=dtype)
    return scales


def autoselect_scales_mix_exp(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    max_class=None,
    mult: float = 1.5,
    tt: float = 1.5,
):
    """
    Automatically select scales for an exponential mixture prior.

    Parameters
    ----------
    betahat : torch.Tensor
        Observed effect size estimates.
    sebetahat : torch.Tensor
        Standard errors of the effect size estimates.
    max_class : int or None, optional
        If provided, number of mixture components (scales) to return.
    mult : float, optional
        Multiplicative step between scales. Default is 1.5.
    tt : float, optional
        Scaling factor for the maximum scale. Default is 1.5.

    Returns
    -------
    torch.Tensor
        1D tensor of selected scales.
    """
    device = betahat.device
    dtype = betahat.dtype

    sigmaamin = torch.maximum(torch.min(sebetahat) / 10.0, torch.tensor(1e-3, device=device, dtype=dtype))
    # Branchless max-scale selection (mirrors autoselect_scales_mix_norm); no
    # host sync from `if torch.all(...)`.
    diff_sq = (betahat**2 - sebetahat**2).clamp_min(0.0)
    no_signal = torch.max(diff_sq) <= 0
    sig_sigmaamax = tt * torch.sqrt(torch.max(betahat**2))
    sigmaamax = torch.where(no_signal, 8.0 * sigmaamin, sig_sigmaamax)

    if mult == 0:
        return torch.stack([torch.tensor(0.0, device=device, dtype=dtype), (sigmaamax / 2.0).to(dtype)], dim=0)

    scales = _calculate_scales(float(sigmaamax), float(sigmaamin), mult, device, dtype)
    if max_class is not None:
        if scales.numel() != max_class:
            scales = torch.linspace(scales.min(), scales.max(), steps=max_class, device=device, dtype=dtype)
            if scales.numel() >= 3 and scales[2] < torch.tensor(1e-2, device=device, dtype=dtype):
                scales[2:] = scales[2:] + torch.tensor(1e-2, device=device, dtype=dtype)
    return scales


# ============================================================
# L-BFGS mixture weight optimizer (alternative to EM)
# ============================================================


@torch.no_grad()
def optimize_pi_logL_lbfgs(
    logL: torch.Tensor,
    penalty: float | torch.Tensor,
    zero_threshold: float = 1e-6,
) -> torch.Tensor:
    """Optimize mixture weights via L-BFGS with softmax reparameterisation.

    An alternative to :func:`optimize_pi_logL` (EM) that produces
    sparse solutions matching R ashr's convex optimizer (mixsqp) to
    3-5 significant figures.

    The simplex constraint is eliminated by optimising unconstrained
    ``z`` where ``x = softmax(z)``.  L-BFGS builds a positive-definite
    BFGS Hessian approximation from gradient history, avoiding the
    near-singular exact Hessian that arises from aliased mixture
    components.

    The Dirichlet penalty is encoded as pseudo-observations following
    R ashr's convention: the likelihood matrix is augmented with
    identity rows weighted by ``(alpha_k - 1)``.

    Parameters
    ----------
    logL : (n, K) tensor of log-likelihoods.
    penalty : float or (K,) tensor, Dirichlet pseudo-count.
    zero_threshold : float
        Components with weight below this are set to exactly zero.
        Default 1e-6: a component at this weight contributes at most
        one-millionth of the mixture density for any observation,
        which is negligible for posterior computation.  L-BFGS drives
        inactive components to ~1e-30 via softmax, so any threshold
        between 1e-10 and 1e-3 gives the same active set.

    Returns
    -------
    torch.Tensor
        (K,) tensor of optimized mixture weights on the simplex.
    """
    assert logL.ndim == 2, "logL must be (n, K)"
    n, K = logL.shape
    device = logL.device
    dtype = logL.dtype

    # Penalty vector.
    if isinstance(penalty, torch.Tensor):
        vec_pen = penalty.to(device=device, dtype=torch.float64)
    else:
        vec_pen = torch.ones(K, device=device, dtype=torch.float64)
        vec_pen[0] = penalty

    # Convert to likelihood space in float64, row-normalised.
    logL_f64 = logL.to(torch.float64)
    row_max = logL_f64.max(dim=1, keepdim=True).values
    L_data = torch.exp(logL_f64 - row_max)

    # Augment with pseudo-observations for Dirichlet penalty,
    # matching R ashr: A <- rbind(diag(K), L); w <- c(prior-1, rep(1,n))
    prior_minus_1 = vec_pen - 1.0
    pen_idx = torch.where(prior_minus_1 > 0)[0]

    if pen_idx.numel() > 0:
        I_rows = torch.zeros(pen_idx.numel(), K, device=device, dtype=torch.float64)
        for ii, ki in enumerate(pen_idx):
            I_rows[ii, ki] = 1.0
        L_aug = torch.cat([I_rows, L_data], dim=0)
        w_aug = torch.cat(
            [
                prior_minus_1[pen_idx],
                torch.ones(n, device=device, dtype=torch.float64),
            ]
        )
    else:
        L_aug = L_data
        w_aug = torch.ones(n, device=device, dtype=torch.float64)

    w_aug = w_aug / w_aug.sum()

    # Solve via L-BFGS in float64.
    z = torch.zeros(K, device=device, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [z],
        lr=1.0,
        max_iter=2000,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    @torch.no_grad()
    def closure():
        z_s = z.data - z.data.max()
        ex = torch.exp(z_s)
        x = ex / ex.sum()
        Lx = (L_aug @ x).clamp(min=1e-300)
        f = -(w_aug * Lx.log()).sum()
        dfdx = -(L_aug.T @ (w_aug / Lx))
        xg = dfdx * x
        z.grad = xg - x * xg.sum()
        return f

    optimizer.step(closure)

    with torch.no_grad():
        z_s = z - z.max()
        pi = torch.exp(z_s)
        pi /= pi.sum()
        pi[pi < zero_threshold] = 0.0
        s = pi.sum()
        if s > 0:
            pi /= s

    return pi.to(dtype)
