import math

import torch


@torch.no_grad()
def optimize_pi_logL(
    logL: torch.Tensor,
    penalty: float | torch.Tensor,
    max_iters: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
    batch_size: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
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
        Print convergence message if True. Default is True.
    batch_size : int or None, optional
        If None, do full-batch; else iterate over mini-batches each epoch. Default is None.
    shuffle : bool, optional
        Whether to shuffle rows each epoch when using batches. Default is False.
    seed : int or None, optional
        RNG seed used when shuffle=True. Default is None.

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
        n_k = torch.clamp(n_k + (vec_pen - 1.0), min=eps.item())
        pi = n_k / n_k.sum()

        # convergence check
        if torch.linalg.norm(pi - pi_old).item() < tol:
            if verbose:
                print(f"Converged after {it} iterations.")
            break

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
    if torch.all(betahat**2 < sebetahat**2):
        sigmaamax = 8.0 * sigmaamin
    else:
        sigmaamax = 2.0 * torch.sqrt(torch.max(betahat**2 - sebetahat**2))

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
    if torch.all(betahat**2 < sebetahat**2):
        sigmaamax = 8.0 * sigmaamin
    else:
        sigmaamax = tt * torch.sqrt(torch.max(betahat**2))

    if mult == 0:
        return torch.stack([torch.tensor(0.0, device=device, dtype=dtype), (sigmaamax / 2.0).to(dtype)], dim=0)

    scales = _calculate_scales(float(sigmaamax), float(sigmaamin), mult, device, dtype)
    if max_class is not None:
        if scales.numel() != max_class:
            scales = torch.linspace(scales.min(), scales.max(), steps=max_class, device=device, dtype=dtype)
            if scales.numel() >= 3 and scales[2] < torch.tensor(1e-2, device=device, dtype=dtype):
                scales[2:] = scales[2:] + torch.tensor(1e-2, device=device, dtype=dtype)
    return scales
