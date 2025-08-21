
import torch
from torch import Tensor
from .torch_utils import logsumexp, safe_log, softmax

import torch
from typing import Optional

def optimize_pi_logL(
    logL: torch.Tensor,
    penalty: float | torch.Tensor,
    max_iters: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    EM algorithm for optimizing mixture weights pi on the simplex given a log-likelihood matrix.

    Args:
        logL: (n, K) tensor with entries logL[j, k] = log l_{jk}.
        penalty: Dirichlet pseudo-count alpha_1 on component 0 (or a length-K vector).
                 In the original code, vec_pen[0] = penalty and others = 1.
        max_iters: number of EM epochs.
        tol: L2 tolerance on pi change for convergence.
        verbose: print convergence message if True.
        batch_size: if None, do full-batch; else iterate over mini-batches each epoch.
        shuffle: whether to shuffle rows each epoch when using batches.
        seed: RNG seed used when shuffle=True.

    Returns:
        pi: (K,) tensor of optimized mixture weights on the simplex.
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
        vec_pen[0] = dtype.type(penalty)

    eps = dtype.type(1e-12)

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
            perm = torch.randperm(n, generator=g, device=device)
            idx_all = perm
        else:
            idx_all = indices

        for start in range(0, n, batch_size):
            idx = idx_all[start:start + batch_size]
            Lb = logL[idx]  # (B, K)

            # E-step: responsibilities r_{jk} ∝ pi_k * exp(logL_{jk})
            log_pi = torch.log(pi + eps)              # (K,)
            log_r = Lb + log_pi.unsqueeze(0)          # (B, K)
            log_norm = torch.logsumexp(log_r, dim=1, keepdim=True)  # (B,1)
            r = torch.exp(log_r - log_norm)           # (B, K)

            # accumulate expected counts
            n_k += r.sum(dim=0)                       # (K,)

        # M-step with Dirichlet prior α (as pseudo-counts)
        n_k = n_k + (vec_pen - 1.0)
        n_k = torch.clamp(n_k, min=eps)
        pi = n_k / n_k.sum()

        # convergence check
        if torch.linalg.norm(pi - pi_old).item() < tol:
            if verbose:
                print(f"Converged after {it} iterations.")
            break

    return pi
