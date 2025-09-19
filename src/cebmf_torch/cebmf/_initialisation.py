from typing import Protocol

import torch
from torch import Tensor

RANDOM_INIT_SCALE = 0.01


class InitialisationStrategy(Protocol):
    """
    Protocol for factor initialization strategies.

    Methods
    -------
    __call__(Y, N, P, K, device)
        Initialize L and F matrices.
    """

    def __call__(self, Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Initialize L and F matrices.

        Parameters
        ----------
        Y : Tensor
            Input data tensor (N, P)
        N : int
            Number of rows
        P : int
            Number of columns
        K : int
            Number of factors
        device : torch.device
            Target device

        Returns
        -------
        tuple of Tensor
            Tuple of (L, F) tensors
        """
        ...


@torch.no_grad()
def _impute_nan(Y: Tensor) -> Tensor:
    """
    Column-mean imputation for NaNs in a tensor (for SVD init only).

    Parameters
    ----------
    Y : Tensor
        Input data tensor with possible NaNs.

    Returns
    -------
    Tensor
        Imputed tensor with NaNs replaced by column means.
    """
    mask = ~torch.isnan(Y)
    if not mask.any():
        return Y
    col_means = torch.nanmean(Y, dim=0)
    Y_imp = Y.clone()
    idx = torch.where(~mask)
    Y_imp[idx] = col_means[idx[1]]
    return Y_imp


@torch.no_grad()
def svd_initialise(Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    SVD-based initialization strategy for factor matrices.

    Parameters
    ----------
    Y : Tensor
        Input data tensor (N, P)
    N : int
        Number of rows
    P : int
        Number of columns
    K : int
        Number of factors
    device : torch.device
        Target device

    Returns
    -------
    tuple of Tensor
        Tuple of (L, F) tensors
    """
    Y_for_init = _impute_nan(Y)
    U, S, Vh = torch.linalg.svd(Y_for_init, full_matrices=False)
    K_actual = min(K, S.shape[0])
    L = (U[:, :K_actual] * S[:K_actual]).contiguous()
    F = Vh[:K_actual, :].T.contiguous()

    # Pad with zeros if K > rank
    if K_actual < K:
        L = torch.cat([L, torch.zeros(N, K - K_actual, device=device)], dim=1)
        F = torch.cat([F, torch.zeros(P, K - K_actual, device=device)], dim=1)

    return L, F


@torch.no_grad()
def random_initialise(Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Random initialization strategy for factor matrices.

    Parameters
    ----------
    Y : Tensor
        Input data tensor (N, P)
    N : int
        Number of rows
    P : int
        Number of columns
    K : int
        Number of factors
    device : torch.device
        Target device

    Returns
    -------
    tuple of Tensor
        Tuple of (L, F) tensors
    """
    L = torch.randn(N, K, device=device) * RANDOM_INIT_SCALE
    F = torch.randn(P, K, device=device) * RANDOM_INIT_SCALE
    return L, F


@torch.no_grad()
def zero_initialise(Y: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Zero initialization strategy for factor matrices.

    Parameters
    ----------
    Y : Tensor
        Input data tensor (N, P)
    N : int
        Number of rows
    P : int
        Number of columns
    K : int
        Number of factors
    device : torch.device
        Target device

    Returns
    -------
    tuple of Tensor
        Tuple of (L, F) tensors
    """
    L = torch.zeros(N, K, device=device)
    F = torch.zeros(P, K, device=device)
    return L, F


# Registry for initialization strategies
INIT_STRATEGIES = {
    "svd": svd_initialise,
    "random": random_initialise,
    "zero": zero_initialise,
}


def user_provided_factors(L: Tensor, F: Tensor, N: int, P: int, K: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Use user-provided factor matrices.

    Parameters
    ----------
    L : Tensor
        User-provided L matrix (N, K)
    F : Tensor
        User-provided F matrix (P, K)
    N : int
        Number of rows of L (for validation)
    P : int
        Number of rows of F (for validation)
    K : int
        Number of factors i.e. columns of L and F (for validation)
    device : torch.device
        Target device

    Returns
    -------
    tuple of Tensor
        Tuple of (L, F) tensors on the specified device.
    """
    if L.shape != (N, K):
        raise ValueError(f"Provided L has shape {L.shape}, expected ({N}, {K})")
    if F.shape != (P, K):
        raise ValueError(f"Provided F has shape {F.shape}, expected ({P}, {K})")
    return L.to(device), F.to(device)
