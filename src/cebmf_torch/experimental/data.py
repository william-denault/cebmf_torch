"""Data preparation for the experimental joint ATAC--RNA example.

Missing modalities are represented by NaN observation rows, never by zero
loadings or NaN neural-network covariates. No solver uses the simulation truth.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AlignedModalities:
    ids: tuple[Hashable, ...]
    atac: Tensor
    rna: Tensor
    side_info: Tensor

    @property
    def atac_observed(self) -> Tensor:
        return torch.isfinite(self.atac).any(dim=1)

    @property
    def rna_observed(self) -> Tensor:
        return torch.isfinite(self.rna).any(dim=1)


def _ids(values, n: int, name: str) -> list[Hashable]:
    result = values.tolist() if isinstance(values, Tensor) else list(values)
    if len(result) != n:
        raise ValueError(f"{name} must have one identifier per row.")
    try:
        if len(set(result)) != n:
            raise ValueError(f"{name} contains duplicate row identifiers.")
    except TypeError as exc:
        raise ValueError(f"{name} must contain hashable scalar identifiers.") from exc
    return result


def align_modalities(
    atac: Tensor,
    rna: Tensor,
    atac_ids: Sequence[Hashable],
    rna_ids: Sequence[Hashable],
    *,
    side_info: Tensor | None = None,
    side_ids: Sequence[Hashable] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> AlignedModalities:
    """Align two matrices to their ID union, preserving ATAC then new RNA IDs.

    Each input has observations in rows and features in columns. NaNs mark
    missing measurements; infinities are rejected. Side information is fixed,
    finite, and must cover every union ID. Its rows are matched using side_ids.
    Extra side-information rows are ignored. No matching by row position occurs.
    """
    a = torch.as_tensor(atac, dtype=dtype, device=device)
    r = torch.as_tensor(rna, dtype=dtype, device=device)
    if a.ndim != 2 or r.ndim != 2 or min(a.shape + r.shape) < 1:
        raise ValueError("Each modality must be a nonempty two-dimensional matrix.")
    if torch.isinf(a).any() or torch.isinf(r).any():
        raise ValueError("Use NaN for missing observations; infinity is not supported.")
    a_ids, r_ids = _ids(atac_ids, len(a), "atac_ids"), _ids(rna_ids, len(r), "rna_ids")
    union = list(dict.fromkeys(a_ids + r_ids))
    row_for = {identifier: i for i, identifier in enumerate(union)}
    aligned = []
    for data, identifiers in ((a, a_ids), (r, r_ids)):
        target = data.new_full((len(union), data.shape[1]), torch.nan)
        positions = torch.tensor([row_for[key] for key in identifiers], device=data.device)
        target[positions] = data
        aligned.append(target)
    if side_info is None:
        if side_ids is not None:
            raise ValueError("side_ids requires side_info.")
        x = a.new_empty((len(union), 0))
    else:
        if side_ids is None:
            raise ValueError("Provide side_ids to align fixed side information explicitly.")
        x = torch.as_tensor(side_info, dtype=dtype, device=device)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError("side_info must be a finite vector or matrix.")
        x_ids = _ids(side_ids, len(x), "side_ids")
        lookup = {key: i for i, key in enumerate(x_ids)}
        missing = [key for key in union if key not in lookup]
        if missing:
            raise ValueError(f"side_info does not cover union IDs: {missing[:5]}")
        positions = torch.tensor([lookup[key] for key in union], device=x.device)
        x = x[positions].clone()
    return AlignedModalities(tuple(union), aligned[0], aligned[1], x)


def simulate_atac_rna(
    n: int = 2000,
    p: int = 1000,
    *,
    seed: int = 1,
    noise_atac: float = 1.5,
    noise_rna: float = 1.5,
    informative_side_info: bool = False,
) -> dict[str, Tensor]:
    """Original two-bit/four-profile simulation, with optional observed context.

    At p=1000 the blocks are exactly [200:250], [500:550], [700:750].
    The default preserves the original random draw order (RNA noise first).
    Informative context changes ATAC activation probabilities; it is generated
    before the loadings rather than derived from hidden simulation truth.
    """
    if n < 2 or p < 20:
        raise ValueError("Use n >= 2 and p >= 20.")
    if noise_atac <= 0 or noise_rna <= 0:
        raise ValueError("Noise standard deviations must be positive.")
    generator = torch.Generator().manual_seed(seed)
    context_generator = torch.Generator().manual_seed(seed + 1009)
    side = torch.randn(n, 2, generator=context_generator)
    if informative_side_info:
        probabilities = torch.sigmoid(1.5 * side)
        a = (torch.rand(n, 2, generator=generator) < probabilities).float()
    else:
        a = torch.stack([torch.randint(0, 2, (n,), generator=generator).float() for _ in range(2)], 1)
    width = max(1, int(p * 0.05))
    starts = [int(p * fraction) for fraction in (0.2, 0.5, 0.7)]
    fa, fr = torch.zeros(p, 2), torch.zeros(p, 4)
    for k, block in enumerate((0, 2)):
        start = starts[block]
        fa[start : start + width, k] = 1
    for k, blocks in enumerate(((0, 1, 2), (1, 2), (0, 2), (0, 1))):
        for block in blocks:
            start = starts[block]
            fr[start : start + width, k] = 1
    r = torch.zeros(n, 4)
    for k, (first, second) in enumerate(((0, 1), (1, 0), (1, 1), (0, 0))):
        r[(a[:, 0] == first) & (a[:, 1] == second), k] = 1
    signal_a, signal_r = a @ fa.T, r @ fr.T
    observed_r = signal_r + noise_rna * torch.randn(n, p, generator=generator)
    observed_a = signal_a + noise_atac * torch.randn(n, p, generator=generator)
    return {
        "atac": observed_a,
        "rna": observed_r,
        "signal_atac": signal_a,
        "signal_rna": signal_r,
        "loadings_atac": a,
        "loadings_rna": r,
        "factors_atac": fa,
        "factors_rna": fr,
        "side_info": side,
    }
