"""Data preparation for the partially paired observation matrices.

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


