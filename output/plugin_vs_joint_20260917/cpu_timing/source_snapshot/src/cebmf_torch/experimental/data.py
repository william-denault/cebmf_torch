"""Legacy simulation; alignment is shared with the public multi-view API."""

import torch
from torch import Tensor
from cebmf_torch.utils.modalities import AlignedModalities, align_modalities

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
