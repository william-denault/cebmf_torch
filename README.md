
[![codecov](https://codecov.io/gh/william-denault/cebmf_torch/branch/5-coverage-statistics-of-tests/graph/badge.svg)](https://codecov.io/gh/william-denault/cebmf_torch)

# cebmf_torch (Pure PyTorch)

A pure-PyTorch rewrite of EBMF/EBNM components:
- No NumPy. No SciPy. No R.
- GPU-accelerated with mini-batch EM for mixture weights.

## Files
- `torch_utils.py`: math helpers, truncated normal moments.
- `torch_utils_mix.py`: scale selection for ash priors.
- `torch_distribution_operation.py`: log-likelihood matrix builders.
- `torch_mix_opt.py`: mini-batch optimizer for mixture weights (Adam / online EM).
- `torch_posterior.py`: posterior mean/variance for Normal & Exponential mixtures.
- `torch_ash.py`: `ash()` entry point (norm/exp).
- `torch_ebnm_point_exp.py`: point-mass + exponential prior solver (autograd).
- `torch_ebnm_point_laplace.py`: point-mass + Laplace prior solver (autograd).
- `torch_main.py`: minimal EBMF class using ash() for L and F updates.

## Installation

Installation is managed with [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone the repository
git clone https://github.com/william-denault/cebmf_torch.git
cd cebmf_torch

# Install the package and dependencies
uv sync

# Run tests
uv run pytest
```

## Quick start

```python
import torch
from cebmf_torch import ash, cEBMF

# Example: ash with normal mixture prior
n = 10000
betahat = torch.randn(n, device='cuda' if torch.cuda.is_available() else 'cpu')
se = torch.full((n,), 0.5, device=betahat.device)
res = ash(betahat, se, prior='norm', method='adam', steps=200, batch_size=8192)
print(res.pi, res.scale)

# Example: EBMF on a small matrix
Y = torch.randn(500, 200, device=betahat.device)
model = cEBMF(Y, K=5, prior_L='norm', prior_F='norm')
fit = model.fit(maxit=10)
print(fit.L.shape, fit.F.shape, fit.tau.item())
```

## Notes

- Everything runs on whichever device your input tensors are on.
- Mini-batch EM for `pi` is implemented via Adam on logits (recommended) or online EM.
- Truncated normal moments are computed analytically in torch without SciPy.
