# cebmf_torch: Empirical Bayes Matrix Factorization in PyTorch

[![codecov](https://codecov.io/gh/william-denault/cebmf_torch/branch/main/graph/badge.svg)](https://codecov.io/gh/william-denault/cebmf_torch)
[![unittest](https://github.com/william-denault/cebmf_torch/actions/workflows/test.yml/badge.svg)](https://github.com/william-denault/cebmf_torch/actions/workflows/test.yml)
[![docs](https://readthedocs.org/projects/cebmf-torch/badge/?version=latest)](https://cebmf-torch.readthedocs.io/en/latest/)

## Overview

**cebmf_torch** is a pure-PyTorch implementation of Empirical Bayes Matrix Factorization (EBMF) and Empirical Bayes Normal Means (EBNM) methods. It is designed for scalable, GPU-accelerated analysis of large datasets, with a focus on genomics and other high-dimensional applications. The package provides flexible prior families, efficient mini-batch EM, and full support for GPU computation.

- **No NumPy. No SciPy. No R.**
- **GPU-accelerated**: All core computations are performed in PyTorch.
- **Flexible priors**: Easily extendable to new prior families.
- **Mini-batch EM**: Fast optimization for large datasets.
- **Posterior inference**: Compute posterior means and variances for all supported models.

## Features

- Empirical Bayes Matrix Factorization (EBMF) with flexible priors
- Empirical Bayes Normal Means (EBNM) solvers (normal, exponential, Laplace, point-mass, etc.)
- GPU support for all operations
- Mini-batch EM and Adam optimizers for mixture weights
- Analytical truncated normal moments (no SciPy dependency)
- Easy-to-use API for both beginners and advanced users

## Installation

Installation is managed with [`uv`](https://docs.astral.sh/uv/getting-started/installation/), a fast Python package manager.

```bash
# Clone the repository
git clone https://github.com/william-denault/cebmf_torch.git
cd cebmf_torch

# Install the package and dependencies
uv sync

# Run tests to verify your installation
uv run pytest
```

### Docker (GPU Support)

Use the public docker image

```bash
docker pull ghcr.io/william-denault/cebmf_torch:latest
```

or clone the repo and build the image yourself

```
docker build .
```

The Docker image includes:

- CUDA 13.0.1 runtime for GPU acceleration
- Python 3.12 with all dependencies
- Development tools (pytest, etc.)


## Quick Start

Here's how to get started with the main functions:


```python
import torch
from cebmf_torch import ash, cEBMF

# Example: ash with normal mixture prior
n = 10000
betahat = torch.randn(n, device='cuda' if torch.cuda.is_available() else 'cpu')
se = torch.full((n,), 0.5, device=betahat.device)
res = ash(betahat, se, prior='norm', batch_size=8192)
print(res.pi0, res.scale)

# Example: EBMF on a small matrix
Y = torch.randn(500, 200, device=betahat.device)
model = cEBMF(Y, K=5, prior_L='norm', prior_F='norm')
fit = model.fit(maxit=10)
print(fit.L.shape, fit.F.shape, fit.tau.item())
```


## Documentation & Examples

- Full documentation: See the [cebmf-torch documentation](https://cebmf-torch.readthedocs.io/en/latest/)
- Example notebooks: See the `examples/` directory for Jupyter notebooks demonstrating typical workflows.
  - To run the example notebooks, first add some additional dependencies with `uv sync --group examples`

## Notes & Tips

- All computations run on the device of your input tensors (CPU or GPU).
- Mini-batch EM for `pi` is implemented via Adam on logits (recommended) or online EM.
- Truncated normal moments are computed analytically in torch (no SciPy required).
- The codebase is modular and easy to extend for new prior families or custom models.

## Contributing & Support

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request on GitHub.

For questions or help, open an issue or contact the maintainer.

