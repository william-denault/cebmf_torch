# cebmf_torch: Empirical Bayes Matrix Factorization in PyTorch

[![codecov](https://codecov.io/gh/william-denault/cebmf_torch/branch/main/graph/badge.svg)](https://codecov.io/gh/william-denault/cebmf_torch)
[![unittest](https://github.com/william-denault/cebmf_torch/actions/workflows/test.yml/badge.svg)](https://github.com/william-denault/cebmf_torch/actions/workflows/test.yml)
[![docs](https://readthedocs.org/projects/cebmf-torch/badge/?version=latest)](https://cebmf-torch.readthedocs.io/en/latest/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pcarbo/pcarbo/blob/master/cEBMF_demo.ipynb)

*See [here](cEBMF_demo.ipynb) for a Jupyter notebook demo of cEBMF.
You can also open up the notebook [directly in Google Colab](https://colab.research.google.com/github/pcarbo/pcarbo/blob/master/cEBMF_demo.ipynb).*

## Documentation & Examples

- Full documentation: See the [cebmf-torch documentation](https://cebmf-torch.readthedocs.io/en/latest/)

- Example notebooks: See the `examples/` directory for Jupyter notebooks demonstrating typical workflows.

- To run the example notebooks, first add some additional dependencies with `uv sync --group examples` 
  (or `pip install ".[examples]"` if using `pip`). 

## Overview

**cebmf_torch** is a pure-PyTorch implementation of Empirical Bayes Matrix Factorization (EBMF) and Empirical Bayes Normal Means (EBNM) methods. It is designed for analysis of large datasets, with a focus on genomics and other high-dimensional applications. The package provides flexible prior families, mini-batch EM, and tensor computation on CPU or CUDA.
 
- **GPU-accelerated**: All core computations are performed in PyTorch.
- **Flexible priors**: Easily extendable to new prior families.
- **Mini-batch EM**: Fast optimization for large datasets.
- **Posterior inference**: Compute posterior means and variances for all supported models.

## Features

- Empirical Bayes Matrix Factorization (EBMF) with flexible priors
- Empirical Bayes Normal Means (EBNM) solvers (normal, exponential, Laplace, point-mass, etc.)
- CUDA tensor computation; see the [device contract](docs/source/device_contract.rst) for setup, reporting, and solver-specific synchronization.
- Mini-batch EM and Adam optimizers for mixture weights
- Analytical truncated normal moments for exponential prior
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

If you wish to not use uv for some reason, then it is also possible to pip
install the package by replacing `uv sync` with `pip install .`.

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




## HMM priors for ordered factors

Use `prior_F="hmm"` to model each factor along the existing column order, or
`prior_L="hmm"` along the row order. `hmm_pos` constrains effects to be
nonnegative; `hmm_neg` constrains them to be nonpositive. All three include
an exact zero state by default and assume equally spaced adjacent entries.

```python
model = cEBMF(
    data=X_obs_RNA,
    prior_L="gbinary",
    prior_F="hmm",  # also "hmm_pos" or "hmm_neg"
    prior_F_kwargs={"penalty": 1.5, "maxiter": 20},
    K=K,
    device=device,
)
model.initialise_factors()
fit = model.fit(maxit=30)
```

For all three HMM priors, `penalty=1` is unpenalized (the default), and values
above 1 favor zero effects.

The HMM ignores `X_f` (or `X_l` for an HMM on L) and self-covariates on that
side. Supplying them produces one warning per affected side at construction.
For actual locations together with other side information, include location
in the covariate matrix and use `emdn` or `spiked_emdn`.

See the [HMM prior guide](docs/source/hmm_priors.rst) for the fSuSiE model,
controls, direct EBNM calls, and numerical validation.

## Conditional loading priors

`self_row_cov=True` learns `p(L_k | L_<k)` with uncertain earlier loadings
and feedback from later loadings. It uses variational updates, refits the
feature priors, and updates unknown noise during each sweep.

```python
model = cEBMF(Y, K=4, prior_L="cgb", prior_F="norm", self_row_cov=True)
result = model.fit(maxit=20)
```

By default, `verbose=True` prints a short message after each completed sweep,
such as `cEBMF sweep 1 completed.` Set `verbose=False` in the constructor to
silence sweep progress. Numbering continues across `fit()` and `iter_once()`
calls and resets when factors are reinitialized. Progress uses a Python counter
without copying GPU tensors to the CPU; approximation warnings remain enabled.

`prior_F="norm"` is an ash Gaussian-scale-mixture prior. Neural training options
stay in `prior_L_kwargs` (for example `n_epochs`, `hidden_dim`, `lr`, `penalty`).
Numerical controls, when needed, are
`conditional_kwargs={"quadrature_points": 24, "parent_samples": 32, "seed": 0}`.

Conditional fitting defaults to `approximation="quadratic"` for faster
processing and warns once when the fitting graph is created. This caches quadratic
child feedback and updates Gaussian-component probabilities and moments
analytically, while still averaging over uncertain parents. Positive feedback
curvature is clipped for stability; approximation accuracy can differ from
quadrature. To select the quadrature method instead, set
`conditional_kwargs={"approximation": "quadrature"}`. With no latent edges the
original cEBMF update is preserved and no approximation warning is emitted.
See the [derivation and limitations](docs/source/quadratic_feedback.rst) and
[matched comparison script](examples/benchmarks/tree/compare_quadratic_feedback.py).
These control integration, not an MCMC chain. The conditional graph keeps its
starting rank and ordering. With no effective latent-covariate edges, the
original cEBMF path is used directly.

### Two observation models, one ATAC-RNA fit

```python
from cebmf_torch import align_modalities, cEBMF, fit_joint

# Match actual cell IDs; missing modalities become NaN observation rows.
data = align_modalities(Y_atac, Y_rna, atac_ids, rna_ids)
settings = dict(prior_L="cgb", prior_F="norm", self_row_cov=True)
atac = cEBMF(data.atac, K=2, **settings)
rna = cEBMF(data.rna, K=4, **settings)
atac_fit, rna_fit = fit_joint(atac, rna, maxit=20)
```

This learns `p(L_ATAC) p(L_RNA | L_ATAC)` and supports paired, ATAC-only and
RNA-only cells. ATAC-only cells' unobserved RNA branches are integrated out
while fitting, then predicted afterward. Passing `atac.L` as fixed RNA
covariates instead is a plug-in analysis, not this joint fit.

Start with the [ATAC-RNA notebook](examples/ATAC_RNA_joint.ipynb), or the
[single-matrix tree example](examples/tree_joint_simple.ipynb).
The [example index](examples/README.md) explains which files to use;
previous exploratory notebooks are preserved in an explicitly labeled archive.
The [conditional inference guide](docs/source/joint_inference.rst) covers the
objective, approximation limits, supported priors and migration.

The default is a dependent prior with a factorized posterior approximation.
Quadrature and parent integration require sensitivity checks; neither a
monotone objective nor predictive improvement is guaranteed by finite fitting.
The old CPU sampler remains available only through explicit, deprecated
`joint_kwargs` for reproducibility. It is not selected by the flag alone.

## Contributing & Support

Contributions, bug reports, and feature requests are welcome! Please open an issue or pull request on GitHub.

For questions or help, open an issue or contact the maintainer.

