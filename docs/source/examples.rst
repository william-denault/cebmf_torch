Examples
========

This section provides practical examples for using cebmf_torch.

.. contents::
   :local:

Quick Start
-----------

Basic EBMF Example
~~~~~~~~~~~~~~~~~~

Here's a simple example demonstrating matrix factorization:

.. code-block:: python

    import torch
    from cebmf_torch import cEBMF
    
    # Generate synthetic data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Y = torch.randn(500, 200, device=device)
    
    # Create and fit model with simple interface
    model = cEBMF(data=Y, K=5, prior_L='norm', prior_F='norm', device=device)
    result = model.fit(maxit=10)
    
    print(f"L shape: {result.L.shape}")  # (500, 5)
    print(f"F shape: {result.F.shape}")  # (200, 5) 
    print(f"Precision: {result.tau.item():.3f}")

Quick Usage Patterns
~~~~~~~~~~~~~~~~~~~~~

The cEBMF interface is designed to be straightforward:

.. code-block:: python

    import torch
    from cebmf_torch import cEBMF

    # Generate synthetic data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Y = torch.randn(500, 200, device=device)
    row_covariates = torch.randn(500, 3, device=device)
    col_covariates = torch.randn(200, 2, device=device)
    
    # Basic usage with defaults
    model = cEBMF(data=Y)  # Uses K=5, normal priors
    
    # Customized priors and rank
    model = cEBMF(data=Y, K=10, prior_L='exp', prior_F='laplace')
    
    # With covariates
    model = cEBMF(data=Y, K=3, X_l=row_covariates, X_f=col_covariates)
    
    # Different noise models
    from cebmf_torch.cebmf import NoiseType
    model = cEBMF(data=Y, K=5, noise_type=NoiseType.ROW_WISE)

Empirical Bayes Normal Means (EBNM)
------------------------------------

Using ash() for shrinkage estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from cebmf_torch import ash
    
    # Example: shrinkage estimation with normal mixture prior
    n = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    betahat = torch.randn(n, device=device)
    se = torch.full((n,), 0.5, device=device)
    
    result = ash(betahat, se, prior='norm', batch_size=8192)
    
    print(f"Null probability: {result.pi0}")
    print(f"Scales: {result.scale}")
    print(f"Posterior means: {result.post_mean[:10]}")

Point-mass + Exponential Prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cebmf_torch import ebnm_point_exp
    
    # Data with some true zeros
    x = torch.tensor([1.0, 0.1, -0.5, 2.0, 0.0])
    s = torch.tensor([1.0, 0.5, 1.2, 0.8, 1.0])
    
    result = ebnm_point_exp(x, s)

    print(f"Posterior means: {result.post_mean}")
    print(f"Null probability: {1.0 - result.pi_slab}")

Advanced Usage
--------------

Handling Missing Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from cebmf_torch import cEBMF
    
    # Create data with missing values
    Y = torch.randn(100, 50)
    Y[10:20, 5:15] = float('nan')  # Missing block
    Y[torch.rand_like(Y) < 0.1] = float('nan')  # Random missing
    
    # cEBMF handles NaN automatically
    model = cEBMF(data=Y, K=3)
    result = model.fit(maxit=20)
    
    # Check convergence
    import matplotlib.pyplot as plt
    plt.plot(result.history_obj)
    plt.xlabel('Iteration')
    plt.ylabel('Negative ELBO')
    plt.title('Convergence Plot')

Using Covariates
~~~~~~~~~~~~~~~~

.. code-block:: python

    from cebmf_torch.cebnm import cash_posterior_means
    
    # Generate covariates
    n = 1000
    p_cov = 5
    X = torch.randn(n, p_cov)
    
    # Generate effects dependent on covariates
    true_beta = torch.tensor([0.5, -0.3, 0.0, 0.8, -0.2])
    signal = X @ true_beta
    
    betahat = signal + torch.randn(n) * 0.1
    sebetahat = torch.full((n,), 0.1)
    
    result = cash_posterior_means(
        X=X,
        betahat=betahat, 
        sebetahat=sebetahat,
        n_epochs=50,
        num_classes=10
    )
    
    print(f"Posterior means shape: {result.post_mean.shape}")

Linear Covariate Adaptive Shrinkage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LC-ASH and PO-LC-ASH use covariates to model the weights of a normal
mixture prior. For a fresh fit, ``ash_init=True`` first fits ASH, retains
components with weight above ``ash_threshold`` (default ``1e-6``), and
uses their weights to initialize the conditional model. This is the only
component-selection cutoff in this initialization path. ``ash_init=False``
instead uses the full automatically selected grid. The grid remains fixed
during conditional fitting.

Selection raises ``ValueError`` if fewer than two components remain, or
if the spike at zero is removed. It does not add components below the
threshold. Inspect the ASH fit and consider lowering ``ash_threshold``;
``0.0`` retains all positive returned weights. A selection failure is not
itself evidence that the data contain no signal. To inspect the initializer
weights, call ``ash(betahat, se, optimizer="lbfgs", zero_threshold=0.0)``
with the same ``mult``. The initializer uses ASH's default spike penalty
of ``10.0``, independently of the conditional solver's ``penalty``.

Numerical floors used in log probabilities and PO-LC-ASH cut-points do
not remove grid components. They can alter very small initialization
weights, so retaining a component does not guarantee exact reproduction
of its ASH weight.

The returned ``model_param`` contains the ordered component scales, final
network parameters, feature means and population standard deviations,
and solver/version identifiers. Passing it to the same solver restores
the fitted prior without repeating ASH, grid selection or estimation of
feature statistics. ``ash_init``, ``ash_threshold`` and ``mult`` are then
ignored. New effect estimates, standard errors and observation counts are
allowed. Covariate columns must keep their meaning, order and units;
tensor inputs cannot identify a column permutation. Missing entries and
columns that had zero standard deviation in training contribute zero.

Each call starts a new Adam optimizer. Use ``n_epochs=0`` to evaluate the
saved prior without further training. To reselect the grid or estimate
new feature statistics, start a fresh fit without ``model_param``.
Older flat network state dictionaries lack scales and feature statistics
and are rejected. Refit without ``model_param`` to obtain reusable state.
The neural parameter dictionary is ``model_param["state_dict"]``;
those coefficients alone do not specify the fitted prior.

The following small example illustrates additional fitting from saved
state. Its short training budget demonstrates the interface, not
convergence. ``po_lcash_posterior_means`` supports the same calling pattern.

.. code-block:: python

    import torch
    from cebmf_torch.cebnm import lcash_posterior_means

    generator = torch.Generator().manual_seed(1)
    X = torch.randn(128, 2, generator=generator)
    betahat = 2.0 * X[:, 0] + 0.2 * torch.randn(128, generator=generator)
    se = torch.full_like(betahat, 0.2)

    first = lcash_posterior_means(
        X, betahat, se, n_epochs=5, device="cpu", verbose=False
    )
    updated = lcash_posterior_means(
        X, betahat + 0.05, se, model_param=first.model_param,
        n_epochs=5, device="cpu", verbose=False
    )
    torch.testing.assert_close(updated.scale, first.scale)

Within cEBMF, fitted state is passed between successive factor updates.
If self-covariates change the number of feature columns after factor
pruning, the saved state is incompatible and the fit raises rather than
silently reinitializing or transferring a subset of coefficients.

Custom Initialization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cebmf_torch import cEBMF
    
    Y = torch.randn(200, 100)
    model = cEBMF(data=Y, K=5)
    
    # Different initialization strategies
    result_svd = model.fit(maxit=10)  # Default: SVD
    
    model.initialise_factors(method='random')
    result_random = model.fit(maxit=10)
    
    model.initialise_factors(method='zero')
    result_zero = model.fit(maxit=10)

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from cebmf_torch import ash
    
    # Large dataset processing with batching
    n = 100000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    betahat = torch.randn(n, device=device)
    se = torch.full((n,), 0.5, device=device)

    # Use smaller batch size for memory efficiency
    result = ash(
        betahat, se, 
        prior='norm',
        batch_size=4096  # Adjust based on GPU memory
    )

Performance Tips
----------------

GPU Acceleration
~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from cebmf_torch import cEBMF
    
    # Always specify device for tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Y = torch.randn(1000, 500, device=device)
    
    # Model automatically inherits device from data or specify explicitly
    model = cEBMF(data=Y, K=10, device=device)
    result = model.fit(maxit=50)

Convergence Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cebmf_torch import cEBMF
    
    Y = torch.randn(300, 200)
    model = cEBMF(data=Y, K=8, allow_backfitting=True)
    
    result = model.fit(maxit=100)
    
    # Check for convergence
    obj_history = result.history_obj
    if len(obj_history) > 10:
        recent_change = abs(obj_history[-1] - obj_history[-10]) / abs(obj_history[-10])
        if recent_change < 1e-6:
            print("Converged!")
        else:
            print(f"Still changing: {recent_change:.2e}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **Memory errors**: Reduce batch_size in ash() or use smaller K
2. **Slow convergence**: Try different initialization methods or increase steps
3. **NaN results**: Check for extreme values in input data
4. **Device mismatches**: Ensure all tensors are on the same device

.. code-block:: python

    import torch
    from cebmf_torch import cEBMF

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Y = torch.randn(300, 200)
    model = cEBMF(data=Y, K=5)

    # Debug device issues
    print(f"Data device: {Y.device}")
    print(f"Model device: {model.device}")
    
    # Fix device mismatches
    Y = Y.to(device)
    model.device = device


Detailed examples
-----------------

Below we include more detailed examples of applying the methods in this package
to a variety of tasks and datasets. These examples are also available as Jupyter
notebooks in the `examples/` directory of the source code.

.. toctree::
   :maxdepth: 1

   notebooks/spiked_emdn.ipynb 
   notebooks/Tiled-clustering model.ipynb
