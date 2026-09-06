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

LC-ASH uses multinomial logistic regression to model the weights of a
zero-centred normal mixture prior from covariates. PO-LC-ASH uses
proportional odds regression. The component grid is the ordered set of
mixture standard deviations; zero represents a point mass at zero (the spike).

For a fresh fit, ``ash_init=True`` first fits adaptive shrinkage (ASH)
without covariates. It retains components with weight above
``ash_threshold`` (default ``1e-6``) and uses their weights to initialize
the regression model. ``ash_init=False`` uses the full automatically
selected component grid. The component standard deviations remain fixed
during regression fitting.

Selection raises ``ValueError`` if fewer than two components remain or
the spike is removed. Inspect the ASH weights and consider lowering
``ash_threshold``; ``0.0`` retains all positive returned weights. To inspect
the initializer weights, call
``ash(betahat, se, optimizer="lbfgs", zero_threshold=0.0)`` with the same
``mult``. The ASH initializer uses a spike penalty of ``10.0``. The
conditional solver's ``penalty`` controls subsequent regression fitting.

The returned ``model_param`` contains the selected component grid, fitted
regression parameters, covariate means and standard deviations used for
standardization, and solver/version identifiers. The regression parameters are the
component-specific coefficients and intercepts for LC-ASH, or the shared
coefficients and parameters defining the ordered cut-points for PO-LC-ASH.
The saved means and standard deviations define the standardized covariates
to which these parameters apply.

Pass ``model_param`` to the same solver to continue fitting with the saved
component grid, regression parameters and covariate standardization.
``ash_init``, ``ash_threshold`` and ``mult`` are then ignored. Effect
estimates, standard errors and the number of observations may change.
Covariate columns must retain their meaning, order and units. Missing
entries and columns with zero standard deviation in the original fit are
set to zero after standardization.

Each call starts a new Adam optimizer. Use ``n_epochs=0`` to calculate
posterior summaries with the saved prior. To select a new grid or estimate
new covariate statistics, start a fresh fit without ``model_param``.
Pass the complete returned dictionary as ``model_param``;
``model_param["state_dict"]`` contains only the regression parameters.

The example below continues fitting from saved state, using five training
epochs per call. ``po_lcash_posterior_means`` supports the same calling pattern.

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
If factor pruning changes the number of covariate columns, the next update
raises ``ValueError`` because the saved state requires the original column count.

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
