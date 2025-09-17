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

    from cebmf_torch import cEBMF
    
    # Basic usage with defaults
    model = cEBMF(data=Y)  # Uses K=5, normal priors
    
    # Customized priors and rank
    model = cEBMF(data=Y, K=10, prior_L='exp', prior_F='laplace')
    
    # With covariates
    model = cEBMF(data=Y, K=3, X_l=row_covariates, X_f=col_covariates)
    
    # Different noise models
    from cebmf_torch.torch_main import NoiseType
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
    print(f"Null probability: {result.pi0_null}")

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
    betahat = torch.randn(n, device='cuda')
    se = torch.full((n,), 0.5, device='cuda')
    
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
   notebooks/model_RNA_ATAC.ipynb
