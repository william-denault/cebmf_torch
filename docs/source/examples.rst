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

Separate options for L and F
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``prior_L_kwargs`` for the row loadings and ``prior_F_kwargs`` for the
column factors. These options are independent even when both sides use the
same prior family. For example:

.. code-block:: python

    import torch
    from cebmf_torch import cEBMF

    Y = torch.randn(50, 20)
    row_covariates = torch.randn(50, 3)
    col_covariates = torch.randn(20, 2)

    model = cEBMF(
        data=Y,
        K=3,
        prior_L="lcash",
        prior_F="lcash",
        X_l=row_covariates,
        X_f=col_covariates,
        prior_L_kwargs={"penalty": 1.0},
        prior_F_kwargs={"penalty": 1.5},
        device="cpu",
    )

For LC-ASH and PO-LC-ASH, ``penalty=1`` disables the extra spike penalty;
larger values encourage the fitted prior to assign more mass to zero. The
example therefore uses different sparsity penalties for L and F. These values
illustrate separate configuration, rather than recommended tuning settings.
A separate Student-t prior regularizes the standardized covariate coefficients.
Its scale and degrees of freedom can also be set independently for L and F.

Omitting a side's options adds no per-side overrides; learned priors still
receive cEBMF's epoch and device settings. Constructing another cEBMF model
does not change the options of this model.

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

``lcash_posterior_means`` uses multinomial logistic regression to model the
weights of a zero-centred normal mixture prior. ``po_lcash_posterior_means``
uses proportional odds regression. Component standard deviations stay fixed;
zero represents the point mass at zero.

For a fresh fit, ``ash_init=True`` calls ``ash`` at penalty 10 and retains
components with weight above ``ash_threshold`` (default ``1e-6``). Zero
initial slopes reproduce the retained, normalized mixture. The conditional
penalty does not affect this initialization. ``ash_init=False`` uses the full
automatically selected grid. Selection raises ``ValueError`` if the spike or
every component is removed. A single retained spike needs no optimization.

Training uses native PyTorch ``torch.optim.LBFGS`` with full batches,
float64 arithmetic and log probabilities. In the summed objective, the null
term uses coefficient ``(penalty-1)/n`` per observation, where ``n`` is the
training count. The default ``penalty=10`` matches ``ash`` at zero slopes;
``penalty=1`` omits the null term.

Standardized covariate coefficients have independent Student-t priors,
evaluated with ``torch.distributions.StudentT``. Defaults are location zero,
``coefficient_prior_scale=1.0`` and ``coefficient_prior_df=5.0``. The scale
and degrees of freedom must be finite and positive. Scale differs from
standard deviation: these defaults give a standard deviation of 1.29 and
95% of the mass between -2.57 and 2.57. See
`Stan's prior-choice guidance <https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations>`_.
Set ``coefficient_prior_scale=None`` to disable this prior.

For ``lcash_posterior_means``, the prior applies to each non-null component's
slope minus the null component's slope. For ``po_lcash_posterior_means``, it
applies to the shared slope vector. Intercepts and cut-points are not
penalized. The summed negative log prior is divided by ``n`` in the mean
training objective. Returned ``loss`` excludes both prior terms and is the
summed negative marginal log likelihood required by cEBMF.

The standalone iteration limit is ``max_iter=500``. Within cEBMF, omitted
or ``None`` inner ``max_iter`` uses ``2 * internal_epoch``; an explicit
value overrides it. A finite fit above the gradient tolerance warns.
Non-finite optimization losses or gradients raise ``FloatingPointError``.
``verbose=True`` prints the final optimization diagnostics.

Pass the complete returned ``model_param`` unchanged to the same solver to
continue fitting. It preserves the fitted prior, component grid and covariate
standardization. ``ash_init``, ``ash_threshold``
and ``mult`` are then ignored. Effect estimates, standard errors and observation
count may change; covariate columns must retain their meaning, order and units.
Missing values and columns with zero training standard deviation contribute zero.

Use ``max_iter=0`` for posterior prediction without optimization or penalties.
To resume the same objective, also pass the same ``penalty``,
``coefficient_prior_scale`` and ``coefficient_prior_df``; these are not stored
in ``model_param``. Each training call starts fresh L-BFGS curvature history.
Start without ``model_param`` to select another grid or estimate new covariate
statistics.

Fitting and saved state use float64. MPS inputs and saved tensors move to CPU
before fitting because MPS does not support float64. Returned posterior summaries,
weights and scales use float32 on the requested device.

The example below continues fitting from saved state, using up to 100 L-BFGS
iterations per call. ``po_lcash_posterior_means`` supports the same calling pattern.

.. code-block:: python

    import torch
    from cebmf_torch.cebnm import lcash_posterior_means

    generator = torch.Generator().manual_seed(1)
    X = torch.randn(128, 2, generator=generator)
    betahat = 2.0 * X[:, 0] + 0.2 * torch.randn(128, generator=generator)
    se = torch.full_like(betahat, 0.2)

    fit_options = dict(
        max_iter=100, penalty=10, coefficient_prior_scale=1.0,
        coefficient_prior_df=5.0, device="cpu", verbose=False
    )
    first = lcash_posterior_means(X, betahat, se, **fit_options)
    updated = lcash_posterior_means(
        X, betahat + 0.05, se, model_param=first.model_param,
        **fit_options
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
