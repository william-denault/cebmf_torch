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

LC-ASH uses multinomial logistic regression to model the weights of a
zero-centred normal mixture prior from covariates. PO-LC-ASH uses
proportional odds regression. The component grid is the ordered set of
mixture standard deviations; zero represents a point mass at zero (the spike).

For a fresh fit, ``ash_init=True`` calls ``ash`` without covariates,
using its default penalty of 10. It retains weights above
``ash_threshold``. The default ``None`` resolves to ``min(1e-6, 0.5/n)``,
where ``n`` is the training observation count. A numeric ``ash_threshold``
sets the cutoff directly.
The initial covariate coefficients are zero, so the initial conditional
mixture equals the retained, normalized mixture from ``ash``.

``ash_init=False`` uses the full automatically selected grid. Component
scales stay fixed during training and warm starts. Selection raises
``ValueError`` if the spike or every component is removed. A single retained
spike returns its exact posterior without optimization.

To inspect the initializer, call
``ash(betahat, se, optimizer="lbfgs", zero_threshold=0.0)`` with the same
``mult`` and float64 inputs. The caller's conditional penalty does not change
this ``ash(..., penalty=10)`` fit. With ``ash_threshold=0``, the numerical
return floor in ``ash`` can retain components whose optimizer weights were zero.

The conditional ``penalty`` defaults to 10 and uses the coefficient
``(penalty-1)/n`` in the full-data objective. At zero slopes this equals
the null penalty used by ``ash`` at the same strength. A value of 1 omits the null term.

Continued training uses the supplied penalty and the number of supplied
observations. Saved-state prediction with ``max_iter=0`` applies no penalty.

Training uses native PyTorch ``torch.optim.LBFGS`` with full batches,
float64 arithmetic and direct log probabilities. Component log likelihoods
are computed once per fit and reused for optimization and posterior summaries.

The standardized covariate coefficients have independent Student-t priors,
evaluated with native ``torch.distributions.StudentT``. The defaults are
``coefficient_prior_scale=1.0`` and ``coefficient_prior_df=5.0``, with location
zero. When the prior is enabled, its scale and degrees of freedom must be
finite and positive. Scale is the
distribution's scale parameter, not its standard deviation. The default has
standard deviation 1.29 and places 95% of its mass between -2.57 and 2.57.
`Stan's prior-choice guidance <https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations>`_
supports Student-t priors with moderate degrees of freedom and scales chosen
for the parameter's units.
Set ``coefficient_prior_scale=None`` to disable this prior.

For ``lcash_posterior_means``, the prior applies to each non-null component's
slope minus the
null component's slope. These contrasts describe component-versus-null log odds;
adding the same slope to all components does not change the model or its prior.
For ``po_lcash_posterior_means``, the prior applies to the shared slope vector.
With two components, these are the same coefficient interpretation. Intercepts and cut-points are
not penalized. Each coefficient refers to a one-standard-deviation covariate
change, using the saved training means and standard deviations.

The summed negative log prior is divided by ``n`` and added to the mean
training objective. It regularizes covariate effects separately from the
null penalty. Returned ``loss`` excludes both prior terms: it is the summed
negative marginal log likelihood required by cEBMF.

``po_lcash_posterior_means`` caps log cut-point gaps at the floating-point
range limit before exponentiation, leaving room for their cumulative sum. This keeps very large,
finite trial gaps from overflowing. Values below the cap are unchanged.
The solver raises ``FloatingPointError`` for a non-finite optimization
loss or gradient and restores the initial parameters before raising.
A finite fit above the gradient tolerance emits ``RuntimeWarning``,
with the stopping reason. To continue fitting, pass the complete
``model_param`` with the same data, ``penalty``, ``coefficient_prior_scale``
and ``coefficient_prior_df``. Prior settings are not stored in ``model_param``.
Fits are quiet by default; ``verbose=True`` prints the final iteration count,
evaluation count, mean objective and maximum gradient.

Fitting and saved state use float64. MPS fits run on CPU because MPS does
not support float64. MPS inputs and saved parameters move to CPU before
conversion to float64. A native PyTorch device context keeps intermediate
allocations on the fitting device and preserves the caller's default device.
Returned posterior summaries, weights and scales use float32 on the requested
device.

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

Each training call with more than one component starts a new L-BFGS optimizer.
Use ``max_iter=0`` to calculate posterior summaries with the saved prior.
To select a new grid or estimate new covariate statistics, start a fresh fit
without ``model_param``.
Pass the complete returned dictionary as ``model_param``;
``model_param["state_dict"]`` contains only the regression parameters.

Within cEBMF, omitted or ``None`` inner ``max_iter`` uses
``2 * internal_epoch``. An explicit zero or positive value overrides it.
The standalone default is 500 iterations. Iteration counts are upper limits;
L-BFGS can stop earlier when it meets a convergence criterion.

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
