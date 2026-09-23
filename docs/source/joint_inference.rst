Conditional loading priors and joint ATAC-RNA fitting
====================================================

See :doc:`device_contract` for CUDA residency, fixed ASH grids, and device
validation. The mathematical and numerical audit is recorded in
``output/review_joint_l/NUMERICAL_DEVICE_AUDIT.md``.

One matrix: ``self_row_cov=True``
--------------------------------

The loading prior is ``p(L) = product_k p(L_k | L_<k, X_l)``. Earlier
loadings are uncertain latent variables. Their expectations enter the own
log-prior, and later loading priors provide feedback to their parents.

.. code-block:: python

   from cebmf_torch import cEBMF

   model = cEBMF(Y, K=4, prior_L="cgb", prior_F="norm",
                 self_row_cov=True, device="cpu")
   result = model.fit(maxit=20)

``fit(maxit)`` performs that many variational sweeps. ``iter_once`` performs
one sweep. Each sweep updates a loading prior and its posterior together,
then its feature prior and moments, and finally unknown observation noise.
``S`` still supplies fixed observation standard errors. No preliminary
independent fit, arbitrary 0.5 threshold, burn-in, or posterior chain is
inserted. The existing initialization supplies starting factors.

A graph with no latent-covariate edges uses the original cEBMF solver
unchanged. This includes a single factor with fixed covariates. Ordinary
unconditional priors do not gain covariate dependence from a flag. HMM
priors retain their existing warning and ignore covariates on their axis.
``self_col_cov=True`` applies the same construction to feature coordinates.

Two cEBMF models
----------------

.. code-block:: python

   from cebmf_torch import align_modalities, cEBMF, fit_joint

   data = align_modalities(Y_atac, Y_rna, atac_ids, rna_ids)
   settings = dict(prior_L="cgb", prior_F="norm", self_row_cov=True,
                   device="cpu")
   atac = cEBMF(data.atac, K=2, **settings)
   rna = cEBMF(data.rna, K=4, **settings)
   atac_fit, rna_fit = fit_joint(atac, rna, maxit=20)

Argument order defines ``p(L_ATAC) p(L_RNA | L_ATAC)``. Within-modality
self-covariates remain controlled by each model's flag. Both loading priors
must be conditional Gaussian-mixture families. Features, ranks and noise
can differ between modalities; device and floating-point dtype must match.
``align_modalities`` aligns actual IDs, not row positions. If supplying
already aligned matrices directly, callers must ensure the same ID order.

The returned results belong to the two models and share the joint objective
history. Continue with ``fit_joint(atac, rna, maxit=5)``. Calling an individual
model's ``fit`` after coupling is rejected because it would change only part
of the graph. To change the graph, construct fresh model objects or reset
both models' factors with ``initialise_factors``.

Missing modalities
------------------

Only observed entries contribute likelihood terms. Paired cells use both
likelihoods. RNA-only cells retain uncertain ATAC parents, with zero direct
ATAC likelihood precision. ATAC-only cells integrate out the entire missing
RNA branch while fitting. Otherwise a factorized approximation could
artificially penalize dependence through an unobserved child. Eliminated
branches are predicted afterward by ancestral integration, never inserted
as observed data in subsequent fitting.

Some representative paired cells are required to learn the relationship;
the existence of a single pair is not a statistical sufficiency guarantee.
The construction assumes ignorable missingness. This Gaussian implementation
is for simulated or appropriately processed observations, not raw counts.

Controls and results
--------------------

Training options remain in ``prior_L_kwargs`` and ``prior_F_kwargs``:
``n_epochs`` (otherwise ``internal_epoch``), ``batch_size``, ``lr``,
``penalty``, and architecture options such as ``hidden_dim``/``n_layers``.
Unsupported conditional options raise an error rather than being ignored.
The learned Gaussian-mixture families are ``cgb``, ``cgb_sharp``,
``cgb_sharp_2``, ``cash``, ``lcash``, ``po_lcash``, ``emdn`` and
``spiked_emdn``. On a nonconditional axis the ordinary prior builder is used,
including ash and the separately implemented HMM priors.

With no external or self covariates, neural priors receive one constant input
column (zero after standardization). Each factor learns one shared prior across
rows; the row-specific likelihood still produces different posterior moments.
CGB has a zero spike and one Gaussian slab; spiked EMDN has a zero spike and
multiple learned Gaussian slabs. A spike ``penalty > 1`` encourages mass at zero
by adding ``(penalty - 1) * sum(log(pi_spike))`` to the maximized objective.
Use ``spiked_emdn``, rather than ``emdn``, for this sparsity penalty.
Plain ``cgb`` rejects ``omega``; that option belongs to the sharp CGB variants.

In a connected graph, CGB effective slab variances are learned through the
profile objective. For sharp families, ``omega`` controls their initial
variance, not a repeated variance-shrinkage operation. ``learn_scales=False``
fixes that variance explicitly. Numerical SD bounds default to ``min_sd=1e-4``
and ``max_sd=1e4``. Scale-mixture grids remain fixed within a conditional fit.
The no-edge route preserves the original scalar solvers, including their
finite-optimization behavior and options.

Sweep progress is controlled separately by ``cEBMF(..., verbose=True)``
(the default). It prints one short completion message per sweep, including
when using ``iter_once()``. Set ``verbose=False`` to silence those messages;
approximation warnings and explicitly enabled inner-prior logging remain
separate. ``fit_joint`` prints one message for each shared sweep if either
model is verbose; set ``verbose=False`` on both models for quiet joint progress.

Numerical settings use ``conditional_kwargs``:

* ``approximation="quadratic"`` (default) uses cached local child feedback and
  analytic Gaussian-component updates for faster processing. Parent uncertainty
  is preserved; see :doc:`quadratic_feedback`. A warning at graph creation
  explains the approximation and suggests
  ``conditional_kwargs={"approximation": "quadrature"}`` to select the
  quadrature method instead. Continuing the same fit does not repeat the warning.
* ``quadrature_points=24``: Gauss-Hermite nodes for each continuous component.
* ``parent_samples=32``: fixed Sobol integration points when several parents
  require integration. A single uncertain parent uses its full stored
  quadrature, and child mixture labels/values use analytic component moments.
* ``seed=0``: the Sobol integration seed. Network initialization uses torch's
  normal random state, so set ``torch.manual_seed`` for a reproducible fit.

The variational family factorizes across loading coordinates, with each
value kept dependent on its own mixture label. This does not make the prior
independent, but it does omit posterior covariance between loading coordinates.
``result.inference`` is ``"conditional_quadratic"`` by default, or
``"conditional_variational"`` when quadrature feedback is selected;
``result.reconstruction`` is ``result.L @ result.F.T``;
``history_obj`` is a quadrature approximation to the negative regularized
ELBO (lower is better). The ash global mixture-weight regularizer is included
on ordinary ash axes. ``history_elbo`` contains the corresponding negatives.
There are no MCMC draws or sampling-error bars in these results.

The exact profile-coordinate derivation is monotone for exact block
maximization. Numerical integration and finite neural training do not give
that unconditional guarantee. The optimizer retains the best local profile
value it evaluates. Compare integration resolutions, seeds and held-out
performance. Rank and ordering stay fixed once a conditional graph is built;
``allow_backfitting`` applies to the ordinary path, not graph pruning.

Current implementation and migration
------------------------------------

The public implementation is in ``cebmf/_conditional.py`` and
``cebmf/multiview.py``. Conditional distribution objects are shared through
``priors/conditional.py``. The updated manuscript is
``output/pdf/atac_rna_joint_inference.pdf``; the short derivation is
``output/pdf/conditional_loading_prior_derivation.pdf``.

The former sampler is isolated under ``cebmf_torch.experimental``.
Explicit old ``joint_kwargs`` selects that deprecated reference path and
emits a warning. Omit it for the new algorithm. Do not translate ``draws`` or
``burnin`` into variational iteration counts. Old sampler outputs and
exploratory notebooks are preserved under ``examples/archive`` and labeled
as historical, not results of the new method.

See ``examples/ATAC_RNA_joint.ipynb`` for the complete readable example and
``examples/tree_joint_simple.ipynb`` for a single matrix. Their simulation
truths never enter fitting or initialization.
