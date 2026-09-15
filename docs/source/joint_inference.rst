Joint inference for self-covariates and ATAC-RNA
===============================================

Automatic cEBMF interface
------------------------

``self_row_cov=True`` or ``self_col_cov=True`` now selects the joint sampler
through ``cEBMF.fit``. Both flags can be enabled. On each enabled axis, a
coordinate's conditional includes every later coordinate's child-prior
contribution. The current implementation uses the existing conditional
mixture networks; it does not add linear-logistic CGB or VampPrior.

.. code-block:: python

   model = cEBMF(
       data=Z, K=10, prior_L="cgb", prior_F="norm",
       self_row_cov=True, S=1.25, device="cpu",
       joint_kwargs={"burnin": 100, "draws": 150, "thin": 2},
   )
   model.initialise_factors()
   result = model.fit(8)
   signal = result.reconstruction
   posterior = result.joint_posterior

``fit(maxit)`` uses ``maxit`` partial Monte Carlo EM rounds, followed by
burn-in and retained posterior draws. ``iter_once()`` instead advances one
joint sweep at fixed fitted parameters and publishes that realized draw.
It does not collect posterior means or learn parameters. ``fit(0)`` collects
draws without further parameter learning. A repeated ``fit`` continues from
the sampler's last state, not from the displayed posterior means.

``result.inference`` is ``"joint"`` for this route. The posterior contains
``loading_draws``, ``factor_draws``, nonzero-component probabilities on both
axes, acceptance fractions, and a log-joint trace. ``result.reconstruction``
averages products within draws; multiplying marginal factor means generally
gives a different result. ``history_obj``/``model.obj`` contain sampled log
joint densities, not an ELBO or a quantity that must improve every sweep.

The joint route supports the eight learned scalar families listed below,
``norm`` (independent Gaussian scale mixtures), and the three HMM priors.
Other priors fail explicitly instead of silently using the old update.
HMM priors still ignore side information and self-covariates on their own
axis; enabling an effective self-covariate flag on the other axis selects
joint inference. Without either effective flag, the original variational
``cEBMF`` route is unchanged.

The sampler currently requires CPU tensors. Missing entries and fixed
heteroscedastic Gaussian observation variances are supported. Each matrix
row/column needs at least one observation for initialization. Unknown noise
is estimated during independent initialization and then frozen. The rank,
factor order, side information, non-neural priors and preprocessing also
remain fixed during joint sampling. Supplying ``initialise_factors(L=..., F=...)``
skips the independent initialization and preserves that starting rank/order.
Without user-supplied factors, a preliminary independent fit chooses the
starting values and rank; its CGB-like axis uses a generalized-binary prior.

``joint_kwargs`` defaults are: ``initialization_iterations=10``,
``pretrain_steps=100``, ``sweeps_per_round=10``, ``steps=30``, ``lr=0.003``,
``burnin=100``, ``draws=100``, ``thin=2``, ``seed=123`` and
``progress_every=20``. Experimental distribution settings may be supplied in
``joint_kwargs['prior_L_kwargs']``/``['prior_F_kwargs']``. Neural architecture
and distribution settings in ordinary prior kwargs are also forwarded.
Legacy neural ``penalty``, ``n_epochs`` and ``batch_size`` settings produce
a warning and do not change the normalized joint-prior objective. HMM kwargs
continue to configure its initial fitted prior.

Ready-to-run notebooks are:

* ``examples/tree_joint_simple.ipynb``: start here for a short CPU example with
  one settings cell, inline simulation and fitting, prediction errors and objective
  plots. Select the ``cebmf`` kernel and run all cells.
* ``examples/tree_joint_walkthrough.ipynb``: a self-contained, step-by-step
  tree example with negative-ELBO, fixed-draw prior-loss and posterior-sampling
  plots. Its independent baseline runs for 30 iterations.
* ``examples/check_tree_consistency_joint.ipynb``: one matrix, corrected tree
  simulation, signed unordered features, and optional hierarchies on both axes.
* ``examples/ATAC_RNA_self_cov_joint.ipynb``: automatic cEBMF calls, with RNA
  conditioned on fixed fitted ATAC loadings.
* ``examples/ATAC_RNA_hmm_joint.ipynb``: the full coupled ATAC/RNA sampler.

**Fixed cross-modality covariates are not uncertain parents.** Separate
``cEBMF`` calls with ``X_l=another_model.L`` do not propagate uncertainty or
feedback between modalities. Changing fixed covariates mid-chain raises an
error. Use ``JointATACRNA`` below for that coupled target and partial overlap.

The corrected tree simulation has seven named programs but only four leaf
profiles, so its signal has rank at most four. Good reconstruction does not
identify the generating seven-program tree. The default total ordering is a
conditional model, not a learned tree topology.

Fully coupled ATAC-RNA example
-----------------------------

The runnable notebook is ``examples/ATAC_RNA_hmm_joint.ipynb``.
It recreates the two ATAC bands and four RNA profiles, and can include
paired, ATAC-only and RNA-only samples. Start with its quick configuration;
set ``QUICK = False`` for the original 2,000 samples and 1,000 features.

From an activated Python environment, at the repository root::

    python -m pip install -e .
    python examples/run_atac_rna_joint.py --quick
    python examples/run_atac_rna_joint.py --paired
    python examples/run_atac_rna_joint.py --quick --prior-rna spiked_emdn --side-info

The first command updates the editable installation. The second is a shorter
diagnostic run; the third runs the full-size, fully paired example.
Restart an existing notebook kernel after updating the installation.

Model and interface
-------------------

For a matched sample, the loading prior is

.. math::

   p(A_i,R_i\mid X_i)=
   \prod_k g^A_k(A_{ik}\mid A_{i,<k},X_i)
   \prod_j g^R_j(R_{ij}\mid A_i,R_{i,<j},X_i).

Feature factors have independent HMM priors along their respective ordered,
unit-spaced feature axes. The sampler uses one directed joint model;
RNA likelihood information reaches ATAC through the RNA child-prior terms.

.. code-block:: python

   from cebmf_torch.experimental import JointATACRNA, align_modalities

   aligned = align_modalities(
       X_obs_ATAC, X_obs_RNA,
       atac_ids=atac_sample_ids,
       rna_ids=rna_sample_ids,
       # Optional fixed information covering every union sample ID:
       # side_info=sample_covariates, side_ids=covariate_sample_ids,
   )
   model = JointATACRNA(
       aligned,
       prior_atac="cgb",
       prior_rna="spiked_emdn",
       factor_prior="hmm_pos",
       sigma_atac=1.5, sigma_rna=1.5,
       initial_k=5,
       initial_hmm_kwargs={"half_grid": 8, "maxiter": 20},
       prior_rna_kwargs={"num_components": 3, "hidden_dim": 16},
   )
   model.fit_prior_parameters(rounds=8, sweeps_per_round=10, steps=30)
   result = model.sample(burnin=100, draws=100, thin=2)
   predicted_rna = result.rna.reconstruction
   rna_loadings = result.rna.L
   rna_factors = result.rna.F
   sample_order = result.row_ids

This separate experimental API currently runs on CPU, with scalar fixed
Gaussian observation noise per modality. The supplied standard deviations
must describe the data being fitted; 1.5 is the known simulation value.
This class couples both modalities. The automatic cEBMF route above handles
joint inference within a single matrix.

Supported scalar families
-------------------------

Both ``prior_atac`` and ``prior_rna`` accept every scalar learned-registry
family, and the two modalities may use different families:

.. list-table::
   :header-rows: 1
   :widths: 22 45 33

   * - Name
     - Conditional distribution
     - Parameters learned in the optional joint phase
   * - ``cgb``
     - Zero atom and one signed Gaussian; neural weights
     - Gate, global mean and effective slab variance
   * - ``cgb_sharp``
     - Same family with an initialized narrow Gaussian
     - Gate and global mean; narrow scale fixed by default
   * - ``cgb_sharp_2``
     - Zero atom and two untruncated Gaussians with opposite-sign means
     - Gate and constrained means; narrow scales fixed by default
   * - ``cash``
     - Zero-centered Gaussian scale mixture; neural weights
     - Gate; scale grid fixed
   * - ``lcash``
     - Same mixture; linear softmax weights
     - Linear coefficients and intercepts; grid fixed
   * - ``po_lcash``
     - Same mixture; proportional-odds weights
     - Ordered cut-points and shared coefficients; grid fixed
   * - ``emdn``
     - Gaussian mixture without an exact spike
     - Conditional weights, means and scales
   * - ``spiked_emdn``
     - Zero atom and conditional Gaussian mixture
     - Conditional weights, means and slab scales

The sharp names correspond to ``cov_sharp_gb_prior.py`` and
``cov_sharp_2gb_prior.py``. The sign of a Gaussian's *mean* does not truncate
its support. Nonnegative feature factors do not force nonnegative loadings.
``factor_prior`` accepts ``hmm``, ``hmm_pos`` or ``hmm_neg``.

``prior_atac_kwargs`` and ``prior_rna_kwargs`` configure the experimental
conditional distributions. Options include ``hidden_dim``, ``n_layers``,
``num_components`` for EMDN families, a fixed increasing ``scales`` grid
starting at zero for CASH families, and ``slab_sd``/``omega``/``learn_scales``
for CGB variants. These are distribution settings, not the training kwargs
of the legacy EBNM fit functions.

The default effective sharp SD is ``slab_sd * sqrt(omega)``, subject to
``min_sd``. Sharp scales stay fixed unless ``learn_scales=True``; in that
case the actual variance is learned and ``omega`` only affects initialization.
Repeated post-update variance sharpening is not an exact M-step.
Learned continuous SDs are bounded by ``min_sd=0.02`` and ``max_sd=20``;
the fixed CASH grid is independent of those bounds.

The experimental model uses normalized priors without the legacy sparsity
penalty. Passing ``penalty`` here is unsupported: a penalty on a gate that
depends on latent parents requires an explicit joint objective and additional
parent terms. HMM initialization settings can still be passed through
``initial_hmm_kwargs``; fitted HMM distributions are frozen for sampling.

What is exact, what is approximate
----------------------------------

At fixed parameters, each loading update proposes a value and a mixture
component from its ordinary conditional normal-means posterior, then applies
a Metropolis correction using every direct child's augmented log prior.
For EMDN children this includes their means and variances, as well as weights.
HMM factors use joint state-path sampling followed by component and value
draws. These kernels preserve the specified augmented posterior; finite
chains need not have converged.

Initial independent ``gbinary``/HMM fits choose factor counts and HMM
parameters. The optional generalized Monte Carlo EM phase trains the loading
priors on joint latent draws. It accepts only improvements of the fixed-draw
complete-prior objective, but does not guarantee an increase in the actual
evidence. HMM parameters, noise, scale grids, ordering and preprocessing
remain fixed. This is joint *latent* inference conditional on fitted
parameters, with partial empirical-Bayes learning, rather than fully Bayesian
parameter inference or optimization over all model parameters.

Posterior collection freezes parameters and discards a new burn-in.
``result.*.reconstruction`` averages ``L @ F.T`` within each draw, preserving
posterior dependence. Multiplying ``result.*.L @ result.*.F.T`` is generally
different. ``reconstruction_sd`` describes latent-signal uncertainty and
excludes observation noise and hyperparameter uncertainty.
``loading_draws`` and ``factor_draws`` retain samples for diagnostics.
``inclusion_probability`` estimates nonzero-component probability; it is
one for EMDN. The logged joint density is not an ELBO and need not increase
during sampling. Acceptance rates alone do not diagnose mixing.

Partial overlap
---------------

Supply true sample IDs. The helper aligns the union and inserts NaN
measurement rows for absent modalities. Fixed side information must be
finite and cover that union; missing measurements never become zero-valued
neural-network inputs.

For an RNA-only sample, no ATAC likelihood is present. Its unobserved ATAC
loadings are still latent:

.. math::

   p(R_i\mid Y^R_i,X_i)\propto
   p(Y^R_i\mid R_i,F^R)
   \int p(R_i\mid A_i,X_i)\,p(A_i\mid X_i)\,dA_i.

Thus an RNA-only row can use fixed side information and the learned
population relationship after integrating over unknown ATAC loadings.
It does not plug in zero or drop the ATAC input dimensions.
RNA can itself inform the missing ATAC state. Conversely, at fixed
parameters, entirely unobserved RNA loadings integrate to one and add no
marginal ATAC likelihood.

The prototype requires some paired samples to initialize the cross-modality
relationship. Zero-overlap data do not generally identify that relationship
without additional assumptions or external information. Partial overlap is
supported, but improvement is not guaranteed for every sample group.

Verification and diagnostics
----------------------------

Run the focused mathematical and integration checks::

    python -m pytest tests/test_joint_conditional.py tests/test_joint_atac_rna.py -q
    python -m pytest tests/test_joint_matrix.py -q

Or run the package suite::

    python -m pytest tests -q

Checks cover native-family correspondence, numerical quadrature, every
scalar parent-child pairing, detailed balance, Monte Carlo moments,
finite-difference conditional scores, HMM path enumeration, truncated-normal
draws, missing-data marginals, and whole-solver parameter freezing.

The full-size example's diagnostic report is
``output/joint_atac_rna_full_diagnostic.json``. In one seed-1 run with 60%
paired samples, paired-row signal MSE improved for both views; the
observed unpaired groups worsened. The independent reference is the initial
``gbinary`` model, so this comparison changes the prior family and inference
method as well as adding dependence. It is not a controlled ablation or a
benchmark against other methods. Use multiple chains with the same fitted
parameters, several simulation seeds, and held-out data before drawing
performance or uncertainty-calibration conclusions.

Variational inference and scores
--------------------------------

The derivation in ``output/pdf/atac_rna_joint_inference.tex`` (with
``joint_implementation_sections.tex``) gives augmented variational updates,
parameter objectives, and approximation options. The implementation supplies
``model.loading_score(u, value, component, a, b)``: an autograd derivative
of the full conditional on a continuous component, including child terms.
It rejects spike components, for which that continuous derivative is
undefined. No learned score or variational optimizer is implemented.

A learned evaluable proposal could accelerate sampling with a full
Metropolis correction. Learning only continuous scores does not recover
spike masses, and conditional energy normalizers may depend on parents.
Those issues must be handled before substituting score matching for
normalized conditional priors.
