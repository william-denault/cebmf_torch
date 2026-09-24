HMM priors for ordered factors
==============================

Use ``prior_F="hmm"`` to fit each factor column ``F[:, k]`` as an ordered
sequence along the **columns of the data matrix**. Use ``prior_L="hmm"`` for
an ordered sequence along its **rows**. Each factor and each side has its own
fitted HMM. Entries must already be in the intended order.

.. list-table:: Prior names
   :header-rows: 1
   :widths: 20 30 50

   * - Name
     - Effect support
     - Components within each state
   * - ``hmm``
     - Real-valued
     - Point mass at the state center and normal components
   * - ``hmm_pos``
     - Nonnegative, including zero
     - Point mass and normals truncated below at zero
   * - ``hmm_neg``
     - Nonpositive, including zero
     - Reflection of the nonnegative model

The sign constraint applies to the latent effects and posterior means, not
just to the state centers. Zero remains possible for both one-sided priors.
Use an initialization with the intended signs when imposing constraints on
both sides; factor signs are otherwise arbitrary in matrix factorization.

RNA/ATAC example
----------------

This is the HMM counterpart of the RNA fit in
``examples/model_RNA_ATAC.ipynb``:

.. code-block:: python

   from cebmf_torch import cEBMF

   mycebmf00R = cEBMF(
       data=X_obs_RNA,
       prior_L="gbinary",
       prior_F="hmm",       # or "hmm_pos" / "hmm_neg"
       K=K,
       device=device,
   )
   mycebmf00R.initialise_factors()
   res_RNA0 = mycebmf00R.fit(maxit=30)

Replace ``X_obs_RNA`` with ``X_obs_ATAC`` for the corresponding ATAC fit.
The HMM assumes unit spacing between consecutive columns. It does not infer
genomic distances, sort columns, or detect chromosome/sequence boundaries.
Avoid joining unrelated sequences into one chain.

When inspecting ``gbinary``/``hmm_pos`` fits, use separate vertical axes for
the factors. Multiplying ``L[:, k]`` by a constant and dividing ``F[:, k]``
by the same constant preserves the reconstructed signal. A large value in
``F`` therefore need not mean a large contribution to the data. A useful
scale-invariant measure of each component's contribution is::

   component_rms = (res.L.square().mean(0) * res.F.square().mean(0)).sqrt()

The cEBMF HMM adapters learn initial state probabilities by default
(``estimate_init=True``). Holding them fixed while learning transitions can
leave an unsupported spike at the first feature, even on an all-zero input.
Set ``prior_F_kwargs={"estimate_init": False}`` to request fixed initial
probabilities. The low-level ``fit_ash_hmm`` retains its fSuSiE-compatible
default of ``estimate_init=False``.

The ``gbinary`` solver compares its fitted mixture with the all-zero boundary
prior. If the boundary fits at least as well, it returns zero moments and
zero slab weight, allowing cEBMF to prune that unused factor. This check
uses marginal likelihood, not a threshold on factor magnitudes.

Separate RNA and ATAC fits do not by themselves impose a hierarchy or shared
loadings between the two datasets.

Side information and the startup warning
----------------------------------------

On an HMM side, all external and self-covariates are ignored. For example,
keeping ``X_f=torch.arange(1, 1001)`` in the example gives the same fit as
omitting ``X_f``. Changing those values does not change spacing or ordering.
One warning is emitted at model construction for each HMM side that has
side information, including enabled self-covariates. No warning is emitted
when no side information was supplied, and the solver does not repeat this
warning during factor updates. The other side's covariates remain usable.

To use actual locations together with other side information, include
location in the covariate matrix and use ``emdn`` or ``spiked_emdn``:

.. code-block:: python

   X_f = torch.cat((locations.reshape(-1, 1), other_features), dim=1)
   model = cEBMF(
       data=X_obs_RNA, prior_L="gbinary", prior_F="spiked_emdn",
       X_f=X_f, K=K, device=device,
   )

Statistical model
-----------------

For the normal-means estimates supplied by a factor update, the model is

.. math::

   \widehat\beta_t \mid \beta_t \sim N(\beta_t, s_t^2), \qquad
   Q_1 \sim \pi, \qquad Q_t \mid Q_{t-1}=j \sim A_{j,\cdot},

   \beta_t \mid Q_t=m \sim \sum_{\ell} \rho_{m\ell} G_{m\ell}.

In the signed model, :math:`G_{m\ell}=N(\mu_m,\sigma_\ell^2)`; a zero scale
denotes an exact atom at :math:`\mu_m`. The nonnegative version truncates
continuous components to :math:`[0,\infty)`. Its marginal emission includes
the truncated-normal normalizing constants, and its posterior moments are
computed under that distribution. The negative version fits to negated
observations and reflects the means and centers back.

The default null state is exactly :math:`\delta_0`. Other states learn their
own mixture weights over a geometric scale grid. A power grid of candidate
centers is screened using independent emission weights before fitting.
Baum--Welch updates learn the transition matrix and mixture weights using
exact forward/backward smoothing. Supported persistent state centers can
move after a short warm-up, within their initial Voronoi cells. Low-occupancy
or nearly duplicated states are deleted only after evaluating the proposed
model's full marginal likelihood.

``post_mean2`` is the posterior second moment, not the square of the mean.
``pi0_null`` is the smoothed posterior probability that the effect is exactly
zero, including any zero-scale component centered at zero. ``loss`` is the
negative marginal log likelihood of the **whole sequence** under the final
fitted HMM. It is used in cEBMF's KL/objective calculation. Dependence within
a factor is retained by the HMM smoother.

``lfsr`` is the local false sign rate at each position, computed from the
full smoothed posterior mixture as
:math:`\min\{P(\beta_t \leq 0 \mid y), P(\beta_t \geq 0 \mid y)\}`.
As in fSuSiE, an exact zero counts in both sign tails. Thus ``lfsr`` can
exceed 0.5 when the posterior has an atom at zero and equals 1 for an exact
null posterior. For ``hmm_pos`` and ``hmm_neg``, it equals ``pi0_null``.

Controls and direct use
-----------------------

Pass options through ``prior_F_kwargs`` or ``prior_L_kwargs``:

.. code-block:: python

   model = cEBMF(
       data=X_obs_RNA, prior_L="gbinary", prior_F="hmm", K=K,
       prior_F_kwargs={"penalty": 1.5, "maxiter": 20, "tolerance": 1e-5},
       device=device,
   )

   from cebmf_torch.cebnm import fit_ash_hmm, hmm_posterior_means

   result = fit_ash_hmm(betahat, sebetahat)
   means = result.post_mean
   variances = (result.post_mean2 - means.square()).clamp_min(0)
   lfsr = result.lfsr          # One local false sign rate per location
   fitted_centers = result.model_param["mu"]
   transitions = result.model_param["transition"]

   # The cEBNM adapter deliberately ignores its first argument, X.
   result = hmm_posterior_means(None, betahat, sebetahat, maxiter=20, penalty=1.5)

``penalty`` works for ``hmm``, ``hmm_pos`` and ``hmm_neg`` on either side.
It must be finite and at least 1. The default ``penalty=1`` preserves the
unpenalized fSuSiE fit; larger values favor zero effects. In the HMM this
means a Dirichlet pseudo-count of ``penalty - 1`` for the zero destination
in every transition row. The same pseudo-count is applied to the initial
zero-state probability when ``estimate_init=True`` and to the zero-scale
weight of the zero state when ``null_state="adaptive"``. With
``shared_mixture=True``, that mixture pseudo-count enters the pooled update.
The magnitude of its effect depends on the state grid and sequence length;
it need not match the strength of the same number in a covariate model.

Fitting and convergence use the penalized objective. The returned ``loss``
is still the negative **unpenalized** marginal log likelihood under the fitted
HMM, preserving the cEBMF KL calculation. ``history`` records log likelihood;
``objective`` and ``objective_history`` report the penalized log likelihood.
State pruning retains fSuSiE's check on the full marginal likelihood.

The following low-level ``fit_ash_hmm`` defaults match the referenced fSuSiE
exact solver. The cEBMF adapters override ``estimate_init`` to ``True``:

.. list-table:: Main fitting controls
   :header-rows: 1
   :widths: 32 18 50

   * - Option
     - Default
     - Meaning
   * - ``mu``, ``prior_sd``
     - ``None``
     - Automatic centers and scales; explicit centers start at zero, and scales start at zero and are nondecreasing.
   * - ``half_grid``
     - ``50``
     - Initial nonnegative candidates including zero; signed fitting also adds their nonzero negatives.
   * - ``grid_shape``, ``grid_expansion``
     - ``3``, ``3``
     - Power-grid shape and multiplier of the maximum absolute observation.
   * - ``prefilter``
     - Automatic
     - Enabled for automatic centers; disabled for supplied centers.
   * - ``topology``
     - ``"full"``
     - All transitions allowed; ``"hub"`` requires transitions between non-null states to pass through zero.
   * - ``stay_probability``
     - ``0.98``
     - Initial diagonal transition probability; transitions are subsequently estimated.
   * - ``null_state``
     - ``"pointmass"``
     - Exact zero null; ``"adaptive"`` allows a learned mixture centered at zero.
   * - ``penalty``
     - ``1.0``
     - Dirichlet preference for zero effects; values above 1 favor the zero state.
   * - ``shared_mixture``, ``estimate_init``
     - ``False``, ``False``
     - Share free-state scale weights or estimate initial state probabilities. Otherwise initial probabilities stay fixed at the initial transition matrix's stationary distribution (renormalized if states are pruned).
   * - ``learn_state_means``
     - ``True``
     - Learn supported centers after ``mean_update_start=3``; zero stays fixed.
   * - ``mean_min_effective_count``
     - ``2``
     - Minimum posterior occupancy for center learning.
   * - ``mean_min_self_transition``
     - ``0.93``
     - Persistence threshold for center learning; movement is damped by persistence.
   * - ``mean_bounds``
     - ``"voronoi"``
     - Keep learned centers in their initial cells; ``"none"`` removes this restriction.
   * - ``prune_states``
     - ``True``
     - Check state deletion starting at iteration 5 and every 5 iterations.
   * - ``prune_max_loglik_loss``
     - ``0.05``
     - Maximum permitted full-sequence log-likelihood loss for a deletion proposal.
   * - ``maxiter``, ``tolerance``
     - ``20``, ``1e-5``
     - Standalone iteration limit and relative objective convergence tolerance.

All supported parameters are listed in :func:`cebmf_torch.cebnm.fit_ash_hmm`.
Inside cEBMF, ``internal_epoch`` supplies the HMM iteration limit unless
``maxiter`` or ``n_epochs`` is explicitly supplied in the side's kwargs;
``maxiter`` takes precedence over ``n_epochs``. Set ``maxiter=0`` to evaluate
specified initial parameters without EM. State-sized initial values
(``init_transition``, ``init_prob``, ``init_rho``) require an explicit ``mu``
and disabled prefiltering. For ``hmm_neg``, explicit non-null centers must
be negative; ``prior_sd`` remains nonnegative.

The adapter accepts ``model_param`` for compatibility but refits from the
current observations on every call, as fSuSiE does. This allows states to
return after earlier pruning and lets automatic grids follow factor rescaling.
``model_param`` stores the final centers, scale grid, transition probabilities,
initial probabilities and mixture weights for inspection. Explicit initial
values can be used with the low-level solver for controlled continuation.

Numerical checks and scope
--------------------------

The port follows ``fsusieR/R/hmm_routines.R`` at commit ``9f0fd55``
(implementation version ``3.0.0-variational``). Despite that version label,
the reference's default inference is exact. This port implements that exact,
single-sequence EBNM path. It does not expose the R implementation's optional
variational approximation, BIC model-selection gate, multiple-sequence IDs,
arbitrary transition masks, arbitrary Dirichlet-prior arrays or post-fit step
decoders. The scalar ``penalty`` maps to a zero-favoring subset of fSuSiE's
Dirichlet-prior controls.

Calculations use float64 for numerical stability and return posterior
summaries on the input device and floating dtype. Negative-tail moments use
the reference's asymptotic formulas. The nonnegative center update maximizes
the same conditional criterion with a bounded golden-section search instead
of R's Brent optimizer. The automatic scale grid also handles the near-null
case where the reference's computed maximum scale falls below its minimum.
The solver requires finite observations and finite positive SEs; a scalar SE
is broadcast. cEBMF handles missing data before constructing these estimates.

Forward/backward recursions are sequential in position; the HMM is not a
mini-batch solver. CPU execution can be preferable for small state grids.
Cached component emissions use memory proportional to positions times states
times scales. Larger candidate grids increase time and memory requirements.

``tests/test_hmm.py`` compares fitted probabilities, moments, transitions,
weights, centers and likelihood histories against saved R outputs. The fixture
generator in ``tests/fixtures/generate_hmm_reference.R`` uses base R only.
Additional checks enumerate all paths of a small HMM, integrate one-sided
posteriors numerically, and exercise cEBMF integration and warning behavior.
