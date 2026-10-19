Point-exponential comparison with R ebnm
========================================

The comparison uses Stephens Lab's ``ebnm`` 1.1-42 at revision
`825edf1 <https://github.com/stephenslab/ebnm/blob/825edf1e308af9f3e594a5482300b9fb99793129/R/point_exponential.R>`_.
Reference values were generated with R ``ebnm`` 1.1.42 and ``ashr`` 2.2.63.
The installed likelihood, posterior-summary and boundary-check function bodies
were checked against that revision. The fixture generator is
``tests/fixtures/generate_point_exp_reference.R``; Python tests read the saved
JSON and do not require R.

Shared model and parameter mapping
----------------------------------

Both implementations fit a mixture of a point mass at ``mu`` and
``mu + Exponential(rate=a)``, observed with independent Gaussian noise of
standard deviation ``s``. The exponential posterior component is a normal
with mean ``x - mu - s**2 * a`` and standard deviation ``s``, truncated below
zero, then shifted back by ``mu``.

.. list-table:: Parameter correspondence
   :header-rows: 1
   :widths: 30 35 35

   * - Quantity
     - Python
     - R ebnm
   * - Exponential component weight
     - ``result.pi_slab``
     - ``fitted_g$pi[2]`` for a two-component prior
   * - Point-mass weight
     - ``1 - result.pi_slab``
     - ``fitted_g$pi[1]``
   * - Exponential rate
     - ``result.scale`` (historical field name)
     - ``1 / fitted_g$scale[2]``
   * - Location
     - ``result.mode``
     - ``fitted_g$shift``
   * - Fixed prior
     - ``par_init=(logit(weight), log(rate), mu)``,
       ``fix_par=(True, True, True)``
     - ``g_init=gammamix(...)``, ``fix_g=TRUE``

The Python return names and calling convention are preserved. In particular,
do not feed R's exponential scale directly into Python's ``log(rate)``.

Numerical tail correction
-------------------------

The direct convolution formula includes two large terms of opposite sign:
``(s*a)**2 / 2`` and ``log(Phi((x-mu)/s - s*a))``. Computing both accurately
does not make their sum accurate when ``s*a`` is large. Likewise,
reconstructing posterior moments from a very negative tilted-normal mean
subtracts large, nearly equal quantities. Both the R source and the previous
Python implementation use these forms. R's rescaling by median standard
error does not change the dimensionless quantity ``s*a``.

For ``t = s*a - (x-mu)/s >= 4``, Python now evaluates the normal-tail
continued fraction and computes positive first and second moments directly.
The continued fraction follows the complementary-error-function expansion
in `NIST DLMF 7.9 <https://dlmf.nist.gov/7.9>`_. Near the uninformative-noise
limit, the marginal density is evaluated as the noise density plus a
``log1p`` correction, preserving small derivatives with respect to the rate.
The exponential mixture posterior uses the same corrected moment helper.

Ordinary-input results are compared with R. Extreme inputs, both numerical
branch transitions, and likelihood gradients are checked against independent
80-digit quadrature in ``tests/fixtures/exp_tail_reference.json``. Its
generator integrates the Gaussian likelihood times the exponential density;
it does not use normal-CDF formulas or the continued fraction. Tests also
check the likelihood and moment bounds from issue #84 and a cEBMF fit with a
retained near-zero factor.

Fitting behavior
----------------

Python now defaults to ``loga_l2=0.0``, matching R's unpenalized maximum
marginal likelihood objective. To retain the previous optional ridge on the
bounded rate parameter, pass ``loga_l2=1e-3`` explicitly. This is the one
intentional default change; existing call signatures remain valid.

``penalty`` defaults to 1 (no spike penalty). Larger values add
``-(penalty - 1) * log(pi_null)`` to the negative summed marginal likelihood,
matching the ASH nullweight convention. Inside cEBMF, pass it through
``prior_L_kwargs`` or ``prior_F_kwargs`` for ``prior="exp"``. The returned
``log_lik`` remains unpenalized; ``regularization_loss`` reports the spike
and optional rate penalties separately. The all-spike boundary comparison
uses the penalized objective.

As in R, infinite standard errors contribute no information to fitting and
their posterior summaries equal the fitted prior moments. The all-spike
boundary is considered only when the mixture weight is estimated and the
location is fixed. A fixed prior is never replaced merely because a different
prior has a higher likelihood. Non-finite objectives, gradients or final
summaries raise ``FloatingPointError`` instead of being silently replaced by
finite fallback values.

Some fitting choices remain different and can affect fitted parameters:

* Python uses PyTorch L-BFGS and autograd; R defaults to ``nlm`` with analytic
  gradients and supports other optimizers.
* Python retains the configurable bounds ``a_bounds=(0.01, 100)`` for an
  estimated rate and its existing initialization. Fixed rates are used
  unchanged. R optimizes the log rate without these bounds and uses a
  data-dependent initialization.
* Python follows the input dtype and device; R uses double precision and
  rescales observations before optimization.
* Python retains its minimum standard error of ``1e-6``. Exact zero-noise
  behavior is not part of the R parity claim.

The legacy ``eps`` and ``tresh_pi0`` arguments remain accepted. Mixture
weights are now evaluated with log-sigmoid formulas, and the all-spike choice
uses the likelihood comparison rather than a weight threshold.
