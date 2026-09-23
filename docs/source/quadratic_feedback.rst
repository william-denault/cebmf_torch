Local quadratic feedback
========================

Conditional fitting uses local quadratic feedback by default for faster
processing::

    model = cEBMF(
        Y, K=4, prior_L="spiked_emdn", prior_F="norm",
        self_row_cov=True,
        prior_L_kwargs={"penalty": 1.1},
    )
    result = model.fit(20)

Plain CGB can use ``prior_L="cgb"`` and
``prior_L_kwargs={"penalty": 1.0511}``. For two aligned modalities, give both
models the same ``conditional_kwargs`` before calling ``fit_joint(atac, rna)``.
A warning at fitting-graph creation explains that this is an approximation
whose results can differ from quadrature. It also shows how to select the
other method: ``conditional_kwargs={"approximation": "quadrature"}``.
The warning is not repeated during sweeps or continuation of the same graph.
An explicit ``approximation="quadratic"`` selects the same default algorithm.

What changes
------------

The prior remains :math:`p(L)=\prod_k p(L_k\mid L_{<k})`.
The approximation changes the inference algorithm, not the conditional model.
For each Gaussian component, it replaces nonlinear feedback from later factors
with a local concave quadratic. Component probabilities, means, variances and
augmented entropy can then be updated analytically. The zero spike stays exact.

Only child feedback is replaced. Uncertain parent inputs are still averaged
using the existing Gaussian quadrature/Sobol scheme. This is not a method that
uses only the overall mean and variance of every parent. Hermite nodes remain
as a representation for subsequent parent integration; they do not determine
the quadratic method's own posterior moments or entropy, which are stored
analytically. Column ash and observation-noise updates are unchanged.

Derivation
----------

Fix one loading coordinate :math:`v=L_{ik}`. Let :math:`z=c` denote its own
mixture label. Under the augmented scalar mean-field posterior, the Gaussian
likelihood and expected own-prior component produce

.. math::

   \exp\{-a v^2/2+bv+\mathbb E\log p(v,z=c\mid L_{i,<k})\}
   =\exp(B_c)\,\mathcal N(v;m_c,A_c^{-1}).

Here :math:`A_c=a+t_c>0`, where :math:`t_c` is the expected prior precision.
The centered calculation of :math:`B_c` is shared with the quadrature method;
it retains prior normalizers and avoids cancellation for narrow slabs.

Define the feedback, including the existing child sparsity penalties, by

.. math::

   h(v)=\sum_{j:\,k\in\mathrm{pa}(j)}
   \mathbb E_{q(L_{ij},Z_{ij})q(L_{i,\mathrm{pa}(j)\setminus k})}
   [\log p(L_{ij},Z_{ij}\mid L_{i,\mathrm{pa}(j)})]
   +\sum_{j:\,k\in\mathrm{pa}(j)}
   (\lambda_j-1)\mathbb E\log\pi_{j0}(L_{i,\mathrm{pa}(j)}).

Only active child factors contribute. Child values and labels are integrated
using their posterior component probabilities, conditional means and variances;
the remaining parent expectation uses the existing integration points.

At the start of the coordinate update, choose a separate expansion point
:math:`\xi_c=\mathbb E_{q_{\mathrm{old}}}[v\mid z=c]` for each component. Set

.. math::

   h_{0c}=h(\xi_c),\qquad g_c=h'(\xi_c),\qquad
   H_c=\min\{h''(\xi_c),0\},

   \widetilde h_c(v)=h_{0c}+g_c(v-\xi_c)
                    +\tfrac12H_c(v-\xi_c)^2.

The derivative, value and anchor are frozen during all inner optimizer steps.
They do not depend on the current factor's own prior parameters during that
coordinate update: the child priors, child posteriors and other parents are
fixed. Thus profiling the current prior needs no third-order differentiation
through the child networks. Each component has its own local approximation;
this is an approximation to the augmented objective, not a globally quadratic
replacement of the neural prior.

Writing :math:`d_c=m_c-\xi_c` and :math:`P_c=A_c-H_c`, Gaussian integration gives

.. math::

   m'_c=\frac{A_c m_c-H_c\xi_c+g_c}{P_c},\qquad
   V'_c=\frac1{P_c},

   \log w_c=B_c+h_{0c}
       +\frac{A_c}{P_c}\left(g_cd_c+\tfrac12H_cd_c^2\right)
       +\frac{g_c^2}{2P_c}
       -\tfrac12\log\left(1-\frac{H_c}{A_c}\right).

These centered expressions avoid subtracting large natural-parameter squares.
Cached child-network evaluations, derivatives and analytic normalizers use
float64 on the model's existing device. This matters when a narrow feedback
term has a mode far from the expansion point: float32 can cancel large terms
and return an incorrect mixture weight even when the posterior mean looks
correct. Training parameters and stored posterior moments retain the model's
dtype. Temporary float64 child-network states do not move to the CPU or alter
the original networks. Their GPU performance cost depends on the device.
The curvature cap guarantees :math:`P_c\ge A_c>0`. It is conservative: even
positive curvature small enough to leave a proper posterior is discarded.
This explicitly modifies the Taylor approximation in locally convex regions.
The number of clipped slab curvatures is retained in each graph history entry.

For the spike, :math:`v=\xi_0=0`, so its unnormalized log mass is
:math:`\mathbb E\log\pi_0+h(0)` without a quadratic approximation. Normalize
all component masses with log-sum-exp. If their probabilities are :math:`r_c`,
then

.. math::

   \mathbb E[v]=\sum_c r_cm'_c,\qquad
   \mathbb E[v^2]=\sum_c r_c\{(m'_c)^2+V'_c\},

   \mathcal H(q(v,z))=-\sum_c r_c\log r_c
      +\sum_{c:\mathrm{slab}}\frac{r_c}{2}\log(2\pi eV'_c).

The spike has zero mean and variance. This is joint value-and-label entropy,
not the marginal entropy of overlapping Gaussian components.

Reduction and checks
--------------------

For concave quadratic feedback, the local approximation is exact everywhere,
regardless of the expansion point. With no children, it reduces to the analytic
Gaussian-mixture normal-means update for the integrated own prior.
With no latent edges anywhere, the public API bypasses this code and calls
the original cEBMF solver, preserving its actual finite optimizer schedule.

Tests independently verify component weights, moments, entropy and gradients
against Gaussian integration; exact spike treatment; clipped convex curvature;
narrow-slab float32 odds; cached derivatives; no-edge equality; partially paired
views, column graphs and checkpoint continuation. CUDA-specific tests cover
device transfers and retained memory when a CUDA runtime is available.

Limitations and interpretation
------------------------------

* A Taylor approximation can be inaccurate for broad or multimodal uncertainty.
  Separate component anchors preserve distinct expansion points, but a nearly
  unused component can have an unrepresentative anchor and later acquire large
  posterior mass. They do not provide an accuracy bound.
* ReLU networks are piecewise smooth. Autograd computes derivatives within the
  current activation region, with its usual convention at a kink. The update
  can cross activation boundaries that the local approximation does not see.
* A proper Gaussian posterior does not guarantee a good approximation. Large
  gradient-driven shifts remain possible even after clipping curvature.
* The local profile score used to select inner optimizer steps belongs to the
  frozen quadratic surrogate. The reported ``history_obj`` instead reevaluates
  the original regularized objective under the stored posterior, using numerical
  parent integration. An improved surrogate need not improve that diagnostic
  or the true ELBO. Objective histories need not be monotone.
* Posterior dependence across coordinates is still omitted by mean-field
  inference. No claim of recovering an identifiable biological tree is made.

The result reports ``inference="conditional_quadratic"``. The diagnostic retains
``objective_kind="negative_regularized_elbo_quadrature"`` because it evaluates
the original prior using numerical parent integration, not the local surrogate.
Use ``examples/benchmarks/tree/compare_quadratic_feedback.py`` for matched
accuracy and timing comparisons with the alternative quadrature method.

Observed instability in the corrected tree example
-------------------------------------------------

The 17 September 2026 audit captured a coordinate update that improved its
quadratic profile score while worsening the evaluated negative objective by
more than one million. The offending component already had negative curvature
and positive posterior precision. It moved far from its expansion point and
acquired substantial posterior mass. Increasing parent integration points from
32 to 256 did not repair the step; quadrature from the same saved state avoided
the large deterioration. This is evidence about that captured step, not a claim
that quadrature fixes every estimation difference between the fitters.

The current quadratic implementation has no trust-region or actual-objective
acceptance check. Its positive-precision guarantee establishes a proper Gaussian
integral, not a reliable global approximation. Consult
``output/plugin_vs_joint_20260917/README.md`` and
``examples/benchmarks/tree/benchmark_plugin_vs_joint.py`` for the full
plug-in comparison, controls, completion counts and frozen source provenance.
