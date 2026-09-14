Ordered loading priors for ATAC and RNA
======================================

With ``self_row_cov=True``, factor ``k`` receives ``L[:, :k]`` as loading
covariates. The index is zero-based: factor 0 has no earlier loading, factor
1 receives column 0, and factor 2 receives columns 0 and 1. The current factor
and future factors are never predictors. Without external covariates, the
first factor receives a constant column so its prior is unconditional.
``self_col_cov=True`` applies the same ordering to F, except that HMM priors
ignore covariates as described in :doc:`hmm_priors`.

For the directed ATAC-to-RNA structure, configure ATAC with
``self_row_cov=True`` and no ``X_l``; configure RNA with
``self_row_cov=True`` and ``X_l=atac.L.detach().clone()``. Each RNA loading
then receives all ATAC loading columns followed by earlier RNA loadings.
Cells must be aligned between the matrices. Do not feed RNA loadings back
as ATAC covariates if this is the intended ordering.

Covariates changed after construction must be assigned through
``rna.covariate.X_l``. Assigning ``rna.X_l`` does not update the fitter's
covariates. If their dimensions or meanings change, discard the affected
saved networks using ``rna.model_state_L = [None] * rna.model.K`` before
continuing. Keep pruning disabled during linked fits to preserve network
input dimensions. Within a single model, pruning automatically discards
saved networks on sides using self-covariates, because their parent columns
have changed.

What the current updates compute
--------------------------------

The desired conditionals define a valid joint prior:

.. math::

   p(A,R) = \prod_k p(A_k \mid A_{<k})
            \prod_k p(R_k \mid R_{<k}, A).

Here A and R denote the ATAC and RNA loading matrices. However, the current
``cEBMF.iter_once`` algorithm conditions on the current posterior means of
the predictors. It does **not** perform variational inference for this
joint prior. The distinction concerns the fitting algorithm, not whether
the proposed prior is a valid probability model.

For example, the full coordinate update for one ATAC loading includes

.. math::

   \log q^*(A_{ik}) = C + \mathbb{E}_{q_{-ik}}\left[
       \log p(Y_A \mid A,F_A)
       + \log p(A_{ik}\mid A_{i,<k})
       + \sum_{h>k}\log p(A_{ih}\mid A_{i,<h})
       + \sum_j\log p(R_{ij}\mid R_{i,<j}, A_i)
   \right].

The last two sums account for downstream ATAC loadings and all RNA loadings
whose priors depend on this ATAC loading. The existing normal-means update
omits those sums and substitutes posterior means for uncertain predictors.
RNA updates analogously omit the contributions of later RNA conditionals.
Even a forward-only dependence in the prior therefore allows RNA evidence
to affect ATAC in full posterior inference. Adding an independently fitted
reverse conditional is not a substitute for those terms.

With neural-network conditional probabilities, the missing terms generally
do not reduce to a Gaussian normal-means likelihood. A full joint solver
needs to evaluate the expected conditional log priors and change its loading
updates accordingly. The existing HMM factor updates can still be useful,
but fixing covariate indexing alone does not supply a joint solver. Do not
interpret the sum of the two models' recorded objectives as a joint ELBO.

Diagnosing the example
----------------------

``examples/ATAC_RNA_hmm_ordered.ipynb`` is an executable example of the current
conditional approximation. It uses the successful independent fits as
initial values, applies the intended predictor ordering, and reports
reconstruction MSE against the appropriate noiseless matrix. Its optional
control permutes the rows of the ATAC predictors while leaving RNA rows
unchanged.

The CGB network learns a Gaussian slab mean and variance as well as the
covariate-dependent spike probabilities. Its slab is not truncated and
does not require positive loadings. The default ten inner epochs at the
cEBMF level may be inadequate when the initial loading amplitudes are large.
The example therefore uses explicit inner training settings and reciprocally
rescales L and F before fitting, preserving both their reconstructed product
and their posterior second moments. This changes optimization coordinates;
it does not constrain the fitted loadings to equal the simulation truth.

In simulations, compare reconstructions against ``X_RNA`` and ``X_ATAC``
respectively. Comparing an RNA reconstruction against ``X_ATAC`` measures
the difference between modalities, not RNA estimation error. On real data,
hold out entries before any fit used by the workflow and evaluate predictions
on those entries. A decrease in the conditional training objectives does
not establish improvement in either reconstruction or a full joint objective.
