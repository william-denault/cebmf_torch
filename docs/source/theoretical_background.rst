Theoretical Background
======================

This section provides the theoretical foundations for the cebmf_torch package.

.. contents::
    :local:

This document summarizes the key statistical ideas behind the cebmf_torch
implementation. It focuses on Empirical Bayes Matrix Factorization (EBMF)
and the closely related Empirical Bayes Normal Means (EBNM) problems that are
used as building blocks.

Empirical Bayes Nornal Means (EBNM)
-----------------------------------

Before describing EBMF, we first describe the simpler EBNM problem.
Suppose we observe independent observations :math:`y_1, \ldots, y_n` with

.. math::
      y_i \sim \mathcal{N}(\theta_i, \sigma^2), \quad i = 1, \ldots, n.

Here, :math:`\theta_i` is the unknown mean and :math:`\sigma^2` is the known variance.
We further assume that the :math:`\theta_i` are drawn from some unknown prior distribution

.. math::
      \theta_i \sim g \in \mathcal{G},

where :math:`\mathcal{G}` is some family of prior distributions.

If we want to find estimates of the :math:`\theta_i`, we could use the maximum likelihood estimates
:math:`\hat{\theta}_i = y_i`. However, in an Empirical Bayes approach, we instead estimate the prior
:math:`g` from the data, and then use this estimated prior to compute the posterior
:math:`p(\theta_i \mid y_i, \hat{g})` and use the posterior mean as our estimate of :math:`\theta_i`.

We thus find the maximum likelihood estimate of :math:`g`:

.. math::
      \hat{g} = \arg \max \prod_i \int p(y_i \mid \theta_i, \sigma) g(\theta_i) {\rm d} \theta_i.

and then compute the posterior distribution :math:`p(\theta_i \mid y_i, \sigma, \hat{g})`.

.. admonition:: TODO

      Add a short worked example for the EBNM problem here: simulate a small
      dataset (:math:`y_i`), estimate a prior :math:`g` (for example using a
      mixture prior or ash), and show how to compute posterior means
      :math:`E(\theta_i \mid y_i, \hat{g})`. See the :doc:`examples` page for
      longer scripts that can be adapted into this section.


Empirical Bayes Matrix Factorization
------------------------------------

Now we have seen how the Empirical Bayes approach can help in parameter estimation,
we now consider the more complex situation of matrix factorization.
Matrix factorization provides a compact, interpretable representation of high-dimensional data.
By approximating :math:`X` with a small number of latent factors we reduce noise, highlight
low-dimensional structure, and often obtain components that are easier to interpret than raw
columns. In many applications (e.g., genomics, recommendation systems, or signal processing)
the factors capture coherent patterns across samples or variables that correspond to
biological states, user preferences, or repeated signals.

Let us suppose we have taken :math:`n` observations of :math:`p` variables, and stored these in a data matrix
:math:`X \in \mathbb{R}^{n \times p}`.
Given this data matrix, EBMF approximates :math:`X` by a low-rank factor
model

.. math::
      X \approx \sum_{k=1}^K L_k F_k ^T + E,

where :math:`L` (:math:`n \times K`) and :math:`F` (:math:`p \times K`) are low-rank factors and :math:`E` is noise. 

In the EBMF factorization the matrix :math:`L_k` contains loading columns
which describe how strongly each of the :math:`n` observations is associated with latent factor
:math:`k`. The matrix :math:`F_k` contains factor columns which describe the
pattern of the factor across the :math:`p` variables.

Empirical Bayes approaches place priors on the columns of :math:`L_k` and :math:`F_k` and estimate prior
hyperparameters from the data. Explicitly, if :math:`l_{k1}, \ldots, l_{kn}` are the
:math:`n` entries of :math:`L_k`, and :math:`f_{k1}, \ldots, f_{kp}` are the
:math:`p` entries of :math:`F_k`, then we say that

.. math::
      l_{k1}, \ldots, l_{kn} \sim g_{k}^{l} \in \mathcal{G}^{l}, \\
      f_{k1}, \ldots, f_{kp} \sim g_{k}^{f} \in \mathcal{G}^{f}.

where :math:`\mathcal{G}^{l}` and :math:`\mathcal{G}^{f}` are families of prior distributions
and :math:`g_{k}^{l}` and :math:`g_{k}^{f}` are the specific priors for factor :math:`k`.


Iterative Fitting
^^^^^^^^^^^^^^^^^

The EBMF model is typically fit using an iterative algorithm that 
loops over the factors :math:`k = 1, \ldots, K`.
Suppose we have an estimate of all factors except :math:`L_k` and :math:`F_k`.
Then we can compute the residual matrix

.. math::
      R_k = X - \sum_{j \neq k} L_j F_j^T = L_k F_k^T + E.

In this case, :math:`R_k` is a noisy observation of the rank-1 matrix :math:`L_k F_k^T`.
By right-multiplying by :math:`F_k` we obtain

.. math::
      R_k F_k = L_k (F_k^T F_k) + E F_k.

and thus the variable :math:`y_i` can be defined to be

.. math::
      y_i = \left(\frac{R_k F_k}{F_k^T F_k}\right)_i = l_{ki} + e_i, \quad e_i \sim \mathcal{N}(0, s_i^2),

where :math:`s_i^2` is the variance of the noise term :math:`E F_k / (F_k^T F_k)`.
This is now exactly the EBNM problem described above, and we can use an EBNM solver to estimate
:math:`g_k^l` and the posterior distribution of :math:`l_{ki}`.
The way of estimating :math:`F_k` is completely analogous.


As a summary, the EBMF approach does the following:

1. Initialize :math:`L` and :math:`F` (for example using SVD).
2. For each factor :math:`k = 1, \ldots, K`
      1. Compute the residual matrix :math:`R_k`.
      2. Solve the EBNM problem to estimate :math:`g_k^l` and the posterior distribution of :math:`l_{ki}`.
      3. Solve the EBNM problem to estimate :math:`g_k^f` and the posterior distribution of :math:`f_{ki}`.
3. Repeat step 2 until convergence.
   

References
----------

