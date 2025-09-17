Theoretical Background
======================

This section provides the theoretical foundations for the cebmf_torch package.

.. contents::
    :local:

This document summarizes the key statistical ideas behind the cebmf_torch
implementation. It focuses on Empirical Bayes Matrix Factorization (EBMF)
and the closely related Empirical Bayes Normal Means (EBNM) problems that are
used as building blocks.

Empirical Bayes Normal Means (EBNM)
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

This can significantly reduce the scatter between a prediction and the truth, when compared
to using the maximum likelihood estimate :math:`\hat{\theta}_i = y_i`.
See the :doc:`notebooks/spiked_emdn` for an example of this.


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
      X \approx \sum_{k=1}^K \ell_k f_k ^T + E,

where :math:`\ell_k` (:math:`n \times 1`) and :math:`f_k` (:math:`p \times 1`) are the loading and factor
vectors for factor :math:`k`, and :math:`E` is noise. In matrix form, this can be written as
:math:`X \approx L F^T + E`, where :math:`L` (:math:`n \times K`) and :math:`F` (:math:`p \times K`) 
are low-rank factors and are formed from the concatenation of the :math:`K` columns of loadings and factors, respectively.
In the EBMF factorization the matrix :math:`L_k` contains loading columns
which describe how strongly each of the :math:`n` observations is associated with latent factor
:math:`k`. The matrix :math:`F_k` contains factor columns which describe the
pattern of the factor across the :math:`p` variables.

Empirical Bayes approaches place priors on the values of :math:`\ell_k` and :math:`f_k` and estimate prior
hyperparameters from the data. Explicitly, if :math:`\ell_{k1}, \ldots, \ell_{kn}` are the
:math:`n` entries of :math:`\ell_k`, and :math:`f_{k1}, \ldots, f_{kp}` are the
:math:`p` entries of :math:`f_k`, then we say that

.. math::
      \ell_{k1}, \ldots, \ell_{kn} \sim g_{k}^{l} \in \mathcal{G}^{l}, \\
      f_{k1}, \ldots, f_{kp} \sim g_{k}^{f} \in \mathcal{G}^{f}.

where :math:`\mathcal{G}^{l}` and :math:`\mathcal{G}^{f}` are families of prior distributions
and :math:`g_{k}^{l}` and :math:`g_{k}^{f}` are the specific priors for factor :math:`k`.


Iterative Fitting
^^^^^^^^^^^^^^^^^

The EBMF model is typically fit using an iterative algorithm that 
loops over the factors :math:`k = 1, \ldots, K`.
Suppose we have an estimate of all factors except :math:`\ell_k` and :math:`f_k`.
Then we can compute the residual matrix

.. math::
      R_k = X - \sum_{j \neq k} \ell_j f_j^T = \ell_k f_k^T + E.

In this case, :math:`R_k` is a noisy observation of the rank-1 matrix :math:`\ell_k f_k^T`.
By right-multiplying by :math:`f_k` we obtain

.. math::
      R_k f_k = \ell_k (f_k^T f_k) + E f_k.

and thus the variable :math:`y_i` can be defined to be

.. math::
      y_i = \left(\frac{R_k f_k}{f_k^T f_k}\right)_i = \ell_{ki} + e_i, \quad e_i \sim \mathcal{N}(0, s_i^2),

where :math:`s_i^2` is the variance of the noise term :math:`E f_k / (f_k^T f_k)`.
This is now exactly the EBNM problem described above, and we can use an EBNM solver to estimate
:math:`g_k^l` and the posterior distribution of :math:`\ell_{ki}`.
The way of estimating :math:`f_k` is completely analogous.


As a summary, the EBMF approach does the following:

1. Initialize :math:`L` and :math:`F` (for example using SVD).
2. For each factor :math:`k = 1, \ldots, K`
      1. Compute the residual matrix :math:`R_k`.
      2. Solve the EBNM problem to estimate :math:`g_k^l` and the posterior distribution of :math:`l_{ki}`.
      3. Solve the EBNM problem to estimate :math:`g_k^f` and the posterior distribution of :math:`f_{ki}`.
3. Repeat step 2 until convergence.


.. admonition:: TODO

      Discuss sparsity and what our outputs are (posterior means)


Key properties
^^^^^^^^^^^^^^

1. Turns out to correspond to a variational approximation; approximate posterior by :math:`q(l, f ) = q(l)q( f )`.
2. This establishes objective function; guarantees convergence
3. Very flexible prior families; implementing new prior family only involves solving EBNM problem.
4. Level of sparsity automatically tuned to data as part of fitting (no CV).
5. Sufficiently efficient for reasonably large problems (no MCMC).
6. If the family of prior contains a delta function, then we can learn the rank :math:`K`.
7. Extend to :math:`K > 1` by iteratively adding/updating factors (deflation/backfitting).


Covariate Empirical Bayes Matrix Factorization (CEBMF)
------------------------------------------------------

In many applications, we have additional covariate information about the rows and/or columns of the data matrix :math:`X`.
For example, if our data matrix contains information about the height, weight etc. of individuals, 
then we may also have information about their age, gender, or other demographic factors, which provides
additional context that may help in the matrix factorization.
We call this problem Covariate Empirical Bayes Matrix Factorization (CEBMF).

In this case, the parameters of our prior distributions on the factors can depend on the covariates.
For example, if we had a simple Gaussian prior on the loadings, we could let the variance depend on the covariates:

.. math::
      l_{k1}, \ldots, l_{kn} \sim \mathcal{N}(0, \sigma_k^2(z_i)), \quad i = 1, \ldots, n.

where :math:`z_i` is the covariate vector for observation :math:`i` and :math:`\sigma_k^2(\cdot)` is some function
that maps covariates to variances. This could be the output of a neural network, or some simpler function such as a linear model.
This problem now has the additional challenge of estimating the function :math:`\sigma_k^2(\cdot)` from the data.

In the code, we define the covariates for :math:`L` to be :code:`X_l` and for :math:`F` to be :code:`X_f`.


Prior families
--------------

We make use of many prior families in the code, which we define below.

ASH (Adaptive Shrinkage)
^^^^^^^^^^^^^^^^^^^^^^^^

In the simplest example, we assume that the prior is a mixture of a point mass at zero and a mixture of zero-mean Gaussians

.. math::
      g(\cdot,) = \pi_0 \delta_0 (\cdot) + \sum_{j=1}^J \pi_j \mathcal{N}(\cdot, 0, \sigma_j^2),

where :math:`\pi_j` are the mixture weights and :math:`\sigma_j^2` are a fixed grid of variances.
Here we learn the parameters :math:`\{\pi_j\}`, but the variances are fixed.

CASH (Covariate Adaptive Shrinkage)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ASH model can be extended to allow the mixture weights to depend on covariates :math:`z`,

.. math::
      g(\cdot, z ) = \pi_0 (z) \delta_0 (\cdot) + \sum_{j=1}^J \pi_j (z) \mathcal{N}(\cdot, 0, \sigma_j^2),

where :math:`\pi_j(z)` are the mixture weights that depend on covariates :math:`z` and :math:`\sigma_j^2` are a fixed grid of variances.
Here we learn the functions :math:`\{\pi_j(z)\}` by fitting neural networks, but the variances are fixed.

EMDN (Empirical Mixture Density Network)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of restricting ourselves to a mixture of zero-mean Gaussians with known variance, 
we can use a more flexible prior family where the entire prior distribution depends on covariates :math:`z` 
through a Mixture Density Network (MDN)

.. math::
      g(\cdot, z_i) = \sum_{j=1}^J \pi_j(z_i) \mathcal{N}(\cdot, \mu_j(z_i), \sigma_j^2(z_i)),

where :math:`\pi_j(z_i)` are the mixture weights, :math:`\mu_j(z_i)` are the means, 
and :math:`\sigma_j^2(z_i)` are the variances of the mixture components.

Spiked-EMDN (Spiked Empirical Mixture Density Network)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The EMDN model can be expanded by including a point mass at zero in the mixture.

.. math::
      g(\cdot, z_i) = \pi_0(z_i) \delta_0(\cdot) + \sum_{j=1}^J \pi_j(z_i) \mathcal{N}(\cdot, \mu_j(z_i), \sigma_j^2(z_i)).

Although this is a special case of the EMDN model with one of the components having
zero variance, it is numerically more stable to treat it separately.


CGB (Covariate Generalized Binary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Spiked-EMDN model can be simplified to a two-component mixture of a point mass at zero and a single Gaussian

.. math::
      g(\cdot, z_i) = \pi(z_i) \delta_0(\cdot) + (1 - \pi(z_i)) \mathcal{N}(\cdot, \mu, \sigma^2).

In this case we learn the function :math:`\pi(z_i)` with a neural network, but simplify our task
by treating :math:`\mu` and :math:`\sigma^2` as independent of :math:`z_i` and learning them as global parameters.
This model is useful when we believe that there are two fundamental populations in the data, and we
want to learn the probability that each observation belongs to one of these populations.

Sharp_CGB (Sharp Covariate Generalized Binary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CBG model can be further simplified by relating the mean and variance of the Gaussian component to be

.. math::
      \sigma = \omega \vert \mu \vert.

This parameterization is useful when we believe that the variance of the non-zero component. We fix
:math:`\omega` to be a small constant and learn the mean :math:`\mu` as a global parameter.
In this case, we again learn the function :math:`\pi(z_i)` with a neural network.

.. admonition:: TODO

      Add the loss function we optimise here and a link to the relevant example


References
----------

