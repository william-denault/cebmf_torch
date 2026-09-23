# A variational joint-L update that reduces to ordinary cEBMF

Derived 15 September 2026; implemented 16 September. See [IMPLEMENTATION.md](./IMPLEMENTATION.md) for implementation status and [NUMERICAL_DEVICE_AUDIT.md](./NUMERICAL_DEVICE_AUDIT.md) for the subsequent adversarial review. [AUDIT.md](./AUDIT.md) describes the older sampler.

## 1. Design requirement

Keep the Gaussian matrix likelihood and an ordinary learned ash prior on F. Let loading priors depend on fixed row covariates and earlier latent loadings. Use a common variational objective for loading inference, loading-prior learning, feature inference, feature-prior learning, and noise estimation.

The defining reduction is stronger than a flag selecting old code:

> Removing all latent-covariate edges must reduce the mathematical update to the original cEBNM subproblem. Its implementation must then reuse the original solver, settings, moment updates, and schedule.

The construction below meets that requirement. Its main operation is a **joint update of one loading column's variational distributions and prior parameters**, obtained by profiling the variational distributions out of the objective. A single complete-data EM M-step would not give this exact algorithmic reduction.

## 2. Model and variational family

For observed entries,

\[
p(Y\mid L,F,\tau)=\prod_{ij\in\Omega}N(Y_{ij};L_i^TF_j,\tau_{ij}^{-1}).
\]

Introduce a component label \(Z_{ik}\) for each loading. For a spike and Gaussian components,

\[
g_{\theta_k}(v,z\mid P_{ik})=
\begin{cases}
\pi_{k0}(P_{ik})\delta_0(dv), &z=0,\\
\pi_{kh}(P_{ik})N(v;\mu_{kh}(P_{ik}),s_{kh}^2(P_{ik}))dv,&z=h>0,
\end{cases}
\]

where \(P_{ik}=(X_i,L_{i,<k})\), or the chosen subset of earlier parents. Every conditional is normalized. Features have independent ash priors, with their mixture weights learned.

Use

\[
q(L,Z,F)=\prod_{i,k}q_{ik}(L_{ik},Z_{ik})\prod_{j,k}q^F_{jk}(F_{jk}).
\tag{1}
\]

The value and its own component label remain dependent within q_ik. This is a factorized **posterior approximation to a joint generative prior on L**. It includes uncertainty in parents and feedback from children. It does not represent posterior covariance between different loading coordinates. That limitation must be stated, rather than calling this exact joint posterior inference.

For overlapping Gaussian components, this augmented mean-field approximation can differ from a mean-field approximation formed after marginalizing labels. They agree with the usual local normal-means posterior when parents are fixed and there are no child terms. In particular, component augmentation does not spoil the required ordinary cEBMF limit.

The unpenalized objective is

\[
\mathcal L(q,\theta,\eta,\tau)=E_q\log p(Y\mid L,F,\tau)
+\sum_{ik}E_q\log g_{\theta_k}(L_{ik},Z_{ik}\mid P_{ik})
+\sum_{jk}E_q\log g^F_{\eta_k}(F_{jk})+H(q).
\tag{2}
\]

Assume the component variances and needed expectations are well defined. Any numerical lower bound on variance is part of the admissible family and must be explicit and consistent between routes.

## 3. The likelihood reduction is still cEBMF's

With \(w_{ij}=M_{ij}\tau_{ij}\), define

\[
\bar R_{ij,-k}=Y_{ij}-\sum_{h\ne k}\bar L_{ih}\bar F_{jh},\qquad
a_{ik}=\sum_jw_{ij}E_q[F_{jk}^2],\quad
b_{ik}=\sum_jw_{ij}\bar R_{ij,-k}\bar F_{jk}.
\tag{3}
\]

Holding the other variational factors fixed, the expected log likelihood involving v=L_ik is

\[
-\tfrac12a_{ik}v^2+b_{ik}v+\mathrm{constant}.
\tag{4}
\]

The denominator is a posterior second moment. There is no need to sample F. Equation (3) is valid under (1), including when the *prior* depends on earlier loadings; prior dependence does not force the chosen q to be correlated.

## 4. Own-prior expectation and child feedback

Define the own-prior term

\[
A_{ik}(v,z;\theta_k)=E_{q(P_{ik})}\log g_{\theta_k}(v,z\mid P_{ik}).
\tag{5}
\]

The expectation is over latent parents only; X_i is fixed. Define

\[
D_{ik}(v)=\sum_{c:\,k\in\mathrm{pa}(c)}
E_{q(L_{ic},Z_{ic})q(P_{ic}\setminus L_{ik})}
\log g_{\theta_c}(L_{ic},Z_{ic}\mid P_{ic}[v]).
\tag{6}
\]

Only direct children appear. Under a total ordering they are all later columns. The exact variational coordinate at fixed parameters is

\[
q^*_{ik}(v,z)\propto
\exp\{-a_{ik}v^2/2+b_{ik}v+A_{ik}(v,z;\theta_k)+D_{ik}(v)\}.
\tag{7}
\]

These are **expected log prior terms**. They are generally neither the log prior evaluated at mean parents nor the log of an arithmetic mixture over sampled-parent priors:

\[
E\log g(v\mid P)\ne\log g(v\mid EP),\qquad
E\log g(v\mid P)\ne\log E g(v\mid P).
\]

Drawing parents once and applying the current MH conditional gives a different update. Likewise, ignoring D(v) gives the old one-way plug-in behavior rather than a coordinate of this joint-model ELBO.

## 5. Joint prior-and-posterior fitting: the key reduction

Write the exponent in (7) as \(\Psi_{ik}(v,z;\theta_k)\), and define

\[
\mathcal Z_{ik}(\theta_k)=\sum_z\int\exp\{\Psi_{ik}(v,z;\theta_k)\}\,d\nu_z(v),
\tag{8}
\]

where \(\nu_0\) is the unit mass at zero and continuous components use Lebesgue measure. Parameters are distinct between columns, as in the current code. Thus D is fixed while updating theta_k, even though it depends on the candidate loading value v.

The Gibbs variational identity gives

\[
\max_{q_{ik}}\{E_{q_{ik}}\Psi_{ik}+H(q_{ik})\}=\log\mathcal Z_{ik}(\theta_k).
\tag{9}
\]

Consequently the cEBNM-like loading update is

\[
\theta_k^{\rm new}\in\arg\max_{\theta_k}\sum_i\log\mathcal Z_{ik}(\theta_k),
\qquad q_{ik}^{\rm new}=q^*_{ik}(\cdot;\theta_k^{\rm new}).
\tag{10}
\]

This optimizes the loading-prior parameters and loading posterior block together. It is the appropriate generalization of cEBMF's local empirical Bayes fit. Approximate optimization is acceptable only with its numerical limitations made explicit; an exact block optimizer gives ELBO nondecrease.

### Proof of the no-self-reference limit

When there are no latent parents or children, A=log g_theta(v,z|X_i) and D=0. For a>0, let x=b/a and s²=1/a. Then

\[
e^{-av^2/2+bv}=\sqrt{2\pi/a}\,e^{b^2/(2a)}N(x;v,1/a).
\]

Therefore

\[
\log\mathcal Z_{ik}(\theta_k)=
\log\!\left[\int N(x;v,s^2)g_{\theta_k}(dv\mid X_i)\right]
+\tfrac12\log(2\pi/a)+b^2/(2a).
\tag{11}
\]

The last two terms do not depend on theta. Equation (10) is **exactly the original cEBNM marginal-likelihood optimization**, and (7) is its posterior. The factor update below is also unchanged. This is the requested mathematical reduction.

In implementation, take this limit analytically and call the existing cEBNM solver. Do not run noisy Monte Carlo or numerical quadrature to rediscover an available analytic answer. Also preserve the original solver's options and initialization. Matching a formal optimum does not imply that two different finite Adam/EM schedules produce identical outputs.

With a=b=0, use (7) directly, without dividing by zero. Without child/parent uncertainty it reduces to the fitted prior; observed-data initialization and handling of completely missing rows remain separate interface decisions.

**Do not normalize exp(A) as if it were an ordinary prior and discard the normalization.** With uncertain parents its integral is generally theta-dependent. Doing so would alter the objective in (10).

## 6. Computation for Gaussian mixtures and a spike

For each continuous component h define

\[
\begin{aligned}
\alpha_h&=a+E_P[s_h(P)^{-2}],\\
\beta_h&=b+E_P[\mu_h(P)s_h(P)^{-2}],\\
\gamma_h&=E_P\log\pi_h(P)-\tfrac12E_P\log(2\pi s_h(P)^2)
-\tfrac12E_P[\mu_h(P)^2s_h(P)^{-2}].
\end{aligned}
\tag{12}
\]

Set m_h=beta_h/alpha_h and V_h=1/alpha_h. Define one-dimensional integrals

\[
I_{rh}=\int v^rN(v;m_h,V_h)e^{D(v)}dv,\qquad r=0,1,2.
\tag{13}
\]

Unnormalized component masses are

\[
W_h=\exp\{\gamma_h+\beta_h^2/(2\alpha_h)\}\sqrt{2\pi/\alpha_h}\,I_{0h},
\qquad W_0=\exp\{E_P\log\pi_0(P)+D(0)\}.
\tag{14}
\]

Then Z=sum_h W_h and r_h=W_h/Z, and

\[
E_qv=\sum_{h>0}r_hI_{1h}/I_{0h},\qquad
E_qv^2=\sum_{h>0}r_hI_{2h}/I_{0h}.
\tag{15}
\]

An unspiked EMDN simply omits h=0. Compute masses in log space. If D is constant, its common contribution cancels in component probabilities, I0 has an analytic value, and ordinary Gaussian moments suffice. With fixed parents and D=0 these expressions equal the existing analytic Gaussian-mixture normal-means posterior.

### Computing D(v)

Let a child's component probability be r_ch and its component-conditional moments be m_ch and T_ch. The Gaussian component's expected log density is

\[
J_{ch}(P)= -\tfrac12\log(2\pi s_{ch}^2(P))
-\frac{T_{ch}-2\mu_{ch}(P)m_{ch}+\mu_{ch}(P)^2}{2s_{ch}^2(P)}.
\tag{16}
\]

Its contribution to D(v) is

\[
\sum_h r_{ch}E_{P\setminus v}[\log\pi_{ch}(P[v])+J_{ch}(P[v])],
\tag{17}
\]

with J_c0=0 for the fixed zero atom. For CGB with a global slab, J does not depend on v and can be omitted from the *v-dependent* part of D. For EMDN, means and variances depend on parents, so J must be retained.

This calculation integrates over child labels and values using their probabilities and moments. Sampling a single child label unnecessarily adds noise here.

### Numerical approximation

The v-integrals are one-dimensional, even when a loading has many parents. Adaptive quadrature or Gauss–Hermite integration around each Gaussian base component can evaluate them. Expectations over several uncertain parents can require separate quadrature or Monte Carlo integration. Hold integration nodes/draws fixed during each local parameter optimization, and check accuracy by refinement.

This does not prove the method will be fast for every architecture or large K. Sharp child constraints can demand more integration accuracy. It does remove whole-matrix posterior MCMC from the necessary algorithm and confines approximation to the new latent-covariate terms. Every no-edge block remains an ordinary cEBNM call.

## 7. Feature and noise updates

Under the factorized family (1), the feature pseudo-data remain

\[
a^F_{jk}=\sum_iw_{ij}E_q[L_{ik}^2],\qquad
b^F_{jk}=\sum_iw_{ij}\left(Y_{ij}-\sum_{h\ne k}\bar L_{ih}\bar F_{jh}\right)\bar L_{ik}.
\tag{18}
\]

Call the existing ash solver on bF/aF with SE 1/sqrt(aF), **refit its mixture weights**, and store first and second posterior moments. Preserve `prior_F_kwargs`. No feature draws, new factor-prior family, or frozen initialization prior are needed.

For unknown constant noise,

\[
\tau^{\rm new}=\frac{|\Omega|}{\sum_{ij\in\Omega}E_q[(Y_{ij}-L_i^TF_j)^2]},
\]

using

\[
E_q[(Y_{ij}-L_i^TF_j)^2]=
(Y_{ij}-\bar L_i^T\bar F_j)^2+
\sum_k\{E[L_{ik}^2]E[F_{jk}^2]-\bar L_{ik}^2\bar F_{jk}^2\}.
\tag{19}
\]

Use the existing row/column versions for their respective noise models and keep known noise fixed.

## 8. Entropy and the objective

For a continuous component in (7),

\[
q(v\mid z=h)=N(v;m_h,V_h)e^{D(v)}/I_{0h}.
\]

Its differential entropy is available from the same integrals:

\[
H_h=\tfrac12\log(2\pi V_h)
+\frac{E_h[(v-m_h)^2]}{2V_h}-E_h[D(v)]+\log I_{0h}.
\tag{20}
\]

The joint value/label entropy is \(-\sum_hr_h\log r_h+\sum_{h>0}r_hH_h\). The zero atom has zero within-component entropy. The feature KL contribution can use the ordinary EBNM identity, as in cEBMF.

Evaluate (2) using the actual stored variational distributions and current parameters, not independent marginals fitted afterward to a full-posterior chain. After later coordinate changes, an earlier q must remain fixed until explicitly updated: caches must not implicitly reevaluate that q with newly changed priors or parent distributions. This is a concrete implementation requirement for representing the tilted distributions.

Accurate block optimization and integrations yield the usual ELBO ascent result. Finite numerical approximations and incomplete/non-ascent neural optimization require error controls and honest reporting. The existing original scalar solvers themselves are not all exact optimizers, as discussed in the audit.

## 9. Sparsity penalties

First validate the replacement with penalty=1. To retain the existing neural spike regularizer consistently, add

\[
R=\sum_{ik}(\lambda_k-1)E_q\log\pi_{k0}(P_{ik}).
\tag{21}
\]

When updating q_ik, the own term is constant in L_ik, but children's penalty terms depend on its candidate value and must enter D(v). When updating theta_k, the own term enters the profile objective (10); it may be added outside log Z because it is independent of v. With fixed covariates, R becomes the original parameter regularizer and the standard penalized cEBNM limit is recovered.

This is a regularized objective, or an evidence bound including the previously described auxiliary observations. It should not be silently identified with the unpenalized model's evidence. Ash's global mixture-weight penalty has its own existing semantics and should likewise be preserved explicitly, not replaced by a neural per-row penalty.

## 10. Full learning schedule

1. Use the existing cEBMF initialization. Preserve supplied factors and their chosen scaling; do not apply an arbitrary 0.5 cutoff or refit a different preliminary model unless explicitly selected.
2. For k=1,...,K, compute aL,bL from current moments.
3. If the loading column has no latent parents or children, run its original cEBNM fit. Otherwise, form expected-prior and child terms and jointly fit theta_k and q_:k through (10). Cache component probabilities, moments, and a fixed representation needed for later integration/entropy.
4. Immediately update F_:k with the existing ash fit and current loading moments. Refresh residuals, preserving the original L_k/F_k ordering.
5. Update unknown noise and evaluate the common objective.
6. Repeat until the chosen stopping rule or iteration budget. No burn-in or final full-matrix posterior sampling stage is required for this variational algorithm.

For compatibility, preserve ordinary rank/pruning behavior when no latent edges exist. With a hierarchy, deleting a column changes child input identities; graph and network changes need an explicit policy. Holding rank fixed during the first controlled hierarchical comparison is defensible if it is also fixed in the comparator. It must not silently change the no-edge route.

## 11. If full within-row posterior dependence is required instead

One can use \(q(L,F)=\prod_iq_i(L_i)\prod_{jk}q^F_{jk}(F_{jk})\). The optimal row block is

\[
q_i^*(L_i)\propto p_\theta(L_i\mid X_i)
\exp\{-\tfrac12L_i^TH_iL_i+h_i^TL_i\},
\]

where \(H_i=\sum_jw_{ij}E[F_jF_j^T]\) and \(h_i=\sum_jw_{ij}Y_{ij}E[F_j]\).

This is a valid alternative, but it changes the F numerator:

\[
b^F_{jk}=\sum_iw_{ij}\left[Y_{ij}E[L_{ik}]
-\sum_{h\ne k}E[L_{ik}L_{ih}]E[F_{jh}]\right].
\tag{22}
\]

The covariance correction relative to (18) is

\[
-\sum_iw_{ij}\sum_{h\ne k}\operatorname{Cov}(L_{ik},L_{ih})\bar F_{jh}.
\]

It is therefore not correct to sample a correlated L posterior and feed only its marginal means/variances into the unchanged F residual formula. The current full-posterior sampler avoids this particular error by updating F using realized L; a hybrid implementation must use (22).

Also, retaining full row blocks does **not** by itself reduce to original mean-field cEBMF when prior edges vanish: the likelihood can still correlate loadings. To preserve the user's exact reduction with a structured approach, explicitly use variational blocks defined by connected components of the prior graph, and split them into singleton columns in the no-edge case. The joint block/prior objective is then a multivariate normal-means generalization; singleton blocks reduce to ordinary cEBNM. This is a more involved alternative than the scalar-coordinate construction above.

## 12. Derivation checks

[verify_replacement_derivation.py](./verify_replacement_derivation.py) checks:

- Equations (12)–(15) without parents/children against analytic Gaussian-mixture posterior probabilities and moments, including a spike and a zero-information observation.
- Binary versions of (7) against exhaustive enumeration of a three-loading Gaussian model, both with and without directed prior dependence; coordinate ELBO changes are checked directly.
- The profile objective (10) against the optimized enumerated ELBO at four prior-parameter settings; their difference is constant to 6.7e-16.
- The extra cross moment in (22) using a simple correlated two-loading distribution. The correct feature numerator is zero; incorrectly multiplying marginal means gives 0.25.

These checks pass to floating-point precision. An implementation would still require integration-error, neural-gradient, end-to-end no-edge equivalence, convergence, and performance tests.
