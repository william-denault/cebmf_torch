# Joint learning of L: mathematical and implementation audit

15 September 2026. Scope: one Gaussian observation matrix, autoregressive neural priors on L, ordinary Gaussian-mixture ash on F. HMM behavior is outside this audit. Reviewed the relevant history in **Add HMM ebnm priors**, the current working tree (including its pre-existing uncommitted changes), and the cEBMF paper.

## Verdict

**The current sampler is substantially more defensible than the claim that it implements the requested cEBMF extension.** Its fixed-parameter Gaussian-mixture proposals and child corrections are mathematically consistent. However, the learning loop changes the inference objective, stops learning the ash prior on F, changes prior/default semantics, and introduces avoidable initialization and Monte Carlo effects.

**The new learning algorithm does not reduce to ordinary cEBMF when latent covariates are removed.** The public interface happens to dispatch to a separate original implementation when both self-covariate flags are false. That is software fallback, not a reduction of the new algorithm. The distinction in the user's latest requirement is essential.

The weak tree results therefore do not establish that learning a joint prior on L is a bad idea. Equally, a correct joint probability model need not outperform a plug-in estimator on every finite sample. We must first compare implementations with the same priors, settings, initialization, and learning criterion.

The requested next step is an audit and replacement derivation, **not implementation of a replacement**. Production source was not changed in this audit. The proposed variational replacement is specified in [REPLACEMENT.md](./REPLACEMENT.md).

## 1. The intended model and the two different algorithms

Write

\[
Y_{ij}\mid L,F\sim N\!\left(\sum_kL_{ik}F_{jk},\tau_{ij}^{-1}\right),\qquad
p_\theta(L\mid X)=\prod_i\prod_k g_{\theta_k}(L_{ik}\mid X_i,L_{i,<k}),
\]

and

\[
p_\eta(F)=\prod_{j,k}\left[\pi_{k0}\delta_0(F_{jk})+
\sum_{h>0}\pi_{kh}N(F_{jk};0,s_{kh}^2)\right].
\]

This is a normalized joint prior on L because the dependency graph is acyclic and each conditional is normalized. It does not require an additional VAE, a new upper latent layer, or an HMM. A total ordering lets every earlier loading predict later loadings; it does not itself learn a tree topology.

### Ordinary cEBMF

With fixed covariates, cEBMF uses variational posterior moments and solves a normal-means empirical Bayes subproblem for both L and F. The paper's equations 14–15 use the **second posterior moment**, not the square of the first moment. See [the original algorithm, Section 3.3.2](https://arxiv.org/html/2505.11639v2#S3.SS3.SSS2).

Let \(w_{ij}=M_{ij}\tau_{ij}\), and use bars for posterior means. For loading k,

\[
a^L_{ik}=\sum_jw_{ij}E_q[F_{jk}^2],\quad
b^L_{ik}=\sum_jw_{ij}\left(Y_{ij}-\sum_{h\ne k}\bar L_{ih}\bar F_{jh}\right)\bar F_{jk}.
\]

The local observation and SE are \(b/a\) and \(a^{-1/2}\). The original code implements these moment formulas in `cebmf.py:479` and `cebmf.py:529`.

### The new code

`experimental/matrix.py:277` instead uses realized F and L, and `matrix.py:287` alternates posterior sweeps with neural complete-data parameter fits. This is partial Monte Carlo EM for a full posterior target. It is not a variational cEBMF coordinate update.

Using realized squares is **correct for Gibbs/MH sampling**. Replacing them with posterior second moments inside that existing full-posterior sampler would make its claimed target wrong. The required change is to derive a variational update, not patch a denominator in isolation.

Even infinite E-step samples do not make one complete-data M-step identical to a collapsed cEBNM fit. For example, take \(x=2\), measurement variance 1, prior \(N(\mu,1)\), and old \(\mu=0\). The marginal-likelihood cEBNM optimum is \(\mu=2\). An exact E-step followed by one M-step gives \(\mu=1\), and the refreshed posterior mean is 1.5 rather than 2. Repeated EM can reach the same optimum in this simple fixed-design problem, but the iterations are different. With uncertain F, full-posterior EM and mean-field cEBMF also optimize different criteria.

## 2. What is correct in the fixed-parameter implementation

For a loading v and its mixture label z, the full conditional is

\[
p(v,z\mid\text{rest})\propto
e^{-av^2/2+bv}g_\theta(v,z\mid P)\,e^{D(v)},
\quad D(v)=\sum_{c\in\mathrm{children}}\log g_{\theta_c}(L_c,Z_c\mid P_c[v]).
\]

The proposal is the normalized first part, q0. Consequently the independence-MH acceptance ratio is exactly

\[
\min\{1,\exp[D(v')-D(v)]\}.
\]

The own-prior and likelihood terms cancel against the proposal ratio. No missing likelihood factor or reversed sign was found. `conditional.py:250` implements this cancellation; `loading.py:21` evaluates every later child's full augmented density. In particular, EMDN child means and variances are included, not just their gate probabilities. Child labels remain fixed during this coordinate move, as required for this augmented-state kernel. Exact zero spikes have their own probability mass.

The matrix residual/statistic calculation also respects masking and elementwise precision. Row updates can be vectorized because fixed F and the specified priors make rows conditionally independent. Fixed input standardization avoids inadvertently coupling the rows through changing batch statistics.

The neural parameter objective, `loading.py:104` and `learning.py:49`, uses matched latent values, parent values, and labels. Maximizing their complete log prior is a legitimate MCEM M-step. A second measurement-error deconvolution here would be wrong. Likewise, an entropy gradient or gradient through the saved MCMC draws is not missing from an M-step: the E-step distribution is held fixed.

The problem is the chosen learning procedure and approximations, not an obvious error in this MH ratio.

## 3. Confirmed mismatches and performance risks

### A. Ash on F is frozen and its settings are discarded — high priority

At `experimental/matrix.py:121`, ash is fitted once with a hard-coded `penalty=1.0`. Its weights and scale grid are placed in `self.fixed`. Later feature updates only draw from that fixed prior; the M-step visits neural `axis.priors`, which is empty for an ash axis.

Ordinary cEBMF refits ash during its F updates. In addition, ordinary `ash()` defaults to `penalty=10` (`ebnm/ash.py:167`). Explicit settings such as `prior_F_kwargs={'penalty': 80, 'mode': 3}` do not reach this joint ash fit. The joint mixture is constructed with zero means irrespective of `mode`.

Thus even using `prior_F='norm'` in both runs does not make their feature priors or their training equivalent. Freezing parameters during final posterior sampling is appropriate; freezing the F prior throughout empirical Bayes learning is the mismatch.

Confirmed numerically: changing those F settings had no effect on the initialized joint prior, including in a probe with nonzero loading columns. Its weights also remained bitwise unchanged throughout the tree learning rounds.

### B. Initialization imposes an absolute 0.5 loading cutoff — high priority

`experimental/matrix.py:115` sets all initial loading values with absolute magnitude below 0.5 to zero for any spiked neural family. This includes signed spiked EMDN. The preceding pretraining treats the initial estimates as observations with an arbitrary common SE of 0.15 (`matrix.py:109`).

These are heuristics, not consequences of the Gaussian observation model or the cEBMF initialization. The 0.5 threshold is particularly inappropriate as a universal rule in a factorization whose likelihood is unchanged under \(L_k\mapsto cL_k,F_k\mapsto F_k/c\). It also changes the starting state before the F prior is fitted, so it can change a prior that is subsequently frozen.

An explicit probe starts with 60 nonzero loadings in (0,0.4); all 60 are zeroed before the first sweep. This does not prove those coordinates can never recover: their later priors can permit nonzero moves. It proves that the supplied initialization is substantially altered without a scale-aware statistical criterion.

### C. The same neural prior names do not have the same defaults or scale learning

`ConditionalMixture` defaults to width 16, one additional hidden layer, and a minimum slab SD of 0.02. The sharp priors fix their slab scales by default (`conditional.py:136–143`). With the current default `omega=0.01`, their initial/fixed SD is about 0.02236 in loading units.

The old sharp CGB solvers instead estimate variance from pseudo-data and multiply it by omega. The new behavior is not an implementation of that old update. It may be more coherent as a fixed prior family, but using the same name obscures a material change in shrinkage.

Neural penalty defaults also change: old CGB uses 1.1 and old spiked EMDN 1.5, whereas the joint training helper defaults to 1.0. The cEBMF-level `internal_epoch` is not consulted by the joint helper; explicit `n_epochs` is consulted. Explicit `n_epochs` also overrides pretraining epochs. Original scalar solvers and joint learning can perform very different numbers of optimizer steps for the same epoch count because joint training uses S times as many saved sample/row pairs.

### D. The current ELBO curve is a separate diagnostic, not the optimized variational objective

`experimental/elbo.py:116` fits independent labelled mixtures to saved marginals, then uses fresh draws from that explicit q to estimate \(E_q[\log p-\log q]\). Subject to its stated support conditions, this is a genuine ELBO construction. It is not the entropy of the MCMC samples, and the code does not make that mistake.

However, the sampler and MCEM loop do not optimize this fitted q. Destroying between-coordinate dependence when constructing q can produce unlikely parent/child combinations and a poor bound even when the sampled joint states fit well. A decreasing complete-data training loss, a higher sampled log joint, and a higher diagnostic ELBO are three different statements.

The ELBO standard error measures only its independent evaluation draws conditional on fitted q. It does not measure chain mixing, the error from learning on ten correlated sweeps, or the approximation gap of q.

### E. Learning uses short, changing chains

The default retains ten successive sweeps per learning round, immediately optimizes neural priors on those states, and continues from the last state. There is no initial E-step equilibration requirement or Monte Carlo accuracy criterion. Selecting the best network on those same saved states establishes improvement of that finite-sample objective only.

This can reinforce an early latent assignment. More final posterior draws cannot correct prior parameters that were learned from a poorly explored E-step. Sampling during learning and sampling after learning must be assessed separately.

### F. The existing example is not the requested plug-in comparison

`examples/run_tree_joint.py:43` compares the joint model against an independent norm/norm baseline. It is not a baseline using the old \(L_k\mid\widehat L_{<k}\) neural update. The original discussion also compared different cold/warm prior families, ranks and penalties. Those comparisons cannot isolate the child correction.

The corrected tree has seven named programs but only four distinct leaf profiles, hence signal rank at most four. Recovering its signal does not uniquely recover the seven-program tree. This is a limitation of a structural-recovery claim, not an excuse for worse signal RMSE.

## 4. Sampling evidence from this audit

The audit script exercises the retained legacy moment-update code through a local subclass that bypasses only automatic sampler dispatch. It does not change production code. Both methods start from the same scaled rank-4 factors. All runs use N=240, P=80, known noise SD 1.25, 5% held-out entries, explicit penalty 1 on both axes, width 16, one additional layer, and 20 neural epochs. The legacy route runs 20 iterations; the joint route runs eight learning rounds, then three chains, each with 200 burn-in sweeps and 600 retained sweeps.

These controls remove several confounders, but do **not** equalize objectives, slab rules, feature-prior learning, or total optimizer work. They are diagnostic comparisons, not a definitive matched-model performance benchmark.

Signal RMSE, lower is better (joint estimates average within-draw products and then pool chains):

| L prior | Data seed | Legacy plug-in | Current joint |
|---|---:|---:|---:|
| spiked EMDN | 1 | 0.18091 | 0.18589 |
| spiked EMDN | 2 | 0.18955 | 0.18515 |
| spiked EMDN | 3 | 0.19468 | 0.18271 |
| sharp CGB, two slabs | 1 | 0.16602 | 0.17172 |
| sharp CGB, two slabs | 2 | 0.16669 | 0.20576 |
| sharp CGB, two slabs | 3 | 0.18136 | 0.18233 |

The sharp-prior seed-2 gap persists after pooling 1,800 retained draws. Individual chain RMSEs are approximately 0.20564, 0.20569 and 0.20620. Comparing 50 with 600 draws changes the error modestly, not enough to close that gap.

Acceptance fractions for that run are about 97%, 99.4%, 100%, and 100%. Nevertheless, the first two loading columns have **zero component switches** across the recorded sweeps. This may reflect concentrated conditional assignments, or trapping; it is not by itself proof of either. Classical split R-hat on selected signal entries is near one, but all three chains started at the same fitted state. Agreement does not rule out a shared inaccessible mode.

The distinction between averaging products and multiplying means is real, but in these runs its effect on RMSE was only around 1e-5 to 1e-4. It does not explain this large sharp-prior gap.

An independently enumerable stress test demonstrates why acceptance is insufficient. Let two binary loadings agree with probability 1-1e-6, give the first a Bernoulli(1/2) prior, and provide no likelihood information. The exact marginal probability of one is 1/2. Starting 400 chains in (0,0), the same proposal/correction mechanism records approximately 50% parent acceptance and 100% child acceptance for 1,000 sweeps, but no parent switches: the empirical marginal remains zero. This is a limiting illustration using two atoms, not the production continuous CGB model.

**Assessment:** finite Monte Carlo error contributes; poor exploration during learning remains plausible; final averaging alone does not explain the demonstrated sharp-prior gap. The structural algorithm and prior changes are already established independently of these mixing questions.

## 5. The old implementation is not an infallible mathematical reference

Matching the old public behavior is an important compatibility requirement. It is distinct from proving that every old inner solver exactly maximizes an ELBO.

For example, `cov_gb_prior.py:122` computes a weighted average of \((\hat l_i-\mu)^2-s_i^2\). With heterogeneous s this is not generally the maximizing variance of the Gaussian marginal likelihood. With slab responsibilities equal to one, \(\hat l=(2,0)\), \(s=(0.1,10)\), \(\mu=0\), and old slab variance 4, that formula returns 1e-6 and reduces the slab log likelihood from -5.3532 to -201.8179. The sharp solver uses the related formula and additional multiplication by omega.

This counterexample identifies a pre-existing limitation; it is not claimed to cause the tree gap. Similarly, old neural losses can be penalized while their reported likelihood omits the penalty. Neither old nor new plotted curves should be treated as unconditional evidence of monotone optimization of the same criterion.

## 6. Required replacement and acceptance criteria

The minimal coherent replacement keeps cEBMF's variational moment updates and learned ash F prior. It changes the L variational coordinate by adding **expected own-prior terms under uncertain parents and expected child-prior terms**. It jointly fits each loading column's prior parameters and variational distributions by maximizing their local log normalizer. With no latent edges, that objective is the original cEBNM marginal log likelihood plus a parameter-independent constant. The complete derivation, including Gaussian mixtures and spikes, is in [REPLACEMENT.md](./REPLACEMENT.md).

Required tests before adopting an implementation:

1. Removing all latent-covariate edges makes the equations reduce to original cEBMF, and dispatches through the same existing EBNM solvers with the same options, initialization, schedule, and RNG state.
2. Tiny enumerated examples verify full-objective coordinate updates, including child feedback, and demonstrate ELBO ascent when numerical integrations/optimizers are accurate.
3. Neural mixed-prior calculations preserve spike mass and include dependence through means and variances.
4. Ash weights on F are learned each fitting iteration, and supplied F settings are respected.
5. The reported objective is the one being optimized, with entropy for the actual variational family.
6. Numerical integration and uncertain-parent approximations are checked for accuracy; independent starts and effective sample diagnostics assess any remaining sampling.
7. Only then compare joint and plug-in fits across seeds at comparable settings and computational budgets.

**Keep a full-posterior MCEM sampler explicitly labelled as a separate experimental method if desired. It should not define the required cEBMF reduction.**

## Evidence and reproducibility

- Targeted existing suite: **204 passed**, 46.45 seconds; three existing TorchScript deprecation warnings.
- [Audit experiments](./audit_joint_l.py), [spiked EMDN results](./results_spiked_emdn.json), [sharp CGB results](./results_cgb_sharp_2.json), [initialization/settings and mixing stress checks](./analytic_checks.json).
- [Replacement derivation verification](./verify_replacement_derivation.py) and [its results](./derivation_checks.json): no-parent Gaussian-mixture probabilities, moments and log normalizers agree with analytic normal-means calculations to 2.0e-15; binary dependent and independent coordinates agree with exhaustive enumeration to 1.2e-16; all tested exact coordinate ELBO changes are nonnegative. The profiled log-normalizer objective also agrees with the directly optimized enumerated ELBO up to a parameter-independent constant, to 6.7e-16. These checks validate the proposed equations, not a completed replacement solver.

From the repository root, using the project's Python environment:

```powershell
& 'C:/Users/willi/miniconda3/envs/cebmf/python.exe' -m pytest tests/test_joint_conditional.py tests/test_joint_matrix.py tests/test_joint_learning_elbo.py tests/test_ordered_covariates.py tests/test_known_variance.py -q
& 'C:/Users/willi/miniconda3/envs/cebmf/python.exe' output/review_joint_l/verify_replacement_derivation.py
& 'C:/Users/willi/miniconda3/envs/cebmf/python.exe' output/review_joint_l/audit_joint_l.py --prior spiked_emdn --seeds 1 2 3
& 'C:/Users/willi/miniconda3/envs/cebmf/python.exe' output/review_joint_l/audit_joint_l.py --prior cgb_sharp_2 --seeds 1 2 3
```
