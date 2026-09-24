# Hierarchical ATAC→RNA Model: Analysis & Formal Specification

## 1. What the Toy Model Represents

### Biological Motivation

The notebook encodes a stylized but mechanistically motivated model of how **chromatin accessibility drives alternative splicing**. The two ATAC peaks (at positions ~225 and ~725) represent two distinct regulatory elements — say, two enhancers or splice-site accessibility regions. The four RNA splicing programs correspond to the four possible combinatorial states of those two elements being open or closed. This is a toy version of the idea that ATAC-seq data encodes a *regulatory context* that determines which RNA isoform program is active in a given cell — a setting that naturally arises in single-cell multi-omics (e.g., 10x Multiome).

The key structural feature is that the **relationship between ATAC and RNA is not additive — it is combinatorial (Boolean)**. This is the heart of why this model is hard.

---

## 2. Formal Generative Model

### 2.1 Notation

Let $N$ be the number of samples (cells), $P$ the number of features (genomic positions, shared for simplicity).

### 2.2 ATAC Layer

Two binary latent loading vectors are drawn independently:
$$
L_{i,1}^{\mathrm{ATAC}},\; L_{i,2}^{\mathrm{ATAC}} \;\overset{\mathrm{iid}}{\sim}\; \mathrm{Bernoulli}(0.5), \quad i = 1, \ldots, N.
$$

Two localized (indicator) factors are fixed:
$$
f_{1,j}^{\mathrm{ATAC}} = \mathbb{1}_{j \in [200,250]}, \qquad f_{2,j}^{\mathrm{ATAC}} = \mathbb{1}_{j \in [700,750]}.
$$

The observed ATAC matrix is:
$$
\boxed{
X_{ij}^{\mathrm{ATAC}} = \sum_{k=1}^{2} L_{i,k}^{\mathrm{ATAC}}\, f_{k,j}^{\mathrm{ATAC}} + \varepsilon_{ij}^{\mathrm{ATAC}}, \qquad \varepsilon_{ij}^{\mathrm{ATAC}} \sim \mathcal{N}(0, \sigma_A^2).
}
$$

In matrix form: $\mathbf{X}^{\mathrm{ATAC}} = \mathbf{L}^{\mathrm{ATAC}} (\mathbf{F}^{\mathrm{ATAC}})^\top + \mathbf{E}^{\mathrm{ATAC}}$, with $\mathbf{L}^{\mathrm{ATAC}} \in \{0,1\}^{N \times 2}$.

### 2.3 Nonlinear Hierarchical Link: The Boolean Map

The **ATAC state** of sample $i$ is the pair $(L_{i,1}^{\mathrm{ATAC}}, L_{i,2}^{\mathrm{ATAC}}) \in \{0,1\}^2$. The RNA loading is a **one-hot encoding** of this joint state via a deterministic map $\phi: \{0,1\}^2 \to \{0,1\}^4$:

$$
\phi(a_1, a_2) =
\begin{cases}
\mathbf{e}_1 & \text{if } (a_1, a_2) = (0, 1) \\
\mathbf{e}_2 & \text{if } (a_1, a_2) = (1, 0) \\
\mathbf{e}_3 & \text{if } (a_1, a_2) = (1, 1) \\
\mathbf{e}_4 & \text{if } (a_1, a_2) = (0, 0)
\end{cases}
$$

Written out component-wise, each RNA loading is a **product of linear terms** in the ATAC loadings:
$$
\boxed{
\begin{aligned}
L_{i,1}^{\mathrm{RNA}} &= (1 - L_{i,1}^{\mathrm{ATAC}})\cdot L_{i,2}^{\mathrm{ATAC}} \\
L_{i,2}^{\mathrm{RNA}} &= L_{i,1}^{\mathrm{ATAC}} \cdot (1 - L_{i,2}^{\mathrm{ATAC}}) \\
L_{i,3}^{\mathrm{RNA}} &= L_{i,1}^{\mathrm{ATAC}} \cdot L_{i,2}^{\mathrm{ATAC}} \\
L_{i,4}^{\mathrm{RNA}} &= (1 - L_{i,1}^{\mathrm{ATAC}}) \cdot (1 - L_{i,2}^{\mathrm{ATAC}})
\end{aligned}
}
$$

This is an inherently **nonlinear** (quadratic in the ATAC loadings) and **non-additive** mapping — the effect of each ATAC factor on RNA depends on the state of the other.

### 2.4 RNA Layer

Four RNA factors are defined as superpositions of three localized indicator regions:
$$
\mathbf{F}^{\mathrm{RNA}} =
\begin{pmatrix}
\mathbf{f}_{\mathrm{mid}} + \mathbf{f}_{\mathrm{lo}} + \mathbf{f}_{\mathrm{hi}} \\
\mathbf{f}_{\mathrm{mid}} + \mathbf{f}_{\mathrm{hi}} \\
\mathbf{f}_{\mathrm{lo}} + \mathbf{f}_{\mathrm{hi}} \\
\mathbf{f}_{\mathrm{lo}} + \mathbf{f}_{\mathrm{mid}}
\end{pmatrix}
$$

where $\mathbf{f}_{\mathrm{lo}} = \mathbb{1}_{[200,250]}$, $\mathbf{f}_{\mathrm{mid}} = \mathbb{1}_{[500,550]}$, $\mathbf{f}_{\mathrm{hi}} = \mathbb{1}_{[700,750]}$.

The observed RNA matrix is:
$$
\boxed{
X_{ij}^{\mathrm{RNA}} = \sum_{k=1}^{4} L_{i,k}^{\mathrm{RNA}}\, f_{k,j}^{\mathrm{RNA}} + \varepsilon_{ij}^{\mathrm{RNA}}, \qquad \varepsilon_{ij}^{\mathrm{RNA}} \sim \mathcal{N}(0, \sigma_R^2).
}
$$

### 2.5 Full Joint Model (Compact Form)

$$
\begin{aligned}
L_{i,k}^{\mathrm{ATAC}} &\overset{\mathrm{iid}}{\sim} \mathrm{Bernoulli}(0.5), \quad k=1,2 \\
\mathbf{L}_{i,\cdot}^{\mathrm{RNA}} &= \phi\!\left(L_{i,1}^{\mathrm{ATAC}}, L_{i,2}^{\mathrm{ATAC}}\right) \quad \text{(deterministic, Boolean)} \\
\mathbf{X}^{\mathrm{ATAC}} &= \mathbf{L}^{\mathrm{ATAC}} (\mathbf{F}^{\mathrm{ATAC}})^\top + \mathbf{E}^{\mathrm{ATAC}} \\
\mathbf{X}^{\mathrm{RNA}} &= \mathbf{L}^{\mathrm{RNA}} (\mathbf{F}^{\mathrm{RNA}})^\top + \mathbf{E}^{\mathrm{RNA}}
\end{aligned}
$$

The hierarchical structure is: $\mathbf{L}^{\mathrm{ATAC}} \longrightarrow \phi(\cdot) \longrightarrow \mathbf{L}^{\mathrm{RNA}}$, where $\phi$ implements a truth table — a classic XOR/AND interaction structure.

---

## 3. Why MOFA+ Cannot Fit This Model

MOFA+ (Multi-Omics Factor Analysis v2) assumes the following generative model across $M$ views:

$$
\mathbf{X}^{(m)} = \mathbf{Z}\, (\mathbf{W}^{(m)})^\top + \mathbf{E}^{(m)}, \quad m = 1, \ldots, M
$$

where $\mathbf{Z} \in \mathbb{R}^{N \times K}$ is the **shared** latent factor matrix, and $\mathbf{W}^{(m)} \in \mathbb{R}^{P \times K}$ are view-specific factor loadings.

The critical constraint is that the sample representations $\mathbf{Z}$ are **identical across views** — only the loadings $\mathbf{W}^{(m)}$ differ. This means:
$$
\mathbf{L}^{\mathrm{ATAC}} = \mathbf{Z}\, (\mathbf{W}^{\mathrm{ATAC}})^\top \qquad \text{and} \qquad \mathbf{L}^{\mathrm{RNA}} = \mathbf{Z}\, (\mathbf{W}^{\mathrm{RNA}})^\top
$$

implying the relationship between the two sets of loadings is necessarily **linear**:
$$
\mathbf{L}^{\mathrm{RNA}} = \mathbf{L}^{\mathrm{ATAC}}\, \underbrace{(\mathbf{W}^{\mathrm{ATAC}})^{-\top} (\mathbf{W}^{\mathrm{RNA}})^\top}_{\text{fixed linear map}}.
$$

**The fundamental incompatibility:** The true link $\phi$ maps $(L_{i,1}^{\mathrm{ATAC}}, L_{i,2}^{\mathrm{ATAC}})$ to a one-hot vector via products (e.g., $L_{i,3}^{\mathrm{RNA}} = L_{i,1}^{\mathrm{ATAC}} \cdot L_{i,2}^{\mathrm{ATAC}}$). This is **quadratic** in $\mathbf{Z}$, which lies strictly outside the linear function class that MOFA+ can represent, regardless of the number of factors $K$.

Concretely: MOFA+ with $K=2$ shared factors would find a $\mathbf{Z}$ that explains the ATAC data well, but the RNA data would require a rank-4 structure that is a nonlinear (Boolean product) function of those 2 factors. No linear map $\mathbf{W}^{\mathrm{RNA}}$ can capture this — the RNA patterns are **not in the column span of $\mathbf{L}^{\mathrm{ATAC}}$**. Increasing $K$ to 4 doesn't help either: you'd recover 4 good RNA factors, but then the 2-dimensional ATAC structure would not align linearly with them.

In short: MOFA+'s shared linear latent space assumption is **structurally misspecified** for any model where the link between modalities involves interactions (products, logic gates, or other nonlinearities) of the latent factors.

---

## 4. cEBMF Formulation

### 4.1 Why cEBMF is a Natural Fit

cEBMF (covariate Empirical Bayes Matrix Factorization) separates factorization from the inter-modality link by placing **covariate-informed priors** on the loadings. The key idea: rather than constraining $\mathbf{L}^{\mathrm{RNA}}$ and $\mathbf{L}^{\mathrm{ATAC}}$ to share the same $\mathbf{Z}$, cEBMF allows the **prior on $\mathbf{L}^{\mathrm{RNA}}$** to be a function of the estimated $\mathbf{L}^{\mathrm{ATAC}}$.

### 4.2 The cEBMF Model for Each Modality

For ATAC:
$$
\mathbf{X}^{\mathrm{ATAC}} = \mathbf{L}^{\mathrm{ATAC}} (\mathbf{F}^{\mathrm{ATAC}})^\top + \mathbf{E}^{\mathrm{ATAC}}
$$
$$
L_{i,k}^{\mathrm{ATAC}} \sim g_k^{\mathrm{ATAC}}(\cdot \mid \mathbf{x}_{i,k}^{(L,\mathrm{ATAC})}), \quad F_{k,j}^{\mathrm{ATAC}} \sim h_k^{\mathrm{ATAC}}(\cdot \mid \mathbf{x}_{j,k}^{(F,\mathrm{ATAC})})
$$

For RNA, with ATAC loadings as covariates:
$$
\mathbf{X}^{\mathrm{RNA}} = \mathbf{L}^{\mathrm{RNA}} (\mathbf{F}^{\mathrm{RNA}})^\top + \mathbf{E}^{\mathrm{RNA}}
$$
$$
\boxed{L_{i,k}^{\mathrm{RNA}} \sim g_k^{\mathrm{RNA}}\!\left(\cdot \;\Big|\; \mathbf{x}_{i,k}^{(L,\mathrm{RNA})}\right), \qquad \mathbf{x}_{i,k}^{(L,\mathrm{RNA})} = \Psi\!\left(\hat{\mathbf{L}}_{i,\cdot}^{\mathrm{ATAC}}\right)}
$$

where $g_k^{\mathrm{RNA}}$ is an adaptive (empirical Bayes) prior family and $\Psi$ is the **covariate feature map**.

### 4.3 The Critical Role of $\Psi$: Capturing the Interaction

This is where the key design choice lies. In the notebook's current implementation, $\Psi$ is the identity: $\mathbf{x}_{i,k}^{(L,\mathrm{RNA})} = \hat{\mathbf{L}}_{i,\cdot}^{\mathrm{ATAC}}$, which is a **linear** embedding of the ATAC state. This captures main effects but not interactions.

To fully capture the Boolean structure $\phi$, the feature map must include **interaction (product) terms**:
$$
\Psi\!\left(\hat{\mathbf{L}}_{i,\cdot}^{\mathrm{ATAC}}\right) = \left(\hat{L}_{i,1}^{\mathrm{ATAC}},\; \hat{L}_{i,2}^{\mathrm{ATAC}},\; \hat{L}_{i,1}^{\mathrm{ATAC}} \cdot \hat{L}_{i,2}^{\mathrm{ATAC}}\right)
$$

This 3-dimensional feature vector is sufficient to linearly recover all four one-hot indicators (since the fourth is determined by the other three: $L_{i,4}^{\mathrm{RNA}} = 1 - L_{i,1}^{\mathrm{RNA}} - L_{i,2}^{\mathrm{RNA}} - L_{i,3}^{\mathrm{RNA}}$). More generally, for $K_A$ ATAC factors, the full interaction feature map has $2^{K_A} - 1$ terms — exponential in the number of ATAC factors, which motivates approximations (kernels, shallow networks) for large $K_A$.

### 4.4 Alternating Inference (as in the Notebook)

The notebook implements a **coordinate-ascent / block-alternating** scheme:

$$
\hat{\mathbf{L}}^{\mathrm{ATAC}} \;\longrightarrow\; \text{prior on } \mathbf{L}^{\mathrm{RNA}} \;\longrightarrow\; \hat{\mathbf{L}}^{\mathrm{RNA}} \;\longrightarrow\; \text{prior on } \mathbf{L}^{\mathrm{ATAC}} \;\longrightarrow\; \cdots
$$

At each step $t$:
$$
\hat{\mathbf{L}}^{\mathrm{RNA},(t)} = \underset{\mathbf{L}^{\mathrm{RNA}}}{\arg\max}\; \mathcal{F}\!\left(\mathbf{X}^{\mathrm{RNA}}, \mathbf{L}^{\mathrm{RNA}}, \mathbf{F}^{\mathrm{RNA}} \mid \mathbf{x}^{(L,\mathrm{RNA})} = \Psi\!\left(\hat{\mathbf{L}}^{\mathrm{ATAC},(t-1)}\right)\right)
$$
$$
\hat{\mathbf{L}}^{\mathrm{ATAC},(t)} = \underset{\mathbf{L}^{\mathrm{ATAC}}}{\arg\max}\; \mathcal{F}\!\left(\mathbf{X}^{\mathrm{ATAC}}, \mathbf{L}^{\mathrm{ATAC}}, \mathbf{F}^{\mathrm{ATAC}} \mid \mathbf{x}^{(L,\mathrm{ATAC})} = \hat{\mathbf{L}}^{\mathrm{RNA},(t)}\right)
$$

where $\mathcal{F}$ is the cEBMF variational ELBO (Evidence Lower BOund). This is not a joint ELBO optimization — the objective being tracked in the notebook (increasing RNA/ATAC loss separately) reflects this: it is a pseudo-likelihood alternating scheme whose convergence properties depend on how tightly the two objectives are coupled.

### 4.5 Summary: Model Hierarchy Diagram

```
Bernoulli(0.5) × Bernoulli(0.5)
        |
        v
   L^{ATAC} ∈ {0,1}^{N×2}   ──────────────────────────────────────────────────────>  X^{ATAC} = L^{ATAC} F^{ATAC,⊤} + E^{ATAC}
        |
        | φ(·)  [nonlinear: products / Boolean logic]
        |   ← current impl: linear covariate pass-through (main effects only)
        |   ← ideal: include L1·L2 interaction term in Ψ
        v
   L^{RNA} ∈ {0,1}^{N×4}   ──────────────────────────────────────────────────────-->  X^{RNA}  = L^{RNA}  F^{RNA,⊤}  + E^{RNA}
```

---

## 5. Key Takeaways

**The toy model:** A two-level hierarchical factor model where the first-level (ATAC) binary factors determine the second-level (RNA) factors through a Boolean truth-table mapping. The nonlinearity is quadratic (products of binary variables), encoding alternative splicing programs driven by combinatorial regulatory element accessibility.

**MOFA+ cannot fit this** because its shared-$\mathbf{Z}$ assumption forces a linear relationship between modality loadings. The truth-table link $\phi$ is quadratic in $\mathbf{Z}$ and thus lies outside MOFA+'s function class entirely.

**cEBMF can fit this**, with the right covariate feature map $\Psi$. The current implementation (linear pass-through of $\hat{\mathbf{L}}^{\mathrm{ATAC}}$) captures main effects and already improves over independent fitting, as shown in the notebook's CDF plots. Adding the pairwise interaction term $\hat{L}_{i,1}^{\mathrm{ATAC}} \cdot \hat{L}_{i,2}^{\mathrm{ATAC}}$ to the covariate vector $\Psi$ would give the prior exactly the information it needs to represent the true nonlinear structure — and this remains within the cEBMF framework without any architectural change.
