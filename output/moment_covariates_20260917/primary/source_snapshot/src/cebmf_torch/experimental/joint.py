"""Experimental learned-prior/HMM joint sampler for the RNA--ATAC notebook.

This conditional empirical Bayes prototype couples two modalities. cEBMF's
self-covariate route shares its loading kernels within one matrix. Loading
updates use normal-means proposals with a child-prior Metropolis correction.
Feature blocks use FFBS.
All parameters are frozen while retained posterior samples are collected.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cebmf_torch import cEBMF
from cebmf_torch.cebnm.hmm import _component_log_emission

from .conditional import ConditionalMixture
from .data import AlignedModalities
from .loading import ConditionalLoadingSampler


@torch.jit.script
def _ffbs(log_emission: Tensor, transition: Tensor, initial: Tensor, uniforms: Tensor) -> Tensor:
    """Sample a whole Markov path from its posterior, not marginal states."""
    n = log_emission.size(0)
    filtered = torch.empty_like(log_emission)
    log_t = transition.log()
    first = initial.log() + log_emission[0]
    filtered[0] = first - torch.logsumexp(first, 0)
    for i in range(1, n):
        current = log_emission[i] + torch.logsumexp(filtered[i - 1, :, None] + log_t, 0)
        filtered[i] = current - torch.logsumexp(current, 0)
    states = torch.empty(n, dtype=torch.long, device=log_emission.device)
    for i in range(n - 1, -1, -1):
        logits = filtered[i]
        if i < n - 1:
            logits = logits + log_t[:, states[i + 1]]
        probabilities = torch.softmax(logits, 0)
        states[i] = (probabilities.cumsum(0) < uniforms[i]).sum().clamp_max(probabilities.numel() - 1)
    return states


def _positive_normal(mean: Tensor, sd: Tensor, generator: torch.Generator) -> Tensor:
    """Truncated normal draws with a stable exponential proposal in the tail."""
    result = torch.empty_like(mean)
    pending = torch.ones_like(mean, dtype=torch.bool)
    # For nonnegative means, normal rejection accepts at least half the draws.
    normal = mean >= 0
    while (pending & normal).any():
        index = (pending & normal).nonzero().flatten()
        value = mean[index] + sd[index] * torch.randn(len(index), generator=generator, dtype=mean.dtype)
        good = value >= 0
        result[index[good]] = value[good]
        pending[index[good]] = False
    # For negative means, sample an offset above the positive standardized
    # truncation point. Returning sd * offset avoids cancellation with mean.
    while pending.any():
        index = pending.nonzero().flatten()
        lower = -mean[index] / sd[index]
        root = torch.sqrt(lower.square() + 4)
        rate = 0.5 * (lower + root)
        uniform = torch.rand(len(index), generator=generator, dtype=mean.dtype).clamp_min(torch.finfo(mean.dtype).tiny)
        offset = -uniform.log() / rate
        log_acceptance = -0.5 * (offset - 2 / (lower + root)).square()
        good = torch.rand(len(index), generator=generator, dtype=mean.dtype).log() < log_acceptance
        result[index[good]] = sd[index[good]] * offset[good]
        pending[index[good]] = False
    return result


def sample_hmm(a: Tensor, b: Tensor, parameters: dict, generator: torch.Generator):
    """Draw (factor values, states, components) conditional on realized L."""
    if parameters.get("effect_support") == "nonpositive":
        reflected = {
            **parameters,
            "mu": -parameters["mu"],
            "effect_support": "nonnegative",
            "nonnegative_state_means": True,
        }
        value, states, components = sample_hmm(a, -b, reflected, generator)
        return -value, states, components
    if (a < 0).any() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("HMM precisions must be nonnegative and statistics finite.")
    if (b[a == 0] != 0).any():
        raise ValueError("A zero-precision likelihood must also have zero linear term.")
    mu, sd = parameters["mu"], parameters["prior_sd"]
    safe_a = torch.where(a > 0, a, 1.0)
    component = _component_log_emission(b / safe_a, safe_a.rsqrt(), mu, sd, parameters["nonnegative_state_means"])
    component[a == 0] = 0  # Missing positions contribute likelihood one.
    log_weights = parameters["mixture_weight"].log()
    emission = torch.logsumexp(component + log_weights, dim=2)
    states = _ffbs(
        emission,
        parameters["transition"],
        parameters["init_prob"],
        torch.rand(len(a), generator=generator, dtype=a.dtype),
    )
    positions = torch.arange(len(a))
    component_prob = torch.softmax(component[positions, states] + log_weights[states], dim=1)
    components = torch.multinomial(component_prob, 1, generator=generator).flatten()
    prior_var = sd[components].square()
    posterior_mean = (mu[states] + b * prior_var) / (1 + a * prior_var)
    posterior_sd = (prior_var / (1 + a * prior_var)).sqrt()
    values = posterior_mean.clone()  # Includes exact atoms.
    continuous = posterior_sd > 0
    if parameters["nonnegative_state_means"]:
        values[continuous] = _positive_normal(posterior_mean[continuous], posterior_sd[continuous], generator)
    else:
        values[continuous] += posterior_sd[continuous] * torch.randn(
            int(continuous.sum()), generator=generator, dtype=a.dtype
        )
    return values, states, components


@dataclass
class ModalityPosterior:
    L: Tensor
    L2: Tensor
    F: Tensor
    F2: Tensor
    reconstruction: Tensor
    reconstruction_sd: Tensor
    loading_draws: Tensor
    factor_draws: Tensor
    inclusion_probability: Tensor


@dataclass
class JointPosterior:
    atac: ModalityPosterior
    rna: ModalityPosterior
    acceptance: Tensor
    log_joint: list[float]
    row_ids: tuple


class JointATACRNA(ConditionalLoadingSampler):
    """CPU reference sampler with arbitrary missing entries and modality rows.

    Priors are initialized from independent gbinary/HMM fits of observed rows.
    An optional Monte Carlo EM phase updates the loading-prior parameters.
    HMM parameters, factor order, scaling, and observation noise then remain
    fixed. Kept draws sample the full latent joint posterior conditional on
    these fitted parameters. Finite chains can still mix poorly.
    """

    def __init__(
        self,
        data: AlignedModalities,
        *,
        sigma_atac: float = 1.5,
        sigma_rna: float = 1.5,
        initial_k: int = 5,
        initial_iterations: int = 10,
        pretrain_steps: int = 100,
        hidden_dim: int = 16,
        seed: int = 123,
        initial_hmm_kwargs: dict | None = None,
        prior_atac: str = "cgb",
        prior_rna: str = "cgb",
        prior_atac_kwargs: dict | None = None,
        prior_rna_kwargs: dict | None = None,
        factor_prior: str = "hmm_pos",
    ):
        if any(t.device.type != "cpu" for t in (data.atac, data.rna, data.side_info)):
            raise ValueError("This experimental reference example currently supports CPU tensors.")
        if sigma_atac <= 0 or sigma_rna <= 0 or not math.isfinite(sigma_atac + sigma_rna):
            raise ValueError("Supply finite positive observation standard deviations.")
        if initial_k < 1 or initial_iterations < 1 or pretrain_steps < 0 or hidden_dim < 1:
            raise ValueError("Invalid initialization size or iteration count.")
        if factor_prior not in ("hmm", "hmm_pos", "hmm_neg"):
            raise ValueError("factor_prior must be hmm, hmm_pos or hmm_neg.")
        if len(data.ids) != len(data.atac) or len(data.rna) != len(data.atac) or len(data.side_info) != len(data.atac):
            raise ValueError("Align modality and side-information rows with align_modalities first.")
        if not torch.isfinite(data.side_info).all() or any(torch.isinf(t).any() for t in (data.atac, data.rna)):
            raise ValueError("Fixed side information must be finite; use NaN only for missing observations.")
        self.data = data
        self.y = [data.atac.double(), data.rna.double()]
        self.side_info = data.side_info.double().clone()
        self.mask = [torch.isfinite(y).double() for y in self.y]
        self.y0 = [torch.nan_to_num(y, nan=0.0) for y in self.y]
        self.tau = [1 / sigma_atac**2, 1 / sigma_rna**2]
        present = [m.bool().any(1) for m in self.mask]
        if not (present[0] & present[1]).any():
            raise ValueError("This demo requires some paired rows to initialize cross-modality priors.")
        if not (present[0] | present[1]).all():
            raise ValueError("Every union row must have some observed data in at least one modality.")
        self.generator = torch.Generator().manual_seed(seed)
        self.initial_models = []
        self.factors, loading_parts, self.hmm = [], [], []
        for m, sigma in enumerate((sigma_atac, sigma_rna)):
            with torch.random.fork_rng():
                torch.manual_seed(seed + m)
                source = cEBMF(
                    self.y[m][present[m]],
                    K=initial_k,
                    prior_L="gbinary",
                    prior_F=factor_prior,
                    prior_F_kwargs=initial_hmm_kwargs or {},
                    S=sigma,
                    device="cpu",
                )
                source.initialise_factors()
                source.fit(initial_iterations)
            if source.model.K < 1:
                raise ValueError("The independent fit retained no factors; increase signal or initialization effort.")
            self.initial_models.append(source)
            scale = source.L.square().sum(0) / source.L.sum(0).clamp_min(1e-12)
            scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
            loadings = source.L.new_full((len(data.ids), source.model.K), 0.5)
            loadings[present[m]] = source.L / scale
            loading_parts.append(loadings)
            self.factors.append(source.F.detach().clone() * scale)
            parameters = []
            for k, state in enumerate(source.model_state_F):
                if state is None:
                    raise ValueError("Missing HMM initialization; increase initial_iterations.")
                saved = {
                    key: value.detach().clone() if isinstance(value, Tensor) else value for key, value in state.items()
                }
                saved["mu"] *= scale[k]
                saved["prior_sd"] *= scale[k]
                parameters.append(saved)
            self.hmm.append(parameters)
        self.ka = loading_parts[0].shape[1]
        self.kr = loading_parts[1].shape[1]
        self.loadings = torch.cat(loading_parts, dim=1)
        self.components = torch.zeros_like(self.loadings, dtype=torch.long)
        self.priors = nn.ModuleList()
        # The directed graph is all earlier ATAC for A, all ATAC plus earlier
        # RNA for R. Fixed covariates are supplied to every node.
        for u in range(self.ka + self.kr):
            eligible = present[0] if u < self.ka else present[0] & present[1]
            inputs = self._inputs(self.loadings, u)
            with torch.random.fork_rng():
                torch.manual_seed(seed + 100 + u)
                kwargs = dict(prior_atac_kwargs or {}) if u < self.ka else dict(prior_rna_kwargs or {})
                kwargs.setdefault("hidden_dim", hidden_dim)
                prior = ConditionalMixture(prior_atac if u < self.ka else prior_rna, inputs[eligible], **kwargs)
            optimizer = torch.optim.Adam(prior.parameters(), lr=0.01)
            # Initialization from independent estimates only; these are not
            # treated as ground truth or retained as posterior observations.
            target = self.loadings[eligible, u]
            a0 = torch.full_like(target, 1 / 0.15**2)
            for _ in range(pretrain_steps):
                optimizer.zero_grad()
                _, evidence = prior(inputs[eligible]).posterior(a0, a0 * target)
                loss = -evidence.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite prior initialization objective.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 10.0)
                optimizer.step()
            self.priors.append(prior)
            with torch.no_grad():
                # Initialize on the appropriate mixed support. A spike label
                # is assigned only to exact zero; EMDN remains continuous.
                if prior.has_spike:
                    self.loadings[:, u] = torch.where(self.loadings[:, u] >= 0.5, self.loadings[:, u], 0.0)
                mixture = prior(self._inputs(self.loadings, u))
                labels = []
                for z in range(mixture.mean.shape[1]):
                    component = torch.full((len(self.loadings),), z, dtype=torch.long)
                    labels.append(mixture.log_prob(self.loadings[:, u], component))
                self.components[:, u] = torch.stack(labels, 1).argmax(1)
        self.residual = [
            (self.y0[m] - self._modality_loadings(m) @ self.factors[m].T) * self.mask[m] for m in range(2)
        ]
        self.hmm_states = [[None] * self.ka, [None] * self.kr]
        self.hmm_components = [[None] * self.ka, [None] * self.kr]
        self.learning_history: list[list[dict]] = []
        with torch.no_grad():
            self._update_factors()  # Establish valid HMM states/components.

    def _modality_loadings(self, m: int) -> Tensor:
        return self.loadings[:, : self.ka] if m == 0 else self.loadings[:, self.ka :]

    def _loading_location(self, u: int) -> tuple[int, int]:
        return (0, u) if u < self.ka else (1, u - self.ka)

    @torch.no_grad()
    def _update_factors(self):
        for m in range(2):
            loading = self._modality_loadings(m)
            for k in range(loading.shape[1]):
                old = self.factors[m][:, k].clone()
                partial = self.residual[m] + torch.outer(loading[:, k], old) * self.mask[m]
                a = self.tau[m] * (self.mask[m].T @ loading[:, k].square())
                b = self.tau[m] * (partial.T @ loading[:, k])
                value, state, component = sample_hmm(a, b, self.hmm[m][k], self.generator)
                self.factors[m][:, k] = value
                self.hmm_states[m][k], self.hmm_components[m][k] = state, component
                self.residual[m] = partial - torch.outer(loading[:, k], value) * self.mask[m]

    @torch.no_grad()
    def log_joint(self) -> float:
        """Augmented joint density at fixed parameters (not an ELBO)."""
        total = self.loadings.new_zeros(())
        for m in range(2):
            total += -0.5 * (
                self.tau[m] * self.residual[m].square().sum()
                + self.mask[m].sum() * math.log(2 * math.pi / self.tau[m])
            )
        for u, prior in enumerate(self.priors):
            total += prior(self._inputs(self.loadings, u)).log_prob(self.loadings[:, u], self.components[:, u]).sum()
        for m in range(2):
            for k, parameter in enumerate(self.hmm[m]):
                state, component = self.hmm_states[m][k], self.hmm_components[m][k]
                total += parameter["init_prob"][state[0]].log()
                total += parameter["transition"][state[:-1], state[1:]].log().sum()
                total += parameter["mixture_weight"][state, component].log().sum()
                sd, mean = parameter["prior_sd"][component], parameter["mu"][state]
                continuous = sd > 0
                value, sd, mean = self.factors[m][continuous, k], sd[continuous], mean[continuous]
                density = -0.5 * math.log(2 * math.pi) - sd.log() - 0.5 * ((value - mean) / sd).square()
                if parameter.get("effect_support") == "nonpositive":
                    density -= torch.special.log_ndtr(-mean / sd)
                elif parameter["nonnegative_state_means"]:
                    density -= torch.special.log_ndtr(mean / sd)
                total += density.sum()
        return float(total)

    @torch.no_grad()
    def sample(self, *, burnin: int = 100, draws: int = 100, thin: int = 2, progress_every: int = 20):
        """Collect posterior draws after freezing all fitted parameters.

        Reconstruction averages L @ F.T *within each draw*, preserving
        posterior dependence. Product of marginal means is not substituted.
        """
        if burnin < 0 or draws < 2 or thin < 1 or progress_every < 0:
            raise ValueError("Use burnin >= 0, draws >= 2, thin >= 1, progress_every >= 0.")
        self.priors.eval()
        saved_l, saved_z, saved_f = [], [], [[], []]
        mean_reconstruction = [torch.zeros_like(y) for y in self.y0]
        m2_reconstruction = [torch.zeros_like(y) for y in self.y0]
        accepted = torch.zeros(len(self.priors), dtype=self.loadings.dtype)
        trace = []
        total_sweeps = burnin + draws * thin
        for sweep in range(total_sweeps):
            acceptance = self.sweep()
            if sweep >= burnin:
                accepted += acceptance
            if sweep >= burnin and (sweep - burnin + 1) % thin == 0:
                saved_l.append(self.loadings.clone())
                saved_z.append(
                    torch.stack(
                        [
                            self.components[:, u] != 0
                            if prior.has_spike
                            else torch.ones(len(self.loadings), dtype=torch.bool)
                            for u, prior in enumerate(self.priors)
                        ],
                        1,
                    )
                )
                count = len(saved_l)
                for m in range(2):
                    saved_f[m].append(self.factors[m].clone())
                    reconstruction = self._modality_loadings(m) @ self.factors[m].T
                    delta = reconstruction - mean_reconstruction[m]
                    mean_reconstruction[m] += delta / count
                    m2_reconstruction[m] += delta * (reconstruction - mean_reconstruction[m])
                trace.append(self.log_joint())
            if progress_every and (sweep + 1) % progress_every == 0:
                print(f"Joint sweep {sweep + 1}/{total_sweeps}; kept {len(saved_l)} draws", flush=True)
        loadings, active = torch.stack(saved_l), torch.stack(saved_z)
        result = []
        for m, columns in enumerate((slice(None, self.ka), slice(self.ka, None))):
            ls, fs = loadings[:, :, columns], torch.stack(saved_f[m])
            result.append(
                ModalityPosterior(
                    ls.mean(0),
                    ls.square().mean(0),
                    fs.mean(0),
                    fs.square().mean(0),
                    mean_reconstruction[m],
                    (m2_reconstruction[m] / draws).clamp_min(0).sqrt(),
                    ls,
                    fs,
                    active[:, :, columns].double().mean(0),
                )
            )
        return JointPosterior(result[0], result[1], accepted / (draws * thin), trace, self.data.ids)
