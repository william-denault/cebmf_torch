"""Joint matrix sampler selected by cEBMF's self-covariate flags.

Both axes reuse the existing neural mixtures and child-prior correction.
Noise and non-neural priors are frozen; neural priors have partial MCEM.
"""

import math
from dataclasses import dataclass
from warnings import warn

import torch
from torch import Tensor, nn

from cebmf_torch.ebnm.ash import ash

from .conditional import SCALAR_PRIORS, ConditionalMixture, GaussianMixture, metropolis_normal_means
from .joint import sample_hmm
from .loading import ConditionalLoadingSampler

HMM_PRIORS = ("hmm", "hmm_pos", "hmm_neg")
SUPPORTED_PRIORS = (*SCALAR_PRIORS, "norm", *HMM_PRIORS)
DEFAULT_OPTIONS = dict(
    initialization_iterations=10, pretrain_steps=100, seed=123,
    sweeps_per_round=10, steps=30, lr=0.003,
    burnin=100, draws=100, thin=2, progress_every=20,
)


def joint_options(options):
    options = dict(options or {})
    extra = set(options) - set(DEFAULT_OPTIONS) - {"prior_L_kwargs", "prior_F_kwargs"}
    if extra:
        raise ValueError(f"Unknown joint_kwargs: {sorted(extra)}.")
    result = {**DEFAULT_OPTIONS, **options}
    for key in ("initialization_iterations", "pretrain_steps", "seed", "burnin", "progress_every"):
        if not isinstance(result[key], int) or result[key] < 0:
            raise ValueError(f"joint_kwargs[{key!r}] must be a nonnegative integer.")
    for key in ("sweeps_per_round", "steps", "thin", "draws"):
        if not isinstance(result[key], int) or result[key] < (2 if key == "draws" else 1):
            raise ValueError(f"Invalid positive iteration count joint_kwargs[{key!r}].")
    if not math.isfinite(result["lr"]) or result["lr"] <= 0:
        raise ValueError("joint_kwargs['lr'] must be finite and positive.")
    return result


def distribution_kwargs(raw, overrides):
    allowed = {
        "hidden_dim", "n_layers", "num_components", "scales", "omega", "slab_sd", "min_sd", "max_sd", "learn_scales",
    }
    result = {key: value for key, value in raw.items() if key in allowed}
    if "n_gaussians" in raw:
        result.setdefault("num_components", raw["n_gaussians"])
    result.update(overrides or {})
    return result


@dataclass
class MatrixPosterior:
    L: Tensor
    L2: Tensor
    F: Tensor
    F2: Tensor
    reconstruction: Tensor
    reconstruction_sd: Tensor
    loading_draws: Tensor
    factor_draws: Tensor
    inclusion_probability_L: Tensor
    inclusion_probability_F: Tensor
    acceptance_L: Tensor
    acceptance_F: Tensor
    log_joint: list[float]


class MatrixAxis(ConditionalLoadingSampler):
    def __init__(self, engine, axis, name, side_info, self_cov, kwargs, builder, hmm_parameters):
        self.engine, self.axis, self.name = engine, axis, name
        self.loadings = engine.values[axis]
        self.self_cov = self_cov
        self.side_info = (
            self.loadings.new_empty((len(self.loadings), 0)) if side_info is None else side_info.double().clone()
        )
        if self.side_info.ndim == 1:
            self.side_info = self.side_info[:, None]
        if (self.side_info.ndim != 2 or len(self.side_info) != len(self.loadings)
                or not torch.isfinite(self.side_info).all()):
            raise ValueError("Side information must be finite and have one row per axis entry.")
        self.priors = nn.ModuleList()
        self.fixed, self.hmm, self.states = [], [], []
        self.components = torch.zeros_like(self.loadings, dtype=torch.long)
        self.learning_history = []
        for k in range(engine.rank):
            a, b, _ = engine.statistics(axis, k)
            if name in SCALAR_PRIORS:
                inputs = self._inputs(self.loadings, k)
                with torch.random.fork_rng():
                    torch.manual_seed(engine.options["seed"] + 100 + 1000 * axis + k)
                    prior = ConditionalMixture(name, inputs, **kwargs)
                target = self.loadings[:, k].clone()
                a0 = torch.full_like(target, 1 / 0.15**2)
                optimizer = torch.optim.Adam(prior.parameters(), lr=0.01)
                with torch.enable_grad():
                    for _ in range(engine.options["pretrain_steps"]):
                        optimizer.zero_grad()
                        _, evidence = prior(inputs).posterior(a0, a0 * target)
                        loss = -evidence.mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite neural prior initialization.")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(prior.parameters(), 10.0)
                        optimizer.step()
                self.priors.append(prior)
                if prior.has_spike:
                    self.loadings[:, k] = torch.where(target.abs() >= 0.5, target, 0.0)
                with torch.no_grad():
                    mix = prior(self._inputs(self.loadings, k))
                    scores = [mix.log_prob(self.loadings[:, k], torch.full((len(target),), z, dtype=torch.long))
                              for z in range(mix.mean.shape[1])]
                    self.components[:, k] = torch.stack(scores, 1).argmax(1)
            elif name == "norm":
                observed = a > 0
                if observed.any():
                    fit = ash(b[observed] / a[observed], a[observed].rsqrt(), prior="norm", penalty=1.0)
                    weight, variance = fit.pi / fit.pi.sum(), fit.scale.square()
                else:
                    weight, variance = a.new_ones(1), a.new_ones(1)
                lw = weight.log().expand(len(a), -1).clone()
                self.fixed.append(GaussianMixture(lw, torch.zeros_like(lw), variance.expand_as(lw).clone()))
            else:
                parameter = hmm_parameters[k] if hmm_parameters else None
                if parameter is None:
                    safe_a = a.clamp_min(1e-12)
                    parameter = builder.fit(None, b / safe_a, safe_a.rsqrt(), device=torch.device("cpu")).model_param
                self.hmm.append({key: value.detach().double().clone() if isinstance(value, Tensor) else value
                                 for key, value in parameter.items()})
                self.states.append(None)
            engine.refresh_residual()

    def _inputs(self, loadings, u):
        return torch.cat((self.side_info, loadings[:, :u]), 1) if self.self_cov else self.side_info

    def child_log_prior(self, u, candidate):
        if not self.self_cov or self.name not in SCALAR_PRIORS:
            return torch.zeros_like(candidate)
        return super().child_log_prior(u, candidate)

    @torch.no_grad()
    def update(self, k):
        a, b, partial = self.engine.statistics(self.axis, k)
        if self.name in HMM_PRIORS:
            value, state, label = sample_hmm(a, b, self.hmm[k], self.engine.generator)
            self.states[k] = state
            acceptance = a.new_ones(())
        elif self.name == "norm":
            posterior, _ = self.fixed[k].posterior(a, b)
            value, label = posterior.sample(self.engine.generator)
            acceptance = a.new_ones(())
        else:
            mix = self.priors[k](self._inputs(self.loadings, k))
            value, label, accept = metropolis_normal_means(
                mix, a, b, self.loadings[:, k], self.components[:, k],
                lambda v: self.child_log_prior(k, v), self.engine.generator,
            )
            acceptance = accept.double().mean()
        self.loadings[:, k] = value
        self.components[:, k] = label
        self.engine.set_residual(self.axis, partial - torch.outer(value, self.engine.values[1 - self.axis][:, k]))
        return acceptance

    def active(self):
        if self.name in SCALAR_PRIORS:
            return self.components != 0 if self.name != "emdn" else torch.ones_like(self.components, dtype=torch.bool)
        return self.loadings != 0  # HMM may have atoms away from zero.

    def log_prior(self):
        total = self.loadings.new_zeros(())
        for k in range(self.engine.rank):
            if self.name in SCALAR_PRIORS:
                mixture = self.priors[k](self._inputs(self.loadings, k))
                total += mixture.log_prob(self.loadings[:, k], self.components[:, k]).sum()
            elif self.name == "norm":
                total += self.fixed[k].log_prob(self.loadings[:, k], self.components[:, k]).sum()
            else:
                p, state, z = self.hmm[k], self.states[k], self.components[:, k]
                total += p["init_prob"][state[0]].log() + p["transition"][state[:-1], state[1:]].log().sum()
                total += p["mixture_weight"][state, z].log().sum()
                sd, mu, value = p["prior_sd"][z], p["mu"][state], self.loadings[:, k]
                atom = sd == 0
                if (value[atom] != mu[atom]).any():
                    return total.new_tensor(-torch.inf)
                sd, mu, value = sd[~atom], mu[~atom], value[~atom]
                density = -0.5 * math.log(2 * math.pi) - sd.log() - 0.5 * ((value - mu) / sd).square()
                if p.get("effect_support") == "nonpositive":
                    density -= torch.special.log_ndtr(-mu / sd)
                elif p["nonnegative_state_means"]:
                    density -= torch.special.log_ndtr(mu / sd)
                total += density.sum()
        return total


class JointMatrix:
    """Stateful joint sampler; cEBMF owns initialization and published results."""

    def __init__(self, owner):
        self.options = joint_options(owner.joint_kwargs)
        self.validate_owner(owner)
        self.names = (owner.model.prior_L, owner.model.prior_F)
        self.rank = owner.model.K
        self.y0 = owner.Y0.double().clone()
        self.mask = owner.mask.double().clone()
        self.precision = owner.tau_map.double().expand_as(self.y0).clone()
        self.weight = self.mask * self.precision
        self.values = [owner.L.double().clone(), owner.F.double().clone()]
        self.generator = torch.Generator().manual_seed(self.options["seed"])
        self.refresh_residual()
        self.axes = []
        settings = (
            (owner.covariate.X_l, owner.covariate.self_row_cov, owner._prior_L_kwargs,
             owner.prior_L_fn, owner.model_state_L),
            (owner.covariate.X_f, owner.covariate.self_col_cov, owner._prior_F_kwargs,
             owner.prior_F_fn, owner.model_state_F),
        )
        for axis, (side, self_cov, raw, builder, states) in enumerate(settings):
            if self.names[axis] in SCALAR_PRIORS and any(key in raw for key in ("penalty", "n_epochs", "batch_size")):
                warn("Joint inference uses normalized priors: legacy penalty/n_epochs/batch_size do not govern "
                     "joint updates; configure learning with joint_kwargs.", stacklevel=3)
            kwargs = distribution_kwargs(raw, self.options.get(f"prior_{'L' if axis == 0 else 'F'}_kwargs"))
            self.axes.append(MatrixAxis(self, axis, self.names[axis], side, self_cov, kwargs, builder, states))
        self.learning_history = []
        self.sweep()  # Establish valid norm/HMM labels.

    @staticmethod
    def validate_owner(owner):
        if torch.device(owner.device).type != "cpu":
            raise ValueError("Self-covariate joint inference currently requires device='cpu'.")
        for name in (owner.model.prior_L, owner.model.prior_F):
            if name not in SUPPORTED_PRIORS:
                raise ValueError(f"Joint inference does not yet support prior {name!r}. "
                                 f"Supported: {SUPPORTED_PRIORS}.")
        if not owner.mask.bool().any(0).all() or not owner.mask.bool().any(1).all():
            raise ValueError("Single-matrix joint initialization needs an observation in every row and column.")

    def check_fixed_inputs(self, owner):
        if (owner.model.prior_L, owner.model.prior_F) != self.names or owner.model.K != self.rank:
            raise ValueError("Prior families and rank are fixed during joint inference; start a new fit.")
        if (owner.covariate.self_row_cov, owner.covariate.self_col_cov) != tuple(a.self_cov for a in self.axes):
            raise ValueError("Self-covariate flags are fixed during joint inference; start a new fit.")
        for axis, current in zip(self.axes, (owner.covariate.X_l, owner.covariate.X_f)):
            current = axis.side_info.new_empty((len(axis.side_info), 0)) if current is None else current.double()
            if current.ndim == 1:
                current = current[:, None]
            if current.shape != axis.side_info.shape or not torch.equal(current, axis.side_info):
                raise ValueError("Fixed side information changed during joint inference. Start a new fit; "
                                 "use JointATACRNA for uncertain cross-modality loadings.")

    def refresh_residual(self):
        self.residual = (self.y0 - self.values[0] @ self.values[1].T) * self.mask

    def set_residual(self, axis, residual):
        self.residual = (residual if axis == 0 else residual.T) * self.mask

    def statistics(self, axis, k):
        residual, weight = (self.residual, self.weight) if axis == 0 else (self.residual.T, self.weight.T)
        v, other = self.values[axis][:, k], self.values[1 - axis][:, k]
        partial = residual + torch.outer(v, other)
        return weight @ other.square(), (weight * partial) @ other, partial

    @torch.no_grad()
    def sweep(self):
        return torch.stack([torch.stack([axis.update(k) for k in range(self.rank)]) for axis in self.axes])

    def fit_prior_parameters(self, rounds):
        for _ in range(rounds):
            values, labels = [[], []], [[], []]
            for _ in range(self.options["sweeps_per_round"]):
                self.sweep()
                for m, axis in enumerate(self.axes):
                    values[m].append(axis.loadings.clone())
                    labels[m].append(axis.components.clone())
            record = []
            with torch.enable_grad():
                for m, axis in enumerate(self.axes):
                    record.append(axis._fit_from_draws(torch.stack(values[m]), torch.stack(labels[m]),
                                                       steps=self.options["steps"], lr=self.options["lr"]))
            self.learning_history.append(record)

    @torch.no_grad()
    def log_joint(self):
        likelihood = -0.5 * (self.weight * self.residual.square()
                             + self.mask * (math.log(2 * math.pi) - self.precision.log())).sum()
        return float(likelihood + sum(axis.log_prior() for axis in self.axes))

    @torch.no_grad()
    def sample(self):
        burnin, draws, thin = (self.options[key] for key in ("burnin", "draws", "thin"))
        for axis in self.axes:
            axis.priors.eval()
        values, active, trace = [[], []], [[], []], []
        accepted = self.y0.new_zeros(2, self.rank)
        mean, m2 = torch.zeros_like(self.y0), torch.zeros_like(self.y0)
        count = 0
        total = burnin + draws * thin
        for step in range(total):
            accept = self.sweep()
            if step >= burnin:
                accepted += accept
            if step >= burnin and (step - burnin + 1) % thin == 0:
                count += 1
                for m, axis in enumerate(self.axes):
                    values[m].append(axis.loadings.clone())
                    active[m].append(axis.active())
                signal = self.values[0] @ self.values[1].T
                delta = signal - mean
                mean += delta / count
                m2 += delta * (signal - mean)
                trace.append(self.log_joint())
            every = self.options["progress_every"]
            if every and (step + 1) % every == 0:
                print(f"Joint sweep {step + 1}/{total}; kept {count} draws", flush=True)
        l, f = (torch.stack(v) for v in values)
        acceptance = accepted / (draws * thin)
        return MatrixPosterior(l.mean(0), l.square().mean(0), f.mean(0), f.square().mean(0),
                               mean, (m2 / draws).clamp_min(0).sqrt(), l, f,
                               torch.stack(active[0]).double().mean(0), torch.stack(active[1]).double().mean(0),
                               acceptance[0], acceptance[1], trace)
