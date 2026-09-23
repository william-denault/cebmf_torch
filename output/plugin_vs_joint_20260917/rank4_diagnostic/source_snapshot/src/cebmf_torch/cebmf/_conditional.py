"""Variational coordinates for p(L_k | L_<k), including child feedback.

The local objective profiles out q(L_k, Z_k): sum_i log integral exp(Psi_ik).
Gaussian own-prior terms are analytic. Gauss--Hermite quadrature integrates
the one-dimensional child tilt; fixed Sobol points integrate uncertain
parents. Stored quadrature weights and entropy describe the *previous*
coordinate until it is explicitly updated, even when other priors change.

The default quadratic path replaces the child tilt with a cached, concave
Taylor approximation per component. Its moments/entropy are analytic; nodes
remain only for uncertain-parent integration. See _quadratic.py.

These are numerical approximations to the continuous augmented ELBO, not an
MCMC algorithm or a discrete prior on quadrature nodes. A graph with no
latent edges is dispatched to the original cEBNM solvers by the public API.
"""

import inspect
import math
from dataclasses import dataclass
from warnings import warn

import torch
from torch import Tensor

from cebmf_torch.priors.conditional import SCALAR_PRIORS, ConditionalMixture
from cebmf_torch.priors.learned import builder_functions
from cebmf_torch.utils.device import require_finite


def integration_options(options=None):
    # New fits default to quadratic. The get(..., "quadrature") checks below
    # preserve the method of older checkpoints saved without this option.
    result = dict(quadrature_points=24, parent_samples=32, seed=0, approximation="quadratic")
    unknown = set(options or {}) - result.keys()
    if unknown:
        raise ValueError(f"Unknown conditional_kwargs: {sorted(unknown)}. Supported: {sorted(result)}.")
    result.update(options or {})
    if result["approximation"] not in ("quadrature", "quadratic"):
        raise ValueError("conditional_kwargs['approximation'] must be 'quadrature' or 'quadratic'.")
    for key, value in result.items():
        if key == "approximation":
            continue
        minimum = 0 if key == "seed" else 2
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"conditional_kwargs[{key!r}] must be an integer >= {minimum}.")
    return result


def hermite_rule(count, reference):
    """Standard-normal Gaussian quadrature, using torch's symmetric eigensolver."""
    off = torch.arange(1, count, dtype=torch.float64, device=reference.device).sqrt()
    matrix = torch.diag(off, 1) + torch.diag(off, -1)
    nodes, vectors = torch.linalg.eigh(matrix)
    return nodes.to(reference), vectors[0].square().to(reference)


@dataclass
class Coordinate:
    values: Tensor                 # rows x mixture components x quadrature nodes
    weights: Tensor                # joint component/integration weights
    entropy: Tensor                # continuous-and-label entropy, not discrete-node entropy
    gaussian_moments: tuple[Tensor, Tensor, Tensor] | None = None

    @property
    def mean(self):
        if self.gaussian_moments is not None:
            mass, mean, _ = self.gaussian_moments
            return (mass * mean).sum(1)
        return (self.weights * self.values).sum((1, 2))

    @property
    def second(self):
        if self.gaussian_moments is not None:
            mass, mean, variance = self.gaussian_moments
            return (mass * (mean.square() + variance)).sum(1)
        return (self.weights * self.values.square()).sum((1, 2))

    def component_centered_moments(self):
        """Mass, conditional mean and variance, without E[V²] - E[V]²."""
        if self.gaussian_moments is not None:
            return self.gaussian_moments
        mass = self.weights.sum(2)
        normalized = self.weights / mass.clamp_min(torch.finfo(mass.dtype).tiny)[:, :, None]
        mean = (normalized * self.values).sum(2)
        variance = (normalized * (self.values - mean[:, :, None]).square()).sum(2)
        return mass, mean, variance

    def quantiles(self, uniform):
        """Select weighted quadrature nodes in component order using fixed uniforms."""
        weights = self.weights.flatten(1)
        cdf = weights.cumsum(1).contiguous()
        cdf[:, -1].fill_(1)
        u = uniform.expand(len(weights), -1).contiguous()
        index = torch.searchsorted(cdf, u).clamp_max(weights.shape[1] - 1)
        return self.values.flatten(1).gather(1, index)

    def detach(self):
        moments = None if self.gaussian_moments is None else tuple(v.detach() for v in self.gaussian_moments)
        return Coordinate(self.values.detach(), self.weights.detach(), self.entropy.detach(), moments)


def tilted_coordinate(a, b, log_weight, inverse_variance, center, log_height,
                      atoms, rule, child_log_prob):
    """Profile integral, moments and entropy; all arguments can carry gradients.

For slabs, E log g(v,z|P) = log_height - inverse_variance*(v-center)²/2.
For fixed zero atoms it is log_weight. Constants are never discarded.
The centered form is essential: natural-parameter completion of the square
subtracts terms of order center² / variance and fails for narrow slabs.
"""
    alpha = a[:, None] + inverse_variance
    alpha = torch.where(atoms, torch.ones_like(alpha), alpha)
    prior_fraction = inverse_variance / alpha
    mean = torch.where(atoms, 0.0, prior_fraction * center + b[:, None] / alpha)
    variance = torch.where(atoms, 0.0, alpha.reciprocal())
    x, weight = rule
    values = mean[:, :, None] + variance.sqrt()[:, :, None] * x
    log_base = (log_height + prior_fraction * (-0.5 * a[:, None] * center.square() + b[:, None] * center)
                + b[:, None].square() / (2 * alpha) + 0.5 * (math.log(2 * math.pi) - alpha.log()))
    log_base = torch.where(atoms, log_weight, log_base)
    tilt = child_log_prob(values)
    log_mass = log_base[:, :, None] + weight.log() + tilt
    log_z = torch.logsumexp(log_mass.flatten(1), 1)
    weights = (log_mass - log_z[:, None, None]).exp()
    # Evaluate the base density in standardized quadrature coordinates. This
    # also avoids subtracting two almost identical values for very narrow q.
    log_normal = -0.5 * (math.log(2 * math.pi) - alpha.log()[:, :, None] + x.square())
    psi = log_base[:, :, None] + log_normal
    psi = torch.where(atoms[:, :, None], log_weight[:, :, None], psi) + tilt
    entropy = -(weights * (psi - log_z[:, None, None])).sum((1, 2))
    return log_z, Coordinate(values, weights, entropy)


class LoadingGraph:
    """One directed graph across loading columns, with independent sample rows."""

    def __init__(self, models, axis, options, cross=False):
        self.models, self.axis, self.options = models, axis, options
        self.nodes = [(model, k) for model in models for k in range(model.model.K)]
        self.parents, self.children, self.priors, self.training = [], [], [], []
        self.active, self.external, self.q = [], [], []
        self.lookup = {(id(model), k): u for u, (model, k) in enumerate(self.nodes)}
        reference = self.values(models[0])
        self.rule = hermite_rule(options["quadrature_points"], reference)
        self.uniform = torch.quasirandom.SobolEngine(len(self.nodes), scramble=True, seed=options["seed"]).draw(
            options["parent_samples"]
        ).to(reference).T
        for u, (model, k) in enumerate(self.nodes):
            self_cov = model.covariate.self_row_cov if axis == 0 else model.covariate.self_col_cov
            self.parents.append([v for v, (parent_model, _) in enumerate(self.nodes[:u])
                                 if (parent_model is model and self_cov) or (parent_model is not model and cross)])
            self.active.append((model.mask if axis == 0 else model.mask.T).bool().any(1).clone())
            external = model.covariate.X_l if axis == 0 else model.covariate.X_f
            if external is None:
                external = reference.new_empty(len(reference), 0)
            elif external.ndim == 1:
                external = external[:, None]
            if external.ndim != 2 or len(external) != len(reference) or not torch.isfinite(external).all():
                raise ValueError("Fixed covariates must be finite and match their model's rows/columns.")
            self.external.append(external.to(reference).clone())
        self.children = [[v for v, parents in enumerate(self.parents) if u in parents]
                         for u in range(len(self.nodes))]
        # Remove unobserved terminal branches from each row's fitting objective.
        for u in reversed(range(len(self.nodes))):
            for parent in self.parents[u]:
                self.active[parent] |= self.active[u]
        # nonzero has a data-dependent output size and synchronizes CUDA. The
        # observation pattern is fixed: compute these indices only at setup.
        self.indices = [active.nonzero().flatten() for active in self.active]
        self.missing_indices = [(~active).nonzero().flatten() for active in self.active]
        for u, (model, k) in enumerate(self.nodes):
            name = model.model.prior_L if axis == 0 else model.model.prior_F
            if name not in SCALAR_PRIORS:
                raise ValueError(f"A coupled loading axis needs a conditional Gaussian-mixture prior; got {name!r}.")
            raw = model._prior_L_kwargs if axis == 0 else model._prior_F_kwargs
            prior, training = self._make_prior(u, name, raw, model.internal_epoch)
            self.priors.append(prior)
            self.training.append(training)
        self.batches = [index.split(config["batch_size"])
                        for index, config in zip(self.indices, self.training)]
        self.prediction_uniform = None
        if any(len(index) for index in self.missing_indices):
            # SobolEngine is CPU-only. Transfer its fixed table once, not on
            # every call to finish()/predict_missing().
            count = max(256, options["parent_samples"])
            self.prediction_uniform = torch.quasirandom.SobolEngine(
                2 * len(self.nodes), scramble=True, seed=options["seed"] + 1
            ).draw(count).to(reference).clamp(1e-6, 1 - 1e-6)
        # Initialize uncertainty using the actual residual likelihood, never pseudo-data at arbitrary SE.
        for u in range(len(self.nodes)):
            a, b = self.statistics(u)
            inputs = self.reference_inputs(u)[:, None, :]
            coeff = self.coefficients(u, inputs)
            if options.get("approximation", "quadrature") == "quadratic":
                from ._quadratic import QuadraticFeedback, quadratic_coordinate

                zero = torch.zeros_like(coeff[0])[:, :, None]
                feedback = QuadraticFeedback(zero, zero, zero, zero, torch.zeros_like(zero, dtype=torch.bool))
                _, q = quadratic_coordinate(a, b, *coeff, self.rule, feedback)
            else:
                _, q = tilted_coordinate(a, b, *coeff, self.rule, lambda v: torch.zeros_like(v))
            self.q.append(q.detach())
        for u, (model, k) in enumerate(self.nodes):
            self.values(model)[:, k] = self.q[u].mean
            (model.L2 if axis == 0 else model.F2)[:, k] = self.q[u].second
        self.history = []

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Object identities change on deserialization; numerical state does not.
        self.lookup = {(id(model), k): u for u, (model, k) in enumerate(self.nodes)}
        if not hasattr(self, "indices"):
            # Checkpoints created before indices/Sobol predictions were cached.
            self.indices = [active.nonzero().flatten() for active in self.active]
            self.missing_indices = [(~active).nonzero().flatten() for active in self.active]
            self.batches = [index.split(config["batch_size"])
                            for index, config in zip(self.indices, self.training)]
        if not hasattr(self, "prediction_uniform"):
            self.prediction_uniform = None
            if any(len(index) for index in self.missing_indices):
                self.prediction_uniform = torch.quasirandom.SobolEngine(
                    2 * len(self.nodes), scramble=True, seed=self.options["seed"] + 1
                ).draw(max(256, self.options["parent_samples"])).to(self.q[0].values).clamp(1e-6, 1 - 1e-6)

    def values(self, model):
        return model.L if self.axis == 0 else model.F

    def reference_inputs(self, u):
        parts = [self.external[u]]
        parts.extend(self.values(self.nodes[v][0])[:, self.nodes[v][1]:self.nodes[v][1] + 1]
                     for v in self.parents[u])
        return torch.cat(parts, 1)

    def _make_prior(self, u, name, raw, epochs):
        if "omega" in raw and name == "cgb":
            raise ValueError("Plain cgb has no omega parameter; use cgb_sharp for omega-controlled sharpening.")
        defaults = {key: value.default for key, value in inspect.signature(builder_functions[name]).parameters.items()}
        allowed = {"n_epochs", "batch_size", "lr", "penalty", "hidden_dim", "n_layers", "n_gaussians",
                   "num_classes", "omega", "scales", "min_sd", "max_sd", "learn_scales", "slab_sd"}
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f"Unsupported conditional {name} settings: {sorted(extra)}; settings are never ignored.")
        kw = {key: raw.get(key, defaults.get(key, fallback)) for key, fallback in
              (("hidden_dim", 32), ("n_layers", 4), ("omega", 0.01))}
        kw["num_components"] = raw.get("n_gaussians", defaults.get("n_gaussians", 3))
        # Effective slab variances are optimized in the common profile objective.
        # omega initializes sharp families; repeated variance shrinkage is not a variational M-step.
        kw.update(min_sd=raw.get("min_sd", 1e-4), max_sd=raw.get("max_sd", 1e4),
                  learn_scales=raw.get("learn_scales", True))
        value = self.values(self.nodes[u][0])[:, self.nodes[u][1]]
        observed = self.active[u]
        scale = value[observed].square().mean().sqrt().clamp_min(0.1)
        kw["slab_sd"] = raw.get("slab_sd", scale * 0.5)
        if name in ("cash", "lcash", "po_lcash"):
            count = raw.get("num_classes", defaults.get("num_classes", 12))
            kw["scales"] = raw.get("scales", torch.cat((scale.new_zeros(1),
                scale * torch.logspace(-2, 1, count - 1, device=scale.device, dtype=scale.dtype))))
        prior = ConditionalMixture(name, self.reference_inputs(u)[observed], **kw)
        with torch.no_grad():
            if name in ("cgb", "cgb_sharp"):
                v = value[observed]
                prior.net.mu_2.copy_(v[v.abs() >= v.abs().median()].mean())
            elif name in ("emdn", "spiked_emdn"):
                centers = torch.linspace(0.1, 0.9, prior.net.mu.out_features, device=value.device, dtype=value.dtype)
                prior.net.mu.bias.copy_(torch.quantile(value[observed], centers))
        config = dict(n_epochs=raw.get("n_epochs", epochs), batch_size=raw.get("batch_size", 128),
                      lr=raw.get("lr", defaults.get("lr", 0.001)),
                      penalty=raw.get("penalty", defaults.get("penalty", 1.0)))
        if not isinstance(config["n_epochs"], int) or config["n_epochs"] < 0:
            raise ValueError("n_epochs must be a nonnegative integer.")
        if config["batch_size"] is None:
            config["batch_size"] = len(value)
        if not isinstance(config["batch_size"], int) or config["batch_size"] < 1:
            raise ValueError("batch_size must be a positive integer or None.")
        if not math.isfinite(config["lr"]) or config["lr"] <= 0:
            raise ValueError("lr must be positive and finite.")
        if not math.isfinite(config["penalty"]) or config["penalty"] < 1:
            raise ValueError("penalty must be finite and >= 1.")
        if name == "emdn" and config["penalty"] != 1:
            raise ValueError("emdn has no spike; use penalty=1.")
        return prior, config

    def statistics(self, u):
        model, k = self.nodes[u]
        model._recompute_residual()
        other, other2 = (model.F, model.F2) if self.axis == 0 else (model.L, model.L2)
        residual = model.R if self.axis == 0 else model.R.T
        mask = model.mask if self.axis == 0 else model.mask.T
        precision = model.tau if model.tau.ndim == 0 else (model.tau_map if self.axis == 0 else model.tau_map.T)
        weight = mask * precision
        partial = residual + torch.outer(self.values(model)[:, k], other[:, k])
        return weight @ other2[:, k], (weight * partial) @ other[:, k]

    @torch.no_grad()
    def parent_draws(self):
        return [q.quantiles(self.uniform[u]) for u, q in enumerate(self.q)]

    def inputs(self, u, index, draws):
        count = self.options["parent_samples"]
        parts = [self.external[u][index, None, :].expand(-1, count, -1)]
        parts.extend(draws[v][index, :, None] for v in self.parents[u])
        return torch.cat(parts, 2)

    def coefficients(self, u, inputs, weights=None):
        n, draws, dim = inputs.shape
        mix = self.priors[u](inputs.reshape(n * draws, dim))
        shape = (n, draws, -1)
        def average(value):
            return value.mean(1) if weights is None else (value * weights[:, :, None]).sum(1)
        log_weight = average(mix.log_weight.reshape(shape))
        mean, variance = mix.mean.reshape(shape), mix.variance.reshape(shape)
        atom = variance[:, 0] == 0
        inv = torch.where(variance > 0, variance.clamp_min(torch.finfo(variance.dtype).tiny).reciprocal(), 0)
        inverse = average(inv)
        center = average(mean * inv) / torch.where(atom, 1, inverse)
        # Compute the precision-weighted spread directly, never as the
        # difference of two large second moments. exp(E log g) is generally
        # unnormalized; this parent-uncertainty term must be retained.
        height = log_weight - 0.5 * average(torch.where(variance > 0, variance, 1).log()
                     + math.log(2 * math.pi) + (mean - center[:, None]).square() * inv)
        height = torch.where(atom, log_weight, height)
        return log_weight, inverse, center, height, atom

    def integrated_inputs(self, u, index, draws, excluding=None):
        """Use the full stored quadrature when at most one parent needs integration."""
        uncertain = [v for v in self.parents[u] if v != excluding]
        if len(uncertain) > 1:
            return self.inputs(u, index, draws), None
        if uncertain:
            parent = self.q[uncertain[0]]
            values, weights = parent.values[index].flatten(1), parent.weights[index].flatten(1)
        else:
            values = self.external[u].new_zeros(len(index), 1)
            weights = torch.ones_like(values)
        count = values.shape[1]
        parts = [self.external[u][index, None, :].expand(-1, count, -1)]
        parts.extend((torch.zeros_like(values) if v == excluding else values)[:, :, None]
                     for v in self.parents[u])
        return torch.cat(parts, 2), weights

    def child_term(self, u, index, values, draws, prior_states=None):
        n, components, points = values.shape
        candidates = components * points
        result = torch.zeros_like(values)
        for child in self.children[u]:
            context, integration_weight = self.integrated_inputs(child, index, draws, excluding=u)
            samples = context.shape[1]
            context = context.to(values)[:, None].expand(-1, candidates, -1, -1).clone()
            position = self.external[child].shape[1] + self.parents[child].index(u)
            context[:, :, :, position] = values.flatten(1)[:, :, None]
            if prior_states is None:
                mix = self.priors[child](context.flatten(0, 2))
            else:
                mix = torch.func.functional_call(self.priors[child], prior_states[child], (context.flatten(0, 2),))
            shape = (n, candidates, samples, -1)
            log_weight, mean, variance = (v.reshape(shape) for v in (mix.log_weight, mix.mean, mix.variance))
            mass, child_mean, child_var = (v[index, None, None, :].to(values)
                                         for v in self.q[child].component_centered_moments())
            safe = torch.where(variance > 0, variance, 1)
            density = -0.5 * (mass * (math.log(2 * math.pi) + safe.log())
                              + mass * (child_var + (child_mean - mean).square()) / safe)
            density = torch.where(variance > 0, density, 0)
            score = (mass * log_weight + density).sum(3)
            score += (self.training[child]["penalty"] - 1) * log_weight[:, :, :, 0]
            average = score.mean(2) if integration_weight is None else (score * integration_weight[:, None]).sum(2)
            result += average.reshape_as(values) * self.active[child][index, None, None]
        return result

    def quadratic_feedback(self, u, index, draws):
        from ._quadratic import local_feedback

        # A separate anchor per mixture label preserves the spike and separated slabs.
        anchor = self.q[u].component_centered_moments()[1][index, :, None]
        # Evaluate the frozen child networks in double precision without changing
        # their parameters or device. The cached derivatives are small; original
        # training weights and stored posteriors keep the observation dtype.
        states = {child: {key: value.detach().to(dtype=torch.float64)
                          for key, value in self.priors[child].state_dict().items()}
                  for child in self.children[u]}
        return local_feedback(lambda value: self.child_term(u, index, value, draws, states), anchor)

    def profile(self, u, index, a, b, draws, feedback=None):
        context, weight = self.integrated_inputs(u, index, draws)
        coeff = self.coefficients(u, context, weight)
        if self.options.get("approximation", "quadrature") == "quadratic":
            from ._quadratic import quadratic_coordinate

            if feedback is None:
                feedback = self.quadratic_feedback(u, index, draws)
            z, q = quadratic_coordinate(a[index], b[index], *coeff, self.rule, feedback)
        else:
            z, q = tilted_coordinate(a[index], b[index], *coeff, self.rule,
                                    lambda values: self.child_term(u, index, values, draws))
        penalty = (self.training[u]["penalty"] - 1) * coeff[0][:, 0]
        return z + penalty, q

    @torch.no_grad()
    def update(self, u):
        a, b = self.statistics(u)
        draws = self.parent_draws()
        config, prior = self.training[u], self.priors[u]
        batches = self.batches[u]
        quadratic = self.options.get("approximation", "quadrature") == "quadratic"
        # Child q, other parents, and child networks stay fixed within this update.
        # Cache derivatives once per batch, outside all own-prior optimizer epochs.
        feedback = [self.quadratic_feedback(u, batch, draws) if quadratic else None for batch in batches]

        def profile(batch, local):
            return self.profile(u, batch, a, b, draws, feedback=local)

        def objective():
            return sum((profile(batch, local)[0].sum() for batch, local in zip(batches, feedback)),
                       a.new_zeros(()))

        before = best = objective()
        saved = {key: value.clone() for key, value in prior.state_dict().items()}
        # Child network parameters stay fixed, but gradients through candidate inputs are required.
        child_flags = [(p, p.requires_grad) for child in self.children[u]
                       for p in self.priors[child].parameters()]
        for parameter, _ in child_flags:
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam([p for p in prior.parameters() if p.requires_grad], lr=config["lr"],
                                     capturable=a.is_cuda)
        try:
            for _ in range(config["n_epochs"]):
                for batch, local in zip(batches, feedback):
                    with torch.enable_grad():
                        optimizer.zero_grad()
                        loss = -profile(batch, local)[0].mean()
                        require_finite(loss, "Nonfinite conditional profile objective.")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(prior.parameters(), 10.0)
                        optimizer.step()
                current = objective()
                improved = torch.isfinite(current) & (current > best)
                best = torch.where(improved, current, best)
                saved = {key: torch.where(improved, value, saved[key])
                         for key, value in prior.state_dict().items()}
        finally:
            prior.load_state_dict(saved)
            for parameter, flag in child_flags:
                parameter.requires_grad_(flag)
        old = self.q[u]
        values, weights, entropy = old.values.clone(), old.weights.clone(), old.entropy.clone()
        moments = tuple(v.clone() for v in old.component_centered_moments()) if quadratic else None
        for batch, local in zip(batches, feedback):
            _, q = profile(batch, local)
            values[batch], weights[batch], entropy[batch] = q.values, q.weights, q.entropy
            if moments is not None:
                for stored, updated in zip(moments, q.gaussian_moments):
                    stored[batch] = updated
        self.q[u] = Coordinate(values, weights, entropy, moments).detach()
        model, k = self.nodes[u]
        self.values(model)[:, k] = self.q[u].mean
        (model.L2 if self.axis == 0 else model.F2)[:, k] = self.q[u].second
        record = dict(node=u, before=before, after=best)
        if quadratic:
            first = 1 if prior.has_spike else 0  # spike feedback is exact, irrespective of curvature
            record["curvature_clipped"] = sum((local.clipped[:, first:].sum() for local in feedback), a.new_zeros(()))
            record["curvature_evaluations"] = sum(local.clipped[:, first:].numel() for local in feedback)
        self.history.append(record)

    @torch.no_grad()
    def negative_prior_entropy(self):
        """Reevaluate current priors against stored q, including changed parents."""
        draws = self.parent_draws()
        result = self.q[0].values.new_zeros(())
        for u, q in enumerate(self.q):
            for index in self.batches[u]:
                context, weight = self.integrated_inputs(u, index, draws)
                logw, inverse, center, height, _ = self.coefficients(u, context, weight)
                mass, mean, variance = (v[index] for v in q.component_centered_moments())
                expected_log_prior = (mass * (height - 0.5 * inverse *
                                              (variance + (mean - center).square()))).sum(1)
                penalty = (self.training[u]["penalty"] - 1) * logw[:, 0]
                result -= (expected_log_prior + q.entropy[index] + penalty).sum()
        return result

    @torch.no_grad()
    def predict_missing(self):
        """Ancestral posterior prediction for integrated-out terminal branches."""
        uniform = self.prediction_uniform
        if uniform is None:
            return
        count = len(uniform)
        draws = []
        for u, (model, k) in enumerate(self.nodes):
            samples = self.q[u].quantiles(uniform[:, 2 * u])
            missing = self.missing_indices[u]
            if len(missing):
                parts = [self.external[u][missing, None, :].expand(-1, count, -1)]
                parts.extend(draws[v][missing, :, None] for v in self.parents[u])
                context = torch.cat(parts, 2)
                n, _, dim = context.shape
                mix = self.priors[u](context.reshape(n * count, dim))
                weights = mix.log_weight.exp()
                means = (weights * mix.mean).sum(1).reshape(n, count).mean(1)
                second = (weights * (mix.variance + mix.mean.square())).sum(1).reshape(n, count).mean(1)
                self.values(model)[missing, k] = means
                (model.L2 if self.axis == 0 else model.F2)[missing, k] = second
                cdf = weights.cumsum(1).contiguous()
                cdf[:, -1].fill_(1)
                choice = torch.searchsorted(cdf, uniform[:, 2 * u].repeat(n)[:, None].contiguous())
                mu = mix.mean.gather(1, choice).flatten()
                sd = mix.variance.gather(1, choice).flatten().sqrt()
                normal = math.sqrt(2) * torch.erfinv(2 * uniform[:, 2 * u + 1].repeat(n) - 1)
                samples[missing] = (mu + sd * normal).reshape(n, count)
            draws.append(samples)


class ConditionalFit:
    """Coordinate schedule shared by a single cEBMF model and coupled views."""

    def __init__(self, models, options=None, cross=False):
        self.models = list(models)
        self.options = integration_options(options)
        self.ranks = [model.model.K for model in models]
        self.row = LoadingGraph(models, 0, self.options, cross=cross) if cross or any(
            m.covariate.self_row_cov and m.model.K > 1 and m.model.prior_L in SCALAR_PRIORS for m in models
        ) else None
        self.columns = {id(m): LoadingGraph([m], 1, self.options) for m in models
                        if m.covariate.self_col_cov and m.model.K > 1 and m.model.prior_F in SCALAR_PRIORS}
        self.history = []
        # Warn at graph creation, not during sweeps or checkpoint continuation.
        graphs = [self.row, *self.columns.values()]
        if self.options["approximation"] == "quadratic" and any(
            any(graph.children) for graph in graphs if graph is not None
        ):
            warn(
                "Conditional fitting uses a local quadratic approximation to child feedback "
                "for faster processing. Results can differ from quadrature. To use the "
                "quadrature method instead, set conditional_kwargs={'approximation': 'quadrature'}.",
                UserWarning, stacklevel=3,
            )

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.columns = {id(graph.models[0]): graph for graph in self.columns.values()}

    @torch.no_grad()
    def step(self):
        if [m.model.K for m in self.models] != self.ranks:
            raise ValueError("The conditional graph's rank changed; initialize a new fit.")
        for model in self.models:
            tau_map = None if model.tau.ndim == 0 else model.tau_map
            for k in range(model.model.K):
                if self.row is None:
                    model._update_L_factor(k, model._partial_residual_masked(k), tau_map, 1e-12)
                else:
                    self.row.update(self.row.lookup[id(model), k])
                if id(model) in self.columns:
                    self.columns[id(model)].update(k)
                else:
                    model._update_F_factor(k, model._partial_residual_masked(k), tau_map, 1e-12)
                model._recompute_residual()
            model.update_tau()
        loss = self.objective()
        self.history.append(loss)
        reported = False
        for model in self.models:
            model.obj.append(loss)
            model._report_sweep(joint=len(self.models) > 1, display=not reported)
            reported = reported or getattr(model, "verbose", True)

    @torch.no_grad()
    def objective(self):
        loss = self.models[0].Y.new_zeros(())
        for model in self.models:
            residual = model._expected_residuals_squared()
            ll = model._compute_constant_loglik(residual) if model.tau.ndim == 0 else model._compute_elementwise_loglik(residual)
            loss -= ll
            if self.row is None:
                loss += model.kl_l.sum() - model.regularization_L.sum()
            if id(model) not in self.columns:
                loss += model.kl_f.sum() - model.regularization_F.sum()
        if self.row is not None:
            loss += self.row.negative_prior_entropy()
        loss += sum((graph.negative_prior_entropy() for graph in self.columns.values()), loss.new_zeros(()))
        return loss

    def finish(self):
        if self.row is not None:
            self.row.predict_missing()
        for graph in self.columns.values():
            graph.predict_missing()
