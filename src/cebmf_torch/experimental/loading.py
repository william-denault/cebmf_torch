"""Shared neural-prior loading moves for the experimental joint samplers.

The total ordering defines a normalized DAG: each loading uses fixed side
information and all earlier loadings. The full augmented child densities
enter every Metropolis correction. Feature updates belong to each sampler.
"""

import math

import torch
from torch import Tensor

from .conditional import metropolis_normal_means


class ConditionalLoadingSampler:
    """Internal common loading/parameter-update implementation."""

    def _inputs(self, loadings: Tensor, u: int) -> Tensor:
        return torch.cat((self.side_info, loadings[:, :u]), dim=1)

    def child_log_prior(self, u: int, candidate: Tensor) -> Tensor:
        """Augmented prior factors for direct children under a candidate parent.

        All later nodes are direct children in this total-order DAG. Using the
        full density is essential for EMDN children (mean and SD also depend on
        parents). The child's component indicator remains fixed in this move.
        """
        current = self.loadings.clone()
        current[:, u] = candidate
        result = candidate.new_zeros(candidate.shape)
        for child in range(u + 1, len(self.priors)):
            mixture = self.priors[child](self._inputs(current, child))
            result += mixture.log_prob(self.loadings[:, child], self.components[:, child])
        return result

    def loading_log_conditional(self, u: int, value: Tensor, component: Tensor, a: Tensor, b: Tensor) -> Tensor:
        """Unnormalized augmented conditional; parameters and other nodes fixed."""
        own = self.priors[u](self._inputs(self.loadings, u))
        return -0.5 * a * value.square() + b * value + own.log_prob(value, component) + self.child_log_prior(u, value)

    def loading_score(self, u: int, value: Tensor, component: Tensor, a: Tensor, b: Tensor) -> Tensor:
        """Exact within-component score, including all child-prior derivatives.

        This is available from autograd; it does not require score matching.
        Undefined for an atomic component, and not a replacement for sampling
        component probabilities. ReLU networks are differentiable almost
        everywhere; at kinks autograd returns its chosen subgradient.
        """
        with torch.enable_grad():
            own = self.priors[u](self._inputs(self.loadings, u))
            index = torch.arange(len(value), device=value.device)
            if (own.variance[index, component] == 0).any():
                raise ValueError("A continuous score is undefined at a spike component.")
            candidate = value.detach().clone().requires_grad_(True)
            log_probability = self.loading_log_conditional(u, candidate, component, a, b)
            return torch.autograd.grad(log_probability.sum(), candidate)[0].detach()

    @torch.no_grad()
    def _update_loading(self, u: int) -> Tensor:
        m, k = self._loading_location(u)
        old = self.loadings[:, u].clone()
        factor = self.factors[m][:, k]
        partial = self.residual[m] + torch.outer(old, factor) * self.mask[m]
        a = self.tau[m] * (self.mask[m] @ factor.square())
        b = self.tau[m] * (partial @ factor)
        mixture = self.priors[u](self._inputs(self.loadings, u))
        value, component, accept = metropolis_normal_means(
            mixture, a, b, old, self.components[:, u], lambda v: self.child_log_prior(u, v), self.generator
        )
        self.loadings[:, u], self.components[:, u] = value, component
        self.residual[m] = partial - torch.outer(value, factor) * self.mask[m]
        return accept.double().mean()

    @torch.no_grad()
    def sweep(self) -> Tensor:
        acceptance = torch.stack([self._update_loading(u) for u in range(len(self.priors))])
        self._update_factors()
        return acceptance

    def fit_prior_parameters(self, *, rounds: int = 3, sweeps_per_round: int = 10, steps: int = 30, lr: float = 0.003):
        """Approximate generalized MCEM for every supported loading-prior family.

        Optimize the average augmented log prior of joint latent draws, with
        no second noise deconvolution. Retain a parameter step only when it
        improves this fixed-draw objective. Finite MCMC samples do not imply
        evidence monotonicity or convergence of the Monte Carlo E-step.
        HMM/noise parameters, scale grids, and preprocessing remain fixed.
        """
        if rounds < 0 or sweeps_per_round < 1 or steps < 1 or lr <= 0:
            raise ValueError("Use rounds >= 0 and positive sweeps_per_round, steps and lr.")
        for _ in range(rounds):
            draws, labels = [], []
            for _ in range(sweeps_per_round):
                self.sweep()
                draws.append(self.loadings.clone())
                labels.append(self.components.clone())
            values, components = torch.stack(draws), torch.stack(labels)
            round_record = self._fit_from_draws(values, components, steps=steps, lr=lr)
            self.learning_history.append(round_record)
        return self

    def _fit_from_draws(self, values: Tensor, components: Tensor, *, steps: int, lr: float):
        round_record = []
        for u, prior in enumerate(self.priors):
            x = torch.cat([self._inputs(draw, u) for draw in values])
            v, z = values[:, :, u].reshape(-1), components[:, :, u].reshape(-1)

            def objective(prior=prior, x=x, v=v, z=z):
                return -prior(x).log_prob(v, z).mean()

            initial = float(objective().detach())
            best = initial
            best_state = {key: value.detach().clone() for key, value in prior.state_dict().items()}
            optimizer = torch.optim.Adam(prior.parameters(), lr=lr)
            for _ in range(steps):
                optimizer.zero_grad()
                loss = objective()
                if not torch.isfinite(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prior.parameters(), 10.0)
                optimizer.step()
                try:
                    updated = float(objective().detach())
                except FloatingPointError:
                    break
                if math.isfinite(updated) and updated < best:
                    best = updated
                    best_state = {key: value.detach().clone() for key, value in prior.state_dict().items()}
            prior.load_state_dict(best_state)
            round_record.append({"before": initial, "after": best})

        return round_record
