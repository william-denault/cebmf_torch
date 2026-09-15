"""Normalized conditional priors and exact normal-means proposals.

Uses the networks from the learned registry, with explicit, persistent scale
parameters and fixed input standardization. No EBNM fitting occurs inside an
MCMC transition. Mixture labels are part of the augmented state; a spike is an
atom, never a narrow Gaussian or a continuous density evaluated at zero.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cebmf_torch.cebnm.cash_solver import CashNet
from cebmf_torch.cebnm.cov_gb_prior import CgbNet
from cebmf_torch.cebnm.cov_sharp_2gb_prior import Cgb2Net
from cebmf_torch.cebnm.emdn import MDN as EmdnNet
from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet
from cebmf_torch.cebnm.spiked_emdn import MDN as SpikedEmdnNet

SCALAR_PRIORS = ("cgb", "cgb_sharp", "cgb_sharp_2", "cash", "lcash", "po_lcash", "emdn", "spiked_emdn")


@dataclass
class GaussianMixture:
    log_weight: Tensor  # N x C, normalized
    mean: Tensor  # N x C
    variance: Tensor  # N x C; exact zero means an atom

    def posterior(self, a: Tensor, b: Tensor):
        """Normal-means mixture and log integral of exp(-a*v²/2+b*v).

        The formula avoids subtracting mu² / variance, works at zero variance
        and zero precision, and includes nonzero atoms without a special case.
        """
        if a.shape != (len(self.mean),) or b.shape != a.shape:
            raise ValueError("a and b must be vectors with one entry per row.")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a < 0).any():
            raise ValueError("Require finite statistics and nonnegative precision.")
        if (b[a == 0] != 0).any():
            raise ValueError("A missing likelihood requires a=b=0.")
        a, b = a[:, None], b[:, None]
        denominator = 1 + a * self.variance
        variance = self.variance / denominator
        mean = (self.mean + b * self.variance) / denominator
        log_integral = (
            -0.5 * torch.log1p(a * self.variance)
            + (-0.5 * a * self.mean.square() + b * self.mean + 0.5 * b.square() * self.variance) / denominator
        )
        logits = self.log_weight + log_integral
        evidence = torch.logsumexp(logits, 1)
        return GaussianMixture(logits - evidence[:, None], mean, variance), evidence

    def sample(self, generator: torch.Generator):
        component = torch.multinomial(self.log_weight.exp(), 1, generator=generator).flatten()
        index = torch.arange(len(component), device=component.device)
        mean, variance = self.mean[index, component], self.variance[index, component]
        value = mean + variance.sqrt() * torch.randn(
            len(mean), dtype=mean.dtype, device=mean.device, generator=generator
        )
        return value, component

    def log_prob(self, value: Tensor, component: Tensor):
        """Augmented log mass/density p(z,v), on label-specific support."""
        index = torch.arange(len(value), device=value.device)
        variance = self.variance[index, component]
        mean = self.mean[index, component]
        atom = variance == 0
        safe = torch.where(atom, 1.0, variance)
        density = -0.5 * (math.log(2 * math.pi) + safe.log() + (value - mean).square() / safe)
        density = torch.where(atom, torch.where(value == mean, 0.0, -torch.inf), density)
        return self.log_weight[index, component] + density

    def marginal_log_prob(self, value: Tensor):
        """Density with respect to Lebesgue measure plus the fixed atoms.

        At an atom only atomic weights contribute, not Gaussian densities.
        Used for initialization/verification, not augmented child corrections.
        """
        atom = self.variance == 0
        at_atom = atom & (value[:, None] == self.mean)
        safe = torch.where(atom, 1.0, self.variance)
        density = -0.5 * (math.log(2 * math.pi) + safe.log() + (value[:, None] - self.mean).square() / safe)
        continuous = (self.log_weight + density).masked_fill(atom, -torch.inf)
        atomic = self.log_weight.masked_fill(~at_atom, -torch.inf)
        return torch.where(at_atom.any(1), torch.logsumexp(atomic, 1), torch.logsumexp(continuous, 1))


class ConditionalMixture(nn.Module):
    """Learned-registry families with a common augmented distribution API.

    CASH/LCASH use fixed zero-centered scale grids; CGB has one global slab;
    CGB_SHARP_2 has global means constrained by sign (untruncated normals).
    EMDN and spiked EMDN have covariate-dependent means and scales as well as
    weights. A small explicit SD floor defines the numerical admissible family.

    Sharp variants fix their narrow slab scales by default. Repeatedly
    multiplying an ML variance estimate by omega is not an EM maximization.
    Setting learn_scales=True instead learns their actual effective variance;
    then omega only affects initialization, not a separate identifiable scale.
    """

    def __init__(
        self,
        name: str,
        reference_inputs: Tensor,
        *,
        hidden_dim: int = 16,
        n_layers: int = 1,
        num_components: int = 3,
        scales=None,
        omega: float = 0.01,
        slab_sd: float = math.sqrt(0.05),
        min_sd: float = 0.02,
        max_sd: float = 20.0,
        learn_scales: bool | None = None,
    ):
        super().__init__()
        if name not in SCALAR_PRIORS:
            raise ValueError(f"Choose a scalar learned prior from {SCALAR_PRIORS}; got {name!r}.")
        if reference_inputs.ndim != 2 or len(reference_inputs) == 0 or not torch.isfinite(reference_inputs).all():
            raise ValueError("reference_inputs must be a nonempty finite matrix.")
        if not 0 < min_sd < max_sd or not 0 < omega <= 1 or not math.isfinite(slab_sd) or slab_sd <= 0:
            raise ValueError("Require 0 < min_sd < max_sd, 0 < omega <= 1, and positive finite slab_sd.")
        if hidden_dim < 1 or n_layers < 0 or num_components < 2:
            raise ValueError("Require hidden_dim >= 1, n_layers >= 0, num_components >= 2.")
        self.name, self.min_sd, self.max_sd = name, min_sd, max_sd
        self.input_dim = reference_inputs.shape[1]
        x = reference_inputs if self.input_dim else reference_inputs.new_ones((len(reference_inputs), 1))
        self.register_buffer("offset", x.mean(0))
        scale = x.std(0, unbiased=False)
        self.register_buffer("input_scale", torch.where(scale > 1e-8, scale, 1.0))
        d = x.shape[1]
        sharp = name in ("cgb_sharp", "cgb_sharp_2")
        self.has_spike = name != "emdn"
        if name in ("cgb", "cgb_sharp", "cgb_sharp_2"):
            self.net = (Cgb2Net if name == "cgb_sharp_2" else CgbNet)(d, hidden_dim, n_layers)
            count = 2 if name == "cgb_sharp_2" else 1
            sd = max(min_sd, slab_sd * (math.sqrt(omega) if sharp else 1))
            learn_scales = not sharp if learn_scales is None else learn_scales
            self.log_slab_sd = nn.Parameter(x.new_full((count,), math.log(sd)), requires_grad=learn_scales)
            if count == 1:
                with torch.no_grad():
                    self.net.mu_2.fill_(1)
        elif name in ("cash", "lcash", "po_lcash"):
            grid = x.new_tensor([0.0, 0.25, 1.0, 2.0]) if scales is None else torch.as_tensor(scales).to(x)
            if (
                grid.ndim != 1
                or len(grid) < 2
                or grid[0] != 0
                or (grid[1:] <= 0).any()
                or not torch.isfinite(grid).all()
            ):
                raise ValueError("The fixed grid must start at zero followed by positive finite SDs.")
            if (grid[1:] <= grid[:-1]).any():
                raise ValueError("The fixed scale grid must be strictly increasing.")
            self.register_buffer("grid", grid)
            if name == "cash":
                self.net = CashNet(d, hidden_dim, len(grid), n_layers)
            else:
                self.net = (LcashNet if name == "lcash" else PropOddsLcashNet)(d, len(grid))
        else:
            self.net = (EmdnNet if name == "emdn" else SpikedEmdnNet)(d, hidden_dim, num_components, n_layers)
            with torch.no_grad():
                self.net.mu.weight.zero_()
                self.net.mu.bias.copy_(torch.linspace(0, 1, self.net.mu.out_features))
                self.net.log_sigma.weight.zero_()
                bias = math.log(slab_sd) if name == "emdn" else math.log(math.expm1(slab_sd))
                self.net.log_sigma.bias.fill_(bias)
        self.to(dtype=x.dtype, device=x.device)

    def standardize(self, inputs: Tensor):
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} fixed-plus-parent covariates.")
        if not torch.isfinite(inputs).all():
            raise ValueError("Latent parents and fixed covariates must be finite; missing observations use a mask.")
        if not self.input_dim:
            inputs = inputs.new_ones((len(inputs), 1))
        return (inputs - self.offset) / self.input_scale

    def forward(self, inputs: Tensor) -> GaussianMixture:
        x = self.standardize(inputs)
        n = len(x)
        net = self.net
        if self.name in ("emdn", "spiked_emdn"):
            h = F.relu(net.fc_in(x))
            for layer in net.hidden_layers:
                h = F.relu(layer(h))
            log_weight = F.log_softmax(net.pi(h), 1)
            mean = net.mu(h)
            if self.name == "emdn":
                log_sd = net.log_sigma(h).clamp(math.log(self.min_sd), math.log(self.max_sd))
                variance = (2 * log_sd).exp()
            else:
                sd = (F.softplus(net.log_sigma(h)) + 1e-6).clamp(self.min_sd, self.max_sd)
                variance = sd.square()
                mean = torch.cat((x.new_zeros(n, 1), mean), 1)
                variance = torch.cat((x.new_zeros(n, 1), variance), 1)
        elif self.name == "lcash":
            log_weight = F.log_softmax(net.linear(x), 1)
            mean = torch.zeros_like(log_weight)
            variance = self.grid.square().expand_as(mean)
        elif self.name == "po_lcash":
            # Stable ordered-logistic probabilities, without CDF subtraction
            # or flooring tiny category weights (which changes the model).
            t = net._get_cutpoints()[None, :] - (x @ net.w)[:, None]
            log_weight = torch.cat(
                (
                    F.logsigmoid(t[:, :1]),
                    F.logsigmoid(t[:, 1:]) + F.logsigmoid(-t[:, :-1]) + torch.log(-torch.expm1(t[:, :-1] - t[:, 1:])),
                    F.logsigmoid(-t[:, -1:]),
                ),
                1,
            )
            log_weight = log_weight - torch.logsumexp(log_weight, 1, keepdim=True)
            mean = torch.zeros_like(log_weight)
            variance = self.grid.square().expand_as(mean)
        else:
            h = F.relu(net.input_layer(x))
            for layer in net.hidden_layers:
                h = F.relu(layer(h))
            logits = net.output_layer(h)
            if self.name in ("cgb", "cgb_sharp"):
                log_weight = torch.cat((F.logsigmoid(-logits), F.logsigmoid(logits)), 1)
                mean = torch.stack((net.mu_2 * 0, net.mu_2)).expand(n, -1)
            elif self.name == "cgb_sharp_2":
                log_weight = F.log_softmax(logits, 1)
                mean = torch.stack((net.raw_mu_pos * 0, F.softplus(net.raw_mu_pos), -F.softplus(net.raw_mu_neg)))
                mean = mean.expand(n, -1)
            else:  # CASH
                log_weight = F.log_softmax(logits, 1)
                mean = torch.zeros_like(log_weight)
            if self.name == "cash":
                variance = self.grid.square().expand_as(mean)
            else:
                log_sd = self.log_slab_sd.clamp(math.log(self.min_sd), math.log(self.max_sd))
                variance = torch.cat((x.new_zeros(1), (2 * log_sd).exp())).expand_as(mean)
        if (
            not torch.isfinite(log_weight).all()
            or not torch.isfinite(mean).all()
            or not torch.isfinite(variance).all()
        ):
            raise FloatingPointError("Nonfinite conditional mixture; reduce learning rate or check input scaling.")
        return GaussianMixture(log_weight, mean, variance)


@torch.no_grad()
def metropolis_normal_means(
    prior: GaussianMixture,
    a: Tensor,
    b: Tensor,
    old: Tensor,
    component: Tensor,
    child_log_prob,
    generator: torch.Generator,
):
    """One augmented Gibbs-within-MH update, simultaneously across independent rows.

    child_log_prob(v) must evaluate every *direct child's* augmented prior
    under the candidate parent value, including its mean/variance dependence.
    Global hyperparameters and all other latent coordinates stay fixed.
    """
    posterior, _ = prior.posterior(a, b)
    proposal, proposed_component = posterior.sample(generator)
    correction = child_log_prob(proposal) - child_log_prob(old)
    if torch.isnan(correction).any():
        raise FloatingPointError("Undefined child-prior ratio; check current state support.")
    accept = torch.rand(len(a), generator=generator, dtype=a.dtype, device=a.device).log() < correction.clamp_max(0)
    return torch.where(accept, proposal, old), torch.where(accept, proposed_component, component), accept
