"""Analytic Gaussian-component updates for frozen local child-feedback fits.

Only the child feedback is approximated here. Expected own-prior coefficients
still average over uncertain parents in _conditional.py. Each component uses
its previous posterior mean as the expansion point; positive curvature is
discarded to make the local feedback concave and the posterior proper.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from cebmf_torch.utils.device import require_finite


@dataclass
class QuadraticFeedback:
    anchor: Tensor
    height: Tensor
    slope: Tensor
    curvature: Tensor
    clipped: Tensor


def local_feedback(function, anchor):
    """Per-entry Taylor derivatives; function must not mix candidate entries.

    Different candidates/rows are independent. Summing outputs therefore gives
    their individual first derivatives, and summing those gives the diagonal
    second derivatives. Derivatives and anchors are frozen for an inner fit:
    there is no third-order differentiation through the child networks. The
    function receives float64 inputs: narrow feedback can otherwise lose its
    component normalizer to cancellation.
    """
    with torch.enable_grad():
        value = anchor.detach().to(dtype=torch.float64).requires_grad_(True)
        height = function(value)
        slope = None
        if height.requires_grad:
            slope = torch.autograd.grad(height.sum(), value, create_graph=True, allow_unused=True)[0]
        if slope is None:
            slope = torch.zeros_like(value)
        curvature = None
        if slope.requires_grad:
            curvature = torch.autograd.grad(slope.sum(), value, allow_unused=True)[0]
        if curvature is None:
            curvature = torch.zeros_like(value)
    for tensor in (height, slope, curvature):
        require_finite(tensor, "Nonfinite local feedback derivative; check prior scales and inputs.")
    return QuadraticFeedback(value.detach(), height.detach(), slope.detach(),
                             curvature.detach().clamp_max(0), curvature.detach() > 0)


def quadratic_coordinate(a, b, log_weight, inverse_variance, center, log_height,
                         atoms, rule, feedback):
    """Profile a concave quadratic tilt analytically, including mixture masses.

    exp(Psi_c(v)) = exp(B_c) N(v; m_c, 1/alpha_c) exp(h_c(v)),
    h_c(v) = h0 + g*(v-t) + H*(v-t)^2/2, H <= 0.
    The spike uses h(0) exactly. Hermite nodes are stored only for subsequent
    uncertain-parent integration, not to calculate these moments or entropy.
    """
    from ._conditional import Coordinate

    dtype = a.dtype
    # These small per-component calculations stay on-device. Float64 is needed
    # when h(anchor) and the integrated quadratic correction nearly cancel.
    a, b, log_weight, inverse_variance, center, log_height = (
        v.to(dtype=torch.float64) for v in (a, b, log_weight, inverse_variance, center, log_height))
    alpha = torch.where(atoms, 1.0, a[:, None] + inverse_variance)
    fraction = inverse_variance / alpha
    mean = torch.where(atoms, 0.0, fraction * center + b[:, None] / alpha)
    log_base = (log_height + fraction * (-0.5 * a[:, None] * center.square() + b[:, None] * center)
                + b[:, None].square() / (2 * alpha) + 0.5 * (math.log(2 * math.pi) - alpha.log()))
    log_base = torch.where(atoms, log_weight, log_base)
    anchor, height, slope, curvature = (v.squeeze(-1) for v in
        (feedback.anchor, feedback.height, feedback.slope, feedback.curvature))
    precision = alpha - curvature
    ratio = alpha / precision
    delta = mean - anchor
    # Centered completion avoids subtracting two large natural-parameter squares.
    mean = ratio * mean + (-curvature / precision) * anchor + slope / precision
    mean = torch.where(atoms, 0.0, mean)
    variance = torch.where(atoms, 0.0, precision.reciprocal())
    correction = (height + ratio * (slope * delta + 0.5 * curvature * delta.square())
                  + 0.5 * slope.square() / precision - 0.5 * torch.log1p(-curvature / alpha))
    log_mass = log_base + torch.where(atoms, height, correction)
    log_z = torch.logsumexp(log_mass, 1)
    log_probability = log_mass - log_z[:, None]
    mass = log_probability.exp()
    entropy = -(mass * log_probability).sum(1) + (
        mass * torch.where(atoms, 0.0, 0.5 * (math.log(2 * math.pi * math.e) - precision.log()))).sum(1)
    nodes, weights = rule
    weights = weights / weights.sum()
    values = mean[:, :, None] + variance.sqrt()[:, :, None] * nodes
    q = Coordinate(values.to(dtype), (mass[:, :, None] * weights).to(dtype), entropy.to(dtype),
                   gaussian_moments=(mass.to(dtype), mean.to(dtype), variance.to(dtype)))
    return log_z.to(dtype), q
