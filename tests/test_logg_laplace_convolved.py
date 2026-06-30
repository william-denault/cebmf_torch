"""Regression test for the public Laplace-convolved-with-normal log density.

``logg_laplace_convolved_with_normal`` is the documented single-source closed
form for log( Laplace(0, rate=a) (x) Normal(0, s^2) ). After the consolidation
refactor it delegates to ``_laplace_slab_terms``; this test pins it to an
independent brute-force convolution so the public helper keeps a guard of its
own (and exercises the otherwise-uncalled wrapper).
"""

import math

import numpy as np
import torch

from cebmf_torch.ebnm.point_laplace import (
    _laplace_slab_terms,
    logg_laplace_convolved_with_normal,
)

CPU = torch.device("cpu")


def _t(v):
    return torch.tensor(float(v), dtype=torch.float64, device=CPU)


def _brute_logg_laplace(x, s, a, n=400_001):
    """log integral of (a/2) e^{-a|theta|} * Normal(x-theta; 0, s^2) d theta."""
    span = 40.0 * s + 40.0 / a
    theta = np.linspace(x - span, x + span, n)
    laplace = 0.5 * a * np.exp(-a * np.abs(theta))
    normal = np.exp(-0.5 * ((x - theta) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    return math.log(np.trapezoid(laplace * normal, theta))


CASES = [
    # (x, s, a)
    (0.0, 1.0, 1.0),
    (1.5, 1.0, 2.0),
    (-2.0, 0.5, 1.0),
    (3.0, 2.0, 0.5),
    (-0.3, 1.0, 3.0),
]


def test_logg_laplace_matches_brute_force():
    for x, s, a in CASES:
        got = float(logg_laplace_convolved_with_normal(_t(x), _t(s), _t(a)))
        ref = _brute_logg_laplace(x, s, a)
        assert abs(got - ref) < 1e-3, f"({x},{s},{a}): got {got}, expected {ref}"


def test_logg_laplace_matches_internal_slab_term():
    # The public wrapper must equal the slab log-density from the shared helper.
    for x, s, a in CASES:
        wrapper = logg_laplace_convolved_with_normal(_t(x), _t(s), _t(a))
        lg, _, _ = _laplace_slab_terms(_t(x), _t(s), _t(a))
        assert torch.equal(wrapper, lg)
