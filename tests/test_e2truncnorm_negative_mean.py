"""Regression tests for my_e2truncnorm second moments with negative / non-zero means.

The previous implementation took ``mean = mean.abs()`` before reconstructing
E[X^2] = mean^2 + 2*mean*sd*E[Z] + sd^2*E[Z^2]. Because the internal ``flip``
only fires when both standardized bounds are positive, the cross term
``2*mean*sd*E[Z]`` kept the wrong sign for left-truncated / negative-mean
intervals. That made the second moment wrong (and sign-asymmetric) and could
drive E[X^2] below E[X]^2, collapsing posterior standard deviations to zero.

These tests use a dependency-free numpy quadrature ground truth plus the exact
reflection-symmetry invariant E[X^2 | a<X<b, mean] == E[X^2 | -b<X<-a, -mean].
"""

import math

import numpy as np
import torch

from cebmf_torch.utils.maths import my_e2truncnorm

CPU = torch.device("cpu")


def _t(v):
    return torch.tensor(float(v), dtype=torch.float64, device=CPU)


def _brute_e2(a, b, mean, sd, n=200_001):
    """E[X^2 | a<X<b] for X~N(mean, sd^2) via fine-grid trapezoidal quadrature."""
    lo = mean - 12.0 * sd if math.isinf(a) else a
    hi = mean + 12.0 * sd if math.isinf(b) else b
    x = np.linspace(lo, hi, n)
    pdf = np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
    z = np.trapezoid(pdf, x)
    m2 = np.trapezoid(x * x * pdf, x)
    return m2 / z


CASES = [
    # (a, b, mean, sd)
    (-np.inf, 0.0, -0.5, 1.0),
    (-np.inf, 0.0, -2.0, 1.0),
    (-np.inf, 2.0, -1.0, 1.0),
    (0.0, np.inf, 0.5, 1.0),
    (-3.0, 3.0, -1.0, 1.0),
    (-np.inf, 0.0, 0.5, 1.0),
    (0.0, np.inf, -1.9, 1.0),
    (1.0, 3.0, 0.5, 1.0),
    (0.0, np.inf, -2.0, 0.5),
    (-2.0, 5.0, 2.0, 1.5),
]


def test_e2truncnorm_matches_quadrature():
    for a, b, mean, sd in CASES:
        got = float(my_e2truncnorm(_t(a), _t(b), _t(mean), _t(sd)))
        ref = _brute_e2(a, b, mean, sd)
        assert abs(got - ref) < 1e-3, f"({a},{b},{mean},{sd}): got {got}, expected {ref}"


def test_e2truncnorm_reflection_symmetry():
    # E[X^2 | a<X<b, N(mean,sd^2)] == E[X^2 | -b<X<-a, N(-mean,sd^2)]
    for a, b, mean, sd in CASES:
        left = float(my_e2truncnorm(_t(a), _t(b), _t(mean), _t(sd)))
        right = float(my_e2truncnorm(_t(-b), _t(-a), _t(-mean), _t(sd)))
        assert abs(left - right) < 1e-6, f"asymmetry for ({a},{b},{mean},{sd}): {left} vs {right}"


def test_e2truncnorm_never_below_first_moment_squared():
    # E[X^2] >= E[X]^2 must always hold; the bug violated this for negative means.
    from cebmf_torch.utils.maths import my_etruncnorm

    for a, b, mean, sd in CASES:
        e1 = float(my_etruncnorm(_t(a), _t(b), _t(mean), _t(sd)))
        e2 = float(my_e2truncnorm(_t(a), _t(b), _t(mean), _t(sd)))
        assert e2 >= e1 * e1 - 1e-9, f"E[X^2]={e2} < E[X]^2={e1 * e1} for ({a},{b},{mean},{sd})"
