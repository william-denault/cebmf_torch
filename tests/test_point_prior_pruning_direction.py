"""Regression test for the H1 fix: PointBuilder.fit must store the
*null* weight in Prior.pi0_null, not the slab weight.

Before the fix, ``priors/point.py:118`` wrote ``pi0_null=obj.pi0`` where
for ``EBNMPointExp`` / ``EBNMLaplaceResult`` / ``EBNMGBResult`` the field
``pi0`` is in fact the *slab* weight (per their own docstrings and the
mixture closure code). This inverted the direction of the cebmf prune
decision (``cebmf.py:482``: prune when ``pi0_null >= prune_thresh``).

After the fix:
- The class field is renamed to ``pi_slab`` (matches the field comments).
- ``priors/point.py`` writes ``pi0_null = 1 - obj.pi_slab`` and
  ``pi_slab = obj.pi_slab``.
- A pure-noise factor (true slab ≈ 0) gets ``pi0_null ≈ 1`` and would be
  pruned at ``prune_thresh = 0.999``.
- A strong-signal factor (true slab ≈ 1) gets ``pi0_null ≈ 0`` and is
  preserved.
"""

import torch

from cebmf_torch.ebnm.generalized_binary import ebnm_gb
from cebmf_torch.priors.base import Prior
from cebmf_torch.priors.point import PointBuilder, PointPriorType


def _simulate_gb(n, pi, mu, omega, s_val, seed):
    """Faithful copy of tests/test_generalized_binary.py::simulate_gb so this
    test does not depend on a sibling test file."""
    g = torch.Generator().manual_seed(seed)
    is_slab = (torch.rand(n, generator=g) < pi).float()
    sigma = omega * mu
    theta = is_slab * (mu + sigma * torch.randn(n, generator=g))
    s = torch.full((n,), s_val)
    x = theta + s * torch.randn(n, generator=g)
    return theta, x, s


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def test_pi_slab_field_is_renamed_on_ebnm_gb():
    """After the rename the EBNM GB result must expose pi_slab and not pi0."""
    _, x, s = _simulate_gb(n=2000, pi=0.3, mu=2.0, omega=0.2, s_val=0.5, seed=10)
    res = ebnm_gb(x, s, omega=0.2)
    assert hasattr(res, "pi_slab"), "EBNMGBResult must expose `pi_slab`"
    assert not hasattr(res, "pi0"), "EBNMGBResult.pi0 must be removed (renamed to pi_slab)"
    assert abs(_to_float(res.pi_slab) - 0.3) < 0.06


def test_point_builder_pi0_null_is_null_for_signal_factor():
    """Strong-signal factor (slab ≈ 0.98). After fit through PointBuilder,
    Prior.pi0_null must be SMALL (~ 0.02), so the factor would NOT be pruned
    at the default prune_thresh = 0.999."""
    _, x, s = _simulate_gb(n=2000, pi=0.98, mu=2.0, omega=0.2, s_val=0.5, seed=20)
    builder = PointBuilder(PointPriorType.GBINARY)
    prior: Prior = builder.fit(X=None, betahat=x, sebetahat=s)
    pi0_null = _to_float(prior.pi0_null)
    pi_slab = _to_float(prior.pi_slab)
    assert pi0_null < 0.2, f"Real-signal factor must have small null weight, got pi0_null={pi0_null}"
    assert pi_slab > 0.8, f"Real-signal factor must have large slab weight, got pi_slab={pi_slab}"
    assert abs(pi0_null + pi_slab - 1.0) < 1e-5
    # Default cebmf prune_thresh
    PRUNE_THRESH = 0.999
    assert not (pi0_null >= PRUNE_THRESH), "Real-signal factor must NOT be pruned"


def test_point_builder_pi0_null_is_null_for_noise_factor():
    """Pure-noise factor (slab ≈ 0.02). After fit through PointBuilder,
    Prior.pi0_null must be LARGE (~ 0.98), so the factor would be pruned
    at any reasonable threshold (e.g. 0.95) — and we explicitly assert it
    crosses 0.85 here, well above the typical 'looks like noise' bar."""
    _, x, s = _simulate_gb(n=2000, pi=0.02, mu=2.0, omega=0.2, s_val=0.5, seed=21)
    builder = PointBuilder(PointPriorType.GBINARY)
    prior: Prior = builder.fit(X=None, betahat=x, sebetahat=s)
    pi0_null = _to_float(prior.pi0_null)
    pi_slab = _to_float(prior.pi_slab)
    assert pi0_null > 0.85, f"Null factor must have large null weight, got pi0_null={pi0_null}"
    assert pi_slab < 0.15
    assert abs(pi0_null + pi_slab - 1.0) < 1e-5


def test_point_builder_pi0_null_consistent_with_ebnm_directly():
    """The Prior.pi_slab returned by the registry path must equal the
    EBNM class's own pi_slab field, exactly. (This catches any future
    drift between the registry's bridging line and the EBNM convention.)"""
    _, x, s = _simulate_gb(n=2000, pi=0.4, mu=2.0, omega=0.2, s_val=0.5, seed=22)
    res = ebnm_gb(x, s, omega=0.2)
    builder = PointBuilder(PointPriorType.GBINARY)
    prior: Prior = builder.fit(X=None, betahat=x, sebetahat=s)
    # Both calls deterministic; if not, this assertion will surface it.
    assert abs(_to_float(prior.pi_slab) - _to_float(res.pi_slab)) < 1e-6
    assert abs(_to_float(prior.pi0_null) - (1.0 - _to_float(res.pi_slab))) < 1e-6
