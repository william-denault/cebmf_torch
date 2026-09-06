"""ASH component selection is explicit and shared by LC-ASH and PO-LC-ASH."""

import importlib

import pytest
import torch

from cebmf_torch.cebnm.lcash import lcash_posterior_means, po_lcash_posterior_means
from cebmf_torch.ebnm.ash import ash

ash_module = importlib.import_module("cebmf_torch.ebnm.ash")
SOLVERS = [lcash_posterior_means, po_lcash_posterior_means]


@pytest.fixture
def controlled_ash(monkeypatch):
    scales = torch.tensor([0.0, 0.5, 2.0])
    weights = torch.tensor([0.6, 0.4 - 5e-7, 5e-7])
    monkeypatch.setattr(ash_module, "autoselect_scales_mix_norm", lambda *args, **kwargs: scales)

    def fitted_weights(logL, penalty, zero_threshold=1e-6):
        # Exercise public ash/config forwarding as well as grid selection.
        assert zero_threshold == 0.0
        return weights.clone()

    monkeypatch.setattr(ash_module, "optimize_pi_logL_lbfgs", fitted_weights)
    return scales, weights


def fit(solver, threshold):
    x = torch.linspace(-2, 2, 16)
    return solver(x[:, None], x, torch.ones_like(x), ash_threshold=threshold, n_epochs=0, verbose=False)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("threshold,expected_K", [(1e-6, 2), (1e-8, 3), (0.0, 3)])
def test_threshold_controls_selection_without_preceding_cutoff(solver, threshold, expected_K, controlled_ash):
    scales, _ = controlled_ash
    result = fit(solver, threshold)
    torch.testing.assert_close(result.scale, scales[:expected_K], rtol=0, atol=0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_weight_equal_to_threshold_is_removed(solver, controlled_ash):
    _, weights = controlled_ash
    # Pass the actual represented weight to test the documented boundary.
    result = fit(solver, float(weights[-1].log().exp()))
    assert result.scale.numel() == 2


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "weights,threshold,count",
    [([0.4, 0.3, 0.3], 0.5, 0), ([1.0, 5e-9, 5e-9], 1e-6, 1), ([5e-9, 1.0, 5e-9], 1e-6, 1)],
)
def test_insufficient_components_raise_without_fallback(solver, weights, threshold, count, controlled_ash):
    _, fitted_weights = controlled_ash
    fitted_weights.copy_(torch.tensor(weights))
    with pytest.raises(ValueError, match=f"retained {count} components") as error:
        fit(solver, threshold)
    message = str(error.value)
    assert "ash_threshold=" in message
    assert "retained scales:" in message
    assert "largest discarded weight:" in message
    assert "consider lowering ash_threshold" in message


@pytest.mark.parametrize("solver", SOLVERS)
def test_two_slabs_without_spike_raise(solver, controlled_ash):
    _, weights = controlled_ash
    weights.copy_(torch.tensor([1e-8, 0.5, 0.5]))
    with pytest.raises(ValueError, match="spike at zero") as error:
        fit(solver, 1e-6)
    assert "spike weight:" in str(error.value)


def test_standalone_ash_lbfgs_default_cutoff_is_preserved():
    x = torch.linspace(-4, 4, 64)
    se = torch.ones_like(x)
    default = ash(x, se, optimizer="lbfgs")
    explicit = ash(x, se, optimizer="lbfgs", zero_threshold=1e-6)
    uncut = ash(x, se, optimizer="lbfgs", zero_threshold=0.0)
    torch.testing.assert_close(default.pi, explicit.pi, rtol=0, atol=0)
    assert (uncut.pi > default.pi).any()
    expected = uncut.pi.clone()
    expected[expected < 1e-6] = 0
    expected /= expected.sum()
    torch.testing.assert_close(default.pi, expected.clamp_min(1e-32), rtol=2e-6, atol=1e-30)
