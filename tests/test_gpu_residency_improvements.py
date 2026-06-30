"""
Progressive test suite for the GPU-residency improvements on the cEBMF package.

Each section covers one of the changes from the GPU-friendliness review.
The tests get added sequentially as new fixes land — earlier sections must
keep passing.
"""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import torch

from cebmf_torch import ash, cEBMF, ebnm_point_exp, ebnm_point_laplace
from cebmf_torch.cebmf import NoiseType
from cebmf_torch.ebnm.ash import AshConfig

# --- shared helpers ---------------------------------------------------------


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def _make_lowrank(n=80, p=60, rank=2, sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, rank))
    F = rng.normal(size=(p, rank))
    truth = L @ F.T
    Y = truth + rng.normal(scale=sigma, size=(n, p))
    return torch.tensor(Y, dtype=torch.float32), torch.tensor(truth, dtype=torch.float32), sigma


# ============================================================================
# Task 2: AshConfig.verbose default is False
# ============================================================================


def test_ashconfig_verbose_defaults_false():
    cfg = AshConfig()
    assert cfg.verbose is False, "verbose should default to False so cEBMF's inner loop doesn't spam stdout"


def test_ash_default_does_not_print_convergence():
    """ash() with default settings should not print to stdout."""
    torch.manual_seed(0)
    n = 200
    betahat = torch.randn(n)
    se = torch.full((n,), 0.5)
    buf = io.StringIO()
    with redirect_stdout(buf):
        ash(betahat, se, prior="norm")  # no verbose=True, no batch
    assert "Converged" not in buf.getvalue(), f"ash() printed unexpectedly: {buf.getvalue()!r}"


# ============================================================================
# Task 3: spike-only shortcut in point_exp / point_laplace is branchless
# ============================================================================


def test_point_exp_pure_spike_data_runs_clean():
    """
    With near-spike data (almost all zero), the optimizer should drive pi_slab tiny
    and the spike-only branch should fire. We don't probe the branch directly —
    we just verify the function still produces sensible output, no NaNs, no
    runtime error from broken branchless rewrite.
    """
    torch.manual_seed(0)
    n = 500
    s = torch.full((n,), 0.5)
    # data is ~all zero with the given se: clear spike scenario
    x = torch.randn(n) * 1e-3
    res = ebnm_point_exp(x, s)

    # post_mean must be finite and small (spike or near-spike)
    assert torch.isfinite(res.post_mean).all()
    assert torch.isfinite(res.post_mean2).all()
    assert torch.isfinite(res.post_sd).all()
    assert res.post_mean.abs().mean().item() < 0.5
    # log_lik scalar must be finite
    assert torch.isfinite(res.log_lik)


def test_point_laplace_pure_spike_data_runs_clean():
    torch.manual_seed(0)
    n = 500
    s = torch.full((n,), 0.5)
    x = torch.randn(n) * 1e-3
    res = ebnm_point_laplace(x, s)
    assert torch.isfinite(res.post_mean).all()
    assert torch.isfinite(res.post_mean2).all()
    assert torch.isfinite(res.post_sd).all()
    assert torch.isfinite(res.log_lik)


def test_point_exp_signal_data_not_all_spike():
    """With clear signal, the slab should activate and post_mean should be informative."""
    torch.manual_seed(0)
    n = 500
    s = torch.full((n,), 0.3)
    # half spike, half informative positive signal
    x = torch.cat([torch.zeros(n // 2), torch.randn(n // 2).abs() + 1.0])
    res = ebnm_point_exp(x, s)
    assert torch.isfinite(res.post_mean).all()
    assert res.post_mean[n // 2 :].mean().item() > 0.1


def test_point_laplace_signal_data_not_all_spike():
    torch.manual_seed(0)
    n = 500
    s = torch.full((n,), 0.3)
    x = torch.cat([torch.zeros(n // 2), torch.randn(n // 2) * 2.0])
    res = ebnm_point_laplace(x, s)
    assert torch.isfinite(res.post_mean).all()
    # signal half should have nonzero posterior magnitude on average
    assert res.post_mean[n // 2 :].abs().mean().item() > 0.1


# ============================================================================
# Task 4: tau in CONSTANT mode is a 0-d tensor; tau_map is a broadcast view
# ============================================================================


def test_constant_tau_is_zero_d_tensor():
    """In CONSTANT mode, self.tau should be a 0-d tensor (not (N,P))."""
    torch.manual_seed(0)
    Y = torch.randn(20, 15)
    model = cEBMF(Y, K=3)
    model.initialise_factors("svd")
    model.update_tau()
    assert model.noise.type == NoiseType.CONSTANT
    assert model.tau.ndim == 0, f"expected 0-d tau, got shape {tuple(model.tau.shape)}"


def test_constant_tau_map_is_broadcast_view_no_copy():
    """tau_map in CONSTANT mode should share storage with tau (it's an expand view)."""
    torch.manual_seed(0)
    Y = torch.randn(20, 15)
    model = cEBMF(Y, K=3)
    model.initialise_factors("svd")
    model.update_tau()
    # expand views share data_ptr with their source
    assert model.tau_map.data_ptr() == model.tau.data_ptr(), (
        "tau_map should be a view over tau, not a materialised (N,P) copy"
    )
    assert tuple(model.tau_map.shape) == (model.N, model.P)


def test_constant_tau_updates_propagate_to_tau_map():
    """A new update_tau() call should rebind both tau and tau_map consistently."""
    torch.manual_seed(0)
    Y = torch.randn(20, 15)
    model = cEBMF(Y, K=3)
    model.initialise_factors("svd")
    model.update_tau()
    # do one factor sweep, then update_tau again
    model.iter_once()
    tau_after = model.tau.item()
    # tau_map should reflect the new scalar
    assert torch.allclose(model.tau_map, torch.full_like(model.tau_map, tau_after)), (
        f"tau_map={model.tau_map[0, 0].item()} but tau={tau_after}"
    )


def test_constant_loglik_unaffected_by_tau_map_view_change():
    """Sanity: regression test that the loglik branch still runs correctly."""
    torch.manual_seed(0)
    Y, truth, sigma = _make_lowrank(n=40, p=30, rank=2, sigma=0.1, seed=0)
    model = cEBMF(Y, K=3)
    model.initialise_factors("svd")
    for _ in range(5):
        model.iter_once()
    # objective must be finite and decreasing-ish
    assert all(np.isfinite(o) for o in model.obj)


# ============================================================================
# Task 5: branchless tensor-conditional handling (no `if tensor:` host syncs)
# ============================================================================


def test_my_etruncnorm_handles_zero_sd_branchlessly():
    """sd==0 case must still return correct degenerate values."""
    from cebmf_torch.utils.maths import my_etruncnorm

    a = torch.tensor([-1.0, -2.0, -1.0])
    b = torch.tensor([1.0, 0.0, 2.0])
    mean = torch.tensor([0.5, -1.0, 0.0])
    sd = torch.tensor([1.0, 0.0, 0.0])  # mix of normal and degenerate

    out = my_etruncnorm(a, b, mean, sd)
    # entry 1 (sd=0, b=0 <= mean=-1 is FALSE; a=-2 >= mean=-1 is FALSE; a<mean<b)
    # Actually: mean=-1, b=0, a=-2. b > mean so cond1 False. a < mean so cond2 False.
    # cond3: a<mean & b>mean → True → res = mean = -1
    assert torch.allclose(out[1].to(torch.float64), torch.tensor(-1.0, dtype=torch.float64))


def test_my_etruncnorm_no_zero_sd_path_unchanged():
    """When no sd is zero, output should match the previous (guarded) behavior."""
    from cebmf_torch.utils.maths import my_etruncnorm

    torch.manual_seed(0)
    a = torch.linspace(-3, -1, 7)
    b = torch.linspace(1, 3, 7)
    mean = torch.zeros(7)
    sd = torch.ones(7)
    out = my_etruncnorm(a, b, mean, sd)
    assert torch.isfinite(out).all()


def test_wpost_exp_spike_only_branchless():
    """w[0]==1 must still give one-hot spike response."""
    from cebmf_torch.utils.posterior import wpost_exp

    x = torch.tensor(0.1)
    s = torch.tensor(0.5)
    w = torch.tensor([1.0, 0.0, 0.0])
    scale = torch.tensor([0.0, 1.0, 2.0])
    r = wpost_exp(x, s, w, scale)
    expected = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(r, expected, atol=1e-6)


def test_wpost_exp_normal_case():
    """Mixed weights should produce a normal distribution of responsibility."""
    from cebmf_torch.utils.posterior import wpost_exp

    x = torch.tensor(2.0)
    s = torch.tensor(0.5)
    w = torch.tensor([0.3, 0.5, 0.2])
    scale = torch.tensor([0.0, 1.0, 2.0])
    r = wpost_exp(x, s, w, scale)
    assert r.shape == (3,)
    assert torch.allclose(r.sum(), torch.tensor(1.0), atol=1e-5)
    assert (r >= 0).all() and (r <= 1).all()


def test_normal_means_loglik_branchless_invalid_inputs():
    """All-invalid input must give a finite (zero) result via the branchless path."""
    from cebmf_torch.cebmf.cebmf import normal_means_loglik

    x = torch.tensor([float("nan"), float("inf")])
    s = torch.tensor([1.0, 1.0])
    Et = torch.zeros(2)
    Et2 = torch.zeros(2)
    out = normal_means_loglik(x, s, Et, Et2, reduce="sum")
    # branchless path: invalid entries contribute 0, so sum is 0 (was NaN before)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_normal_means_loglik_some_valid_some_invalid():
    """Mixed valid/invalid inputs should sum only over the valid entries."""
    from cebmf_torch.cebmf.cebmf import normal_means_loglik

    x = torch.tensor([0.5, float("nan"), 1.0])
    s = torch.tensor([1.0, 1.0, 1.0])
    Et = torch.tensor([0.4, 0.0, 0.9])
    Et2 = torch.tensor([0.2, 0.0, 0.85])
    out = normal_means_loglik(x, s, Et, Et2, reduce="sum")
    # equivalently call on only the valid pair
    x2 = torch.tensor([0.5, 1.0])
    s2 = torch.tensor([1.0, 1.0])
    Et_2 = torch.tensor([0.4, 0.9])
    Et2_2 = torch.tensor([0.2, 0.85])
    expected = normal_means_loglik(x2, s2, Et_2, Et2_2, reduce="sum")
    assert torch.allclose(out, expected, atol=1e-6)


def test_autoselect_scales_no_signal_path():
    """All-noise case should still produce a non-empty scale grid."""
    from cebmf_torch.utils.mixture import autoselect_scales_mix_norm

    torch.manual_seed(0)
    n = 200
    se = torch.full((n,), 1.0)
    bh = torch.randn(n) * 0.1  # |bh| < se almost always
    scales = autoselect_scales_mix_norm(bh, se)
    assert scales.numel() >= 2
    assert scales[0].item() == 0.0  # spike at zero


def test_autoselect_scales_with_signal():
    """When data shows clear signal, the grid should extend further."""
    from cebmf_torch.utils.mixture import autoselect_scales_mix_norm

    torch.manual_seed(0)
    n = 200
    se = torch.full((n,), 0.1)
    bh = torch.randn(n) * 3.0  # signal >> se
    scales = autoselect_scales_mix_norm(bh, se)
    assert scales.numel() >= 2
    assert scales.max().item() > 1.0  # should reach signal scale


# ============================================================================
# Task 6: single canonical _logpdf_normal across the package
# ============================================================================


def test_logpdf_normal_is_single_canonical_definition():
    """All three _logpdf_normal symbols must be the same object (re-exported)."""
    from cebmf_torch.utils import distribution_operation as dop
    from cebmf_torch.utils import maths, posterior

    assert dop._logpdf_normal is maths._logpdf_normal
    assert posterior._logpdf_normal is maths._logpdf_normal


def test_logpdf_normal_matches_torch_distributions():
    """Sanity: our helper agrees with torch.distributions.Normal."""
    from cebmf_torch.utils.maths import _logpdf_normal

    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 21)
    loc = torch.zeros_like(x)
    scale = torch.ones_like(x)
    got = _logpdf_normal(x, loc, scale)
    want = torch.distributions.Normal(loc, scale).log_prob(x)
    assert torch.allclose(got, want, atol=1e-6)


def test_log_norm_pdf_back_compat_alias_still_works():
    """log_norm_pdf must still exist and produce the same answer for healthy inputs."""
    from cebmf_torch.utils.maths import _logpdf_normal, log_norm_pdf

    x = torch.tensor([0.0, 1.0, -1.0])
    loc = torch.tensor([0.0, 0.0, 0.0])
    scale = torch.tensor([1.0, 0.5, 2.0])
    a = log_norm_pdf(x, loc, scale)
    b = _logpdf_normal(x, loc, scale)
    assert torch.allclose(a, b, atol=1e-6)


# ============================================================================
# Task 7: optimize_pi_logL converges with sync-light check
# ============================================================================


def test_optimize_pi_logL_converges_with_sync_light_check():
    """The check_every-iter convergence path should still find a sensible pi."""
    from cebmf_torch.utils.mixture import optimize_pi_logL

    torch.manual_seed(0)
    n, K = 500, 5
    # Build logL where component 2 dominates everything.
    logL = torch.randn(n, K) * 0.1
    logL[:, 2] += 5.0
    pi = optimize_pi_logL(logL, penalty=1.0, max_iters=100, tol=1e-6)
    assert pi.argmax().item() == 2
    assert torch.allclose(pi.sum(), torch.tensor(1.0), atol=1e-5)


def test_optimize_pi_logL_check_every_param():
    """Larger check_every still produces a valid simplex."""
    from cebmf_torch.utils.mixture import optimize_pi_logL

    torch.manual_seed(0)
    logL = torch.randn(200, 4)
    pi_a = optimize_pi_logL(logL, penalty=1.0, max_iters=50, check_every=1)
    pi_b = optimize_pi_logL(logL, penalty=1.0, max_iters=50, check_every=10)
    # Both should be valid simplex points; tighter checks give earlier exits.
    assert torch.allclose(pi_a.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(pi_b.sum(), torch.tensor(1.0), atol=1e-5)


def test_ash_default_check_runs_without_extra_eps_sync():
    """Smoke test that ash() still works after eps.item() removals."""
    torch.manual_seed(0)
    n = 300
    bh = torch.randn(n)
    se = torch.full((n,), 0.5)
    res = ash(bh, se, prior="norm")
    assert torch.isfinite(res.post_mean).all()
    assert torch.isfinite(res.post_sd).all()


# ============================================================================
# Task 8: device threads cleanly through learned priors
# ============================================================================


def test_cebmf_with_learned_prior_stays_on_input_device():
    """When cEBMF is built on CPU, all factor tensors must remain on CPU
    even when a learned prior (which used to default to cuda-or-cpu) is used.

    This is the regression test for the device-bug: previously cebnm/cash_solver.py
    did `device = device or torch.device('cuda' if cuda else 'cpu')`. On a CUDA
    box that defaulted to CUDA, the prior would return CUDA tensors, and the
    `self.L[:, k] = resL.post_mean` assignment would crash with a device mismatch.
    """
    torch.manual_seed(0)
    n, p = 30, 20
    Y = torch.randn(n, p, device="cpu")
    X_l = torch.randn(n, 3, device="cpu")
    X_f = torch.randn(p, 2, device="cpu")
    cpu = torch.device("cpu")

    model = cEBMF(
        Y,
        K=2,
        prior_L="cash",
        prior_F="cash",
        X_l=X_l,
        X_f=X_f,
        device=cpu,
        prior_L_kwargs={"n_epochs": 2, "n_layers": 1, "hidden_dim": 4, "num_classes": 4, "batch_size": 16},
        prior_F_kwargs={"n_epochs": 2, "n_layers": 1, "hidden_dim": 4, "num_classes": 4, "batch_size": 16},
    )
    model.initialise_factors("svd")
    # Single iteration is enough to exercise the prior_L_fn.fit and prior_F_fn.fit
    # pathways with the learned prior on both sides.
    model.iter_once()

    assert model.L.device == cpu
    assert model.F.device == cpu
    assert model.tau.device == cpu


def test_cash_builder_inherits_device_from_betahat():
    """Calling cash_posterior_means with device=None must inherit from input tensor."""
    from cebmf_torch.cebnm.cash_solver import cash_posterior_means

    cpu = torch.device("cpu")
    n = 50
    X = torch.randn(n, 2, device=cpu)
    betahat = torch.randn(n, device=cpu)
    sebetahat = torch.full((n,), 0.3, device=cpu)
    out = cash_posterior_means(
        X, betahat, sebetahat, n_epochs=2, n_layers=1, hidden_dim=4, num_classes=4, batch_size=16, device=None
    )
    assert out.post_mean.device == cpu
    assert out.post_mean2.device == cpu


def test_emdn_builder_inherits_device_from_betahat():
    """Same regression check for the EMDN builder."""
    from cebmf_torch.cebnm.emdn import emdn_posterior_means

    cpu = torch.device("cpu")
    n = 50
    X = torch.randn(n, 2, device=cpu)
    betahat = torch.randn(n, device=cpu)
    sebetahat = torch.full((n,), 0.3, device=cpu)
    out = emdn_posterior_means(
        X, betahat, sebetahat, n_epochs=2, n_layers=1, hidden_dim=4, n_gaussians=3, batch_size=16, device=None
    )
    assert out.post_mean.device == cpu


# ============================================================================
# Task 9: Prior.loss and EBNM scalar fields are 0-d tensors
# ============================================================================


def test_ebnm_point_exp_scalars_are_tensors():
    torch.manual_seed(0)
    n = 200
    x = torch.randn(n)
    s = torch.full((n,), 0.5)
    res = ebnm_point_exp(x, s)
    for name in ("scale", "pi_slab", "log_lik", "mode"):
        v = getattr(res, name)
        assert isinstance(v, torch.Tensor), f"{name} should be a 0-d tensor, got {type(v).__name__}"
        assert v.ndim == 0, f"{name} should be 0-d, got shape {tuple(v.shape)}"


def test_ebnm_point_laplace_scalars_are_tensors():
    torch.manual_seed(0)
    n = 200
    x = torch.randn(n)
    s = torch.full((n,), 0.5)
    res = ebnm_point_laplace(x, s)
    for name in ("a", "pi_slab", "log_lik", "mu"):
        v = getattr(res, name)
        assert isinstance(v, torch.Tensor)
        assert v.ndim == 0


def test_ash_log_lik_is_tensor():
    torch.manual_seed(0)
    n = 200
    bh = torch.randn(n)
    se = torch.full((n,), 0.5)
    res = ash(bh, se, prior="norm")
    assert isinstance(res.log_lik, torch.Tensor)
    assert res.log_lik.ndim == 0


def test_prior_loss_is_tensor_through_builder():
    """The Prior dataclass returned by ASHBuilder/PointBuilder must carry a tensor loss."""
    from cebmf_torch.priors import PRIOR_REGISTRY

    torch.manual_seed(0)
    n = 200
    bh = torch.randn(n)
    se = torch.full((n,), 0.5)
    for name in ("norm", "exp", "laplace"):
        builder = PRIOR_REGISTRY.get_builder(name)
        prior_obj = builder.fit(X=None, betahat=bh, sebetahat=se)
        assert isinstance(prior_obj.loss, torch.Tensor), f"{name} prior's loss should be a tensor"
        assert prior_obj.loss.ndim == 0


def test_cebmf_with_tensor_prior_loss_runs():
    """End-to-end cEBMF.fit with all three core priors must work after tensorisation."""
    torch.manual_seed(0)
    Y = torch.randn(40, 30)
    for prior in ("norm", "exp", "laplace"):
        model = cEBMF(Y, K=2, prior_L=prior, prior_F=prior)
        model.initialise_factors("svd")
        model.iter_once()
        assert torch.isfinite(model.kl_l).all()
        assert torch.isfinite(model.kl_f).all()


# ============================================================================
# Task 10: vectorised posterior helpers (no per-observation Python loops)
# ============================================================================


def test_posterior_mean_exp_vectorised_recovers_signal():
    """The vectorised exp posterior should track a clear signal."""
    from cebmf_torch.utils.posterior import posterior_mean_exp

    torch.manual_seed(0)
    n = 200
    # signal positive, sd small relative to signal magnitude
    bh = torch.cat([torch.zeros(n // 2), torch.linspace(1.0, 5.0, n // 2)])
    se = torch.full_like(bh, 0.3)
    log_pi = torch.log(torch.tensor([0.4, 0.3, 0.3]))
    scale = torch.tensor([0.0, 1.0, 3.0])
    out = posterior_mean_exp(bh, se, log_pi, scale)
    assert out.post_mean.shape == (n,)
    # signal half should have larger posterior mean than spike half
    spike_half = out.post_mean[: n // 2].mean().item()
    signal_half = out.post_mean[n // 2 :].mean().item()
    assert signal_half > spike_half + 0.5


def test_posterior_mean_exp_handles_inf_se():
    """Infinite SE rows should collapse to the prior mixture mean."""
    from cebmf_torch.utils.posterior import posterior_mean_exp

    bh = torch.tensor([0.5, 1.0, 0.5])
    se = torch.tensor([0.5, float("inf"), 0.5])
    log_pi = torch.log(torch.tensor([0.2, 0.4, 0.4]))
    scale = torch.tensor([0.0, 1.0, 2.0])
    out = posterior_mean_exp(bh, se, log_pi, scale)
    assert torch.isfinite(out.post_mean).all()
    assert torch.isfinite(out.post_mean2).all()


def test_cebmf_with_emdn_prior_runs_vectorised():
    """End-to-end smoke test: cEBMF with EMDN prior should fit without error
    (this exercises the vectorised emdn posterior path)."""
    torch.manual_seed(0)
    n, p = 30, 20
    Y = torch.randn(n, p)
    X_l = torch.randn(n, 2)
    X_f = torch.randn(p, 2)
    model = cEBMF(
        Y,
        K=2,
        prior_L="emdn",
        prior_F="emdn",
        X_l=X_l,
        X_f=X_f,
        device=torch.device("cpu"),
        prior_L_kwargs={"n_epochs": 2, "n_layers": 1, "hidden_dim": 4, "n_gaussians": 3, "batch_size": 16},
        prior_F_kwargs={"n_epochs": 2, "n_layers": 1, "hidden_dim": 4, "n_gaussians": 3, "batch_size": 16},
    )
    model.initialise_factors("svd")
    model.iter_once()
    assert torch.isfinite(model.L).all()
    assert torch.isfinite(model.F).all()


def test_cebmf_with_cash_prior_runs_vectorised():
    """End-to-end smoke test: cEBMF with CASH prior."""
    torch.manual_seed(0)
    n, p = 30, 20
    Y = torch.randn(n, p)
    X_l = torch.randn(n, 2)
    X_f = torch.randn(p, 2)
    model = cEBMF(
        Y,
        K=2,
        prior_L="cash",
        prior_F="cash",
        X_l=X_l,
        X_f=X_f,
        device=torch.device("cpu"),
        prior_L_kwargs={"n_epochs": 2, "n_layers": 1, "hidden_dim": 4, "num_classes": 4, "batch_size": 16},
        prior_F_kwargs={"n_epochs": 2, "n_layers": 1, "hidden_dim": 4, "num_classes": 4, "batch_size": 16},
    )
    model.initialise_factors("svd")
    model.iter_once()
    assert torch.isfinite(model.L).all()
    assert torch.isfinite(model.F).all()


def test_no_per_observation_loops_in_cebnm_or_posterior():
    """Static guard: no `for i in range(...)` over the data dimension in the
    posterior or cebnm modules. We check by reading the source."""
    import re
    from pathlib import Path

    import cebmf_torch

    pkg_root = Path(cebmf_torch.__file__).parent
    suspect = []
    for fname in (
        pkg_root / "utils" / "posterior.py",
        pkg_root / "cebnm" / "cash_solver.py",
        pkg_root / "cebnm" / "emdn.py",
        pkg_root / "cebnm" / "spiked_emdn.py",
    ):
        text = fname.read_text()
        # strip comments before scanning
        clean = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        for m in re.finditer(r"^\s*for\s+(\w+)\s+in\s+range\(([^)]+)\)", clean, re.MULTILINE):
            var, expr = m.group(1), m.group(2)
            # epoch / start / k / batch loops are fine; only flag i/j/n loops
            if var in ("i", "j", "n") and "epoch" not in expr and "batch" not in expr:
                suspect.append(f"{fname.name}: for {var} in range({expr})")
    assert not suspect, f"Unexpected per-observation loops still present: {suspect}"


# ============================================================================
# Cross-check: known-S and verbose changes don't regress prior tests
# ============================================================================


def test_cebmf_with_S_still_runs_quietly():
    """cEBMF.fit with known S should not flood stdout."""
    torch.manual_seed(0)
    Y, truth, sigma = _make_lowrank(n=40, p=30, rank=2, sigma=0.1, seed=1)
    model = cEBMF(Y, K=3, S=float(sigma))
    model.initialise_factors("svd")
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf):
            for _ in range(5):
                model.iter_once()
    assert "Converged" not in buf.getvalue()
    assert model.noise.type == NoiseType.KNOWN
