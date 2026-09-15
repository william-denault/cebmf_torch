"""Joint dispatch and independent full-density checks for both matrix axes."""

import math

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.experimental.conditional import SCALAR_PRIORS


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model_for(name="cgb", axis=0, both=False, **options):
    generator = torch.Generator().manual_seed(51)
    y = torch.randn(10, 8, generator=generator, dtype=torch.float64)
    y[1, 2] = torch.nan
    settings = dict(initialization_iterations=0, pretrain_steps=2, sweeps_per_round=2,
                    steps=2, burnin=2, draws=4, thin=2, progress_every=0)
    settings.update(options)
    model = cEBMF(
        y, K=2, prior_L=name if axis == 0 or both else "norm",
        prior_F=name if axis == 1 or both else "norm",
        self_row_cov=axis == 0 or both, self_col_cov=axis == 1 or both,
        X_l=torch.linspace(-1, 1, 10), X_f=torch.linspace(-1, 1, 8),
        S=torch.linspace(0.7, 1.4, 80).reshape(10, 8), device="cpu", joint_kwargs=settings,
    )
    model.initialise_factors()
    return model


@pytest.mark.parametrize("name", SCALAR_PRIORS)
@pytest.mark.parametrize("axis", [0, 1])
def test_full_joint_loading_ratio_on_either_axis(name, axis):
    model = model_for(name, axis)
    model._ensure_joint_sampler()
    engine = model.joint_sampler
    node = engine.axes[axis]
    a, b, _ = engine.statistics(axis, 0)
    old = node.loadings[:, 0].clone()
    old_z = node.components[:, 0].clone()
    proposed = torch.linspace(-0.3, 1.4, len(old), dtype=torch.float64)
    z = torch.ones(len(old), dtype=torch.long)  # a continuous component for every family
    conditional_difference = (
        node.loading_log_conditional(0, proposed, z, a, b)
        - node.loading_log_conditional(0, old, old_z, a, b)
    ).sum().item()
    before = engine.log_joint()
    node.loadings[:, 0] = proposed
    node.components[:, 0] = z
    engine.refresh_residual()
    # Direct joint evaluation includes both axis priors and every observed
    # heteroscedastic likelihood, independently of the normal-means reduction.
    assert engine.log_joint() - before == pytest.approx(conditional_difference, abs=1e-8)


@pytest.mark.parametrize("axis,both", [(0, False), (1, False), (0, True)])
def test_fit_dispatch_joint_moments_and_frozen_parameters(axis, both):
    model = model_for("spiked_emdn", axis, both)
    model._ensure_joint_sampler()
    engine = model.joint_sampler
    engine.fit_prior_parameters(1)
    assert all(step["after"] <= step["before"] for view in engine.learning_history[0] for step in view)
    before = [{key: value.clone() for key, value in a.priors.state_dict().items()} for a in engine.axes]
    precision = engine.precision.clone()
    result = model.fit(0)
    assert result.inference == "joint" and result.joint_posterior is model.joint_posterior
    post = result.joint_posterior
    products = post.loading_draws @ post.factor_draws.transpose(1, 2)
    torch.testing.assert_close(result.reconstruction, products.mean(0))
    torch.testing.assert_close(post.reconstruction_sd.square(), products.var(0, unbiased=False))
    torch.testing.assert_close(model._expected_residuals_squared(),
                               ((model.Y0 - products).square() * model.mask).mean(0))
    model._update_fitted_value()
    torch.testing.assert_close(model.Y_fit, result.reconstruction)
    torch.testing.assert_close(model.L2, post.loading_draws.square().mean(0))
    torch.testing.assert_close(model.F2, post.factor_draws.square().mean(0))
    for a, saved in zip(engine.axes, before):
        for key, value in a.priors.state_dict().items():
            assert torch.equal(value, saved[key])
    assert torch.equal(engine.precision, precision)
    assert all(math.isfinite(v) for v in result.history_obj)
    assert post.acceptance_L[-1] == post.acceptance_F[-1] == 1
    model.iter_once()
    torch.testing.assert_close(model.L, engine.values[0])
    torch.testing.assert_close(model.F, engine.values[1])
    torch.testing.assert_close(engine.residual, (engine.y0 - model.L @ model.F.T) * engine.mask)
    assert model.joint_posterior is None


@pytest.mark.parametrize("axis", [0, 1])
def test_hmm_opposite_a_hierarchical_axis(axis):
    m = model_for(axis=axis)
    from cebmf_torch.priors import PRIOR_REGISTRY
    kwargs = dict(mu=[0, 1], prior_sd=[0, 0.4], maxiter=1, learn_state_means=False)
    if axis == 0:
        m.model.prior_F = "hmm_pos"
        m.prior_F_fn = PRIOR_REGISTRY.get_builder("hmm_pos")
        m.prior_F_fn.set_kwargs(**kwargs)
        m._prior_F_kwargs = kwargs
        m.covariate.X_f = None
    else:
        m.model.prior_L = "hmm_pos"
        m.prior_L_fn = PRIOR_REGISTRY.get_builder("hmm_pos")
        m.prior_L_fn.set_kwargs(**kwargs)
        m._prior_L_kwargs = kwargs
        m.covariate.X_l = None
    result = m.fit(1)
    assert ((result.F if axis == 0 else result.L) >= 0).all()
    assert all(math.isfinite(v) for v in result.history_obj)


def test_fixed_side_information_cannot_silently_change_target():
    m = model_for()
    m.iter_once()
    m.covariate.X_l += 1
    with pytest.raises(ValueError, match="Fixed side information changed"):
        m.iter_once()


def test_joint_refuses_unsupported_prior_and_does_not_fall_back():
    m = cEBMF(torch.ones(6, 5), K=1, prior_L="exp", self_row_cov=True, device="cpu")
    m.initialise_factors()
    with pytest.raises(ValueError, match="does not yet support"):
        m.fit(1)
    assert m.joint_sampler is None


def test_explicit_initialization_keeps_rank_and_resets_existing_sampler():
    m = model_for(initialization_iterations=3)
    l, f = m.L.clone(), m.F.clone()
    m.initialise_factors(L=l, F=f)
    m.fit(0)
    old = m.joint_sampler
    assert m.model.K == 2
    m.initialise_factors(L=l, F=f)
    assert m.joint_sampler is None
    m.iter_once()
    assert m.joint_sampler is not old


def test_without_self_covariates_existing_variational_path_is_unchanged():
    m = cEBMF(torch.eye(6), K=1, device="cpu")
    m.initialise_factors()
    result = m.fit(1)
    assert result.inference == "variational"
    assert result.joint_posterior is None and m.joint_sampler is None


def test_independent_initialization_preserves_known_precision_and_disabled_pruning():
    m = model_for(initialization_iterations=1)
    m.model.allow_backfitting = False
    precision = m.tau_map.double().clone()
    m.fit(0)
    assert m.model.K == 2
    torch.testing.assert_close(m.joint_sampler.precision, precision)
    with pytest.raises(ValueError, match="Rank is frozen"):
        m._prune_indices([1])
    with pytest.raises(ValueError, match="noise is frozen"):
        m.update_tau()


@pytest.mark.parametrize("setting", [{"draws": 1}, {"burnin": -1}, {"lr": float("nan")}, {"typo": 3}])
def test_invalid_joint_settings_fail_before_sampling(setting):
    m = model_for(**setting)
    with pytest.raises(ValueError):
        m.fit(0)
    assert m.joint_sampler is None
