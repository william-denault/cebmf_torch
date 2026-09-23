"""Independent identities and public regression checks for conditional learning."""

import math

import pytest
import torch

from cebmf_torch import align_modalities, cEBMF, fit_joint
from cebmf_torch.cebmf._conditional import hermite_rule, tilted_coordinate
from cebmf_torch.priors.conditional import GaussianMixture


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(y, **kwargs):
    options = dict(K=2, prior_L="cgb", prior_F="norm", self_row_cov=True,
                   allow_backfitting=False, device="cpu", S=1., internal_epoch=2,
                   prior_L_kwargs=dict(hidden_dim=5, n_layers=0, penalty=1, lr=0.01),
                   prior_F_kwargs=dict(penalty=1),
                   conditional_kwargs=dict(approximation="quadrature", quadrature_points=12, parent_samples=16))
    options.update(kwargs)
    return cEBMF(y, **options)


def test_no_child_profile_equals_analytic_mixture_evidence_and_moments():
    dtype = torch.float64
    lw = torch.tensor([[.3, .4, .3], [.1, .2, .7]], dtype=dtype).log()
    mu = torch.tensor([[0, -1, 2], [0, 1, 3]], dtype=dtype)
    var = torch.tensor([[0, .4, 2], [0, .8, .1]], dtype=dtype)
    a, b = torch.tensor([0., 2.], dtype=dtype), torch.tensor([0., 3.], dtype=dtype)
    inv = torch.where(var > 0, var.clamp_min(1e-30).reciprocal(), 0)
    constant = lw - .5 * (torch.where(var > 0, var, 1).log() + math.log(2 * math.pi))
    constant = torch.where(var == 0, lw, constant)
    z, q = tilted_coordinate(a, b, lw, inv, mu, constant, var == 0,
                            hermite_rule(24, a), lambda v: torch.zeros_like(v))
    posterior, expected_z = GaussianMixture(lw, mu, var).posterior(a, b)
    weights = posterior.log_weight.exp()
    torch.testing.assert_close(z, expected_z, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(q.mean, (weights * posterior.mean).sum(1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(q.second, (weights * (posterior.mean.square() + posterior.variance)).sum(1),
                               atol=1e-12, rtol=1e-12)
    entropy = -(weights * posterior.log_weight).sum(1) + (
        weights[:, 1:] * .5 * (math.log(2 * math.pi * math.e) + posterior.variance[:, 1:].log())).sum(1)
    torch.testing.assert_close(q.entropy, entropy, atol=1e-12, rtol=1e-12)


def test_quadratic_child_tilt_has_exact_gaussian_solution_and_gradient():
    a, b = torch.tensor([2.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)
    theta = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    lw = torch.zeros(1, 1, dtype=a.dtype)
    inv, beta = torch.ones_like(lw), theta.expand_as(lw)
    constant = torch.full_like(lw, -.5 * math.log(2 * math.pi))
    z, q = tilted_coordinate(a, b, lw, inv, beta, constant, torch.zeros_like(lw, dtype=torch.bool),
                            hermite_rule(48, a), lambda v: -.5 * v.square() + .3 * v)
    expected_mean = (b + theta + .3) / 4
    torch.testing.assert_close(q.mean, expected_mean, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(q.second - q.mean.square(), torch.full_like(a, .25), atol=1e-10, rtol=1e-10)
    gradient, = torch.autograd.grad(z.sum(), theta)
    torch.testing.assert_close(gradient, expected_mean.sum() - theta, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("prior", ["cgb", "cgb_sharp", "spiked_emdn"])
def test_single_factor_flag_reduces_to_original_cebmf_bit_for_bit(prior):
    torch.manual_seed(10)
    y = torch.randn(16, 9, dtype=torch.float64)
    x = torch.linspace(-1, 1, len(y), dtype=y.dtype)[:, None]
    results = []
    for flag in (False, True):
        torch.manual_seed(42)
        row = dict(hidden_dim=5, n_layers=0, lr=0.01,
                   penalty=1.1 if prior == "spiked_emdn" else 1.0511)
        if prior == "cgb_sharp":
            row["omega"] = 0.01
        m = model(y, K=1, prior_L=prior, X_l=x, self_row_cov=flag,
                  prior_L_kwargs=row, conditional_kwargs={})
        m.initialise_factors()
        result = m.fit(2)
        assert m.conditional_fit is None and m.joint_sampler is None
        results.append((result.L.clone(), result.F.clone(), m.L2.clone(), result.history_obj))
    for first, second in zip(*results):
        assert torch.equal(first, second) if isinstance(first, torch.Tensor) else first == second


@pytest.mark.parametrize("self_cov", [False, True])
@pytest.mark.parametrize("rank", [1, 2])
def test_plain_cgb_rejects_omega_instead_of_silently_ignoring_it(self_cov, rank):
    with pytest.raises(ValueError, match="Plain cgb has no omega.*cgb_sharp"):
        model(torch.randn(12, 6), K=rank, self_row_cov=self_cov,
              prior_L_kwargs={"penalty": 1.0511, "omega": 0.01}).fit(1)


@pytest.mark.parametrize("prior,penalty", [("cgb", 1.0511), ("spiked_emdn", 1.1)])
def test_no_covariates_fit_shared_prior_but_row_specific_posteriors(prior, penalty):
    from cebmf_torch.priors.learned import LearnedBuilder, LearnedPriorType

    xhat = torch.linspace(-3, 3, 16)
    se = torch.full_like(xhat, 0.3)
    outputs = []
    for covariates in (None, torch.ones(16, 1)):
        torch.manual_seed(57)
        builder = LearnedBuilder(LearnedPriorType(prior), penalty=penalty,
                                 n_epochs=2, hidden_dim=5, n_layers=0, batch_size=16)
        result = builder.fit(covariates, xhat, se, device=torch.device("cpu"))
        torch.testing.assert_close(result.pi0_null, result.pi0_null[:1].expand_as(result.pi0_null))
        assert result.post_mean.max() > result.post_mean.min()
        outputs.append(result)
    torch.testing.assert_close(outputs[0].post_mean, outputs[1].post_mean, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0].post_mean2, outputs[1].post_mean2, rtol=0, atol=0)


def test_fixed_covariates_preserve_original_path_and_training_options():
    torch.manual_seed(5)
    y = torch.randn(14, 8)
    m = model(y, self_row_cov=False, X_l=torch.ones(14, 1), conditional_kwargs={})
    m.initialise_factors()
    result = m.fit(2)
    assert result.inference == "variational"
    assert m.conditional_fit is None and m.joint_sampler is None
    assert m.prior_F_fn.kwargs["penalty"] == 1


@pytest.mark.parametrize("axis", ["row", "column", "both"])
def test_variational_sweep_refits_ash_and_noise_without_a_sampler(axis):
    torch.manual_seed(3)
    y = torch.randn(16, 10, dtype=torch.float64)
    kwargs = dict(S=None, self_row_cov=axis != "column", self_col_cov=axis != "row")
    if axis != "row":
        kwargs.update(prior_F="cgb", prior_F_kwargs=dict(hidden_dim=5, n_layers=0, penalty=1))
    m = model(y, **kwargs)
    m.initialise_factors()
    tau = m.tau.clone()
    result = m.fit(2)
    assert result.inference == "conditional_variational"
    assert m.joint_sampler is None and result.joint_posterior is None
    assert not torch.equal(tau, m.tau)
    assert all(math.isfinite(x) for x in result.history_obj)
    assert (m.L2 >= m.L.square() - 1e-8).all()
    assert (m.F2 >= m.F.square() - 1e-8).all()
    assert len(m.conditional_fit.history) == 2
    assert all(s["after"] >= s["before"] for s in (m.conditional_fit.row.history if axis != "column"
                                                   else m.conditional_fit.columns[id(m)].history))


def paired_models():
    torch.manual_seed(7)
    data = align_modalities(torch.randn(10, 8, dtype=torch.float64), torch.randn(9, 7, dtype=torch.float64),
                            range(10), range(3, 12))
    a, r = model(data.atac, K=1), model(data.rna, K=1)
    fit_joint(a, r, maxit=1)
    return data, a, r


def test_partially_paired_views_collapse_missing_children_and_infer_missing_parents():
    data, a, r = paired_models()
    graph = a.conditional_fit.row
    assert graph.active[0].all()  # RNA-only cells still need latent ATAC parents.
    assert torch.equal(graph.active[1], data.rna_observed)
    index = torch.arange(3)
    draws = graph.parent_draws()
    values = torch.randn(3, 2, 12, dtype=torch.float64)
    assert torch.equal(graph.child_term(0, index, values, draws), torch.zeros_like(values))
    precision, numerator = graph.statistics(0)
    assert (precision[~data.atac_observed] == 0).all() and (numerator[~data.atac_observed] == 0).all()
    assert torch.isfinite(a.L).all() and torch.isfinite(r.L).all()
    assert a.obj == r.obj
    fit_joint(a, r, maxit=1)
    assert len(a.obj) == 2
    with pytest.raises(ValueError, match="coupled"):
        r.fit(1)


def test_unobserved_rna_parameters_do_not_change_atac_only_cell_profile():
    _, a, _ = paired_models()
    graph = a.conditional_fit.row
    index, draws = torch.arange(3), graph.parent_draws()
    precision, numerator = graph.statistics(0)
    before, _ = graph.profile(0, index, precision, numerator, draws)
    with torch.no_grad():
        graph.priors[1].net.mu_2.add_(50)
        graph.priors[1].net.output_layer.bias.add_(10)
    after, _ = graph.profile(0, index, precision, numerator, draws)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_exact_single_parent_integration_and_frozen_coordinate():
    _, a, _ = paired_models()
    graph = a.conditional_fit.row
    q = graph.q[0]
    saved = (q.values.clone(), q.weights.clone(), q.entropy.clone())
    indices = graph.active[1].nonzero().flatten()
    context, weights = graph.integrated_inputs(1, indices, graph.parent_draws())
    torch.testing.assert_close(weights.sum(1), torch.ones(len(indices), dtype=weights.dtype))
    torch.testing.assert_close(context[:, :, -1], q.values[indices].flatten(1))
    graph.update(1)
    for old, current in zip(saved, (q.values, q.weights, q.entropy)):
        assert torch.equal(old, current)


def test_unknown_options_are_rejected_instead_of_ignored():
    y = torch.eye(6, dtype=torch.float64)
    m = model(y, prior_L_kwargs=dict(beta_prior=(2, 3)))
    with pytest.raises(ValueError, match="never ignored"):
        m.fit(1)
    m = model(y, conditional_kwargs=dict(burnin=10))
    with pytest.raises(ValueError, match="Unknown conditional_kwargs"):
        m.fit(1)
