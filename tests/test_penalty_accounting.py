"""Cross-prior penalty scaling and independent reconstruction of fitted costs."""

import math
import warnings

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebnm import cov_sharp_2gb_prior
from cebmf_torch.cebnm.hmm import _log_penalty, fit_ash_hmm
from cebmf_torch.ebnm.point_exp import ebnm_point_exp
from cebmf_torch.priors.registry import PRIOR_REGISTRY


@pytest.fixture(scope="module", autouse=True)
def small_cpu_fits():
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_num_threads(threads)
    torch.set_default_dtype(dtype)


@pytest.mark.parametrize("n,batch", [(7, 7), (70, 7), (70, 3)])
@pytest.mark.parametrize("penalty", [1.0, 3.0])
def test_cgb2_penalty_value_and_gradient_use_full_training_count(n, batch, penalty):
    q = torch.linspace(0.2, 0.8, batch, dtype=torch.float64, requires_grad=True)
    y = torch.linspace(-1, 1, batch, dtype=torch.float64)
    args = (q, 0.6 * (1 - q), 0.4 * (1 - q), 1.0, -1.0, 0.5, 0.5, y, torch.ones_like(y))
    loss = cov_sharp_2gb_prior.cgb2_loss
    difference = batch * (loss(*args, penalty=penalty, n_obs=n) - loss(*args, penalty=1, n_obs=n))
    expected = -(penalty - 1) / n * q.log().sum()
    torch.testing.assert_close(difference, expected)
    torch.testing.assert_close(torch.autograd.grad(difference, q)[0], torch.autograd.grad(expected, q)[0])
    if batch == n:
        torch.testing.assert_close(loss(*args, penalty=penalty), loss(*args, penalty=penalty, n_obs=n))


@pytest.mark.parametrize(
    "states,estimate_init,adaptive", [(3, False, False), (3, True, False), (7, True, True), (1, True, True)]
)
def test_hmm_total_budget_matches_one_null_weight(states, estimate_init, adaptive):
    q = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    init = torch.cat((q[None], ((1 - q) / max(states - 1, 1)).expand(states - 1)))
    if states == 1:
        init = torch.ones_like(init)
    rho = torch.stack((q, 1 - q)).expand(states, -1)
    transition = init.expand(states, -1)
    null_state = "adaptive" if adaptive else "pointmass"
    actual = -_log_penalty(transition, init, rho, 3, estimate_init, null_state)
    expected = -2 * q.log()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, q)[0], torch.autograd.grad(expected, q)[0])
    assert _log_penalty(transition, init, rho, 1, estimate_init, null_state) == 0


def test_hmm_normalized_budget_is_used_in_em_not_only_reporting():
    y, se = torch.tensor([0.1, 0.2, 2.0, -0.4], dtype=torch.float64), torch.tensor(0.5, dtype=torch.float64)
    options = {
        "mu": [0, 2],
        "prior_sd": [0],
        "init_prob": [0.6, 0.4],
        "init_transition": [[0.8, 0.2], [0.3, 0.7]],
        "estimate_init": True,
        "learn_state_means": False,
        "prune_states": False,
    }
    initial = fit_ash_hmm(y, se, maxiter=0, **options)
    result = fit_ash_hmm(y, se, maxiter=1, penalty=4, **options)
    # Two transition rows and one initial distribution share 4-1=3 counts.
    expected = initial.state_probability[0].clone()
    expected[0] += 1
    expected /= expected.sum()
    torch.testing.assert_close(result.model_param["init_prob"], expected)
    assert result.iterations == 1


CONDITIONAL = ["cash", "cgb", "cgb_sharp", "cgb_sharp_2", "lcash", "po_lcash", "spiked_emdn"]


def options_for(prior, penalty):
    if prior in ("lcash", "po_lcash"):
        return {"penalty": penalty, "max_iter": 2, "ash_init": False, "coefficient_prior_scale": None}
    return {"penalty": penalty, "n_epochs": 1, "n_layers": 0, "hidden_dim": 4, "batch_size": 3}


@pytest.mark.parametrize("prior", CONDITIONAL)
@pytest.mark.parametrize("penalty", [1.0, 3.0])
def test_conditional_reported_regularization_and_outer_objective(prior, penalty, monkeypatch):
    torch.manual_seed(43)
    y = torch.randn(7, 11, dtype=torch.float64)
    counts = []
    original = cov_sharp_2gb_prior.cgb2_loss

    def record(*args, **kwargs):
        counts.append(kwargs["n_obs"])
        return original(*args, **kwargs)

    monkeypatch.setattr(cov_sharp_2gb_prior, "cgb2_loss", record)
    model = cEBMF(
        y,
        K=1,
        prior_L=prior,
        prior_F=prior,
        X_l=y[:, :1],
        X_f=y[0, :, None],
        prior_L_kwargs=options_for(prior, penalty),
        prior_F_kwargs=options_for(prior, penalty),
        allow_backfitting=False,
        device="cpu",
    )
    model.initialise_factors()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*L-BFGS returned a finite fit.*", category=RuntimeWarning)
        model.fit(1)
    regularization = y.new_zeros(())
    for side in ("L", "F"):
        q = getattr(model, f"pi0_{side}")[0].double()
        expected = -(penalty - 1) * q.log().mean()
        torch.testing.assert_close(getattr(model, f"regularization_{side.lower()}")[0], expected, rtol=2e-6, atol=2e-6)
        regularization += expected
    # Independent likelihood reconstruction, retaining the KLs separately.
    sse = (y - model.L @ model.F.T).square().sum()
    sse += (model.L2 @ model.F2.T - model.L.square() @ model.F.square().T).sum()
    nll = 0.5 * (y.numel() * (math.log(2 * math.pi) - model.tau.log()) + model.tau * sse)
    expected = nll + model.kl_l.sum() + model.kl_f.sum() + regularization
    assert model.obj[-1] == pytest.approx(expected.item(), abs=1e-5)
    if prior == "cgb_sharp_2":
        assert set(counts) == {7, 11}


@pytest.mark.parametrize("prior", ["lcash", "po_lcash"])
def test_coefficient_prior_remains_separate_and_is_reported(prior):
    x = torch.linspace(-2, 2, 12, dtype=torch.float64)
    builder = PRIOR_REGISTRY.get_builder(prior)
    builder.set_kwargs(penalty=1, max_iter=2, ash_init=False, coefficient_prior_scale=0.7, coefficient_prior_df=3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*L-BFGS returned a finite fit.*", category=RuntimeWarning)
        result = builder.fit(x[:, None], x, torch.ones_like(x))
    state = result.model_param["state_dict"]
    slopes = state["linear.weight"][1:] - state["linear.weight"][:1] if prior == "lcash" else state["w"]
    normalizer = math.lgamma(2) - math.lgamma(1.5) - 0.5 * math.log(3 * math.pi) - math.log(0.7)
    expected = -(normalizer - 2 * torch.log1p(slopes.square() / (3 * 0.7**2))).sum()
    torch.testing.assert_close(result.regularization_loss, expected)
    # Zero-iteration prediction preserves moments but still scores the requested penalty.
    builder.set_kwargs(penalty=1, max_iter=0, coefficient_prior_scale=None)
    unpenalized = builder.fit(x[:, None], x, torch.ones_like(x), model_param=result.model_param)
    assert unpenalized.regularization_loss == 0
    torch.testing.assert_close(unpenalized.post_mean, result.post_mean, atol=0, rtol=0)


def test_spiked_emdn_beta_prior_uses_summed_scale():
    torch.manual_seed(44)
    x = torch.linspace(-2, 2, 12)
    builder = PRIOR_REGISTRY.get_builder("spiked_emdn")
    builder.set_kwargs(**options_for("spiked_emdn", 1), beta_prior=(2, 3))
    result = builder.fit(x[:, None], x, torch.ones_like(x))
    q = result.pi0_null
    expected = -(q.log() + 2 * torch.log1p(-q)).sum()
    torch.testing.assert_close(result.regularization_loss, expected)


@pytest.mark.parametrize("penalty", [1.0, 4.0])
def test_point_exponential_penalty_matches_independent_fixed_scale_optimum(penalty):
    x = torch.tensor([-0.3, 0.1, 0.4, 1.0, 2.0, 3.0], dtype=torch.float64)
    se = torch.full_like(x, 0.5)
    fit = ebnm_point_exp(
        x, se, par_init=(0.0, 0.0, 0.0), fix_par=(False, True, True), max_iter=100, tol=1e-12, penalty=penalty
    )
    # ExGaussian log density for rate=1, independent of the package helper.
    slab_log = 0.5 * se.square() - x + torch.special.log_ndtr(x / se - se)
    spike_log = torch.distributions.Normal(0, se).log_prob(x)
    q = (1 - fit.pi_slab).detach().requires_grad_()
    likelihood = torch.logaddexp(q.log() + spike_log, torch.log1p(-q) + slab_log).sum()
    objective = -likelihood - (penalty - 1) * q.log()
    assert 0 < q < 1
    assert abs(torch.autograd.grad(objective, q)[0].item()) < 2e-5
    torch.testing.assert_close(fit.log_lik, likelihood)
    torch.testing.assert_close(fit.regularization_loss, -(penalty - 1) * q.log())


@pytest.mark.parametrize("prior", ["exp", "laplace"])
def test_point_builder_reports_optional_rate_and_weight_penalties(prior):
    x = torch.linspace(0, 3, 12, dtype=torch.float64)
    builder = PRIOR_REGISTRY.get_builder(prior)
    options = {"max_iter": 30, "loga_l2": 0.2}
    options["penalty" if prior == "exp" else "pen_pi0"] = 3 if prior == "exp" else 0.3
    builder.set_kwargs(**options)
    result = builder.fit(None, x, torch.ones_like(x))
    rate = math.exp(result.model_param["par_init"][1])
    raw = math.log((rate - 0.01) / (100 - rate))
    q = result.pi0_null
    expected = 0.2 * raw**2 - (2 * q.log() if prior == "exp" else 0.3 * (q.log() + torch.log1p(-q)))
    torch.testing.assert_close(result.regularization_loss, expected)
