"""Independent checks of regularized targets, minibatches and mixed ELBOs."""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtr

from cebmf_torch import cEBMF
from cebmf_torch.experimental.conditional import SCALAR_PRIORS, ConditionalMixture, GaussianMixture
from cebmf_torch.experimental.elbo import HMMVariationalBlock, estimate_elbo, fit_marginal_mixture
from cebmf_torch.experimental.learning import fit_mixture_network


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_model(name="cgb", axis=0, prior_kwargs=None, **settings):
    generator = torch.Generator().manual_seed(921)
    y = torch.randn(8, 6, generator=generator, dtype=torch.float64)
    y[2, 3] = torch.nan
    options = dict(initialization_iterations=0, pretrain_steps=1, sweeps_per_round=3,
                   steps=2, burnin=2, draws=4, thin=1, progress_every=0, elbo_draws=4)
    options.update(settings)
    kwargs = {f"prior_{'L' if axis == 0 else 'F'}_kwargs": prior_kwargs or {}}
    model = cEBMF(y, K=2, prior_L=name if axis == 0 else "norm",
                  prior_F=name if axis == 1 else "norm",
                  self_row_cov=axis == 0, self_col_cov=axis == 1,
                  X_l=torch.linspace(-1, 1, 8), X_f=torch.linspace(-1, 1, 6),
                  S=torch.linspace(0.8, 1.4, 48).reshape(8, 6), device="cpu", joint_kwargs=options, **kwargs)
    model.initialise_factors()
    return model


@pytest.mark.parametrize("name", [name for name in SCALAR_PRIORS if name != "emdn"])
@pytest.mark.parametrize("axis", [0, 1])
@torch.no_grad()
def test_penalized_mh_correction_matches_direct_joint_density(name, axis):
    model = small_model(name, axis, {"penalty": 1.7})
    model._ensure_joint_sampler()
    engine, node = model.joint_sampler, model.joint_sampler.axes[axis]
    a, b, _ = engine.statistics(axis, 0)
    old, old_z = node.loadings[:, 0].clone(), node.components[:, 0].clone()
    new, new_z = torch.linspace(-0.4, 1.2, len(old), dtype=torch.float64), torch.ones_like(old_z)
    own = node.priors[0](node._inputs(node.loadings, 0))
    proposal, _ = own.posterior(a, b)

    def direct():
        value = engine.log_likelihood(engine.values)
        for m, block in enumerate(engine.axes):
            if m != axis:
                value += block.log_prior()
                continue
            for k, prior in enumerate(node.priors):
                mixture = prior(node._inputs(node.loadings, k))
                value += mixture.log_prob(node.loadings[:, k], node.components[:, k]).sum()
                # Evaluate the specified auxiliary likelihood independently.
                value += 0.7 * mixture.log_weight[:, 0].sum()
        return float(value)

    correction = float((node.child_log_prior(0, new) - node.child_log_prior(0, old)).sum())
    before = direct()
    node.loadings[:, 0], node.components[:, 0] = new, new_z
    difference = direct() - before
    reverse_proposal = float((proposal.log_prob(old, old_z) - proposal.log_prob(new, new_z)).sum())
    assert difference + reverse_proposal == pytest.approx(correction, abs=1e-8)
    engine.refresh_residual()
    assert engine.log_joint() == pytest.approx(direct(), abs=1e-8)


@pytest.mark.parametrize("penalty", [1.0, 2.0, 4.0])
def test_spike_penalty_has_known_intercept_only_optimum(penalty):
    torch.manual_seed(92)
    x = torch.empty(100, 0, dtype=torch.float64)
    prior = ConditionalMixture("cgb", x)
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    prior.net.output_layer.bias.requires_grad_(True)
    labels = torch.cat((torch.zeros(75, dtype=torch.long), torch.ones(25, dtype=torch.long)))
    record = fit_mixture_network(prior, x, labels.double(), components=labels, n_epochs=350,
                                batch_size=None, lr=0.06, penalty=penalty,
                                generator=torch.Generator().manual_seed(2))
    expected_spike = (0.75 + penalty - 1) / penalty
    assert prior(x).log_weight[0, 0].exp().item() == pytest.approx(expected_spike, abs=1e-5)
    assert record["after"] <= record["before"]


@pytest.mark.parametrize("axis", [0, 1])
def test_explicit_epochs_and_batch_size_control_each_prior(axis):
    model = small_model("spiked_emdn", axis, {"penalty": 1.1, "n_epochs": 2, "batch_size": 7}, steps=19)
    result = model.fit(1)
    axis_size = (8, 6)[axis]
    for record in model.joint_sampler.learning_history[0][axis]:
        assert record["n_epochs"] == record["completed_epochs"] == 2
        assert record["batch_size"] == 7
        assert record["optimizer_steps"] == 2 * math.ceil(3 * axis_size / 7)
        assert record["penalty"] == 1.1
        assert record["after"] <= record["before"]
    assert len(result.history_obj) == len(result.history_elbo) == 2
    assert [e.phase for e in result.history_elbo] == ["learning", "posterior"]
    assert result.history_obj == [e.loss for e in result.history_elbo]
    assert result.objective_kind == "negative_regularized_elbo_mc"
    assert result.elbo_estimate is result.joint_posterior.elbo
    assert len(result.joint_posterior.log_joint) == 4
    assert all(e.regularized_elbo <= e.elbo for e in result.history_elbo)
    assert all(e.regularized_standard_error >= 0 for e in result.history_elbo)


def test_full_objective_is_independent_of_batch_partition():
    x = torch.linspace(-1, 1, 19, dtype=torch.float64)[:, None]
    prior = ConditionalMixture("cgb", x)
    labels = torch.arange(19) % 2
    scores = []
    for batch in [None, 1, 7, 19, 100]:
        record = fit_mixture_network(prior, x, labels.double(), components=labels, n_epochs=0,
                                    batch_size=batch, lr=0.01, penalty=1.8,
                                    generator=torch.Generator().manual_seed(4))
        scores.append(record["before"])
        assert record["optimizer_steps"] == 0
    assert max(scores) - min(scores) < 1e-12


@pytest.mark.parametrize("kwargs", [{"penalty": 0.9}, {"penalty": float("nan")}, {"batch_size": 0},
                                    {"batch_size": True}, {"n_epochs": -1}, {"n_epochs": 1.5}])
def test_bad_training_controls_rejected_before_joint_setup(kwargs):
    model = small_model(prior_kwargs=kwargs)
    with pytest.raises(ValueError):
        model.fit(0)
    assert model.joint_sampler is None


def test_no_spike_penalty_for_unspiked_emdn():
    with pytest.raises(ValueError, match="no spike"):
        small_model("emdn", prior_kwargs={"penalty": 1.1}).fit(0)


def test_explicit_q_preserves_atoms_and_continuous_entropy():
    reference = GaussianMixture(tensor([[0.4, 0.3, 0.3]]).log(), tensor([[0, 1, -1]]), tensor([[0, 2, 3]]))
    q = fit_marginal_mixture(tensor([[0], [1], [3], [-1]]), torch.tensor([[0], [1], [1], [2]]), reference)
    torch.testing.assert_close(q.log_weight.exp(), tensor([[0.25, 0.5, 0.25]]))
    torch.testing.assert_close(q.mean, tensor([[0, 2, -1]]))
    torch.testing.assert_close(q.variance, tensor([[0, 1, 3]]))
    count = 50000
    repeated = GaussianMixture(q.log_weight.expand(count, -1), q.mean.expand(count, -1), q.variance.expand(count, -1))
    value, component = repeated.sample(torch.Generator().manual_seed(813))
    assert (value[component == 0] == 0).all()
    entropy_samples = -repeated.log_prob(value, component)
    weights = q.log_weight.exp()[0]
    expected = (-(weights * q.log_weight[0]).sum()
                + (weights[1:] * (2 * math.pi * math.e * q.variance[0, 1:]).log() / 2).sum())
    assert abs(float(entropy_samples.mean() - expected)) < 6 * float(entropy_samples.std() / math.sqrt(count))


@pytest.mark.parametrize("support", ["real", "positive", "negative"])
def test_hmm_q_normalizer_against_quadrature_and_path_enumeration(support):
    parameters = dict(mu=tensor([0, 0.8]), prior_sd=tensor([0, 0.4]),
                      transition=tensor([[0.8, 0.2], [0, 1]]), init_prob=tensor([0.4, 0.6]),
                      mixture_weight=tensor([[0.7, 0.3], [0.2, 0.8]]), nonnegative_state_means=support != "real")
    if support == "negative":
        parameters["mu"] *= -1
        parameters["effect_support"] = "nonpositive"
    a, b = tensor([0, 2, 1]), tensor([0, 0.7, -0.3])
    q = HMMVariationalBlock(a, b, parameters)
    emission = torch.zeros(3, 2, dtype=torch.float64)
    for i, state, label in itertools.product(range(3), range(2), range(2)):
        mu, sd = float(parameters["mu"][state]), float(parameters["prior_sd"][label])
        if sd == 0:
            integral = math.exp(-float(a[i]) * mu**2 / 2 + float(b[i]) * mu)
        else:
            lower, upper = {"positive": (0, math.inf), "negative": (-math.inf, 0),
                            "real": (-math.inf, math.inf)}[support]
            truncation = ndtr(mu / sd) if support == "positive" else ndtr(-mu / sd) if support == "negative" else 1

            def integrand(v):
                exponent = -float(a[i]) * v**2 / 2 + float(b[i]) * v - (v - mu)**2 / (2 * sd**2)
                return math.exp(exponent) / (math.sqrt(2 * math.pi) * sd * truncation)

            integral = quad(integrand, lower, upper, epsabs=1e-11)[0]
        emission[i, state] += parameters["mixture_weight"][state, label] * integral
    expected = 0.0
    for path in itertools.product(range(2), repeat=3):
        expected += float(parameters["init_prob"][path[0]] * emission[0, path[0]]
                          * parameters["transition"][path[0], path[1]] * emission[1, path[1]]
                          * parameters["transition"][path[1], path[2]] * emission[2, path[2]])
    assert float(q.log_normalizer) == pytest.approx(math.log(expected), abs=1e-10)
    generator = torch.Generator().manual_seed(18)
    for _ in range(20):
        value, label, state, score = q.sample(generator)
        assert math.isfinite(float(score))
        assert not ((state[:-1] == 1) & (state[1:] == 0)).any()
        if support == "positive":
            assert (value >= 0).all()
        if support == "negative":
            assert (value <= 0).all()
        assert (value[label == 0] == parameters["mu"][state[label == 0]]).all()
    neutral = HMMVariationalBlock(torch.zeros_like(a), torch.zeros_like(b), parameters)
    assert float(neutral.log_normalizer) == pytest.approx(0, abs=1e-14)


class TinyAxis:
    name = "norm"

    def __init__(self, prior):
        self.fixed = [prior]

    def log_prior(self, value, label, state=None, *, regularized=True):
        return self.fixed[0].log_prob(value[:, 0], label[:, 0]).sum()


def tiny_engine(left, right):
    y, weight = tensor([[0.4]]), tensor([[2.0]])

    def likelihood(values):
        return (-0.5 * ((y - values[0] @ values[1].T).square() * weight
                        + math.log(2 * math.pi) - weight.log())).sum()

    return SimpleNamespace(axes=[TinyAxis(left), TinyAxis(right)], y0=y, weight=weight, rank=1,
                           options={"elbo_draws": 6000}, elbo_generator=torch.Generator().manual_seed(29),
                           log_likelihood=likelihood)


def test_discrete_elbo_against_exhaustive_enumeration():
    left = GaussianMixture(tensor([[0.6, 0.4]]).log(), tensor([[0, 1]]), tensor([[0, 0]]))
    right = GaussianMixture(tensor([[0.3, 0.7]]).log(), tensor([[0, 1]]), tensor([[0, 0]]))
    engine = tiny_engine(left, right)
    labels = [torch.tensor([[[0]], [[1]], [[1]], [[1]]]), torch.tensor([[[0]], [[0]], [[1]], [[1]]])]
    values = [z.double() for z in labels]
    estimate = estimate_elbo(engine, values, labels)
    exact_elbo, evidence = 0.0, 0.0
    for l, f in itertools.product(range(2), repeat=2):
        sample = [tensor([[l]]), tensor([[f]])]
        log_joint = float(engine.log_likelihood(sample) + left.log_weight[0, l] + right.log_weight[0, f])
        q = [0.25, 0.75][l] * 0.5
        exact_elbo += q * (log_joint - math.log(q))
        evidence += math.exp(log_joint)
    assert exact_elbo <= math.log(evidence)
    assert abs(estimate.elbo - exact_elbo) < 6 * estimate.standard_error
    assert estimate.regularized_elbo == estimate.elbo and estimate.penalty_term == 0


def test_continuous_elbo_against_analytic_gaussian_moments():
    prior = GaussianMixture(tensor([[1.0]]).log(), tensor([[0.0]]), tensor([[1.0]]))
    engine = tiny_engine(prior, prior)
    means, variances = [0.4, -0.1], [0.2, 0.5]
    values = [tensor([[[mu - math.sqrt(var)]], [[mu + math.sqrt(var)]]]) for mu, var in zip(means, variances)]
    labels = [torch.zeros_like(v, dtype=torch.long) for v in values]
    estimate = estimate_elbo(engine, values, labels)
    second = [mu**2 + var for mu, var in zip(means, variances)]
    residual_second = 0.4**2 - 2 * 0.4 * means[0] * means[1] + second[0] * second[1]
    log_likelihood = -0.5 * (2 * residual_second + math.log(math.pi))
    log_prior = -math.log(2 * math.pi) - sum(second) / 2
    entropy = sum(math.log(2 * math.pi * math.e * var) / 2 for var in variances)
    expected = log_likelihood + log_prior + entropy
    assert abs(estimate.elbo - expected) < 6 * estimate.standard_error


def test_elbo_evaluation_does_not_change_chain_or_parameters():
    model = small_model(prior_kwargs={"penalty": 1.2})
    model._ensure_joint_sampler()
    engine = model.joint_sampler
    values = [value.clone() for value in engine.values]
    labels = [axis.components.clone() for axis in engine.axes]
    residual, rng = engine.residual.clone(), engine.generator.get_state().clone()
    state = [{k: v.clone() for k, v in axis.priors.state_dict().items()} for axis in engine.axes]
    estimate = engine.record_elbo()
    assert estimate.regularized_elbo <= estimate.elbo
    assert torch.equal(rng, engine.generator.get_state())
    assert torch.equal(residual, engine.residual)
    for m, axis in enumerate(engine.axes):
        assert torch.equal(engine.values[m], values[m])
        assert torch.equal(axis.components, labels[m])
        assert all(torch.equal(v, state[m][k]) for k, v in axis.priors.state_dict().items())


@pytest.mark.parametrize("name", ["hmm", "hmm_pos", "hmm_neg"])
def test_elbo_integration_with_all_hmm_supports(name):
    y = torch.randn(8, 6, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    model = cEBMF(
        y, K=2, prior_L="cgb", prior_F=name, self_row_cov=True, S=1, device="cpu",
        prior_L_kwargs={"penalty": 1.2, "n_epochs": 1, "batch_size": 7},
        prior_F_kwargs={"mu": [0, -1 if name == "hmm_neg" else 1], "prior_sd": [0, 0.4],
                        "maxiter": 0, "learn_state_means": False},
        joint_kwargs={"initialization_iterations": 0, "sweeps_per_round": 2,
                      "burnin": 2, "draws": 4, "thin": 1, "elbo_draws": 4, "progress_every": 0},
    )
    model.initialise_factors()
    result = model.fit(1)
    assert all(math.isfinite(e.elbo) and math.isfinite(e.regularized_elbo) for e in result.history_elbo)
    if name == "hmm_pos":
        assert (result.F >= 0).all()
    elif name == "hmm_neg":
        assert (result.F <= 0).all()
