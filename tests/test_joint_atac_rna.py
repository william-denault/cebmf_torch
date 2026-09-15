"""Whole-solver, partial-overlap and HMM sampling reference checks."""

import itertools
import math

import pytest
import torch

from cebmf_torch.cebnm.hmm import fit_ash_hmm
from cebmf_torch.experimental.conditional import SCALAR_PRIORS
from cebmf_torch.experimental.data import align_modalities, simulate_atac_rna
from cebmf_torch.experimental.joint import JointATACRNA, _ffbs, _positive_normal, sample_hmm


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_align_ids_partial_overlap_and_side_information():
    result = align_modalities(
        tensor([[10], [20]]),
        tensor([[200], [300]]),
        ["a", "b"],
        ["b", "c"],
        side_info=tensor([3, 1, 2]),
        side_ids=["c", "a", "b"],
    )
    assert result.ids == ("a", "b", "c")
    torch.testing.assert_close(result.atac[:, 0], tensor([10, 20, torch.nan]), equal_nan=True)
    torch.testing.assert_close(result.rna[:, 0], tensor([torch.nan, 200, 300]), equal_nan=True)
    torch.testing.assert_close(result.side_info[:, 0], tensor([1, 2, 3]))
    assert result.atac_observed.tolist() == [True, True, False]
    assert result.rna_observed.tolist() == [False, True, True]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"atac_ids": ["a", "a"]}, "duplicate"),
        ({"rna_ids": ["a"]}, "one identifier"),
        ({"side_info": tensor([1, 2])}, "side_ids"),
        ({"side_ids": ["a", "b"]}, "side_info"),
        ({"side_info": tensor([1, 2]), "side_ids": ["a", "b"]}, "cover"),
        ({"side_info": tensor([1, torch.nan, 3]), "side_ids": ["a", "b", "c"]}, "finite"),
        ({"atac": tensor([[1], [torch.inf]])}, "infinity"),
    ],
)
def test_bad_alignment_rejected(kwargs, match):
    args = {"atac": tensor([[1], [2]]), "rna": tensor([[3], [4]]), "atac_ids": ["a", "b"], "rna_ids": ["b", "c"]}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        align_modalities(**args)


def test_simulation_matches_original_rng_order_and_blocks():
    original = simulate_atac_rna(20, 1000)
    generator = torch.Generator().manual_seed(1)
    a = torch.stack([torch.randint(0, 2, (20,), generator=generator).float() for _ in range(2)], 1)
    torch.testing.assert_close(a, original["loadings_atac"])
    torch.testing.assert_close(
        original["rna"], original["signal_rna"] + 1.5 * torch.randn(20, 1000, generator=generator)
    )
    torch.testing.assert_close(
        original["atac"], original["signal_atac"] + 1.5 * torch.randn(20, 1000, generator=generator)
    )
    assert original["factors_atac"].sum().item() == 100
    for i in range(20):
        bits = tuple(a[i].int().tolist())
        k = [(0, 1), (1, 0), (1, 1), (0, 0)].index(bits)
        assert original["loadings_rna"][i, k] == 1
        assert original["loadings_rna"][i].sum() == 1


def test_ffbs_full_path_probabilities_against_enumeration():
    emission = tensor([[0.8, 0.1], [0.2, 0.6], [0.4, 0.3]])
    transition, initial = tensor([[0.85, 0.15], [0.4, 0.6]]), tensor([0.2, 0.8])
    paths = list(itertools.product(range(2), repeat=3))
    weights = tensor(
        [
            float(
                initial[p[0]]
                * emission[0, p[0]]
                * transition[p[0], p[1]]
                * emission[1, p[1]]
                * transition[p[1], p[2]]
                * emission[2, p[2]]
            )
            for p in paths
        ]
    )
    weights /= weights.sum()
    generator = torch.Generator().manual_seed(729)
    counts = torch.zeros(8, dtype=torch.float64)
    for _ in range(12000):
        path = _ffbs(emission.log(), transition, initial, torch.rand(3, generator=generator, dtype=torch.float64))
        counts[int(path[0] * 4 + path[1] * 2 + path[2])] += 1
    torch.testing.assert_close(counts / counts.sum(), weights, atol=0.012, rtol=0)
    # Structural zeros are respected and a length-one sequence is valid.
    for _ in range(10):
        path = _ffbs(
            emission.log(), torch.eye(2, dtype=torch.float64), tensor([1, 0]), torch.rand(3, dtype=torch.float64)
        )
        assert not path.any()
    assert _ffbs(tensor([[0, 0]]), transition, tensor([0, 1]), tensor([0.5])).item() == 1


@pytest.mark.parametrize("mean", [-20.0, -3.0, 0.0, 2.0])
def test_truncated_normal_sampling_against_quadrature(mean):
    generator = torch.Generator().manual_seed(220)
    values = _positive_normal(
        torch.full((30000,), mean, dtype=torch.float64), torch.ones(30000, dtype=torch.float64), generator
    )
    # Integrate the unnormalized positive-half density directly; dropping
    # exp(-mean**2/2) avoids tiny normalization constants in the negative tail.
    grid = torch.linspace(0, max(12.0, mean + 12), 120001, dtype=torch.float64)
    density = (mean * grid - grid.square() / 2).exp()
    density /= torch.trapezoid(density, grid)
    expected = float(torch.trapezoid(grid * density, grid))
    var = float(torch.trapezoid(grid.square() * density, grid)) - expected**2
    assert (values >= 0).all()
    assert abs(float(values.mean()) - expected) < 6 * math.sqrt(var / len(values))
    assert abs(float(values.var()) - var) < max(0.03 * var, 1e-5)


@pytest.mark.parametrize("positive", [False, True])
def test_hmm_draws_match_existing_fixed_parameter_smoother(positive):
    y, se = tensor([-0.4, 0.7, 1.4]), tensor([0.6, 0.4, 0.7])
    fit = fit_ash_hmm(
        y, se, mu=[0, 1], prior_sd=[0, 0.5], maxiter=0, nonnegative_state_means=positive, init_rho=[[1, 0], [0.3, 0.7]]
    )
    generator = torch.Generator().manual_seed(719)
    draws = torch.stack(
        [sample_hmm(se.square().reciprocal(), y / se.square(), fit.model_param, generator)[0] for _ in range(6000)]
    )
    if positive:
        assert (draws >= 0).all()
    torch.testing.assert_close(draws.mean(0), fit.post_mean, atol=0.035, rtol=0)
    torch.testing.assert_close(draws.square().mean(0), fit.post_mean2, atol=0.06, rtol=0)


def test_missing_hmm_positions_have_exactly_neutral_likelihood():
    params = {
        "mu": tensor([0, 2]),
        "prior_sd": tensor([0]),
        "transition": tensor([[0.8, 0.2], [0.1, 0.9]]),
        "init_prob": tensor([0.3, 0.7]),
        "mixture_weight": tensor([[1], [1]]),
        "nonnegative_state_means": True,
    }
    generator = torch.Generator().manual_seed(71)
    draws = torch.stack([sample_hmm(tensor([0, 0]), tensor([0, 0]), params, generator)[0] for _ in range(5000)])
    torch.testing.assert_close(draws.mean(0), tensor([1.4, 1.38]), atol=0.04, rtol=0)
    with pytest.raises(ValueError, match="zero linear"):
        sample_hmm(tensor([0]), tensor([1]), params, generator)


@pytest.mark.parametrize("name", SCALAR_PRIORS)
def test_entire_sampler_partial_rows_learning_and_frozen_posterior(name):
    sim = simulate_atac_rna(60, 40, noise_atac=0.3, noise_rna=0.3)
    # IDs: 0..9 have no RNA; 50..59 have no ATAC. Reverse RNA order so
    # erroneous positional alignment cannot pass this integration test.
    ai, ri = torch.arange(50), torch.arange(59, 9, -1)
    a, r = sim["atac"][ai].clone(), sim["rna"][ri].clone()
    a[0, 0] = torch.nan
    data = align_modalities(a, r, ai, ri, side_info=sim["side_info"], side_ids=range(60))
    solver = JointATACRNA(
        data,
        sigma_atac=0.3,
        sigma_rna=0.3,
        initial_k=2,
        initial_iterations=2,
        pretrain_steps=3,
        prior_atac=name,
        prior_rna=name,
        initial_hmm_kwargs={"maxiter": 2, "half_grid": 4},
    )
    solver.fit_prior_parameters(rounds=1, sweeps_per_round=3, steps=4)
    assert all(item["after"] <= item["before"] for item in solver.learning_history[0])
    before = {key: value.clone() for key, value in solver.priors.state_dict().items()}
    result = solver.sample(burnin=3, draws=6, thin=2, progress_every=0)
    for key, value in solver.priors.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(math.isfinite(v) for v in result.log_joint)
    assert (result.acceptance >= 0).all() and (result.acceptance <= 1).all()
    assert result.acceptance[-1] == 1  # terminal node is an exact Gibbs step
    for m, posterior in enumerate((result.atac, result.rna)):
        assert torch.isfinite(posterior.L).all() and torch.isfinite(posterior.F).all()
        assert posterior.L.shape[0] == 60
        assert (posterior.F >= 0).all()
        exact_residual = (solver.y0[m] - solver._modality_loadings(m) @ solver.factors[m].T) * solver.mask[m]
        torch.testing.assert_close(solver.residual[m], exact_residual, atol=1e-10, rtol=1e-10)
        per_draw_reconstruction = posterior.loading_draws @ posterior.factor_draws.transpose(1, 2)
        torch.testing.assert_close(posterior.reconstruction, per_draw_reconstruction.mean(0))
        torch.testing.assert_close(
            posterior.reconstruction_sd.square(), per_draw_reconstruction.var(0, unbiased=False)
        )
        torch.testing.assert_close(posterior.L2, posterior.loading_draws.square().mean(0))
        assert ((posterior.inclusion_probability >= 0) & (posterior.inclusion_probability <= 1)).all()
        # Missing latent rows are sampled, rather than fixed zero placeholders.
        missing = ~torch.isfinite(solver.y[m]).any(1)
        assert posterior.loading_draws[:, missing].var(0).sum() > 0


def test_no_paired_rows_initialization_fails_clearly():
    data = align_modalities(tensor([[1, 2]]), tensor([[3, 4]]), ["a"], ["b"])
    with pytest.raises(ValueError, match="paired rows"):
        JointATACRNA(data)


def test_negative_hmm_samples_are_exact_reflections():
    fit = fit_ash_hmm(
        tensor([0.2, 1.2]), tensor([0.5, 0.4]), mu=[0, 1], prior_sd=[0, 0.5], nonnegative_state_means=True, maxiter=0
    )
    negative = {
        **fit.model_param,
        "mu": -fit.model_param["mu"],
        "effect_support": "nonpositive",
        "nonnegative_state_means": False,
    }
    for seed in range(20):
        positive = sample_hmm(
            tensor([4, 6.25]), tensor([0.8, 7.5]), fit.model_param, torch.Generator().manual_seed(seed)
        )
        reflected = sample_hmm(tensor([4, 6.25]), tensor([-0.8, -7.5]), negative, torch.Generator().manual_seed(seed))
        torch.testing.assert_close(positive[0], -reflected[0])
        assert (reflected[0] <= 0).all()
        assert torch.equal(positive[1], reflected[1]) and torch.equal(positive[2], reflected[2])
