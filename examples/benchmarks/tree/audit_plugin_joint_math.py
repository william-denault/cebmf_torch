"""Independent algebra and benchmark-control checks; no production mutations."""

import json
import math
from pathlib import Path

import torch
from cebmf_torch import cEBMF
from cebmf_torch.cebmf._conditional import hermite_rule

from benchmark_tree_priors import simulate
from tree_update_ablation import AblationGraph


def run():
    torch.set_num_threads(1)
    dtype = torch.float64
    # Independent Gaussian integration of the proposed variance correction.
    m, variance, w, child, noise_variance = .7, .3, 1.4, -.2, .4
    nodes, weights = hermite_rule(64, torch.tensor(0., dtype=dtype))
    parents = m + math.sqrt(variance) * nodes
    log_density = -.5 * (math.log(2 * math.pi * noise_variance)
                        + (child - w * parents).square() / noise_variance)
    numerical = (weights * log_density).sum()
    plugin = -.5 * (math.log(2 * math.pi * noise_variance)
                   + (child - w * m) ** 2 / noise_variance)
    corrected = plugin - w * w * variance / (2 * noise_variance)
    torch.testing.assert_close(numerical, numerical.new_tensor(corrected), rtol=1e-13, atol=1e-13)

    # Exact Gaussian posterior: prior U~N(0,1), V|U~N(U,.1),
    # independently observe U and V with variance 1 and observed values (1,1).
    precision = torch.tensor([[12., -10.], [-10., 11.]], dtype=dtype)
    covariance = torch.linalg.inv(precision)
    mean = covariance @ torch.ones(2, dtype=dtype)
    # Mean-field has the same optimal mean in this fixed-parameter Gaussian
    # example, but different variances and zero covariance. No claim that
    # mean-field always gives worse means is implied by this illustration.
    mf_variance = precision.diag().reciprocal()
    exact_mstep = (mean.prod() + covariance[0, 1]) / (mean[0].square() + covariance[0, 0])
    mf_mstep = mean.prod() / (mean[0].square() + mf_variance[0])
    mean_plugin = mean[1] / mean[0]

    # Verify the simulation literally matches the user's RNG sequence.
    torch.manual_seed(1)
    masks = [torch.randint(0, 2, (200,), dtype=torch.float32) for _ in range(7)]
    f = torch.stack([mask * torch.randn(200) for mask in masks])
    l = torch.zeros(1000, 7)
    l[:, 0] = 1
    l[:500, 1], l[500:, 2] = 1, 1
    for group in range(4):
        l[250 * group:250 * (group + 1), group + 3] = 1
    signal = l @ f
    observed = signal + 1.25 * torch.randn(1000, 200)
    y, truth, loading, _ = simulate(1, 'corrected')
    assert torch.equal(y, observed) and torch.equal(truth, signal)
    rank = int(torch.linalg.matrix_rank(loading.double()))
    assert rank == 4

    # With point parents and no feedback, the modern profile must be ordinary
    # normal-means evidence for its CURRENT parameters. This checks that the
    # ablation actually removes both proposed uncertainty modifications.
    torch.manual_seed(18)
    model = cEBMF(torch.randn(24, 12, dtype=dtype), K=3, self_row_cov=True,
                  prior_L='spiked_emdn', device='cpu', verbose=False,
                  prior_L_kwargs={'hidden_dim': 5, 'n_layers': 0, 'penalty': 1.1},
                  conditional_kwargs={'approximation': 'quadrature'})
    model.initialise_factors()
    model.fit(0)
    graph = model.conditional_fit.row
    graph.__class__ = AblationGraph
    graph.parents_at_mean, graph.feedback_enabled = True, False
    maximum_error = 0.
    for u in range(3):
        index = graph.indices[u]
        a, b = graph.statistics(u)
        z, q = graph.profile(u, index, a, b, graph.parent_draws())
        mix = graph.priors[u](graph.reference_inputs(u)[index])
        posterior, expected_z = mix.posterior(a[index], b[index])
        expected_z += .1 * mix.log_weight[:, 0]
        expected_mean = (posterior.log_weight.exp() * posterior.mean).sum(1)
        torch.testing.assert_close(z, expected_z, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(q.mean, expected_mean, rtol=1e-11, atol=1e-11)
        maximum_error = max(maximum_error, float((z - expected_z).abs().max().detach()))
    result = dict(simulation_identical=True, signal_loading_rank=rank,
                  gaussian_expected_log_density=float(numerical),
                  plugin_log_density=plugin, variance_correction=corrected-plugin,
                  exact_gaussian_posterior_mean=mean.tolist(),
                  exact_gaussian_posterior_covariance=covariance.tolist(),
                  mean_field_variance=mf_variance.tolist(),
                  exact_em_one_step_slope=float(exact_mstep),
                  mean_field_em_one_step_slope=float(mf_mstep),
                  plugin_one_step_slope=float(mean_plugin),
                  point_parent_no_feedback_evidence_max_error=maximum_error,
                  note='Illustrative one-step parameter updates, not an accuracy benchmark.')
    path = Path('output/plugin_vs_joint_20260917/math_checks.json')
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
