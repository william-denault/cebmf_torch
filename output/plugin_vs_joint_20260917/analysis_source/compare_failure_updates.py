"""Compare numerical updates from one saved pre-failure coordinate state."""

import argparse
import json
from pathlib import Path

import torch
from cebmf_torch.cebmf._conditional import hermite_rule
from tree_update_ablation import AblationGraph


@torch.no_grad()
def set_parent_points(graph, count):
    graph.options['parent_samples'] = count
    graph.uniform = torch.quasirandom.SobolEngine(len(graph.nodes), scramble=True, seed=0).draw(count).to(graph.q[0].values).T


@torch.no_grad()
def evaluate(model):
    graph = model.conditional_fit.row
    old = graph.options['parent_samples']
    scores = {}
    for count in (32, 256):
        set_parent_points(graph, count)
        scores[str(count)] = float(model.conditional_fit.objective())
    set_parent_points(graph, old)
    return scores


def run():
    torch.set_num_threads(1)
    folder = Path('output/plugin_vs_joint_20260917/failure_trace_seed07')
    trace = json.loads((folder / 'trace.json').read_text())
    u = trace['records'][-1]['coordinate']
    result_path = folder / 'branches.json'
    records = json.loads(result_path.read_text())['records'] if result_path.exists() else []
    for method, count, points in [('quadratic', 32, 24), ('quadratic', 256, 24),
                                  ('quadrature', 32, 24), ('quadrature', 32, 96),
                                  ('no_feedback', 32, 24)]:
        if any(r['method'] == method and r['parents'] == count and r['points'] == points for r in records):
            continue
        model = torch.load(folder / 'failure_before.model.pt', map_location='cuda', weights_only=False)
        graph = model.conditional_fit.row
        graph.options['approximation'] = 'quadrature' if method == 'quadrature' else 'quadratic'
        graph.rule = hermite_rule(points, model.L)
        graph.options['quadrature_points'] = points
        if points != graph.q[u].values.shape[-1]:
            # The stored update buffer must match the new node count. Regrid
            # the updated coordinate's exact old Gaussian mixture without
            # changing its component masses, means, variances or entropy.
            q = graph.q[u]
            mass, mean, variance = q.component_centered_moments()
            nodes, weights = graph.rule
            q.values = mean[:, :, None] + variance.sqrt()[:, :, None] * nodes
            q.weights = mass[:, :, None] * (weights / weights.sum())
        set_parent_points(graph, count)
        if method == 'no_feedback':
            graph.__class__ = AblationGraph
            graph.feedback_enabled = False
        before = evaluate(model)
        before_l = model.L.clone()
        torch.manual_seed(555)
        graph.update(u)
        after = evaluate(model)
        row = dict(method=method, parents=count, points=points, coordinate=u,
                   before=before, after=after, difference={key: after[key]-before[key] for key in before},
                   max_loading_shift=float((model.L-before_l).abs().max()),
                   surrogate_improvement=float(graph.history[-1]['after']-graph.history[-1]['before']))
        records.append(row)
        torch.save(model, folder / f'branch_{method}_p{count}_q{points}.model.pt')
        result_path.write_text(json.dumps(dict(complete=False, records=records), indent=2))
        print(json.dumps(row), flush=True)
    result_path.write_text(json.dumps(dict(complete=True, records=records), indent=2) + '\n')


def inspect_saved_failure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    folder = Path('output/plugin_vs_joint_20260917/failure_trace_seed07')
    before = torch.load(folder / 'failure_before.model.pt', map_location='cpu', weights_only=False)
    after = torch.load(folder / 'branch_quadratic_p32_q24.model.pt', map_location='cpu', weights_only=False)
    u, child = 4, 5
    scores = []
    with torch.no_grad():
        for model in (before, after):
            g = model.conditional_fit.row
            context, weights = g.integrated_inputs(child, g.indices[child], g.parent_draws())
            _, inverse, center, height, _ = g.coefficients(child, context, weights)
            mass, mean, variance = g.q[child].component_centered_moments()
            scores.append(-(mass * (height - .5 * inverse * (variance + (mean-center).square()))).sum(1))
    deterioration = scores[1] - scores[0]
    row = int(deterioration.argmax())
    graph = before.conditional_fit.row
    mass, mean, variance = after.conditional_fit.row.q[u].component_centered_moments()
    component = int(mass[row].argmax())
    index = torch.tensor([row])
    draws = graph.parent_draws()
    local = graph.quadratic_feedback(u, index, draws)
    anchor, height, slope, curvature = [getattr(local, key)[0, component, 0] for key in
                                        ('anchor', 'height', 'slope', 'curvature')]
    candidate = mean[row, component].double()
    states = {j: {key: value.detach().double() for key, value in graph.priors[j].state_dict().items()}
              for j in graph.children[u]}
    with torch.no_grad():
        actual = graph.child_term(u, index, candidate.reshape(1, 1, 1), draws, states).item()
        approx = height + slope * (candidate-anchor) + .5 * curvature * (candidate-anchor).square()
        a, _ = graph.statistics(u)
        record = dict(row=row, coordinate=u, component=component, expansion_anchor=float(anchor),
                      new_component_mean=float(candidate), new_component_variance=float(variance[row, component]),
                      previous_component_mass=float(graph.q[u].component_centered_moments()[0][row, component]),
                      new_component_mass=float(mass[row, component]),
                      curvature=float(curvature), curvature_was_clipped=bool(local.clipped[0, component, 0]),
                      feedback_actual_at_new_mean=actual, feedback_quadratic_at_new_mean=float(approx),
                      child_negative_score_increase=float(deterioration.sum()),
                      likelihood_precision=float(a[row]), feature_column_norm=float(before.F[:, u].norm()),
                      note='Frozen saved parameters evaluated on CPU; coordinate/component/row indices are zero-based.')
        (folder / 'failure_geometry.json').write_text(json.dumps(record, indent=2) + '\n')
        grid = torch.linspace(float(candidate)-2, float(anchor)+1, 500, dtype=torch.float64)
        true_curve = graph.child_term(u, index, grid.reshape(1, 1, -1), draws, states).flatten()
        surrogate = height + slope * (grid-anchor) + .5 * curvature * (grid-anchor).square()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for ax in axes:
        ax.plot(grid.numpy(), true_curve.numpy(), label='Evaluated child feedback', color='#137c73')
        ax.plot(grid.numpy(), surrogate.numpy(), '--', label='Local quadratic', color='#bd493d')
        ax.axvline(float(anchor), color='#777777', ls=':', label='Expansion anchor')
        ax.axvline(float(candidate), color='#292929', ls='-.', label='New component mean')
        ax.set(xlabel='Candidate loading value', ylabel='Expected child log-density')
        ax.grid(alpha=.2)
    axes[0].set(yscale='symlog', title='Full range (symmetric logarithmic y-axis)')
    axes[1].set(ylim=(-10, 5), title='Zoom near the scores used by the surrogate')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Failure replay: factor {u+1}, row {row+1}, component {component+1}; curvature already negative')
    fig.savefig(folder / 'failure_geometry.png', dpi=180)
    plt.close(fig)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    if parser.parse_args().inspect_only:
        inspect_saved_failure()
    else:
        run()
