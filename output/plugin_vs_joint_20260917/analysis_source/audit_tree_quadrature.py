"""Frozen-fit numerical sensitivity, separate from full optimization benchmarks.

Compare the coordinate update at identical fitted parameters/posteriors.
More points are numerical references, not an exact posterior or exact ELBO.
The stored parent mixture representation remains the fitted 24-node rule.
"""

import argparse
import json
from pathlib import Path

import torch

from cebmf_torch.cebmf._conditional import hermite_rule
from benchmark_tree_priors import PluginCGB  # noqa: F401 -- checkpoint class
from tree_update_ablation import AblationGraph  # noqa: F401 -- checkpoint class


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    settings = [('quadratic', 24, 32, 0), ('quadrature', 24, 32, 0),
                ('quadrature', 64, 32, 0), ('quadrature', 96, 32, 0),
                ('quadratic', 24, 256, 0), ('quadratic', 24, 256, 7),
                ('quadrature', 96, 256, 0)]
    records = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for stage in ('cold', 'warm'):
            path = Path(args.input) / 'current' / f'seed{seed:02d}_{stage}.model.pt'
            model = torch.load(path, map_location=args.device, weights_only=False)
            graph = model.conditional_fit.row
            index = torch.tensor([0, 124, 250, 374, 500, 624, 750, 874], device=model.L.device)
            for u in (0, 3, 5):
                a, b = graph.statistics(u)
                for approximation, points, parents, scramble in settings:
                    graph.options.update(approximation=approximation, quadrature_points=points,
                                         parent_samples=parents, seed=scramble)
                    graph.rule = hermite_rule(points, model.L)
                    graph.uniform = torch.quasirandom.SobolEngine(
                        len(graph.nodes), scramble=True, seed=scramble).draw(parents).to(model.L).T
                    z, q = graph.profile(u, index, a, b, graph.parent_draws())
                    record = dict(seed=seed, stage=stage, coordinate=u, approximation=approximation,
                                  points=points, parents=parents, scramble=scramble,
                                  rows=index.cpu().tolist(), mean=q.mean.cpu().tolist(),
                                  second=q.second.cpu().tolist(), log_z=z.cpu().tolist(),
                                  likelihood_se=a[index].rsqrt().cpu().tolist())
                    records.append(record)
                    output.write_text(json.dumps(dict(complete=False, records=records), indent=2))
                reference = records[-1]
                baseline = records[-len(settings)]
                delta = (torch.tensor(reference['mean']) - torch.tensor(baseline['mean'])).abs()
                print(f'seed={seed} stage={stage} node={u} quadratic/ref mean delta '
                      f'max={float(delta.max()):.5g} average={float(delta.mean()):.5g}', flush=True)
    output.write_text(json.dumps(dict(complete=True, records=records), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/plugin_vs_joint_20260917/primary')
    parser.add_argument('--output', default='output/plugin_vs_joint_20260917/numerical_sensitivity.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    run(parser.parse_args())
