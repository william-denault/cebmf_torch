"""Check local posterior integration at an already fitted, fixed model state.

This measures numerical integration sensitivity, not the error of the
factorized variational posterior and not a new end-to-end benchmark.
"""
import argparse
import json
from pathlib import Path

import torch

from cebmf_torch.cebmf._conditional import hermite_rule


@torch.no_grad()
def check(path):
    # Only load checkpoints generated locally by benchmark_tree_priors.py.
    model = torch.load(path, map_location="cpu", weights_only=False)
    graph = model.conditional_fit.row
    g = torch.Generator().manual_seed(904)
    indices = torch.unique(torch.cat((torch.randperm(model.N, generator=g)[:64],
                                     torch.tensor([250, 500, 750]))))
    stats = [graph.statistics(u) for u in range(len(graph.nodes))]
    results = []
    settings = [(8, 8, 0), (16, 32, 0), (32, 128, 0), (48, 256, 0),
                (32, 128, 1), (8, 8, 1)]
    for quadrature, parents, seed in settings:
        graph.options["parent_samples"] = parents
        graph.rule = hermite_rule(quadrature, model.L)
        graph.uniform = torch.quasirandom.SobolEngine(len(graph.nodes), scramble=True, seed=seed).draw(parents).to(model.L).T
        draws = graph.parent_draws()
        means, seconds, logz = [], [], []
        for u, (a, b) in enumerate(stats):
            coordinates = [graph.profile(u, batch, a, b, draws) for batch in indices.split(8)]
            means.append(torch.cat([q.mean for _, q in coordinates]))
            seconds.append(torch.cat([q.second for _, q in coordinates]))
            logz.append(torch.cat([z for z, _ in coordinates]))
        results.append(dict(quadrature=quadrature, parents=parents, integration_seed=seed,
                            mean=torch.stack(means, 1), second=torch.stack(seconds, 1),
                            logz=torch.stack(logz, 1)))
        print(path.name, quadrature, parents, seed, "evaluated", flush=True)
    reference = results[3]
    scale = model.L[indices].square().mean(0).sqrt().clamp_min(.01)
    record = dict(checkpoint=str(path), evaluated_rows=indices.tolist(), reference="48 quadrature points, 256 parent points, seed 0", results=[])
    for result in results:
        change = (result["mean"] - reference["mean"]) / scale
        record["results"].append(dict(
            quadrature=result["quadrature"], parents=result["parents"], integration_seed=result["integration_seed"],
            relative_mean_rms_by_column=change.square().mean(0).sqrt().tolist(),
            relative_mean_rms=float(change.square().mean().sqrt()),
            mean_abs_logz_difference=float((result["logz"] - reference["logz"]).abs().mean()),
            reconstruction_rms_change=float(((result["mean"]-reference["mean"]) @ model.F.T).square().mean().sqrt()),
        ))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("output/tree_prior_benchmark/integration.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = [check(path) for path in args.checkpoint]
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
