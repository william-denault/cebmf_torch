"""Replay a selected unstable fit and save the first large ELBO deterioration.

This is a diagnostic replay after observing an outlier, not an additional
independent accuracy replicate. Serialization occurs only at diagnostic
boundaries and is not used for timing comparisons.
"""

import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import torch
from cebmf_torch import cEBMF
from cebmf_torch.cebmf._conditional import LoadingGraph
from benchmark_plugin_vs_joint import fit_stage
from benchmark_tree_priors import simulate


class FoundDeterioration(Exception):
    pass


def run(args):
    torch.set_num_threads(1)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    observed, truth, _, _ = simulate(args.seed, 'corrected')
    observed, truth = observed.cuda(), truth.cuda()
    torch.manual_seed(args.seed)
    initializer = cEBMF(observed, K=6, device='cuda', allow_backfitting=False, verbose=False)
    initializer.initialise_factors()
    initial_l, initial_f = initializer.L.clone(), initializer.F.clone()
    del initializer
    original = LoadingGraph.update
    records = []
    def traced_update(graph, u):
        model = graph.models[0]
        sweep = model._sweeps_completed + 1
        if sweep < args.start:
            return original(graph, u)
        before = float(model.conditional_fit.objective())
        torch.save(model, folder / 'latest_before.model.pt')
        original(graph, u)
        after = float(model.conditional_fit.objective())
        record = dict(sweep=sweep, coordinate=u, negative_objective_before=before,
                      negative_objective_after=after, increase=after-before,
                      surrogate_improvement=float(graph.history[-1]['after'] - graph.history[-1]['before']))
        records.append(record)
        (folder / 'trace.json').write_text(json.dumps(dict(complete=False, records=records), indent=2))
        print(json.dumps(record), flush=True)
        if after - before > 1000:
            shutil.copy2(folder / 'latest_before.model.pt', folder / 'failure_before.model.pt')
            torch.save(model, folder / 'failure_after.model.pt')
            (folder / 'trace.json').write_text(json.dumps(dict(complete=True, records=records), indent=2))
            raise FoundDeterioration
    options = argparse.Namespace(output=str(folder / 'replay'), omega=.01, batch=None,
                                 epochs=None, parents=None, quadrature=None, fixed_rank=False)
    try:
        with patch.object(LoadingGraph, 'update', traced_update):
            fit_stage(observed, truth, initial_l, initial_f, 'current', 'cold', 'spiked_emdn',
                      30, args.seed, options, 'diagnostic-replay-of-primary')
    except FoundDeterioration:
        print('Saved before/after states for the first large objective deterioration.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--start', type=int, default=23)
    parser.add_argument('--output', default='output/plugin_vs_joint_20260917/failure_trace_seed07')
    run(parser.parse_args())
