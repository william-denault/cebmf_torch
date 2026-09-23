"""Warm-stage controls, all starting from the same current CGB precursor.

Separate precursor quality from the sharp-prior update. A plug-in restart
from the graph-replaced means tests whether that disturbance is recoverable.
Fixing modern scales is a distinct sharpness control, not the legacy omega
variance rule. No changes are made to the package or primary runs.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from cebmf_torch import cEBMF
from benchmark_tree_priors import PluginCGB, simulate
from benchmark_plugin_vs_joint import metrics, sync


def run(args):
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        precursor = torch.load(Path(args.input) / 'current' / f'seed{seed:02d}_cgb.model.pt',
                               map_location=args.device, weights_only=False)
        _, truth, _, _ = simulate(seed, 'corrected')
        truth = truth.to(precursor.L)
        for mode in args.modes:
            path = output / f'{mode}_seed{seed:02d}.json'
            if path.exists() and json.loads(path.read_text()).get('complete'):
                continue
            torch.manual_seed(100000 + seed * 10 + 2)
            options = dict(K=precursor.model.K, prior_L='cgb_sharp_2', prior_F='norm',
                           self_row_cov=True, allow_backfitting=False, device=args.device,
                           verbose=False, prior_L_kwargs={'penalty': 1.0511, 'omega': .01},
                           conditional_kwargs={'approximation': 'quadratic'})
            initial_l, initial_f = precursor.L.clone(), precursor.F.clone()
            if mode == 'plugin_after_reset':
                reset = cEBMF(precursor.Y, **options)
                reset.initialise_factors(L=initial_l.clone(), F=initial_f.clone())
                reset.fit(0)
                initial_l, initial_f = reset.L.clone(), reset.F.clone()
                del reset
                torch.manual_seed(100000 + seed * 10 + 2)
            if mode == 'current_fixed_scales':
                options['prior_L_kwargs']['learn_scales'] = False
            cls = cEBMF if mode == 'current_fixed_scales' else PluginCGB
            model = cls(precursor.Y, **options)
            model.initialise_factors(L=initial_l, F=initial_f)
            before = metrics(model, truth, 0, 0)
            model.fit(0)
            record = dict(seed=seed, mode=mode, complete=False, precursor='primary/current',
                          fixed_rank=True, initialization=before,
                          history=[metrics(model, truth, 0, 0)])
            elapsed = 0.
            for sweep in range(1, 21):
                sync(model.L.device)
                start = time.perf_counter()
                model.iter_once()
                sync(model.L.device)
                elapsed += time.perf_counter() - start
                record['history'].append(metrics(model, truth, sweep, elapsed))
                path.write_text(json.dumps(record, indent=2))
            record['complete'] = True
            path.write_text(json.dumps(record, indent=2) + '\n')
            print(f'{mode} seed={seed} initial={before["signal_rmse"]:.5f} '
                  f'final={record["history"][-1]["signal_rmse"]:.5f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/plugin_vs_joint_20260917/primary')
    parser.add_argument('--output', default='output/plugin_vs_joint_20260917/warm_diagnostics')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--modes', nargs='+', choices=['plugin_common', 'plugin_after_reset', 'current_fixed_scales'],
                        default=['plugin_common', 'plugin_after_reset', 'current_fixed_scales'])
    run(parser.parse_args())
