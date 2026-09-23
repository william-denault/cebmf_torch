"""Matched, sequential CPU comparison of quadrature and quadratic feedback.

Uses the tree driver's options, family-specific penalties and checkpoints.
Reports complete-fit accuracy and mean sweep time excluding the first sweep.
This is a small diagnostic, not a general accuracy or CUDA performance claim.
"""

import json
import hashlib
import statistics
from pathlib import Path

import torch

from benchmark_tree_priors import parser, run


def compare(args):
    output = Path(args.output)
    for approximation in ("quadrature", "quadratic"):
        args.approximation = approximation
        args.output = str(output / approximation)
        for scenario in args.scenarios:
            for seed in args.seeds:
                for method in args.methods:
                    run(args, seed, scenario, method)
    records = []
    for approximation in ("quadrature", "quadratic"):
        for path in sorted((output / approximation).glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("complete"):
                records.append(record)
    # Require one common protocol, avoiding accidental aggregation of reruns with
    # different budgets or hyperparameters in the same output directory.
    ignored = {"method", "seed", "scenario", "approximation", "prior_L", "prior_L_kwargs"}
    protocols = {json.dumps({k: v for k, v in r["config"].items() if k not in ignored}, sort_keys=True)
                 for r in records}
    if len(protocols) != 1:
        raise ValueError("Mixed protocols found; use a fresh --output directory for each comparison.")
    summary = []
    for scenario in args.scenarios:
        for method in args.methods:
            for approximation in ("quadrature", "quadratic"):
                group = [r for r in records if (r['config']['scenario'], r['config']['method'], r['config']['approximation'])
                         == (scenario, method, approximation)]
                if sorted(r['config']['seed'] for r in group) != sorted(args.seeds):
                    raise ValueError("Unexpected seeds or duplicate configurations; use a fresh output directory.")
                times = [(r['history'][-1]['seconds'] - r['history'][1]['seconds']) / (len(r['history']) - 2)
                         for r in group]
                errors = [r['history'][-1]['signal_rmse'] for r in group]
                clipped = sum(r.get('curvature_clipped', 0) for r in group)
                evaluations = sum(r.get('curvature_evaluations', 0) for r in group)
                summary.append(dict(scenario=scenario, method=method, approximation=approximation,
                                    seeds=args.seeds, rmse_mean=statistics.mean(errors),
                                    rmse_sd=statistics.stdev(errors) if len(errors) > 1 else 0.,
                                    seconds_per_sweep_median=statistics.median(times),
                                    curvature_clipped_fraction=clipped / evaluations if evaluations else None))
    root = Path(__file__).resolve().parents[3]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((root / 'src/cebmf_torch').rglob('*.py'))}
    result = dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(), device='cpu',
                  threads=torch.get_num_threads(), protocol=json.loads(next(iter(protocols))),
                  source_sha256=source_hashes,
                  note='Sequential runs; timing excludes the first sweep. Small finite-budget diagnostic.', results=summary)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    lines = ['# Quadratic feedback comparison', '', result['note'], '',
             '| Simulation | Prior/update | Feedback | Signal RMSE (mean ± SD) | Seconds/sweep (median) | Clipped curvature |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    for r in summary:
        clipped = '-' if r['curvature_clipped_fraction'] is None else f"{100*r['curvature_clipped_fraction']:.2f}%"
        lines.append(f"| {r['scenario']} | {r['method']} | {r['approximation']} | "
                     f"{r['rmse_mean']:.4f} ± {r['rmse_sd']:.4f} | {r['seconds_per_sweep_median']:.3f} | {clipped} |")
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    p = parser()
    p.description = __doc__
    p.set_defaults(seeds=[1, 2, 3], scenarios=['corrected'], methods=['cgb_self', 'spiked_self'],
                   n=200, p=60, rank=4, steps=12, epochs=2, hidden=12, layers=0,
                   quadrature=16, parents=16, output='output/tree_prior_benchmark/quadratic_20260916/validated')
    args = p.parse_args()
    if args.steps < 2:
        p.error('Use at least two sweeps for a timing that excludes the first sweep.')
    torch.set_num_threads(1)
    compare(args)
