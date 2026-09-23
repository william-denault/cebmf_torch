"""Corrected user simulation: current joint fitting versus sequential plug-in.

The primary protocol keeps the user's N/P/K, penalties and 30/10/20 sweeps.
SVD factors are shared and each stage gets a reproducible neural RNG seed, so
the cold run's RNG consumption cannot change the subsequent warm-start branch.
Truth is used only to score fits. All methods keep ash feature priors.
"""

import argparse
import hashlib
import inspect
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from cebmf_torch import cEBMF
from cebmf_torch.priors.learned import builder_functions

from benchmark_tree_priors import PluginCGB, simulate
from tree_update_ablation import AblationGraph


METHODS = ['current', 'plugin', 'parent_only', 'mean_feedback', 'mean_only', 'quadrature', 'fixed_sharp']


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def metrics(model, truth, sweep, elapsed):
    with torch.no_grad():
        estimate = model.L @ model.F.T
        row = dict(sweep=sweep, seconds=elapsed,
                   signal_rmse=float((estimate - truth).square().mean().sqrt()),
                   observation_rmse=float((estimate - model.Y).square().mean().sqrt()),
                   sigma=float(model.tau.rsqrt().mean()), rank=model.model.K,
                   loading_variance=float((model.L2 - model.L.square()).clamp_min(0).mean()))
        if model.obj:
            row['objective'] = float(model.obj[-1])
        if model.conditional_fit:
            graph = model.conditional_fit.row
            latest = graph.history[-model.model.K:] if sweep else []
            if latest and 'curvature_clipped' in latest[0]:
                row['curvature_clipped'] = float(sum(r['curvature_clipped'] for r in latest))
                row['curvature_evaluations'] = sum(r['curvature_evaluations'] for r in latest)
            scales = [p.log_slab_sd.exp() for p in graph.priors if hasattr(p, 'log_slab_sd')]
            if scales:
                all_scales = torch.cat(scales)
                row['slab_sd_min'] = float(all_scales.min())
                row['slab_sd_max'] = float(all_scales.max())
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError(f'Nonfinite metric: {row}')
        return row


def prior_options(prior, method, args):
    options = dict(penalty=1.1 if prior == 'spiked_emdn' else 1.051 if prior == 'cgb' else 1.0511)
    if prior == 'cgb_sharp_2':
        options['omega'] = args.omega
        if method == 'fixed_sharp':
            options['learn_scales'] = False
    if args.batch is not None:
        options['batch_size'] = args.batch
    if args.epochs is not None:
        options['n_epochs'] = args.epochs
    return options


def fit_stage(y, truth, initial_l, initial_f, method, stage, prior, steps, seed, args, signature):
    folder = Path(args.output) / method
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'seed{seed:02d}_{stage}'
    path = folder / f'{stem}.json'
    model_path = folder / f'{stem}.model.pt'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing['signature'] != signature:
            raise ValueError(f'{path} uses a different protocol/source; choose a new output directory.')
        if existing.get('complete') and model_path.exists():
            print(f'SKIP {method} seed={seed} {stage}', flush=True)
            return torch.load(model_path, map_location=y.device, weights_only=False)
    torch.manual_seed(100000 + seed * 10 + {'cold': 0, 'cgb': 1, 'warm': 2}[stage])
    row = prior_options(prior, method, args)
    cls = PluginCGB if method == 'plugin' else cEBMF
    numerical = dict(approximation='quadrature' if method == 'quadrature' else 'quadratic')
    if args.parents is not None:
        numerical['parent_samples'] = args.parents
    if args.quadrature is not None:
        numerical['quadrature_points'] = args.quadrature
    sync(y.device)
    start = time.perf_counter()
    model = cls(y, K=initial_l.shape[1], prior_L=prior, prior_F='norm', self_row_cov=True,
                allow_backfitting=not args.fixed_rank, prior_L_kwargs=row, conditional_kwargs=numerical,
                device=y.device, verbose=False)
    model.initialise_factors(L=initial_l.clone(), F=initial_f.clone())
    before = metrics(model, truth, 0, 0)
    model.fit(0)
    if method in ('parent_only', 'mean_feedback', 'mean_only'):
        graph = model.conditional_fit.row
        graph.__class__ = AblationGraph
        graph.parents_at_mean = method in ('mean_feedback', 'mean_only')
        graph.feedback_enabled = method == 'mean_feedback'
    sync(y.device)
    setup_time = time.perf_counter() - start
    scalar_defaults = {k: p.default for k, p in inspect.signature(builder_functions[prior]).parameters.items()
                       if k in ('n_epochs', 'hidden_dim', 'n_layers', 'batch_size', 'n_gaussians', 'lr')}
    scalar_defaults['n_epochs'] = model.internal_epoch
    scalar_defaults.update(row)
    resolved = model.conditional_fit.row.training[0] if model.conditional_fit else scalar_defaults
    record = dict(signature=signature, seed=seed, method=method, stage=stage, prior=prior,
                  prior_kwargs=row, scalar_network_defaults=scalar_defaults, resolved_training=resolved,
                  conditional_options=model.conditional_fit.options if model.conditional_fit else None,
                  fixed_rank=args.fixed_rank, initial_rank=model.model.K, steps=steps,
                  initialization=before, setup_seconds=setup_time, complete=False,
                  history=[metrics(model, truth, 0, 0)])
    elapsed = 0.
    print(f'START {method} seed={seed} {stage} K={model.model.K} initial={before["signal_rmse"]:.5f} '
          f'after_setup={record["history"][0]["signal_rmse"]:.5f} settings={resolved}', flush=True)
    for sweep in range(1, steps + 1):
        sync(y.device)
        start = time.perf_counter()
        model.iter_once()
        sync(y.device)
        elapsed += time.perf_counter() - start
        record['history'].append(metrics(model, truth, sweep, elapsed))
        path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        if sweep == 1 or sweep % 5 == 0 or sweep == steps:
            current = record['history'][-1]
            print(f'{method:13s} seed={seed:02d} {stage:4s} sweep={sweep:02d}/{steps:02d} '
                  f'RMSE={current["signal_rmse"]:.5f} K={model.model.K} time={elapsed:.1f}s', flush=True)
    # Device-to-host serialization is an explicit boundary, outside fit timing.
    torch.save(model, model_path)
    record.update(complete=True, fit_seconds=elapsed,
                  checkpoint=str(model_path), inference=model._conditional_result().inference if model.conditional_fit else 'plugin')
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return model


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    if device.type == 'cuda':
        torch.ones(2, device=device).square()
        sync(device)
    root = Path(__file__).resolve().parents[3]
    files = list((root / 'src/cebmf_torch').rglob('*.py')) + [Path(__file__).resolve(),
            Path(__file__).with_name('tree_update_ablation.py'), Path(__file__).with_name('benchmark_tree_priors.py')]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    config = {k: v for k, v in vars(args).items() if k not in ('seeds', 'methods', 'output', 'stages')}
    signature = hashlib.sha256(json.dumps(dict(config=config, hashes=hashes), sort_keys=True).encode()).hexdigest()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Output belongs to another source/protocol; use a new directory.')
    metadata = dict(signature=signature, config=config, source_sha256=hashes, python=sys.executable,
                    torch=torch.__version__, hip=torch.version.hip,
                    gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                    protocol='Matched SVD and independent per-stage neural seeds; original user data and budgets.')
    manifest.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    for file in files:
        target = folder / 'source_snapshot' / file.relative_to(root)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    print(json.dumps({k: v for k, v in metadata.items() if k != 'source_sha256'}, indent=2), flush=True)
    for seed in args.seeds:
        y, truth, _, _ = simulate(seed, 'corrected', args.n, args.p)
        y, truth = y.to(device), truth.to(device)
        torch.manual_seed(seed)
        initializer = cEBMF(y, K=args.rank, device=device, allow_backfitting=False, verbose=False)
        initializer.initialise_factors()
        initial_l, initial_f = initializer.L.clone(), initializer.F.clone()
        del initializer
        methods = args.methods if seed % 2 else list(reversed(args.methods))
        for method in methods:
            if 'cold' in args.stages:
                cold = fit_stage(y, truth, initial_l, initial_f, method, 'cold', 'spiked_emdn',
                                 args.steps[0], seed, args, signature)
                del cold
            if 'warm' in args.stages:
                precursor = fit_stage(y, truth, initial_l, initial_f, method, 'cgb', 'cgb',
                                      args.steps[1], seed, args, signature)
                if precursor.model.K == 0:
                    raise RuntimeError(f'{method} seed {seed}: precursor pruned all factors.')
                warm = fit_stage(y, truth, precursor.L, precursor.F, method, 'warm', 'cgb_sharp_2',
                                 args.steps[2], seed, args, signature)
                del precursor, warm


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 31)))
    p.add_argument('--methods', choices=METHODS, nargs='+', default=['current', 'plugin'])
    p.add_argument('--stages', choices=['cold', 'warm'], nargs='+', default=['cold', 'warm'])
    p.add_argument('--steps', type=int, nargs=3, default=[30, 10, 20])
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--p', type=int, default=200)
    p.add_argument('--rank', type=int, default=6)
    p.add_argument('--batch', type=int, default=None, help='Override batch size equally; omitted preserves each solver default.')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--parents', type=int, default=None)
    p.add_argument('--quadrature', type=int, default=None)
    p.add_argument('--omega', type=float, default=.01)
    p.add_argument('--fixed-rank', action='store_true', help='Disable pruning for the plug-in control too.')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', default='output/plugin_vs_joint_20260917/primary')
    args = p.parse_args()
    if min(args.steps) < 1 or args.n % 4 or args.rank < 2:
        p.error('Require positive stage lengths, n divisible by four, and rank >= 2.')
    run(args)
