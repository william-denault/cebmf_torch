"""Paired tree simulation: plug-in means versus means and raw second moments."""

import argparse
import hashlib
import inspect
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
from cebmf_torch import cEBMF
from cebmf_torch.priors.learned import builder_functions

from benchmark_tree_priors import simulate
from plugin_moments import MomentPluginCEBMF, frozen_elbo_terms


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


@torch.no_grad()
def metrics(model, truth, sweep, elapsed):
    row = dict(sweep=sweep, seconds=elapsed, rank=model.model.K,
               signal_rmse=float((model.L @ model.F.T - truth).square().mean().sqrt()),
               observation_rmse=float((model.L @ model.F.T - model.Y).square().mean().sqrt()),
               noise_sd=float(model.tau.rsqrt()),
               loading_variance=float((model.L2 - model.L.square()).clamp_min(0).mean()))
    if sweep:
        row.update(frozen_elbo_terms(model))
    if not all(math.isfinite(v) for v in row.values()):
        raise FloatingPointError(f'Nonfinite metric: {row}')
    return row


def fit_stage(y, truth, l, f, method, stage, seed, args, signature):
    folder = Path(args.output) / method
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'seed{seed:02d}_{stage}.json'
    checkpoint = path.with_suffix('.model.pt')
    if path.exists():
        old = json.loads(path.read_text())
        if old['signature'] != signature:
            raise ValueError('Source or settings changed; use a new output directory.')
        if old.get('complete') and checkpoint.exists():
            print(f'SKIP {method} seed={seed} {stage}', flush=True)
            return torch.load(checkpoint, map_location=y.device, weights_only=False)
        if old.get('failed'):
            raise RuntimeError('A failed stage is retained, not automatically retried.')
    prior, steps, penalty = {
        'cold': ('spiked_emdn', args.steps[0], 1.1),
        'cgb': ('cgb', args.steps[1], 1.051),
        'warm': ('cgb_sharp_2', args.steps[2], 1.0511),
    }[stage]
    torch.manual_seed(100000 + 10 * seed + {'cold': 0, 'cgb': 1, 'warm': 2}[stage])
    options = dict(penalty=penalty)
    if prior == 'cgb_sharp_2':
        options['omega'] = .01
    sync(y.device)
    start = time.perf_counter()
    model = MomentPluginCEBMF(y, K=l.shape[1], prior_L=prior, prior_F='norm',
                              self_row_cov=True, allow_backfitting=True,
                              prior_L_kwargs=options, prior_F_kwargs={'penalty': 10.},
                              row_cov_moments=method, device=y.device, verbose=False)
    model.initialise_factors(L=l.clone(), F=f.clone())
    model.fit(0)
    sync(y.device)
    defaults = {k: v.default for k, v in inspect.signature(builder_functions[prior]).parameters.items()
                if k in ('n_epochs', 'batch_size', 'hidden_dim', 'n_layers', 'lr', 'n_gaussians')}
    defaults.update(options, n_epochs=model.internal_epoch)
    record = dict(signature=signature, method=method, stage=stage, seed=seed,
                  prior=prior, prior_kwargs=options, resolved_training=defaults,
                  initial_rank=model.model.K, steps=steps, complete=False,
                  setup_seconds=time.perf_counter() - start,
                  objective_kind='unpenalized_fitted_frozen_covariate_elbo',
                  history=[metrics(model, truth, 0, 0.)])
    elapsed = 0.
    print(f'START {method} seed={seed} {stage} K={model.model.K}', flush=True)
    try:
        for sweep in range(1, steps + 1):
            sync(y.device)
            start = time.perf_counter()
            model.iter_once()
            sync(y.device)
            elapsed += time.perf_counter() - start
            row = metrics(model, truth, sweep, elapsed)
            record['history'].append(row)
            save_json(path, record)
            if sweep == 1 or sweep % 5 == 0 or sweep == steps:
                print(f'{method:11s} seed={seed:02d} {stage:4s} {sweep:02d}/{steps} '
                      f'RMSE={row["signal_rmse"]:.5f} ELBO={row["elbo"]:.2f} '
                      f'K={model.model.K} time={elapsed:.1f}s', flush=True)
        torch.save(model, checkpoint)
        record.update(complete=True, fit_seconds=elapsed, checkpoint=str(checkpoint),
                      final_network_state_entries=sum(v.numel() for state in model.model_state_L
                                                      if state for v in state.values()))
        save_json(path, record)
        return model
    except Exception as error:
        record.update(failed=True, error=str(error), failed_sweep=len(record['history']),
                      traceback=traceback.format_exc())
        save_json(path, record)
        raise


def run(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.ones(1, device=device).square()
        sync(device)
    root = Path(__file__).resolve().parents[3]
    files = list((root / 'src/cebmf_torch').rglob('*.py')) + [Path(__file__).resolve(),
            Path(__file__).with_name('plugin_moments.py'), Path(__file__).with_name('benchmark_tree_priors.py')]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    config = {k: v for k, v in vars(args).items() if k not in ('seeds', 'methods', 'output')}
    signature = hashlib.sha256(json.dumps(dict(config=config, source=hashes), sort_keys=True).encode()).hexdigest()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Source/settings mismatch; choose a fresh output directory.')
    metadata = dict(signature=signature, config=config, source_sha256=hashes,
                    python=sys.executable, torch=torch.__version__, hip=torch.version.hip,
                    gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None)
    save_json(manifest, metadata)
    for source in files:
        target = folder / 'source_snapshot' / source.relative_to(root)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(json.dumps({k: v for k, v in metadata.items() if k != 'source_sha256'}, indent=2), flush=True)
    failures = []
    for seed in args.seeds:
        y, truth, _, _ = simulate(seed, 'corrected', args.n, args.p)
        y, truth = y.to(device), truth.to(device)
        torch.manual_seed(seed)
        initial = cEBMF(y, K=args.rank, device=device, verbose=False)
        initial.initialise_factors()
        l, f = initial.L.clone(), initial.F.clone()
        del initial
        for method in args.methods if seed % 2 else args.methods[::-1]:
            try:
                cold = fit_stage(y, truth, l, f, method, 'cold', seed, args, signature)
                del cold
            except Exception as error:
                failures.append((seed, method, 'cold', str(error)))
                print(f'FAILED cold {seed} {method}: {error}', flush=True)
            try:
                precursor = fit_stage(y, truth, l, f, method, 'cgb', seed, args, signature)
                warm = fit_stage(y, truth, precursor.L, precursor.F, method, 'warm', seed, args, signature)
                del precursor, warm
            except Exception as error:
                failures.append((seed, method, 'warm pipeline', str(error)))
                print(f'FAILED warm pipeline {seed} {method}: {error}', flush=True)
    if failures:
        raise RuntimeError(f'Failed stages retained: {failures}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 31)))
    parser.add_argument('--methods', choices=['mean', 'mean_second'], nargs='+', default=['mean', 'mean_second'])
    parser.add_argument('--steps', type=int, nargs=3, default=[30, 10, 20])
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--p', type=int, default=200)
    parser.add_argument('--rank', type=int, default=6)
    parser.add_argument('--output', default='output/moment_covariates_20260917/primary')
    args = parser.parse_args()
    if min(args.steps) < 1 or args.n % 4 or args.rank < 2:
        parser.error('Positive sweep counts, n divisible by four, and rank >= 2 required.')
    run(args)
