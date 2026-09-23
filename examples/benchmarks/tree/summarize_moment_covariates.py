"""Analyze paired moment-covariate fits and their frozen-prior ELBOs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('mean', 'mean_second')
STAGES = ('cold', 'cgb', 'warm')


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                median=float(np.median(a)), minimum=float(a.min()), maximum=float(a.max()))


def difference(second, first):
    a = np.asarray(second) - np.asarray(first)
    rng = np.random.default_rng(9182026)
    draws = a[rng.integers(len(a), size=(20000, len(a)))].mean(1)
    return dict(**describe(a), ci95=np.quantile(draws, [.025, .975]).tolist(),
                positive=int((a > 0).sum()), negative=int((a < 0).sum()))


def run(args):
    folder = Path(args.input)
    primary = folder / 'primary'
    manifest = json.loads((primary / 'manifest.json').read_text())
    root = Path(__file__).resolve().parents[3]
    for relative, expected in manifest['source_sha256'].items():
        for path in (root / relative, primary / 'source_snapshot' / relative):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    runs, failures, pending = {}, [], []
    for method in METHODS:
        for seed in range(1, 31):
            for stage, steps in zip(STAGES, manifest['config']['steps']):
                path = primary / method / f'seed{seed:02d}_{stage}.json'
                r = json.loads(path.read_text()) if path.exists() else {}
                key = method, stage, seed
                if r.get('failed'):
                    failures.append(dict(method=method, stage=stage, seed=seed, error=r['error']))
                elif r.get('complete'):
                    assert r['signature'] == manifest['signature']
                    assert len(r['history']) == steps + 1
                    assert all(h['sweep'] == j for j, h in enumerate(r['history']))
                    assert all(math.isfinite(v) for h in r['history'] for v in h.values()), path
                    assert path.with_suffix('.model.pt').is_file()
                    runs[key] = r
                else:
                    pending.append(key)
    if args.require_complete and (pending or failures):
        raise RuntimeError(f'{len(pending)} pending and {len(failures)} failed stages: {pending}, {failures}')
    summary = dict(complete=not pending and not failures, complete_stages=len(runs), pending=pending,
                   failures=failures, stages={}, source_files_verified=len(manifest['source_sha256']))
    rows = []
    for (method, stage, seed), r in sorted(runs.items()):
        rows.append(dict(method=method, stage=stage, seed=seed, **r['history'][-1],
                         fit_seconds=r['fit_seconds'], uncontended_timing=seed == 1,
                         final_network_state_entries=r['final_network_state_entries']))
    if not rows:
        return
    with (folder / 'final_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired_data = {}
    for stage in STAGES:
        seeds = [s for s in range(1, 31) if all((m, stage, s) in runs for m in METHODS)]
        if not seeds:
            continue
        records = {m: [runs[m, stage, s] for s in seeds] for m in METHODS}
        final = {m: [r['history'][-1] for r in records[m]] for m in METHODS}
        scores = ('signal_rmse', 'elbo', 'penalized_elbo', 'expected_log_likelihood',
                  'kl_loading', 'kl_feature', 'row_regularization', 'column_regularization', 'rank')
        result = dict(seeds=seeds, methods={}, differences={})
        for m in METHODS:
            result['methods'][m] = {score: describe([h[score] for h in final[m]]) for score in scores}
            result['methods'][m]['final_covariates_current'] = sum(h['current_covariates_match_fitted'] for h in final[m])
            decreases = [sum(b['elbo'] < a['elbo'] - 1 for a, b in zip(r['history'][1:-1], r['history'][2:])
                             if a['rank'] == b['rank'] and a['current_covariates_match_fitted']
                             and b['current_covariates_match_fitted']) for r in records[m]]
            result['methods'][m]['fits_with_elbo_decrease_over_one'] = sum(n > 0 for n in decreases)
            result['methods'][m]['elbo_decreasing_steps_over_one'] = sum(decreases)
        for score in scores:
            result['differences'][score] = difference([h[score] for h in final['mean_second']],
                                                     [h[score] for h in final['mean']])
        dr = np.array([b['signal_rmse'] - a['signal_rmse'] for a, b in zip(final['mean'], final['mean_second'])])
        de = np.array([b['elbo'] - a['elbo'] for a, b in zip(final['mean'], final['mean_second'])])
        result['higher_elbo_and_lower_rmse_second'] = int(((de > 0) & (dr < 0)).sum())
        result['elbo_rmse_disagree_seeds'] = [s for s, e, r in zip(seeds, de, dr) if e * r > 0]
        valid = [i for i in range(len(seeds))
                 if all(final[m][i]['current_covariates_match_fitted'] for m in METHODS)]
        result['stale_final_inputs'] = {m: [s for s, h in zip(seeds, final[m])
                                           if not h['current_covariates_match_fitted']] for m in METHODS}
        if valid:
            result['elbo_current_inputs_only'] = dict(
                seeds=[seeds[i] for i in valid],
                difference=difference([final['mean_second'][i]['elbo'] for i in valid],
                                      [final['mean'][i]['elbo'] for i in valid]))
        summary['stages'][stage] = result
        paired_data[stage] = records
        for seed in seeds:
            if stage != 'warm':
                assert runs['mean', stage, seed]['history'][0] == runs['mean_second', stage, seed]['history'][0]
            else:
                for method in METHODS:
                    assert abs(runs[method, stage, seed]['history'][0]['signal_rmse'] -
                               runs[method, 'cgb', seed]['history'][-1]['signal_rmse']) < 1e-6
    fitted = [h for r in runs.values() for h in r['history'][1:]]
    summary['max_absolute_package_score_discrepancy'] = max(abs(h['package_score_discrepancy']) for h in fitted)
    summary['minimum_cached_kl'] = min(min(h['min_loading_kl'], h['min_feature_kl']) for h in fitted)
    if all((m, s, 1) in runs for m in METHODS for s in STAGES):
        summary['serial_seed1_seconds'] = {m: dict(cold=runs[m, 'cold', 1]['fit_seconds'],
                                                  warm=sum(runs[m, s, 1]['fit_seconds'] for s in ('cgb', 'warm')))
                                          for m in METHODS}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# Posterior-moment covariate benchmark', '',
              'Complete: 30 paired seeds, 180 fitted stages.' if summary['complete'] else
              f'**IN PROGRESS: {len(runs)}/180 fitted stages complete.**', '',
              'Only original scalar plug-in fitting is compared. Second means raw E[L²], not E[L]².', '',
              '## Reconstruction', '', '| Stage | Means only RMSE | Means + second RMSE | Difference [95% CI] | Second wins |',
              '|---|---:|---:|---:|---:|']
    for stage, r in summary['stages'].items():
        d = r['differences']['signal_rmse']
        a, b = [r['methods'][m]['signal_rmse']['mean'] for m in METHODS]
        report.append(f'| {stage} | {a:.5f} | {b:.5f} | {d["mean"]:+.5f} [{d["ci95"][0]:+.5f}, {d["ci95"][1]:+.5f}] | {d["negative"]}/{d["n"]} |')
    report += ['', 'Differences are second-moment minus mean-only. Negative RMSE differences favor second moments.',
               'Intervals use 20,000 paired bootstrap resamples of the simulation seeds.', '',
               '## Fitted frozen-covariate ELBO', '', 'Higher is better; these exclude training penalties.', '',
               '| Stage | Means only | Means + second | Difference [95% CI] | Second higher |', '|---|---:|---:|---:|---:|']
    for stage, r in summary['stages'].items():
        d = r['differences']['elbo']
        a, b = [r['methods'][m]['elbo']['mean'] for m in METHODS]
        report.append(f'| {stage} | {a:.2f} | {b:.2f} | {d["mean"]:+.2f} [{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}] | {d["positive"]}/{d["n"]} |')
    report += ['', '## Where the ELBO difference comes from', '',
               '| Stage | Δ expected log likelihood | Δ loading KL | Δ feature KL | Δ penalized score [95% CI] |',
               '|---|---:|---:|---:|---:|']
    for stage, r in summary['stages'].items():
        d = r['differences']
        p = d['penalized_elbo']
        report.append(f'| {stage} | {d["expected_log_likelihood"]["mean"]:+.2f} | {d["kl_loading"]["mean"]:+.2f} | '
                      f'{d["kl_feature"]["mean"]:+.2f} | {p["mean"]:+.2f} [{p["ci95"][0]:+.2f}, {p["ci95"][1]:+.2f}] |')
    report += ['', 'Δ ELBO = Δ expected log likelihood − Δ loading KL − Δ feature KL.',
               'The penalized score additionally includes the row spike penalty and ASH penalty.', '',
               '## Agreement, rank and score checks', '',
               '| Stage | Both ELBO and RMSE favor second | Seeds where ELBO and RMSE disagree | Mean rank: first / second |',
               '|---|---:|---|---:|']
    for stage, r in summary['stages'].items():
        ranks = [r['methods'][m]['rank']['mean'] for m in METHODS]
        report.append(f'| {stage} | {r["higher_elbo_and_lower_rmse_second"]}/{len(r["seeds"])} | '
                      f'{r["elbo_rmse_disagree_seeds"]} | {ranks[0]:.2f} / {ranks[1]:.2f} |')
    report += ['', f'Maximum discrepancy between recomputed ELBO and negative package objective: '
               f'{summary["max_absolute_package_score_discrepancy"]:.6f}.',
               f'Minimum cached individual loading/feature KL: {summary["minimum_cached_kl"]:.6f}.', '',
               '| Stage | Method | Current fitted inputs at endpoint | Fits with an ELBO decrease > 1 unit |',
               '|---|---|---:|---:|']
    for stage, r in summary['stages'].items():
        for m in METHODS:
            v = r['methods'][m]
            report.append(f'| {stage} | {m} | {v["final_covariates_current"]}/{len(r["seeds"])} | {v["fits_with_elbo_decrease_over_one"]}/{len(r["seeds"])} |')
    report += ['', 'Decreases count consecutive fitted sweeps with unchanged rank and current fitted inputs.',
               'These moving-covariate algorithms are not guaranteed to ascend a fixed joint ELBO.', '',
               'The score holds the fitted covariates fixed. It is not the autoregressive joint ELBO,',
               'and it is not a held-out score or a complexity-adjusted comparison of neural networks.',
               'See [PROTOCOL.md](PROTOCOL.md) for the exact definition and limitations.', '']
    for stage, r in summary['stages'].items():
        if any(r['stale_final_inputs'].values()):
            valid = r['elbo_current_inputs_only']
            d = valid['difference']
            flagged = '; '.join(f'{m}, seeds {ss}' for m, ss in r['stale_final_inputs'].items() if ss)
            report += [f'**Final pruning flag ({stage}):** {flagged}. '
                       'These scores use retained priors from their last fits; inputs changed through pruning. '
                       f'Excluding the flagged pairs leaves {len(valid["seeds"])} pairs and an ELBO difference of '
                       f'{d["mean"]:+.2f} [{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}].', '']
    if summary.get('serial_seed1_seconds'):
        report += ['## Seed-1 timing reference', '', 'Only this seed ran without another benchmark worker.', '',
                   '| Method | Cold seconds | CGB + warm seconds |', '|---|---:|---:|']
        for m, v in summary['serial_seed1_seconds'].items():
            report.append(f'| {m} | {v["cold"]:.2f} | {v["warm"]:.2f} |')
    (folder / 'RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'mean': '#137c73', 'mean_second': '#7352a2'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, stage, title in zip(axes, STAGES, ['Cold spiked EMDN', 'CGB precursor', 'Warm sharp CGB']):
        if stage not in paired_data:
            continue
        records = paired_data[stage]
        a, b = [np.array([r['history'][-1]['signal_rmse'] for r in records[m]]) for m in METHODS]
        jitter = np.random.default_rng(42).uniform(-.06, .06, len(a))
        for j, x, y in zip(jitter, a, b):
            ax.plot([j, 1 + j], [x, y], color='#c5cbd0', lw=.65)
        for pos, (m, v) in enumerate(zip(METHODS, (a, b))):
            ax.scatter(pos + jitter, v, s=20, color=colors[m], alpha=.8, zorder=3)
            ax.plot([pos - .13, pos + .13], [v.mean()] * 2, color='black', lw=2)
        ax.set(title=title, xticks=[0, 1], xticklabels=['Mean', 'Mean + second'], ylabel='Signal RMSE; lower is better')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Original plug-in cEBMF; paired simulation seeds')
    fig.savefig(folder / 'paired_rmse.png', dpi=180)
    fig.savefig(folder / 'paired_rmse.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, stage in zip(axes, STAGES):
        if stage not in paired_data:
            continue
        histories = {m: np.array([[h['elbo'] for h in r['history'][1:]] for r in paired_data[stage][m]]) for m in METHODS}
        delta = histories['mean_second'] - histories['mean']
        x = np.arange(1, delta.shape[1] + 1)
        ax.axhline(0, color='black', lw=.8, ls='--')
        ax.plot(x, np.median(delta, axis=0), color=colors['mean_second'])
        ax.fill_between(x, *np.quantile(delta, [.25, .75], axis=0), color=colors['mean_second'], alpha=.2)
        ax.set(title=stage, xlabel='Sweep', ylabel='ELBO difference: mean + second − mean')
        ax.grid(alpha=.2)
    fig.suptitle('Paired fitted-score differences: median and middle 50%; positive favors second moments')
    fig.savefig(folder / 'elbo_difference_trajectories.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, stage, title in zip(axes, STAGES, ['Cold spiked EMDN', 'CGB precursor', 'Warm sharp CGB']):
        if stage not in paired_data:
            continue
        final = {m: [r['history'][-1] for r in paired_data[stage][m]] for m in METHODS}
        delta_rmse = [b['signal_rmse'] - a['signal_rmse'] for a, b in zip(final['mean'], final['mean_second'])]
        delta_elbo = [b['elbo'] - a['elbo'] for a, b in zip(final['mean'], final['mean_second'])]
        ax.axhline(0, color='black', lw=.8, ls='--')
        ax.axvline(0, color='black', lw=.8, ls='--')
        ax.scatter(delta_rmse, delta_elbo, s=25, color=colors['mean_second'], alpha=.8)
        ax.set(title=title, xlabel='RMSE difference; left favors second',
               ylabel='ELBO difference; up favors second')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.grid(alpha=.2)
    fig.suptitle('Final paired differences: mean + second minus mean; each point is one seed')
    fig.savefig(folder / 'elbo_vs_rmse_differences.png', dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'pending'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/moment_covariates_20260917')
    parser.add_argument('--require-complete', action='store_true')
    run(parser.parse_args())
