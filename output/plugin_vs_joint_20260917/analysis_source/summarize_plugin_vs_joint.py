"""Summarize completed paired fits; never silently count incomplete runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_tree_priors import simulate


def read_runs(folder):
    runs = {}
    for path in folder.glob('*/seed*.json'):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # A live sweep may be writing its progress file.
        if record.get('complete'):
            runs[record['method'], record['stage'], record['seed']] = record
    return runs


def descriptive(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()),
                sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max()))


def paired(first, second):
    delta = np.asarray(first) - np.asarray(second)
    rng = np.random.default_rng(9182026)
    estimates = delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(1)
    return dict(**descriptive(delta), bootstrap_mean_ci95=np.quantile(estimates, [.025, .975]).tolist(),
                first_wins=int((delta < 0).sum()), second_wins=int((delta > 0).sum()),
                relative_mean_difference=float(np.mean(first) / np.mean(second) - 1))


def numerical_summary(folder):
    records = {}
    for path in folder.glob('numerical_seed*.json'):
        data = json.loads(path.read_text())
        if not data['complete']:
            continue
        for r in data['records']:
            key = tuple(r[k] for k in ('seed', 'stage', 'coordinate', 'approximation', 'points', 'parents', 'scramble'))
            records[key] = r
    comparisons = {
        'quadratic_vs_quadrature96': (('quadratic', 24, 32, 0), ('quadrature', 96, 32, 0)),
        'quadrature24_vs_96': (('quadrature', 24, 32, 0), ('quadrature', 96, 32, 0)),
        'parent_points32_vs_256': (('quadratic', 24, 32, 0), ('quadratic', 24, 256, 0)),
        'parent_scramble0_vs_7_at256': (('quadratic', 24, 256, 0), ('quadratic', 24, 256, 7)),
    }
    result = {}
    for stage in ('cold', 'warm'):
        result[stage] = {}
        for name, (first, second) in comparisons.items():
            deltas, scaled = [], []
            for seed in range(1, 6):
                for u in (0, 3, 5):
                    prefix = (seed, stage, u)
                    if prefix + first not in records or prefix + second not in records:
                        continue
                    a, b = records[prefix + first], records[prefix + second]
                    delta = np.abs(np.array(a['mean'])-np.array(b['mean']))
                    deltas.extend(delta)
                    scaled.extend(delta / np.array(a['likelihood_se']))
            if deltas:
                result[stage][name] = dict(n=len(deltas), max_mean_change=float(np.max(deltas)),
                                          rms_mean_change=float(np.sqrt(np.mean(np.square(deltas)))),
                                          max_change_in_likelihood_se=float(np.max(scaled)))
    return result


def plot_factor_contributions(folder, seed):
    panels = []
    for stage in ('cold', 'warm'):
        for method in ('plugin', 'current'):
            checkpoint = folder / 'primary' / method / f'seed{seed:02d}_{stage}.model.pt'
            if not checkpoint.exists():
                return
            model = torch.load(checkpoint, map_location='cpu', weights_only=False)
            # Absorb the feature-column scale into L. This leaves a nearly
            # unused factor nearly invisible instead of normalizing its noise
            # up to unit variance. Orientation changes are display-only.
            j = model.F.abs().argmax(0)
            signs = model.F[j, torch.arange(model.model.K)].sign()
            amplitude = model.L * model.F.square().mean(0).sqrt() * signs
            panels.append((method, stage, amplitude.detach().numpy()))
    maximum = max(np.max(np.abs(value)) for _, _, value in panels)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
    for ax, (method, stage, value) in zip(axes.flat, panels):
        shown = ax.imshow(value, aspect='auto', interpolation='nearest', cmap='RdBu_r',
                          vmin=-maximum, vmax=maximum)
        ax.set(title=f'{method.capitalize()} / {stage} / K={value.shape[1]}', xlabel='Factor', ylabel='Row',
               xticks=np.arange(value.shape[1]), xticklabels=np.arange(1, value.shape[1]+1),
               yticks=[0, 249, 499, 749, 999], yticklabels=[1, 250, 500, 750, 1000])
        for boundary in (249.5, 499.5, 749.5):
            ax.axhline(boundary, color='black', lw=.5, alpha=.4)
    fig.colorbar(shown, ax=list(axes.flat), shrink=.8, label='Loading × feature-column RMS (display sign adjusted)')
    fig.suptitle(f'Seed {seed}: contribution patterns; factors are not matched across fits')
    fig.savefig(folder / f'factor_contributions_seed{seed:02d}.png', dpi=180)
    plt.close(fig)


def run(args):
    torch.set_num_threads(1)
    folder = Path(args.input)
    primary, controls = read_runs(folder / 'primary'), read_runs(folder / 'ablations')
    required = {(method, stage, seed) for method in ('plugin', 'current')
                for stage in ('cold', 'cgb', 'warm') for seed in range(1, 31)}
    missing = sorted(required - primary.keys())
    if args.require_complete and missing:
        raise RuntimeError(f'{len(missing)} primary stages are missing: {missing}')
    summary = dict(primary_complete=not missing, complete_primary_stages=len(primary), missing=missing,
                   primary={}, comparisons={}, controls={}, warm_diagnostics={})
    failure_path = folder / 'failures.json'
    summary['failures'] = json.loads(failure_path.read_text()) if failure_path.exists() else []
    rows = []
    for (method, stage, seed), record in sorted(primary.items()):
        final = record['history'][-1]
        rows.append(dict(method=method, stage=stage, seed=seed, rmse=final['signal_rmse'], rank=final['rank'],
                         observation_rmse=final['observation_rmse'], noise_sd=final['sigma'],
                         initial_rmse=record['initialization']['signal_rmse'],
                         after_setup_rmse=record['history'][0]['signal_rmse'],
                         fit_seconds=record['fit_seconds'], uncontended_timing=seed == 1))
    with (folder / 'final_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for stage in ('cold', 'cgb', 'warm'):
        seeds = [s for s in range(1, 31) if all((m, stage, s) in primary for m in ('current', 'plugin'))]
        if not seeds:
            continue
        values = {method: [primary[method, stage, s]['history'][-1]['signal_rmse'] for s in seeds]
                  for method in ('current', 'plugin')}
        summary['primary'][stage] = {m: descriptive(v) for m, v in values.items()}
        summary['primary'][stage]['seeds'] = seeds
        summary['comparisons'][stage] = paired(values['current'], values['plugin'])
    for method in ('plugin', 'current', 'parent_only', 'mean_feedback', 'mean_only'):
        source = primary if method == 'current' else controls
        summary['controls'][method] = {}
        for stage in ('cold', 'cgb', 'warm'):
            seeds = [s for s in range(1, 6) if (method, stage, s) in source]
            if seeds:
                summary['controls'][method][stage] = dict(
                    **descriptive([source[method, stage, s]['history'][-1]['signal_rmse'] for s in seeds]),
                    seeds=seeds)
    for mode in ('plugin_common', 'plugin_after_reset', 'current_fixed_scales'):
        records = [json.loads(p.read_text()) for p in (folder / 'warm_diagnostics').glob(f'{mode}_seed*.json')]
        records = [r for r in records if r.get('complete')]
        if records:
            summary['warm_diagnostics'][mode] = dict(
                **descriptive([r['history'][-1]['signal_rmse'] for r in records]),
                seeds=sorted(r['seed'] for r in records))
    oracle = []
    for seed in range(1, 31):
        y, truth, _, _ = simulate(seed, 'corrected')
        prediction = y.reshape(4, 250, 200).mean(1, keepdim=True).expand(-1, 250, -1).reshape_as(y)
        oracle.append(float((prediction - truth).square().mean().sqrt()))
    summary['known_leaf_group_mean_oracle'] = descriptive(oracle)
    summary['serial_seed1_fit_seconds'] = {
        m: dict(cold=primary[m, 'cold', 1]['fit_seconds'],
                warm_pipeline=sum(primary[m, stage, 1]['fit_seconds'] for stage in ('cgb', 'warm')))
        for m in ('current', 'plugin')}
    summary['numerical_sensitivity'] = numerical_summary(folder)
    rank4 = read_runs(folder / 'rank4_diagnostic')
    summary['rank4_posthoc'] = [dict(method=m, stage=stage, seed=seed, rmse=r['history'][-1]['signal_rmse'])
                               for (m, stage, seed), r in sorted(rank4.items())]
    summary['late_deterioration'] = []
    for (method, stage, seed), record in sorted(primary.items()):
        best = min(record['history'][1:], key=lambda h: h['signal_rmse'])
        final = record['history'][-1]
        if final['signal_rmse'] > best['signal_rmse'] + .05:
            summary['late_deterioration'].append(dict(method=method, stage=stage, seed=seed,
                                                      best_sweep=best['sweep'], best_rmse=best['signal_rmse'],
                                                      final_rmse=final['signal_rmse']))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# Numerical results', '',
              'Primary comparison complete: 30 paired seeds.' if not missing else
              f'**IN PROGRESS: {len(primary)}/180 primary stages complete. Do not treat this as the final benchmark.**',
              '', 'Signal reconstruction RMSE; lower is better. SD is across simulation seeds.', '',
              '| Pipeline | Method | Paired seeds | Mean | SD | Median | Maximum |',
              '|---|---|---:|---:|---:|---:|---:|']
    for stage in ('cold', 'cgb', 'warm'):
        if stage not in summary['primary']:
            continue
        for method in ('plugin', 'current'):
            stat = summary['primary'][stage][method]
            sd = f'{stat["sd"]:.5f}' if stat['sd'] is not None else '—'
            report.append(f'| {stage} | {method} | {stat["n"]} | {stat["mean"]:.5f} | {sd} | '
                          f'{stat["median"]:.5f} | {stat["maximum"]:.5f} |')
    report += ['', '## Paired differences', '',
               'Current minus plug-in. Positive differences favor plug-in. Confidence intervals use',
               '20,000 paired bootstrap resamples of the same simulation seeds.', '',
               '| Pipeline | Mean difference | 95% interval | Current wins | Plug-in wins |',
               '|---|---:|---:|---:|---:|']
    for stage, stat in summary['comparisons'].items():
        lo, hi = stat['bootstrap_mean_ci95']
        report.append(f'| {stage} | {stat["mean"]:.5f} | [{lo:.5f}, {hi:.5f}] | '
                      f'{stat["first_wins"]}/{stat["n"]} | {stat["second_wins"]}/{stat["n"]} |')
    report += ['', '## Large late deterioration', '',
               'Descriptive list of fits ending more than 0.05 RMSE above an earlier fitted sweep.',
               'The final value remains the primary endpoint; earlier truth-based minima are diagnostic only.', '',
               '| Method | Stage | Seed | Earlier sweep | Earlier RMSE | Final RMSE |',
               '|---|---|---:|---:|---:|---:|']
    for r in summary['late_deterioration']:
        report.append(f'| {r["method"]} | {r["stage"]} | {r["seed"]} | {r["best_sweep"]} | '
                      f'{r["best_rmse"]:.5f} | {r["final_rmse"]:.5f} |')
    report += ['', '## Rank-6, batch-128 diagnostic subset', '',
               'Each cell shows mean RMSE and completed seeds out of the five planned seeds.',
               'See PROTOCOL.md for what each control changes.', '',
               '| Method | Cold | CGB precursor | Warm |', '|---|---:|---:|---:|']
    for method, stages in summary['controls'].items():
        cells = []
        for stage in ('cold', 'cgb', 'warm'):
            failed = sum(r['group'] == 'ablations' and r['method'] == method and r['stage'] == stage
                         for r in summary['failures'])
            cell = f'{stages[stage]["mean"]:.5f} ({stages[stage]["n"]}/5)' if stage in stages else 'pending'
            if failed:
                cell += f'; {failed} failed'
            cells.append(cell)
        report.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    if summary['failures']:
        report += ['', '### Numerical failures', '',
                   'Failed fits are not silently retried or included as successful endpoints.',
                   'Their last available sweep is not substituted for the requested final sweep.', '',
                   '| Group | Method | Stage | Seed | Failed sweep | Reason |',
                   '|---|---|---|---:|---:|---|']
        for r in summary['failures']:
            report.append(f'| {r["group"]} | {r["method"]} | {r["stage"]} | {r["seed"]} | '
                          f'{r["failed_sweep"]} | {r["error"]} |')
    report += ['', '## Shared-precursor warm controls', '',
               'These all start from the current CGB precursor and keep rank 6.', '',
               '| Sharp-stage control | Mean RMSE | Completed seeds |', '|---|---:|---:|']
    for mode, stat in summary['warm_diagnostics'].items():
        report.append(f'| {mode} | {stat["mean"]:.5f} | {stat["seeds"]} |')
    report += ['', '`plugin_after_reset` starts a fresh scalar fit from the graph-replaced means,',
               'including fresh noise and posterior-moment initialization. It tests recoverability,',
               'not the isolated causal effect of replacing means in an otherwise identical modern fit.',
               '`current_fixed_scales` freezes initial slab scales; it does not restore the legacy',
               'repeated-omega variance update.']
    report += ['', '## Frozen-fit numerical sensitivity', '',
               'Maximum absolute coordinate-mean change across the sampled rows and coordinates.',
               'These are local checks at fitted states, not full refits or guarantees about earlier sweeps.', '',
               '| Change | Cold | Warm |', '|---|---:|---:|']
    for name in summary['numerical_sensitivity']['cold']:
        cells = [f'{summary["numerical_sensitivity"][stage][name]["max_mean_change"]:.6g}' for stage in ('cold', 'warm')]
        report.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    if summary['rank4_posthoc']:
        report += ['', '## Post-hoc rank-four diagnostic', '',
                   'Seeds 1 and 7 selected after observing the instability; fixed rank and batch 128.', '',
                   '| Seed | Method | Stage | RMSE |', '|---:|---|---|---:|']
        for r in summary['rank4_posthoc']:
            report.append(f'| {r["seed"]} | {r["method"]} | {r["stage"]} | {r["rmse"]:.5f} |')
    report += ['', '## Uncontended GPU timing', '',
               'Seed 1 only. Other jobs share the GPU, so their timings are not used as serial speed comparisons.', '',
               '| Method | Cold, seconds | CGB + warm, seconds |', '|---|---:|---:|']
    for method, timing in summary['serial_seed1_fit_seconds'].items():
        report.append(f'| {method} | {timing["cold"]:.1f} | {timing["warm_pipeline"]:.1f} |')
    report += ['', f'Known-leaf-group sample-mean reference, 30 seeds: {np.mean(oracle):.5f} RMSE.',
               'This uses true group labels for comparison only; it is not a formal lower bound.', '',
               'Final endpoints use all requested sweeps. Early best RMSEs are not substituted for failed late fits.',
               'The two fitters have different objectives; their objective values are not compared as model scores.', '']
    (folder / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(plugin='#137c73', current='#bd493d')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout='constrained')
    for ax, stage, title in zip(axes, ('cold', 'warm'), ('Cold: spiked EMDN, 30 sweeps',
                                                       'Warm: CGB 10 + sharp CGB 20 sweeps')):
        seeds = summary['primary'][stage]['seeds']
        values = {m: np.array([primary[m, stage, s]['history'][-1]['signal_rmse'] for s in seeds])
                  for m in colors}
        jitter = np.random.default_rng(42).uniform(-.055, .055, len(seeds))
        for i, seed in enumerate(seeds):
            ax.plot([jitter[i], 1+jitter[i]], [values['plugin'][i], values['current'][i]], color='#c5cbd0', lw=.7, zorder=1)
        for x, method in enumerate(colors):
            ax.scatter(x+jitter, values[method], color=colors[method], alpha=.75, s=22, zorder=2)
            mean = values[method].mean()
            ax.plot([x-.12, x+.12], [mean, mean], color='black', lw=2.2, zorder=3)
            ax.annotate(f'mean {mean:.4f}', (x, mean), xytext=(8, 7), textcoords='offset points', fontsize=9)
        ax.axhline(np.mean(oracle), ls=':', color='#555555', lw=1, label='Known-group mean oracle')
        ax.set(xticks=[0, 1], xticklabels=['Plug-in', 'Current quadratic'], xlim=(-.3, 1.55), title=title,
               ylabel='Noiseless-signal RMSE (lower is better)')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'Corrected tree: N=1,000, P=200, initial K=6; {len(summary["primary"]["warm"]["seeds"])} paired seeds')
    fig.savefig(folder / 'paired_rmse.png', dpi=180)
    fig.savefig(folder / 'paired_rmse.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for ax, stage, title in zip(axes, ('cold', 'cgb', 'warm'), ('Cold spiked EMDN', 'CGB precursor', 'Warm sharp CGB')):
        seeds = summary['primary'][stage]['seeds']
        for method in colors:
            history = np.array([[r['signal_rmse'] for r in primary[method, stage, s]['history'][1:]] for s in seeds])
            x = np.arange(1, history.shape[1]+1)
            ax.plot(x, np.median(history, axis=0), color=colors[method], label=method)
            ax.fill_between(x, *np.quantile(history, [.25, .75], axis=0), color=colors[method], alpha=.14)
        ax.set(title=title, xlabel='Sweep', ylabel='Signal RMSE')
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Median trajectories; shading is the middle 50% of simulation seeds')
    fig.savefig(folder / 'trajectories.png', dpi=180)
    plt.close(fig)
    for seed in (1, 7):
        plot_factor_contributions(folder, seed)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('missing',)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/plugin_vs_joint_20260917')
    parser.add_argument('--require-complete', action='store_true')
    run(parser.parse_args())
