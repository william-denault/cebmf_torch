"""Check provenance, matched starts, and every planned benchmark outcome."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(folder, allow_incomplete=False):
    root = Path(__file__).resolve().parents[3]
    failures = json.loads((folder / 'failures.json').read_text())
    failed = {(r['group'], r['method'], r['stage'], r['seed']): r for r in failures}
    assert len(failed) == len(failures), 'Duplicate failure records'
    plans = {
        'primary': (['plugin', 'current'], range(1, 31)),
        'ablations': (['plugin', 'parent_only', 'mean_feedback', 'mean_only'], range(1, 6)),
        'rank4_diagnostic': (['plugin', 'current'], [1, 7]),
    }
    result = dict(complete=False, groups={}, pending=[], recorded_failures=len(failed))
    completed = {}
    for group, (methods, seeds) in plans.items():
        manifest = json.loads((folder / group / 'manifest.json').read_text())
        for relative, expected in manifest['source_sha256'].items():
            for source in (root / relative, folder / group / 'source_snapshot' / relative):
                assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, str(source)
        count, failure_count = 0, 0
        for method in methods:
            for seed in seeds:
                for stage, steps in [('cold', 30), ('cgb', 10), ('warm', 20)]:
                    key = group, method, stage, seed
                    path = folder / group / method / f'seed{seed:02d}_{stage}.json'
                    try:
                        record = json.loads(path.read_text()) if path.exists() else {}
                    except json.JSONDecodeError:
                        record = {}  # A running worker may be writing the file.
                    if key in failed:
                        assert not record.get('complete'), f'Failure silently replaced: {key}'
                        assert record['history'][-1]['sweep'] == failed[key]['last_saved_sweep']
                        failure_count += 1
                        continue
                    if not record.get('complete'):
                        result['pending'].append(key)
                        continue
                    assert record['signature'] == manifest['signature'], key
                    assert (record['method'], record['stage'], record['seed']) == (method, stage, seed)
                    assert record['steps'] == steps
                    assert [h['sweep'] for h in record['history']] == list(range(steps + 1)), key
                    assert all(math.isfinite(v) for h in record['history'] for v in h.values()), key
                    assert path.with_suffix('.model.pt').is_file(), key
                    if group != 'primary' or method == 'current':
                        rank = 4 if group == 'rank4_diagnostic' else 6
                        assert all(h['rank'] == rank for h in record['history']), key
                        assert record['resolved_training']['batch_size'] == 128, key
                    completed[key] = record
                    count += 1
        result['groups'][group] = dict(completed=count, failed=failure_count,
                                       planned=len(methods) * len(seeds) * 3,
                                       source_hashes_verified=len(manifest['source_sha256']))
    for seed in range(1, 31):
        for stage in ('cold', 'cgb'):
            keys = [('primary', m, stage, seed) for m in ('plugin', 'current')]
            if all(k in completed for k in keys):
                assert completed[keys[0]]['initialization'] == completed[keys[1]]['initialization'], keys
        for method in ('plugin', 'current'):
            precursor, warm = ('primary', method, 'cgb', seed), ('primary', method, 'warm', seed)
            if precursor in completed and warm in completed:
                before = completed[warm]['initialization']
                after = completed[precursor]['history'][-1]
                assert abs(before['signal_rmse'] - after['signal_rmse']) < 1e-6
                assert before['rank'] == after['rank']
    warm_count = 0
    for mode in ('plugin_common', 'plugin_after_reset', 'current_fixed_scales'):
        for seed in range(1, 6):
            path = folder / 'warm_diagnostics' / f'{mode}_seed{seed:02d}.json'
            record = json.loads(path.read_text()) if path.exists() else {}
            if record.get('complete'):
                assert len(record['history']) == 21
                assert all(math.isfinite(v) for h in record['history'] for v in h.values())
                warm_count += 1
            else:
                result['pending'].append(('warm_diagnostics', mode, seed))
    result['groups']['warm_diagnostics'] = dict(completed=warm_count, planned=15)
    for name in ('numerical_seed01.json', 'numerical_seed02_to05.json',
                 'failure_trace_seed07/trace.json', 'failure_trace_seed07/branches.json'):
        assert json.loads((folder / name).read_text())['complete'], name
    result['complete'] = not result['pending']
    (folder / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not allow_incomplete and result['pending']:
        raise RuntimeError('Some planned outcomes are still missing; see validation.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/plugin_vs_joint_20260917')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    verify(Path(args.input), args.allow_incomplete)
