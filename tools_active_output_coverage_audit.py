"""Unsigned raw output accounting only. 手工研究原型，不是成品."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
from pathlib import Path
import numpy as np
from run_sensory_tuning_annotation_audit import sha, unique_join
from engine import memory_guard as mg

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/flybody_loop'


def totals(mask, syn):
    return {'rows': int(np.count_nonzero(mask)), 'synapses': float(syn[mask].sum(dtype=np.float64))}


def main():
    source = OUT/'active_edge_transmitter_audit.json'
    events = OUT/'flybody_connectome_event_audit.json'
    pointer = HERE/'outputs/neural_muscle_closure/current_run.json'
    p = json.loads(pointer.read_text())
    bundle = pointer.parent/p['path']
    identity_path, manifest_path = bundle/'neural_identity.json', bundle/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == p['manifest_sha256']
    assert sha(identity_path) == manifest['files']['neural_identity.json']['sha256']
    identity = json.loads(identity_path.read_text())
    active = json.loads(source.read_text())
    event_data = json.loads(events.read_text())
    missing = {r['root_id'] for r in active['roots'] if r['fired'] and not r['cached_ann_nt_verified']}
    assert len(missing) == 34
    tier = set(map(str, identity['tier_root_ids']))
    motor = {x['root_id'] for x in event_data['arms'][0]['diagnostics']['motor_cells']}
    assert len(motor) == 202 and motor <= tier
    cache = HERE/'data/flywire/banc_connectome.npz'
    paths = [source, events, pointer, identity_path, manifest_path, cache, Path(__file__)]
    hashes = {str(x): sha(x) for x in paths}
    with np.load(cache, allow_pickle=False) as a:
        roots = a['root_ids']
        joined = unique_join(roots, missing)
        verified = a['ann_nt_verified']
        assert all(str(verified[i]) == '' for i in joined.values())
        pre, post, syn = a['pre'], a['post'], a['syn']
        inside = np.isin(roots, np.asarray(list(tier), dtype=roots.dtype))
        motor_mask = np.isin(roots, np.asarray(list(motor), dtype=roots.dtype))
        rows = []
        direct_edges = []
        for root in sorted(missing):
            selected = pre == joined[root]
            in_tier = selected & inside[post]
            outside = selected & ~inside[post]
            to_motor = in_tier & motor_mask[post]
            to_other = in_tier & ~motor_mask[post]
            r = {'root_id': root, 'all_outputs': totals(selected, syn),
                 'outside_tier': totals(outside, syn), 'inside_tier_excluded_missing_verified': totals(in_tier, syn),
                 'inside_to_motor': totals(to_motor, syn), 'inside_to_nonmotor': totals(to_other, syn)}
            for key in ('rows', 'synapses'):
                assert np.isclose(r['all_outputs'][key], r['outside_tier'][key]+r['inside_tier_excluded_missing_verified'][key])
                assert np.isclose(r['inside_tier_excluded_missing_verified'][key], r['inside_to_motor'][key]+r['inside_to_nonmotor'][key])
            rows.append(r)
            for index in np.flatnonzero(to_motor):
                direct_edges.append({'source_root_id': root, 'target_root_id': str(roots[int(post[index])]),
                                     'raw_synapses': float(syn[index]), 'raw_row_index': int(index),
                                     'sign': None, 'modelled': False})
    keys = ['all_outputs', 'outside_tier', 'inside_tier_excluded_missing_verified', 'inside_to_motor', 'inside_to_nonmotor']
    summary = {k: {q: sum(r[k][q] for r in rows) for q in ('rows', 'synapses')} for k in keys}
    assert hashes == {str(x): sha(x) for x in paths}
    result = {'label': '手工研究原型，不是成品', 'status': 'RAW_UNSIGNED_OUTPUT_COVERAGE_NOT_SIGN_REPAIR',
              'missing_verified_fired_roots': len(missing), 'tier_neurons': len(tier), 'motor_cells': len(motor),
              'summary': summary, 'per_root': rows, 'raw_direct_motor_edges': direct_edges,
              'unique_direct_motor_targets': len({e['target_root_id'] for e in direct_edges}),
              'input_sha256': hashes,
              'limits': ['Raw cache rows and synapse counts, before per-edge cap or conductance scaling.',
                         'Outside-tier and inside-tier categories are disjoint; outside targets may also lack verified labels.',
                         'Direct motor rows are a subset of inside-tier excluded rows, not an additional loss category.',
                         'No sign or receptor action inferred from unsigned counts or predicted NT.',
                         'No missing edges were inserted; no counterfactual recovery or standing claim.']}
    (OUT/'active_output_coverage_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = ['Outside tier', 'Inside nonmotor\nmissing verified', 'Inside motor\nmissing verified']
    for ax, q in zip(axes, ('rows', 'synapses')):
        vals = [summary[k][q] for k in ('outside_tier', 'inside_to_nonmotor', 'inside_to_motor')]
        ax.bar(labels, vals)
        ax.set_ylabel('Raw '+q)
        for i, v in enumerate(vals):
            ax.text(i, v, str(int(v)), ha='center', va='bottom')
        ax.tick_params(axis='x', labelsize=9)
    fig.suptitle('Outputs of 34 fired roots lacking verified NT\nMANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; unsigned raw counts')
    fig.tight_layout()
    fig.savefig(OUT/'active_output_coverage_audit.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'summary': summary, 'unique_direct_motor_targets': result['unique_direct_motor_targets']}, indent=2))


if __name__ == '__main__':
    with mg.guarded_run('active_output_coverage', need_mb=2200., cap_mb=4608.):
        main()
