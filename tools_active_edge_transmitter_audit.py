"""Actual fired-root NT provenance. 手工研究原型，不是成品; no sign filling."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from run_sensory_tuning_annotation_audit import sha, unique_join
from run_touch_transmitter_source_audit import stream_neurons, normalized_nt
from tools_banc_channel_attribute_audit import normalized

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/flybody_loop'


def sign(label):
    if label == 'acetylcholine':
        return 'excitatory'
    if label in ('gaba', 'glutamate'):
        return 'inhibitory'
    return 'excluded'


def main():
    event_path = OUT/'flybody_connectome_event_audit.json'
    events = json.loads(event_path.read_text())
    fired = {x['root_id'] for a in events['arms'] for x in a['spike_events']}
    focus = {x['root_id'] for a in events['arms'] for x in a['focus_motor_cells']}
    targets = fired | focus
    cache = HERE/'data/flywire/banc_connectome.npz'
    csv = HERE/'data/flywire/banc_neurons.csv.gz'
    feather = HERE/'data/banc888/banc_888_meta.feather'
    paths = [event_path, cache, csv, feather, HERE/'engine/neck_cut_data.py', Path(__file__)]
    hashes = {str(p): sha(p) for p in paths}
    with np.load(cache, allow_pickle=False) as a:
        indices = unique_join(a['root_ids'], targets)
        labels = {r: str(a['ann_nt_verified'][i]) for r, i in indices.items()}
    raw, _, _ = stream_neurons(csv, targets)
    df = pd.read_feather(feather, columns=['banc_888_id', 'neurotransmitter_verified', 'neurotransmitter_predicted', 'cell_type'], use_threads=False)
    joined = unique_join(df['banc_888_id'], targets)
    rows = []
    for root in sorted(targets):
        assert len(raw[root]) == 1, 'Missing/duplicate CSV root'
        values = raw[root][0]['values']
        row = df.iloc[joined[root]]
        csv_verified = values['Verified NT type']
        feather_verified = normalized(row['neurotransmitter_verified'])
        predicted = normalized(row['neurotransmitter_predicted'])
        rows.append({'root_id': root, 'fired': root in fired, 'focus_motor': root in focus,
                     'cell_type': normalized(row['cell_type']), 'cached_ann_nt_verified': labels[root],
                     'csv_values_verbatim': values, 'feather_verified': feather_verified,
                     'feather_predicted': predicted, 'runtime_output_sign': sign(labels[root]),
                     'csv_cache_exact_agree': csv_verified == labels[root],
                     'feather_cache_normalized_agree': normalized_nt(feather_verified) == normalized_nt(labels[root]),
                     'predicted_verified_agree': normalized_nt(predicted) == normalized_nt(labels[root]) if predicted and labels[root] else None})
    checks = []
    for arm in events['arms']:
        for edge in arm['diagnostics']['fired_source_to_motor_edges']:
            source = edge['source_root_id']
            checks.append({'arm': arm['arm'], **edge, 'cached_presynaptic_label': labels[source],
                           'matches_runtime_rule': edge['kind'] == sign(labels[source])})
    assert all(x['csv_cache_exact_agree'] for x in rows)
    assert all(x['matches_runtime_rule'] for x in checks)
    assert hashes == {str(p): sha(p) for p in paths}
    result = {'label': '手工研究原型，不是成品', 'status': 'MEASURED_ACTIVE_SOURCE_PROVENANCE_NOT_RECEPTOR_VALIDATION',
              'unique_fired_roots': len(fired), 'roots_audited': len(rows), 'roots': rows,
              'motor_edge_records_checked': len(checks), 'edge_checks': checks,
              'summary': {'fired_cached_labels': dict(Counter(labels[r] or '<missing>' for r in fired)),
                          'csv_cache_mismatches': sum(not r['csv_cache_exact_agree'] for r in rows),
                          'feather_cache_disagreements': sum(not r['feather_cache_normalized_agree'] for r in rows),
                          'edge_sign_mismatches': sum(not e['matches_runtime_rule'] for e in checks)},
              'input_sha256': hashes,
              'limits': ['verified is a dataset field name, not independent experimental verification by this audit.',
                         'Source label determines outgoing edge sign, not incoming drive.',
                         'ACh -> excitation; GABA/glutamate -> inhibition is a model receptor mapping, not target-specific receptor evidence.',
                         'Unknown/mixed/histamine/modulator labels remain excluded; no predicted fallback.',
                         'Motor edge record counts may repeat identical edges across paired arms.',
                         'No source or core model changed; no standing or calibration claim.']}
    dest = OUT/'active_edge_transmitter_audit.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = result['summary']['fired_cached_labels']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(counts), list(counts.values()))
    ax.set_ylabel('Unique fired roots across paired input arms')
    ax.set_title('Cached presynaptic NT labels; no predicted fallback\nMANUAL RESEARCH PROTOTYPE, NOT A PRODUCT')
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(OUT/'active_edge_transmitter_audit.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'roots_audited': len(rows), 'unique_fired_roots': len(fired), 'motor_edge_records_checked': len(checks), **result['summary']}, indent=2))


if __name__ == '__main__':
    main()
