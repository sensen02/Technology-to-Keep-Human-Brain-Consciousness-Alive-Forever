"""Evidence strata for 34 fired roots lacking verified NT. Read-only; no edge repair."""
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


def evidence_state(verified, predicted, csv_predicted, cell_type):
    if verified:
        return 'verified_present'
    if predicted or csv_predicted:
        return 'prediction_only'
    return 'no_transmitter_evidence_in_audited_fields'


def main():
    coverage = json.loads((OUT/'active_output_coverage_audit.json').read_text())
    event = json.loads((OUT/'flybody_connectome_event_audit.json').read_text())
    missing = {r['root_id'] for r in json.loads((OUT/'active_edge_transmitter_audit.json').read_text())['roots']
               if r['fired'] and not r['cached_ann_nt_verified']}
    assert len(missing) == 34
    cache = HERE/'data/flywire/banc_connectome.npz'
    csv = HERE/'data/flywire/banc_neurons.csv.gz'
    feather = HERE/'data/banc888/banc_888_meta.feather'
    paths = [OUT/'active_output_coverage_audit.json', OUT/'active_edge_transmitter_audit.json',
             OUT/'flybody_connectome_event_audit.json', cache, csv, feather, Path(__file__)]
    hashes = {str(x): sha(x) for x in paths}
    with np.load(cache, allow_pickle=False) as a:
        idx = unique_join(a['root_ids'], missing)
        cache_pred = {r: normalized(a['ann_nt_predicted'][i]) if 'ann_nt_predicted' in a else None for r, i in idx.items()}
        # Cache schemas may not contain prediction; explicit null is evidence absence.
    raw, _, _ = stream_neurons(csv, missing)
    d = pd.read_feather(feather, columns=['banc_888_id','cell_type','cell_sub_class','cell_class','super_class',
        'body_part_sensory','body_part_effector','cell_function','cell_function_detailed',
        'neurotransmitter_verified','neurotransmitter_predicted','manc_cell_type'], use_threads=False)
    motor_targets = {e['target_root_id'] for e in coverage['raw_direct_motor_edges']}
    pos = unique_join(d['banc_888_id'], missing | motor_targets)
    target_annotations = {r: {c: normalized(d.iloc[pos[r]][c]) for c in
        ['cell_type', 'cell_sub_class', 'super_class', 'body_part_effector', 'cell_function_detailed']}
        for r in motor_targets}
    rows = []
    for root in sorted(missing):
        row = d.iloc[pos[root]]
        assert len(raw[root]) == 1, 'Missing or duplicate raw CSV root'
        vals = raw[root][0]['values']
        assert not vals['Verified NT type'], 'Raw CSV verified field unexpectedly nonempty'
        verified = normalized(row['neurotransmitter_verified'])
        predicted = normalized(row['neurotransmitter_predicted'])
        csv_pred = normalized_nt(vals.get('Predicted NT type'))
        rows.append({'root_id': root, 'cell_type': normalized(row['cell_type']),
            'cell_sub_class': normalized(row['cell_sub_class']), 'cell_class': normalized(row['cell_class']),
            'super_class': normalized(row['super_class']), 'body_part_sensory': normalized(row['body_part_sensory']),
            'body_part_effector': normalized(row['body_part_effector']), 'cell_function': normalized(row['cell_function']),
            'cell_function_detailed': normalized(row['cell_function_detailed']), 'manc_cell_type': normalized(row['manc_cell_type']),
            'cached_predicted_if_present': cache_pred[root], 'feather_verified': verified,
            'feather_predicted': predicted, 'csv_predicted': csv_pred,
            'evidence_state': evidence_state(verified, predicted, csv_pred, row['cell_type']),
            'prediction_fields_agree': (normalized_nt(predicted) == csv_pred) if predicted and csv_pred else None,
            'direct_motor_targets': []})
    by = {x['root_id']: x for x in rows}
    for e in coverage['raw_direct_motor_edges']:
        by[e['source_root_id']]['direct_motor_targets'].append({**e, 'target_annotation': target_annotations[e['target_root_id']]})
    # Include all output totals from coverage for context.
    totals = {x['root_id']: x for x in coverage['per_root']}
    for x in rows:
        x['output_partition'] = totals[x['root_id']]
    assert hashes == {str(x): sha(x) for x in paths}
    result = {'label':'手工研究原型，不是成品', 'status':'EVIDENCE_STRATA_NOT_NT_REPAIR',
              'roots': rows, 'summary': {'roots': len(rows),
                  'evidence_states': dict(Counter(x['evidence_state'] for x in rows)),
                  'direct_motor_source_roots': sum(bool(x['direct_motor_targets']) for x in rows),
                  'direct_motor_edge_rows': sum(len(x['direct_motor_targets']) for x in rows),
                  'direct_motor_unique_targets': len({e['target_root_id'] for x in rows for e in x['direct_motor_targets']})},
              'limits': ['Prediction-only is not verified evidence and is not used to sign edges.',
                         'Cell type/function is identity context, not a transmitter assay.',
                         'CSV and feather are provenance cross-checks, not independent physiological validation.',
                         'No missing edge was inserted and all direct motor edge signs remain null.',
                         'Counts are specific to the paired LF diagnostic experiment.'], 'input_sha256': hashes}
    dest = OUT/'missing_nt_evidence_audit.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    states = list(result['summary']['evidence_states'])
    vals = [result['summary']['evidence_states'][s] for s in states]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(states, vals)
    ax.set_ylabel('Fired roots lacking verified label')
    ax.set_title('Evidence strata for missing verified transmitter roots\nMANUAL RESEARCH PROTOTYPE, NOT A PRODUCT')
    for i,v in enumerate(vals): ax.text(i,v,str(v),ha='center',va='bottom')
    fig.tight_layout(); fig.savefig(OUT/'missing_nt_evidence_audit.png', dpi=150); plt.close(fig)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
