"""Unique v888 annotation join and event accounting. 手工研究原型，不是成品."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from run_sensory_tuning_annotation_audit import unique_join, sha
from tools_banc_channel_attribute_audit import normalized

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/flybody_loop'


def main():
    source = OUT/'flybody_connectome_event_audit.json'
    result = json.loads(source.read_text())
    targets = {x['root_id'] for a in result['arms'] for x in a['diagnostics']['fired_nonsensory_cells']}
    targets.update(x['root_id'] for a in result['arms'] for x in a['focus_motor_cells'])
    targets.update(e['source_root_id'] for a in result['arms'] for f in a['focus_motor_cells'] for e in f['incoming_fired_edges'])
    meta = HERE/'data/banc888/banc_888_meta.feather'
    before = sha(meta)
    df = pd.read_feather(meta, use_threads=False)
    positions = unique_join(df['banc_888_id'], targets)
    columns = ['cell_type', 'cell_sub_class', 'cell_class', 'super_class', 'side', 'nerve',
               'body_part_sensory', 'body_part_effector', 'cell_function', 'cell_function_detailed',
               'neurotransmitter_predicted', 'neurotransmitter_verified', 'manc_cell_type']
    annotations = {root: {c: normalized(df.iloc[i][c]) for c in columns} for root, i in positions.items()}
    assert before == sha(meta)
    summaries = []
    for arm in result['arms']:
        event_counts = Counter(e['root_id'] for e in arm['spike_events'])
        assert sum(event_counts.values()) == arm['spike_totals']['all']
        for x in arm['diagnostics']['fired_nonsensory_cells']:
            assert event_counts[x['root_id']] == x['spikes']
        focus = []
        for f in arm['focus_motor_cells']:
            rows = f['trace']
            diag = next(x for x in arm['diagnostics']['motor_cells'] if x['root_id'] == f['root_id'])
            peak = max(-65., max(x['voltage_mV'] for x in rows))
            assert abs(peak-diag['peak_post_step_voltage_mV']) < 1e-10
            excitatory_events = [x['time_start_ms'] for x in rows if x['arriving_exc_uS'] > 0]
            inhibitory_events = [x['time_start_ms'] for x in rows if x['arriving_inh_uS'] > 0]
            focus.append({'root_id': f['root_id'], 'annotation': annotations[f['root_id']],
                          'peak_voltage_mV': peak, 'peak_end_ms': max(rows, key=lambda x: x['voltage_mV'])['time_end_ms'],
                          'excitatory_arrival_start_ms': excitatory_events,
                          'inhibitory_arrival_start_ms': inhibitory_events,
                          'incoming_fired_edges': f['incoming_fired_edges']})
        summaries.append({'arm': arm['arm'], 'focus': focus,
                          'fired_nonsensory_annotations': [{**x, 'annotation': annotations[x['root_id']]}
                                                       for x in arm['diagnostics']['fired_nonsensory_cells']]})
    output = {'label': '手工研究原型，不是成品', 'join_key': 'banc_888_id',
              'joined_unique_roots': len(positions), 'annotations': annotations, 'arms': summaries,
              'input_sha256': {str(source): sha(source), str(meta): before, str(Path(__file__)): sha(__file__)},
              'limits': ['Annotation labels do not establish causal function or verified transmitter.',
                         'Arrival times are interval START; voltages are interval END.',
                         'Private ring slot inspected read-only for diagnostics; core network unmodified.',
                         'Five highest post-step voltage peaks per arm; ties retain motor index order.',
                         'No claim of standing, natural tuning, or that increased input would solve the loop.']}
    (OUT/'flybody_connectome_event_annotations.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, arm in enumerate(result['arms'][1:]):
        f = arm['focus_motor_cells'][0]
        rows = f['trace']
        axes[0, col].plot([x['time_end_ms'] for x in rows], [x['voltage_mV'] for x in rows])
        axes[0, col].axhline(-50, color='red', linestyle='--', label='Threshold')
        axes[0, col].set_title(arm['arm']+'\nMotor root '+f['root_id'])
        axes[0, col].set_ylabel('Post-step voltage (mV)')
        for key, label in [('arriving_exc_uS', 'Excitatory'), ('arriving_inh_uS', 'Inhibitory')]:
            axes[1, col].step([x['time_start_ms'] for x in rows], [x[key] for x in rows], where='post', label=label)
        axes[1, col].set_ylabel('Arrival increment (uS)')
        axes[1, col].set_xlabel('Time (ms)')
        axes[1, col].legend()
    fig.suptitle('Focus motor event timing; 20 ms input, 100 ms observation\nMANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; no parameter changes')
    fig.tight_layout()
    fig.savefig(OUT/'flybody_focus_motor_events.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'joined_unique_roots': len(positions), 'focus': [{ 'arm': a['arm'], **{k: v for k, v in a['focus'][0].items() if k != 'incoming_fired_edges'}} for a in summaries]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
