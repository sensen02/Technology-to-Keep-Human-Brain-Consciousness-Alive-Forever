"""Injected vs propagable sensory input.

Read-only. Joins the per-cell sensory currents recorded by
`run_sensory_frontend_balance.py` with the BANC verified-transmitter labels, so a
channel's drive is reported twice: total injected current, and the part injected into
cells that actually have a verified transmitter (and can therefore propagate through
the closure's verified-presynapse filter).
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
FRONT = HERE / 'outputs/model_branches/sensory_frontend_balance/sensory_frontend_balance_traces.npz'
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
OUT = HERE / 'outputs/model_branches/propagable_input'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import pandas as pd
    f = pd.read_feather(FEATHER, columns=['banc_888_id', 'neurotransmitter_verified'])

    def is_verified(v):
        # A missing label arrives as NaN, the empty string, or None; 'nan' is NOT a label.
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        text = str(v).strip()
        return bool(text) and text.lower() not in {'nan', 'none', '<na>', 'null'}

    verified = {}
    for root, v in zip(f['banc_888_id'], f['neurotransmitter_verified']):
        verified[str(root)] = is_verified(v)
    with np.load(FRONT) as z:
        cur = z['cell_current_mean_abs_nA']
        ch = np.asarray([str(x) for x in z['cell_channel']])
        leg = np.asarray([str(x) for x in z['cell_leg']])
        roots = np.asarray([str(x) for x in z['cell_root_id']])
    mask = np.asarray([verified.get(r, False) for r in roots])
    per_channel, per_leg = {}, {}
    for name in sorted(set(ch)):
        sel = ch == name
        total = float(cur[sel].sum())
        prop = float(cur[sel & mask].sum())
        per_channel[name] = dict(cells=int(sel.sum()), verified_cells=int((sel & mask).sum()),
                                 injected_current_nA=total, propagable_current_nA=prop,
                                 propagable_fraction_of_channel=prop / total if total else None)
    for name in sorted(set(leg)):
        sel = leg == name
        total = float(cur[sel].sum())
        prop = float(cur[sel & mask].sum())
        per_leg[name] = dict(cells=int(sel.sum()), verified_cells=int((sel & mask).sum()),
                             injected_current_nA=total, propagable_current_nA=prop)
    grand_inj = float(cur.sum())
    grand_prop = float(cur[mask].sum())
    # Cross-check against the independent per-channel coverage audit.
    reference = json.loads((HERE / 'outputs/model_branches/channel_transmitter_audit/channel_transmitter_audit.json').read_text())
    cross_check = {}
    for name, ref in reference['per_channel'].items():
        got = per_channel[name]['verified_cells']
        cross_check[name] = dict(this_audit=got, channel_audit=ref['verified_nonempty'],
                                 agrees=got == ref['verified_nonempty'])
    assert all(v['agrees'] for v in cross_check.values()), cross_check
    result = dict(
        status='MEASURED_INJECTED_VS_PROPAGABLE_INPUT',
        label='manual read-only evidence-weighted input audit, not a tuning result',
        question='How much of the injected sensory current lands on cells that can actually propagate it?',
        method=dict(definition='propagable = current delivered to a cell whose BANC Verified NT label is non-empty; '
                               'the closure filters synapses on verified presynaptic labels, so an unlabelled cell cannot pass its spikes on',
                    source_trace=str(FRONT), source_trace_sha256=sha(FRONT),
                    feather=str(FEATHER), feather_sha256=sha(FEATHER),
                    note='current is the mean absolute injected current over 2001 samples, summed over cells'),
        totals=dict(injected_current_nA=grand_inj, propagable_current_nA=grand_prop,
                    propagable_share=grand_prop / grand_inj if grand_inj else None,
                    sensory_cells=int(cur.size), verified_sensory_cells=int(mask.sum())),
        per_channel=per_channel, per_leg=per_leg, cross_check_vs_channel_audit=cross_check,
        interpretation=['Most injected current need not be effective: a channel with no verified label is a dead end.',
                        'This is an evidence-weighted measure of drive, not a claim about what the animal senses.'],
        limitations=['One published run, one pose, one tier; no new dynamics.',
                     'Verified-label coverage is a data property of BANC, not a biological statement about transmitter presence.',
                     'No channel gain, reference, or current scale was changed.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'propagable_input.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'cells'} for k, v in per_channel.items()}, indent=2))
    print('totals', json.dumps(result['totals']))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('propagable_input', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
