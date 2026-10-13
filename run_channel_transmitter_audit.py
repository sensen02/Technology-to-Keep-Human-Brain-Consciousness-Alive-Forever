"""Per-channel transmitter-evidence audit for the current closure's sensory cells.

Read-only. For every sensory channel actually used by the published run, count how
many of its cells have a Verified NT label in the BANC feather, and how many are
blank. No label is invented, filled, or interpreted.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, signal, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
OUT = HERE / 'outputs/model_branches/channel_transmitter_audit'
LOOKUP = HERE / 'outputs/neural_rebuild/sensory_channel_lookup.json'
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
CURRENT = HERE / 'outputs/neural_muscle_closure/current_run.json'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import pandas as pd
    feather = pd.read_feather(FEATHER, columns=['banc_888_id', 'cell_class', 'cell_sub_class',
                                                'neurotransmitter_verified', 'neurotransmitter_predicted'])
    current = json.loads(CURRENT.read_text())
    run = HERE / 'outputs/neural_muscle_closure' / current['path']
    identity = json.loads((run / 'neural_identity.json').read_text())
    tier_roots = [str(x) for x in identity['tier_root_ids']]
    tier_set = set(tier_roots)
    lookup = json.loads(LOOKUP.read_text())
    by_root = lookup['leg_channel_by_root']
    verified = ['' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
                for v in feather['neurotransmitter_verified']]
    predicted = ['' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
                 for v in feather['neurotransmitter_predicted']]
    ids = [str(x) for x in feather['banc_888_id']]
    rows = dict(zip(ids, zip(verified, predicted)))
    channels = sorted({v[1] for v in by_root.values()})
    per_channel = {}
    for ch in channels:
        roots = [r for r, v in by_root.items() if v[1] == ch and r in tier_set]
        stat = dict(tier_cells=len(roots), verified_nonempty=0, verified_empty=0, absent_from_feather=0,
                    verified_values={}, predicted_only_nonempty=0)
        for r in roots:
            if r not in rows:
                stat['absent_from_feather'] += 1
                continue
            v, p = rows[r]
            if v:
                stat['verified_nonempty'] += 1
                stat['verified_values'][v] = stat['verified_values'].get(v, 0) + 1
            else:
                stat['verified_empty'] += 1
                if p:
                    stat['predicted_only_nonempty'] += 1
        stat['verified_fraction'] = stat['verified_nonempty'] / stat['tier_cells'] if stat['tier_cells'] else None
        per_channel[ch] = stat
    result = dict(
        status='MEASURED_PER_CHANNEL_VERIFIED_TRANSMITTER_COVERAGE',
        label='manual read-only evidence-coverage audit, not a transmitter assignment',
        question='For each sensory channel the published run actually uses, how many cells have a Verified NT label?',
        scope=dict(run_id=current['run_id'], tier_neurons=len(tier_roots),
                   channels=channels, total_channel_cells=int(sum(s['tier_cells'] for s in per_channel.values())),
                   lookup=str(LOOKUP), lookup_sha256=sha(LOOKUP),
                   feather=str(FEATHER), feather_sha256=sha(FEATHER),
                   identity=str(run / 'neural_identity.json')),
        per_channel=per_channel,
        interpretation=['An empty Verified NT cell is not evidence of "no transmitter"; it is absence of a verified label.',
                        'Predicted-only labels are counted separately and were never promoted to Verified.',
                        'The closure\'s synaptic filter acts on verified presynaptic labels, so a channel with near-zero coverage cannot drive the network through that filter.'],
        limitations=['Counts are for the published run tier; other tiers/datasets are not implied.',
                     'cell_class/sub_class of the leg cells are not used to alter any label.',
                     'No polarity, sign, tuning, or segment identity is inferred anywhere.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'channel_transmitter_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(per_channel, indent=2))
    return result


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s channel-audit deadline')))
    signal.alarm(60)
    with closure.mg.guarded_run('channel_transmitter_audit', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
    signal.alarm(0)
