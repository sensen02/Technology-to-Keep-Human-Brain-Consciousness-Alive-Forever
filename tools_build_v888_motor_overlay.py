"""Turn the BANC v888 type-level resolutions into a motor-mapping OVERLAY for the runner.

`outputs/mn_resolution_v888/mn_resolution_v888.json` lists, per actuator, the motor-neuron
roots whose v888 cell type matches that actuator's muscle.  This tool maps those roots onto
local indices inside the 12,000-neuron tier actually simulated and writes an overlay with
two clearly separated parts:

  * EXACT_MUSCLE_NAME        - the documented cell type equals the actuator's muscle name,
                               so the pool is that muscle's motor pool (the only part that
                               is an ontology-level match).
  * PAIR_LEVEL_CLUSTER_ONLY  - the model's a/b actuators share ONE documented pool.  Driving
                               both clusters with the same pool double counts muscle
                               capacity, so this part is written but must stay DISABLED
                               unless a caller asks for it explicitly and accepts the caveat.

Roots absent from the tier are reported, never substituted.  Nothing here grants runtime
authorisation: the overlay is an input the caller opts into, and the strict entry point still
refuses to run without explicit flags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'outputs/mn_resolution_v888/mn_resolution_v888.json'
OUT = ROOT / 'outputs/motor_overlay_v888'


def digest(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    from muscle_wiring import build_mn_muscle_wiring
    _, _, built = build_mn_muscle_wiring(verbose=False)
    roots = np.asarray(built['data']['root_ids'])
    gi = np.asarray(built['global_index'])
    local_of = {int(roots[g]): i for i, g in enumerate(gi)}

    src = json.loads(SRC.read_text())
    exact, pair, missing = {}, {}, {}
    mn_locals = []
    for row in src['newly_resolved']:
        act = row['actuator']
        have, miss = [], []
        for c in row['candidates']:
            li = local_of.get(int(c['root_id']))
            if li is None:
                miss.append(c['root_id'])
            else:
                have.append(li)
                mn_locals.append(li)
        have = sorted(set(have))
        if row['resolution_level'] == 'EXACT_MUSCLE_NAME':
            exact[act] = have
        else:
            pair[act] = have
        if miss:
            missing[act] = sorted(set(miss))

    report = {
        'status': 'V888_MOTOR_OVERLAY',
        'source': {'resolution': str(SRC), 'sha256': digest(SRC)},
        'tier_fingerprint': built['fingerprint'], 'tier_size': int(built['n']),
        'exact_muscle_name': exact,
        'pair_level_cluster_only': pair,
        'missing_from_tier': missing,
        'counts': {'exact_actuators': len(exact), 'pair_level_actuators': len(pair),
                   'exact_locals': int(sum(len(v) for v in exact.values())),
                   'pair_level_locals': int(sum(len(v) for v in pair.values())),
                   'unique_motor_neurons': int(len(set(mn_locals)))},
        'rules': {
            'exact_muscle_name': 'the documented v888 cell type equals the actuator muscle name; '
                                 'usable as that muscle motor pool without extra assumptions',
            'pair_level_cluster_only': 'REQUIRES an explicit opt-in: the model a/b actuators share one '
                                       'documented pool, so driving both double counts capacity',
            'missing': 'roots absent from the 12000-neuron tier are reported and never substituted'},
        'limitations': ['No individual anatomical validation; type-name-to-muscle is an ontology mapping.',
                        'No cross-specimen identity claim: the v888 annotations cover the same roots this '
                        'project already uses.',
                        'This file grants no runtime authorisation by itself.'],
        'scope': 'mapping overlay only; nothing is simulated here',
    }
    (out / 'motor_overlay_v888.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    c = report['counts']
    ax.bar(['exact\nactuators', 'pair-level\nactuators'],
           [c['exact_actuators'], c['pair_level_actuators']],
           color=['tab:green', 'tab:orange'])
    for i, v in enumerate([c['exact_actuators'], c['pair_level_actuators']]):
        ax.text(i, v + 0.1, str(v), ha='center')
    ax.set_ylabel('actuators')
    ax.set_title('BANC v888 motor overlay: %d exact + %d pair-level (opt-in)\n'
                 'runs from %d unique motor neurons' % (c['exact_actuators'], c['pair_level_actuators'],
                                                        c['unique_motor_neurons']))
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout(); fig.savefig(out / 'motor_overlay_v888.png', dpi=110)

    print(json.dumps(report['counts'], indent=2))
    print('exact:', list(exact), '\npair-level:', list(pair))
    print(out)


if __name__ == '__main__':
    main()
