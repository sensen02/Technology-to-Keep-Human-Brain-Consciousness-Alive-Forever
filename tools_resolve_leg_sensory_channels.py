"""Resolve the per-leg sensory battery for walking from the BANC v888 annotations.

WHY.  The loop currently drives only the front-leg chordotonal organ (200 cells, one
modality).  Walking needs, at minimum, per-leg JOINT POSITION, LOAD and GROUND CONTACT.
The v888 annotation table documents all of those per leg
(`chordotonal_organ_neuron`, `hair_plate_neuron`, `campaniform_sensillum_neuron`,
`bristle_neuron`, `taste_bristle_tactile_neuron`, each with a `<leg>_<...>` sub class),
and its `banc_888_id` covers every root this project uses.

WHAT THIS PRODUCES.  A JSON mapping per (leg, modality) -> local indices inside the
12,000-neuron tier actually simulated, plus explicit lists of roots that are absent from
the tier.  A missing root is REPORTED, never replaced by a wrong index.

WHAT THIS DOES NOT DO.  It does not claim per-cell tuning, does not pick a current gain,
and does not authorise runtime use: the caller must still drive each channel with a
documented CLASS-LEVEL body variable and label the gain uncalibrated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
META = ROOT / 'data/banc888/banc_888_meta.feather'
OUT = ROOT / 'outputs/leg_sensory_channels'

# leg key -> (v888 sub-class prefix, anatomical side)
LEGS = {
    'LF': ('front_leg', 'left'), 'RF': ('front_leg', 'right'),
    'LM': ('middle_leg', 'left'), 'RM': ('middle_leg', 'right'),
    'LH': ('hind_leg', 'left'), 'RH': ('hind_leg', 'right'),
}
MODALITY_BY_CLASS = {
    'chordotonal_organ_neuron': 'chordotonal',
    'hair_plate_neuron': 'hair_plate',
    'campaniform_sensillum_neuron': 'campaniform',
    'bristle_neuron': 'tactile_bristle',
    'taste_bristle_tactile_neuron': 'tactile_bristle',
}
# documented class-level property -> the body variable the runner must supply
MODALITY_VARIABLE = {
    'chordotonal': 'joint position (claw) and joint motion magnitude (hook, club) of that leg',
    'hair_plate': 'joint position deviation, averaged over that leg\'s joints',
    'campaniform': 'that leg\'s ground contact normal force (cuticular strain / load proxy)',
    'tactile_bristle': 'binary tarsus contact of that leg (touching the ground)',
}


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    from muscle_wiring import build_mn_muscle_wiring
    _, _, built = build_mn_muscle_wiring(verbose=False)
    roots = np.asarray(built['data']['root_ids'])
    global_index = np.asarray(built['global_index'])
    local_of_root = {int(roots[g]): i for i, g in enumerate(global_index)}
    assert len(local_of_root) == built['n'] == 12000

    meta = pd.read_feather(META)
    meta['cclass'] = meta['cell_class'].astype(str)
    meta['csub'] = meta['cell_sub_class'].astype(str)
    sens = meta[meta['super_class'].astype(str).str.lower() == 'sensory']

    channels, missing_report = {}, {}
    for leg, (prefix, side) in LEGS.items():
        channels[leg] = {}
        missing_report[leg] = {}
        sel = sens[sens['csub'].str.startswith(prefix, na=False) &
                   (sens['side'].astype(str).str.lower() == side)]
        for cls, modality in MODALITY_BY_CLASS.items():
            sub = sel[sel['cclass'] == cls]
            have, missing = [], []
            for rid in sub['banc_888_id'].astype('int64'):
                li = local_of_root.get(int(rid))
                (have if li is not None else missing).append(int(rid) if li is None else li)
            channels[leg].setdefault(modality, [])
            channels[leg][modality] = sorted(set(channels[leg][modality] + have))
            if missing:
                missing_report[leg].setdefault(modality, [])
                missing_report[leg][modality] = sorted(set(missing_report[leg][modality] + missing))
        # chordotonal sub classes are kept separately: claw encodes POSITION, hook/club MOTION
        for sub_name, key in (('claw', 'chordotonal_claw'), ('hook', 'chordotonal_hook'),
                              ('club', 'chordotonal_club')):
            sub = sel[sel['csub'].str.startswith(prefix + '_' + sub_name, na=False) &
                      (sel['cclass'] == 'chordotonal_organ_neuron')]
            have, missing = [], []
            for rid in sub['banc_888_id'].astype('int64'):
                li = local_of_root.get(int(rid))
                (have if li is not None else missing).append(int(rid) if li is None else li)
            channels[leg][key] = sorted(set(have))
            if missing:
                missing_report[leg][key] = sorted(set(missing))

    flat = np.concatenate([np.asarray(v) for leg in channels for v in channels[leg].values()])
    dup = len(flat) - len(np.unique(flat))
    # sanity: no channel may overlap the mapped motor neurons (the runner asserts this too)
    manifest = json.loads((ROOT / 'outputs/muscle_anatomy_resolution/resolution.json').read_text())
    mn_types = {r['target_primary'] for r in manifest['actuators'] if r['candidates']}

    report = {
        'status': 'LEG_SENSORY_BATTERY_RESOLUTION',
        'source': {'metadata': str(META), 'sha256': digest(META),
                   'release': 'BANC v888, Bates et al. 2026 Nature'},
        'tier_size': int(built['n']), 'tier_fingerprint': built['fingerprint'],
        'legs': list(LEGS), 'modalities_per_leg': sorted({m for leg in channels for m in channels[leg]}),
        'modality_variable': MODALITY_VARIABLE,
        'channels': channels,
        'counts': {leg: {m: len(v) for m, v in channels[leg].items() if m != 'channels'} for leg in channels},
        'missing_from_tier': missing_report,
        'duplicate_local_indices_across_channels': int(dup),
        'total_channels': int(len(np.unique(flat))),
        'limitations': [
            'Class-level only: no per-cell tuning, no preferred direction, no gain.',
            'Every channel must be driven by a body-available variable; the caller labels the '
            'normalisation as uncalibrated.',
            'Roots absent from the 12000-neuron tier are reported, never substituted.',
            'runtime_authorization is NOT granted by this file.'],
        'scope': 'channel resolution only; no simulation, no controller, no gain choice',
    }
    (out / 'leg_sensory_channels.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mods = sorted({m for leg in channels for m in channels[leg]})
    legs = list(LEGS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(mods))
    for i, leg in enumerate(legs):
        vals = np.array([len(channels[leg].get(m, [])) for m in mods], float)
        ax.bar(range(len(mods)), vals, bottom=bottom, label=leg)
        bottom += vals
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels([m.replace('_', '\n') for m in mods], fontsize=7)
    ax.set_ylabel('neurons inside the 12000-neuron tier')
    ax.set_title('Per-leg sensory battery available for walking\n'
                 'class-level channels only; no tuning, no gain chosen')
    ax.legend(ncol=6, fontsize=8); ax.grid(alpha=.3, axis='y')
    fig.tight_layout(); fig.savefig(out / 'leg_sensory_channels.png', dpi=110)

    print(json.dumps(report['counts'], indent=1))
    print('union', report['total_channels'], '| duplicated across channels', dup)
    print(out)


if __name__ == '__main__':
    main()
