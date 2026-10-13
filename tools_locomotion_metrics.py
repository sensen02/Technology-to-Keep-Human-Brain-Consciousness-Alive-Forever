"""Locomotion metrics for the ground closed loop: measure walking, do not assume it.

Reads a whole-body run produced by diagnostics/diag_autonomous_sensorimotor.py and
reports the six metrics that define "walking" for this project:

  upright_fraction      fraction of the run with the torso axis close to vertical
  distance_traveled_mm  horizontal displacement of the root
  tripod_index          overlap of the alternating-tripod stance sets
                        {LF,LH,RM} vs {RF,RH,LM}; 0 = never alternating, 1 = perfect
                        alternation, negative = the two sets are co-active
  tarsus_only_fraction  fraction of intervals whose ground contact set is tarsus-only
  six_feet_fraction     fraction of intervals with all six feet on the ground
  active_control        every interval's muscle command came from the spike translation

WHY THIS EXISTS FIRST.  Without a fixed metric, "it looks like it walks" is unfalsifiable
and thresholds get chosen after the fact.  The thresholds are frozen in
thresholds.json by this tool on first use and are PROJECT CONVENTIONS, not biological
standards; the tool never tunes anything and never drives a body.

It also prints the same metrics for every cut/intervention branch of the run, so a
metric that improves only because a branch changed cannot be mistaken for walking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = {
    'upright_fraction': 0.8,
    'distance_traveled_mm': 2.0,      # about one body length for this model
    'tripod_index': 0.25,
    'tripod_index_note': 'signed: negative means the alternating sets are co-active (anti-alternation)',
    'tarsus_only_fraction': 0.9,
    'six_feet_fraction': 0.0,         # reported only, no pass line
    'upright_up_z_min': 0.9,
    'note': 'PROJECT CONVENTIONS frozen before any tuning, not biological standards',
}


def metrics_for(block, thresholds):
    body = block['body_per_step']
    t = np.array([b['time_s'] for b in body])
    up = np.array([b.get('torso_up_z', np.nan) for b in body])
    x = np.array([b.get('root_x_mm', np.nan) for b in body])
    y = np.array([b.get('root_y_mm', np.nan) for b in body])
    legs = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
    foot = {L: np.array([bool(b.get('foot_contact', {}).get(L, False)) for b in body]) for L in legs}

    half = t >= 0.5 * t[-1]
    upright = up[half] >= thresholds['upright_up_z_min']
    upright_fraction = float(np.mean(upright)) if upright.size else float('nan')

    # the first body record is the pre-step state and may lack the positional fields, so use
    # the first finite sample rather than index 0
    fin = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if fin.size >= 2:
        i0 = fin[0]
        dist = float(np.hypot(x[-1] - x[i0], y[-1] - y[i0]))
    else:
        dist = float('nan')

    setA = foot['LF'] & foot['LH'] & foot['RM']
    setB = foot['RF'] & foot['RH'] & foot['LM']
    onlyA = float(np.mean(setA[half] & ~setB[half])) if half.any() else float('nan')
    onlyB = float(np.mean(setB[half] & ~setA[half])) if half.any() else float('nan')
    both = float(np.mean(setA[half] & setB[half])) if half.any() else float('nan')
    neither = float(np.mean(~setA[half] & ~setB[half])) if half.any() else float('nan')
    tripod_index = onlyA + onlyB - 2.0 * both

    recs = [r['contacts'] for r in block['commands'] if r.get('contacts')]
    tr = np.array([r['time_s'] for r in block['commands'] if r.get('contacts')])
    if recs:
        late = tr >= 0.5 * tr[-1]
        late_recs = [r for r, m in zip(recs, late) if m] or recs
        tarsus_only_fraction = float(np.mean([r['tarsus_only'] for r in late_recs]))
        six_feet_fraction = float(np.mean([len(r['legs_with_foot_contact']) == 6 for r in late_recs]))
    else:
        tarsus_only_fraction = six_feet_fraction = float('nan')

    active_control = bool(all(r['command_origin_audit']['status'] == 'PASS'
                              for r in block['commands']))

    out = {'upright_fraction': upright_fraction,
           'distance_traveled_mm': dist,
           'tripod_index': float(tripod_index),
           'tripod_only_A_fraction': onlyA, 'tripod_only_B_fraction': onlyB,
           'tripod_both_fraction': both, 'tripod_neither_fraction': neither,
           'tarsus_only_fraction': tarsus_only_fraction,
           'six_feet_fraction': six_feet_fraction,
           'active_control': active_control,
           'final_height_mm': float(body[-1]['height_mm']),
           'initial_height_mm': float(body[0]['height_mm'])}
    checks = {'upright_fraction': out['upright_fraction'] >= thresholds['upright_fraction'],
              'distance_traveled_mm': (np.isfinite(out['distance_traveled_mm'])
                                       and out['distance_traveled_mm'] >= thresholds['distance_traveled_mm']),
              # SIGNED on purpose: a NEGATIVE index means the two tripod sets are co-active,
              # which is anti-alternation, so it must not pass through an absolute value.
              'tripod_index': out['tripod_index'] >= thresholds['tripod_index'],
              'tarsus_only_fraction': out['tarsus_only_fraction'] >= thresholds['tarsus_only_fraction'],
              'active_control': out['active_control']}
    out['walking_criteria'] = checks
    out['walks'] = bool(all(checks.values()))
    return out, (t, x, y, up, foot)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='exploratory.json from a whole-body run')
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    run = args.run.resolve()
    out = (args.out_dir.resolve() if args.out_dir else run.parent)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or run.parent.name

    thr_path = ROOT / 'outputs/locomotion_metrics/thresholds.json'
    thr_path.parent.mkdir(parents=True, exist_ok=True)
    if thr_path.exists():
        thresholds = json.loads(thr_path.read_text())
        frozen = 'reused the thresholds frozen earlier'
    else:
        thresholds = dict(DEFAULT_THRESHOLDS)
        thr_path.write_text(json.dumps(thresholds, indent=2))
        frozen = 'FROZE the thresholds now; they must not be edited after seeing results'

    report = json.loads(run.read_text())
    summary = {'label': label, 'run': str(run), 'thresholds': thresholds,
               'threshold_provenance': frozen, 'body_model': report.get('body_model'),
               'sensory_mode': report.get('sensory_mode'), 'cases': {}}
    plot_data = {}
    for case, block in report['cases'].items():
        m, series = metrics_for(block, thresholds)
        summary['cases'][case] = m
        plot_data[case] = series
    summary['walks'] = bool(summary['cases'].get('intact', {}).get('walks', False))
    summary['scope'] = ('measurement only; no controller, no target, no tuning. Thresholds are '
                        'project conventions frozen before tuning, not biological standards.')

    (out / f'locomotion_metrics_{label}.json').write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for case, (t, x, y, up, foot) in plot_data.items():
        axs[0, 0].plot(t, up, label=case, lw=1)
        axs[0, 1].plot(x, y, label=case, lw=1)
        axs[1, 1].plot(t, x - x[0], label=case, lw=1)
    axs[0, 0].axhline(thresholds['upright_up_z_min'], ls='--', c='gray')
    axs[0, 0].set_xlabel('time (s)'); axs[0, 0].set_ylabel('torso up . world z')
    axs[0, 0].set_title('Upright'); axs[0, 0].legend(fontsize=7); axs[0, 0].grid(alpha=.3)
    axs[0, 1].set_xlabel('x (mm)'); axs[0, 1].set_ylabel('y (mm)')
    axs[0, 1].set_title('Horizontal path'); axs[0, 1].legend(fontsize=7); axs[0, 1].grid(alpha=.3)
    axs[1, 1].set_xlabel('time (s)'); axs[1, 1].set_ylabel('x displacement (mm)')
    axs[1, 1].set_title('Distance traveled'); axs[1, 1].legend(fontsize=7); axs[1, 1].grid(alpha=.3)

    t, x, y, up, foot = plot_data['intact']
    legs = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
    M = np.array([[1.0 if foot[L][i] else 0.0 for L in legs] for i in range(len(t))]).T
    axs[1, 0].imshow(M, aspect='auto', cmap='Greys', interpolation='nearest',
                     extent=[t[0], t[-1], len(legs) - .5, -.5])
    axs[1, 0].set_yticks(range(len(legs))); axs[1, 0].set_yticklabels(legs, fontsize=8)
    axs[1, 0].set_xlabel('time (s)'); axs[1, 0].set_title('Foot contact (intact): black = down')
    fig.suptitle(f'Locomotion metrics: {label}   walks={summary["walks"]}   '
                 '(measurement only, no controller)')
    fig.tight_layout(); fig.savefig(out / f'locomotion_metrics_{label}.png', dpi=110)

    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items() if kk in
                          ('upright_fraction', 'distance_traveled_mm', 'tripod_index',
                           'tarsus_only_fraction', 'six_feet_fraction', 'active_control', 'walks')}
                      for k, v in summary['cases'].items()}, indent=2))
    print('walks:', summary['walks'], '| thresholds:', thr_path)
    print(out)


if __name__ == '__main__':
    main()
