"""Verify fresh deterministic replays, then summarize coverage and plot measured loads."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def verify_arrays(out, replay):
    for name in ('BODY.json', 'NEURAL.json', 'STATIC.json'):
        assert (out / name).read_bytes() == (replay / name).read_bytes(), name
    files = sorted(out.glob('*.npz'))
    assert len(files) == 18
    compared = 0
    for path in files:
        with np.load(path) as first, np.load(replay / path.name) as second:
            assert first.files == second.files
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key], err_msg=f'{path.name}/{key}')
                compared += 1
    return compared


def make_plot(out, body):
    fig, grid = plt.subplots(2, 2, figsize=(11, 6), sharex=True, constrained_layout=True)
    for col, (pose, title) in enumerate((('keyframe', 'Canonical keyframe'),
                                         ('archived_six_feet', 'Archived six-foot pose'))):
        for mode, linestyle, label in (('native', '-', 'Native floor'),
                                        ('zero_floor_control', '--', 'Zero floor')):
            with np.load(out / f'{mode}_{pose}.npz') as data:
                t = data['time'] * 1000
                ground = data['ground_load_w']
                grid[0, col].plot(t, ground[:, :-1].sum(axis=1), linestyle, color='#216750',
                                 label=f'Feet · {label}')
                grid[0, col].plot(t, ground[:, -1], linestyle, color='#973c27',
                                 label=f'Non-foot · {label}')
                grid[1, col].plot(t, data['torso'][:, 0], linestyle, color='#294a87', label=label)
        grid[0, col].axhspan(.8, 1.2, color='#216750', alpha=.09)
        grid[0, col].set_title(title)
        grid[1, col].set_xlabel('Time (ms)')
        grid[0, col].axvline(100, linewidth=.8, color='#777777')
        grid[1, col].axvline(100, linewidth=.8, color='#777777')
        first = body['trials']['native_' + pose]['first_nonfoot_s']
        if first is not None:
            grid[0, col].axvline(first * 1000, linewidth=.8, color='#973c27', alpha=.65)
    grid[0, 0].set_ylabel('Ground normal force / body weight')
    grid[1, 0].set_ylabel('Whole-body COM height (mm)')
    grid[0, 1].legend(loc='upper right', fontsize=8)
    fig.suptitle('Canonical muscle body · zero applied command · native samples')
    fig.savefig(out / 'BODY_SUPPORT.png', dpi=150)
    plt.close(fig)


def main(out, replay):
    compared = verify_arrays(out, replay)
    body = json.loads((out / 'BODY.json').read_text())
    neural = json.loads((out / 'NEURAL.json').read_text())
    wiring = json.loads((out / 'WIRING.json').read_text())
    rows = wiring['motor_manifest']['rows']
    active_by_probe = {}
    indices = {i for row in rows for i in row['local_indices']}
    for label in neural['trials']:
        with np.load(out / f'neural_{label}.npz') as data:
            counts = data['spike_count']
        active_by_probe[label] = dict(
            spiking_mapped_cells=sum(bool(counts[i]) for i in indices),
            actuator_pools_with_spikes=sum(bool(counts[row['local_indices']].sum()) for row in rows))
        for row in rows:
            row.setdefault('spikes_by_neural_probe', {})[label] = int(counts[row['local_indices']].sum())
    reachability = {}
    for level in ('raw_cache', 'unsigned_tier', 'modeled_tier'):
        key = f'minimum_edges_{level}'
        reachability[level] = dict(
            same_leg=sum(row[key][row['leg']] is not None for row in rows),
            descending=sum(row[key]['crossing_descending'] is not None for row in rows))
    static = json.loads((out / 'STATIC.json').read_text())
    summary = dict(static_diagnostics=static, verification='array_exact', compared_arrays=compared,
                   body_runs=len(body['trials']), standing_runs=sum(x['standing'] for x in body['trials'].values()),
                   native_physics_steps=sum(x['steps'] for x in body['trials'].values()),
                   neural_probe_runs=len(neural['trials']), neural_ticks=neural['ticks_per_trial'] * len(neural['trials']),
                   muscle_pool_status=dict(Counter(row['status'] for row in rows)),
                   muscle_pool_reachability=reachability, muscle_pools_active_by_probe=active_by_probe)
    for row in rows:
        row['muscle_transmission'] = {pose: next(m for m in probe['muscles'] if m['actuator'] == row['actuator'])
                                      for pose, probe in body['muscle_probes'].items()}
    for name, content in (('SUMMARY.json', summary), ('MUSCLE_CHAIN.json', rows)):
        (out / name).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    make_plot(out, body)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('replay', type=Path)
    args = parser.parse_args()
    main(args.output, args.replay)
