"""Verify replay and render the measured body failure and neural cut traces."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def compare(original, replay):
    count = 0
    with np.load(original, allow_pickle=False) as a, np.load(replay, allow_pickle=False) as b:
        assert a.files == b.files
        for field in a.files:
            np.testing.assert_array_equal(a[field], b[field], err_msg=f'{original.name}/{field}')
            count += 1
    return count


def main():
    replay = HERE / 'replay'
    body_count = sum(compare(path, replay / path.name) for path in sorted(HERE.glob('BODY_*.npz')))
    neural = HERE / 'neural'
    neural_count = sum(compare(path, replay / 'neural' / path.name) for path in sorted(neural.glob('*.npz')))
    for name in ('BODY_OBSERVATIONS.json', 'SEATING.json', 'neural/NEURAL_VERIFICATION.json'):
        assert (HERE / name).read_bytes() == (replay / name).read_bytes()
    assert body_count == 33 and neural_count == 18
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for tag, style, color in (('archived_six_feet', 'Zero command', '#246e68'),
                              ('archived_six_feet_constant_LP', 'Constant muscle input', '#a84a39')):
        with np.load(HERE / f'BODY_{tag}.npz') as a:
            axes[0].plot(a['time'] * 1000, a['ground_load_w'][:, :-1].sum(axis=1), label=style, color=color)
    axes[0].axhline(.8, linewidth=.7, linestyle='--', color='#888888')
    axes[0].set(xlabel='Time (ms)', ylabel='Foot normal force / body weight', title='Same archived six-foot initial pose')
    axes[0].legend(frameon=False, fontsize=8)
    for sign, column, color in (('exc', 1, '#28717c'), ('inh', 2, '#9f433a')):
        with np.load(neural / f'{sign}_restored.npz') as a, np.load(neural / f'{sign}_single_edge_cut.npz') as b:
            difference = a['target_states'][:, column] - b['target_states'][:, column]
            t = .5 * (1 + np.arange(difference.size))
            axes[1].plot(t[t <= 25], difference[t <= 25], color=color, label=sign.upper(), drawstyle='steps-post')
    axes[1].axvline(17, linewidth=.8, linestyle='--', color='#777777')
    axes[1].set(xlabel='Time (ms)', ylabel='Restored minus cut conductance (µS)', title='Two local edge interventions')
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(HERE / 'DIAGNOSTICS.png', dpi=150)
    plt.close(fig)
    (HERE / 'REPLAY_CHECK.json').write_text(json.dumps(dict(body_arrays_exact=body_count,
        neural_arrays_exact=neural_count, metadata_exact=3), indent=2) + '\n')


if __name__ == '__main__':
    main()
