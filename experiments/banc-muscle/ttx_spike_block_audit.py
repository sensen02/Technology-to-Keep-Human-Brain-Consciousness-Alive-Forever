"""Independently audit and plot the paired virtual spike-block run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('control', 'all_neurons', 'sensory_roots', 'motor_roots')
LABELS = {'control': 'Control', 'all_neurons': 'All point neurons',
          'sensory_roots': 'Selected sensory roots', 'motor_roots': 'Motor pool roots'}
COLORS = {'control': '#1967a0', 'all_neurons': '#b54432',
          'sensory_roots': '#289167', 'motor_roots': '#8e52a6'}
HANDOFF = 5000


def main(path):
    result = json.loads((path / 'RESULT.json').read_text())
    arrays = {}
    for arm in ARMS:
        with np.load(path / f'{arm}.npz', allow_pickle=False) as saved:
            arrays[arm] = {key: saved[key] for key in saved.files}
    first = arrays['control']
    for arm in ARMS[1:]:
        for key in ('qpos', 'qvel', 'activation', 'loads', 'torso', 'depth_mm'):
            np.testing.assert_array_equal(first[key][:HANDOFF + 1], arrays[arm][key][:HANDOFF + 1])
        for key in ('neural_command', 'all_spikes', 'sensory_spikes', 'motor_spikes'):
            np.testing.assert_array_equal(first[key][:HANDOFF // 5], arrays[arm][key][:HANDOFF // 5])
    for key in ('qpos', 'qvel', 'activation', 'loads', 'torso', 'applied_command'):
        np.testing.assert_array_equal(arrays['all_neurons'][key], arrays['motor_roots'][key])
    assert arrays['all_neurons']['all_spikes'][HANDOFF // 5:].sum() == 0
    assert arrays['sensory_roots']['sensory_spikes'][HANDOFF // 5:].sum() == 0
    assert arrays['motor_roots']['motor_spikes'][HANDOFF // 5:].sum() == 0
    assert all(not a['teacher_on'][HANDOFF:].any() for a in arrays.values())
    for arm, a in arrays.items():
        np.testing.assert_allclose(np.diff(a['time_s']), .0001, rtol=0, atol=1e-12)
        assert not any(np.isnan(v).any() for v in a.values())
        assert result['arms'][arm]['post_neural_spikes'] == int(a['all_spikes'][HANDOFF // 5:].sum())
        assert result['arms'][arm]['post_motor_spikes'] == int(a['motor_spikes'][HANDOFF // 5:].sum())
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for arm in ARMS:
        a = arrays[arm]
        t = (a['time_s'][HANDOFF:] - .5) * 1000
        c = COLORS[arm]
        ax[0].plot(t, a['loads'][HANDOFF:, :6].sum(axis=1), color=c, label=LABELS[arm], lw=1.3)
        ax[1].plot(t[:-1], a['applied_command'][HANDOFF:].sum(axis=1), color=c, lw=1.3)
        ax[2].plot(t, a['torso'][HANDOFF:, 0], color=c, lw=1.3)
    ax[0].axhspan(.8, 1.2, color='#e8e8e8', zorder=-1)
    ax[0].set_ylabel('Six-foot load / body weight')
    ax[1].set_ylabel('Sum of 90 muscle controls')
    ax[2].set_ylabel('COM height (mm)')
    ax[2].set_xlabel('Time after neural handoff (ms)')
    ax[0].legend(ncol=2, loc='upper right', fontsize=9)
    for axis in ax:
        axis.grid(alpha=.2)
        axis.set_xlim(0, 250)
    fig.suptitle('Virtual spike-generation block on the same BANC muscle handoff')
    fig.savefig(path / 'PAIRED_RESPONSE.png', dpi=160)
    plt.close(fig)
    audit = {'pre_handoff_physics_exact': True,
             'pre_handoff_neural_exact': True,
             'all_and_motor_body_exact': True,
             'targeted_post_spikes_zero': True,
             'no_teacher_after_handoff': True,
             'control_replay_from_run': result['control_exact_legacy_replay'],
             'plot': str(path / 'PAIRED_RESPONSE.png')}
    (path / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    main(parser.parse_args().result_dir)
