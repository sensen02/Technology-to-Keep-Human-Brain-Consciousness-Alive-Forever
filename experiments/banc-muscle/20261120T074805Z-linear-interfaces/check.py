"""Verify replay, trainable parameter bounds and sign routing, then plot the recorded body traces."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assay import HERE, MODEL, Interfaces, mj


def check():
    src, replay = HERE / 'verified', HERE / 'replayed'
    count = 0
    for source in sorted(src.glob('*.npz')):
        with np.load(source) as first, np.load(replay / source.name) as second:
            assert first.files == second.files
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])
                count += 1
    assert json.loads((src/'RESULT.json').read_text()) == json.loads((replay/'RESULT.json').read_text())
    interface = Interfaces(mj.MjModel.from_xml_path(str(MODEL)))
    interface.restore(src / 'PARAMETERS.npz')
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / 'probe.npz'
        values = interface.parameters()
        values['exc_fraction'][0:2] = [-.08, .08]
        values['vnc_gain_uS'][0:2] = [.00025, -.00025]
        values['feco_gain'][0] = -.01
        np.savez(path, **values)
        interface.restore(path)
        exc, inh = interface.connectivity(interface.fractions)
        for i in range(2):
            row = np.searchsorted(interface.network[0].indptr, i, side='right') - 1
            col = interface.network[0].indices[i]
            assert exc[row,col] == interface.network[0].data[i] * (1 + values['exc_fraction'][i])
        assert exc[interface.post[0], interface.pre[0]] == .00025
        assert inh[interface.post[1], interface.pre[1]] == .00025
        assert inh[interface.post[0], interface.pre[0]] == 0 and exc[interface.post[1], interface.pre[1]] == 0
        assert interface.feco_gain[0] == -.01
        for invalid in (.080001, -.080001, np.nan):
            values['exc_fraction'][0] = invalid
            np.savez(path, **values)
            try:
                interface.restore(path)
            except ValueError:
                continue
            raise AssertionError('invalid checkpoint accepted')
    print('replayed arrays including checkpoint', count, 'candidate sources / FeCO',
          len(np.unique(interface.pre)), len(interface.feco))
    fig, axes = plt.subplots(2, 2, figsize=(9,5), sharex=True, layout='constrained')
    for j, pose in enumerate(('archived_six_feet','keyframe')):
        for name, label in [('sensory_cut_checkpoint','cut'),('gaps_-8','92%'),('gaps_checkpoint','100%'),('gaps_+8','108%')]:
            with np.load(src / f'{pose}_{name}.npz') as trace:
                t = trace['time'] * 1000
                axes[0,j].plot(t, trace['torso'][:,0] / trace['torso'][0,0], label=label)
                axes[1,j].plot(t, trace['ground_load_w'][:,-1])
        axes[0,j].set_title(('Archived six-foot pose','Keyframe pose')[j])
        axes[0,j].axhline(.9, color='gray', ls=':', lw=1)
        axes[1,j].set_xlabel('Time (ms)')
        axes[1,j].set_xlim(0,60)
    axes[0,0].set_ylabel('COM height / initial height')
    axes[1,0].set_ylabel('Nonfoot load / body weight')
    axes[0,1].legend(fontsize=8)
    fig.savefig(HERE/'CONTACT.png', dpi=145)


if __name__ == '__main__':
    check()
