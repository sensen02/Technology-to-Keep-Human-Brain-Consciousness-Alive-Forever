"""Plot the engineering runs using their saved voltage and spike timestamps."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
out = HERE / 'replay'
proof = json.loads((out / 'VERIFICATION.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
for mode, name, color in (('augmented', 'Published edge present', '#276b57'),
                          ('edge_cut', 'One added edge cut', '#9f4e43')):
    with np.load(out / f'leg_campaniform_left_{mode}.npz', allow_pickle=False) as data:
        v = data['target_mV']
        t = (.5 + np.arange(len(v)) * .5)
        view = t <= 30
        axes[0].plot(t[view], v[view], label=name, color=color)
first = proof['edge_interventions'][0]['first_target_response_end_ms']
axes[0].axvline(first, color='#8d8d8d', linestyle='--', linewidth=.8)
axes[0].set(xlabel='Simulated time (ms)', ylabel='Target membrane potential (mV)',
            title='One added foreleg afferent → motor edge')
axes[0].legend(frameon=False, fontsize=8)
response = proof['adaptive_single_cell']['spike_times_ms']
axes[1].eventplot([response['fixed'], response['adaptive']], lineoffsets=[1., 0.], linelengths=.5,
                  colors=['#365681', '#bd662b'])
axes[1].set(xlim=(0, 320), xlabel='Simulated time (ms)', yticks=[0, 1],
            yticklabels=['Adaptive', 'Fixed'], title='Same isolated cell and 45 pA step')
fig.suptitle('BANC circuit and threshold tests with simulated current input')
fig.savefig(out / 'PROBES.png', dpi=150)
