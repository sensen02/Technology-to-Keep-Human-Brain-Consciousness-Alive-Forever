"""Round13 summary figure from saved results; zero simulation steps."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-wide-module/20261107T233459Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/banc-wide-module/20261107T233459Z')
r = json.loads((SRC / 'RESULT.json').read_text()); l = json.loads((SRC / 'RESULT_LATERAL.json').read_text()); g = json.loads((SRC / 'RESULT_GAIN.json').read_text())
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8), layout='constrained')
tags = ['frozen20k', 'banc_strict']
x = np.arange(len(tags)); w = .35
ax[0].bar(x - w / 2, [r['results'][t]['tripod_alternation_score'] for t in tags], w, label='tripod alternation score')
ax[0].bar(x + w / 2, [r['results'][t]['ipsilateral_score'] for t in tags], w, label='ipsilateral correlation')
ax[0].axhline(.30, color='tab:green', ls='--', label='positive threshold +0.30')
ax[0].axhline(0, color='k', lw=.8); ax[0].set_xticks(x, ['20k control\n435,126 edges', 'whole strict\n1,223,208 edges'])
ax[0].set(ylabel='score / correlation', title='More cells does NOT restore alternation'); ax[0].legend(fontsize=8)
for i, t in enumerate(tags):
    c = l['results'][t]['conditions']
    ax[1].bar(np.arange(6) + (i - .5) * w, [c['only_' + L]['driven_leg_hz'] or 0. for L in LEGS], w, label=t)
ax[1].set_xticks(np.arange(6), LEGS); ax[1].set(ylabel='driven leg MN rate (Hz)', title='Only LF sensory drive recruits motors'); ax[1].legend(fontsize=8)
labels = ['LM 1x', 'LM 4x', 'LM 16x', 'LF 1x', 'LF 4x', 'LF 16x']
keys = ['LM_gain1', 'LM_gain4', 'LM_gain16', 'LF_gain1', 'LF_gain4', 'LF_gain16']
ax[2].bar(np.arange(6), [g['results'][k]['driven_leg_hz'] for k in keys], color=['tab:red'] * 3 + ['tab:blue'] * 3)
ax[2].axhline(2., color='tab:orange', ls=':', label='2 Hz reference')
ax[2].set_xticks(np.arange(6), labels, rotation=25, ha='right'); ax[2].set(ylabel='driven leg MN rate (Hz)', title='Gain control: LM needs 16x for 0.9 Hz'); ax[2].legend(fontsize=8)
fig.suptitle('Round13 whole-graph module and single-leg drive | uncalibrated reconstruction, no physics')
fig.savefig(OUT / 'WIDE_AND_LATERAL.png', dpi=150); plt.close(fig)
print('ok')
