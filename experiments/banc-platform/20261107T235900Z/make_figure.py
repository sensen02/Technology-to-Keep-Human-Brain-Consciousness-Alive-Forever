"""Round14 figure: labelled ANN movement + platform state machine; zero simulation steps."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-platform/20261107T235900Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/banc-platform/20261107T235900Z')
tags = ['reachable_base', 'reachable_hungry', 'unreachable']
labels = ['reachable, energy .6', 'reachable, energy .3 (hungry)', 'unreachable (3.0 ahead)']
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8), layout='constrained')
colors = ['tab:blue', 'tab:orange', 'tab:red']
for i, t in enumerate(tags):
    z = np.load(SRC / f'PLATFORM_{t}.npz')
    r = json.loads((SRC / f'RESULT_{t}.json').read_text())
    feed = z['decision'] == 'feed'
    ax[0].plot(z['time_s'], z['energy'], color=colors[i], label=labels[i])
    ax[0].scatter(z['time_s'][feed], z['energy'][feed], s=3, color=colors[i], alpha=.6)
    ax[1].plot(z['time_s'], z['mouth_to_food'], color=colors[i], label=labels[i])
    ax[1].axhline(r['radius'], color='k', ls=':', lw=.8)
    if r['arrived']:
        ax[1].axvline(r['arrival']['time_s'], color=colors[i], ls='--', lw=.8)
ax[0].set(xlabel='time (s)', ylabel='declared energy', title='Energy: intake only while the mouth is in the food patch')
ax[0].legend(fontsize=8); ax[0].set_ylim(0, .65)
ax[1].set(xlabel='time (s)', ylabel='mouth-to-food distance (native)', title='Arrival: only the reachable conditions enter the patch radius')
ax[1].set_yscale('log'); ax[1].legend(fontsize=8)
counts = []
for t in tags:
    r = json.loads((SRC / f'RESULT_{t}.json').read_text())
    counts.append([r['decisions']['feed'], r['decisions']['forage'], r['decisions']['rest']])
counts = np.asarray(counts)
x = np.arange(len(tags)); w = .25
for j, name in enumerate(('feed', 'forage', 'rest')):
    ax[2].bar(x + (j - 1) * w, counts[:, j], w, label=name)
ax[2].set_xticks(x, ['reachable .6', 'hungry .3', 'unreachable'], rotation=15, ha='right')
ax[2].set(ylabel='5 ms ticks', title='Engineering selector decisions (declared rule, not neural)'); ax[2].legend(fontsize=8)
fig.suptitle('Platform v2 | movement = LABELLED replacement ANN (slip criterion FAILS, not connectome) | feeding = virtual gate | selector = engineering rule')
fig.savefig(OUT / 'PLATFORM_WITH_ANN.png', dpi=150); plt.close(fig)
print('ok')
