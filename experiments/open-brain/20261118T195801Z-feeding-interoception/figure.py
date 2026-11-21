"""Plot actual execution and anatomical candidate screening traces."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def body_csv(directory, name):
    with (directory / (name + '.csv')).open() as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def main(directory):
    body = directory / 'BODY_FINAL'
    fed = body_csv(body, 'nutrition')
    inert = body_csv(body, 'same_volume_no_nutrition')
    blocked = body_csv(body, 'blocked')
    eon = [json.loads((directory / f'EON_FINAL{seed}/RESULT.json').read_text())
           for seed in (127, 509)]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.8), layout='constrained')
    t = fed['time_s']
    a = axes[0, 0]
    a.plot(t, fed['crop'], color='#257b6b', lw=2, label='Crop fill, both foods')
    a.plot(t, fed['intake_feedback'], color='#285ba6', label='Intake feedback, both foods')
    a.plot(t, blocked['intake_feedback'], '--', color='#986b40', label='Intake blocked')
    a.set(title='Same volume, same mechanical feedback', xlabel='Time (s)', ylabel='Fraction', xlim=(0, 7))
    a.legend(loc='upper right', fontsize=8)
    a = axes[0, 1]
    a.plot(t, fed['reserve'], color='#257b6b', lw=2, label='Nutritive')
    a.plot(t, inert['reserve'], color='#bd6548', lw=2, label='Zero calorie')
    a.set(title='Absorption and reserve diverge after intake', xlabel='Time (s)', ylabel='Reserve (simulation units)')
    a.legend(frameon=False)
    a = axes[1, 0]
    with np.load(directory / 'MOUTH/TRACE.npz', allow_pickle=False) as z:
        a.plot(z['time'], np.cumsum(z['ingested'][:, 0]), label='Near', lw=2, color='#257b6b')
        a.plot(z['time'], np.cumsum(z['ingested'][:, 1]), '--', label='Far and blocked', color='#bd6548')
        a.fill_between(z['time'], 0., .22, where=z['command'], alpha=.12, color='#285ba6', label='Feed on')
    a.set(title='Virtual droplet sampled at the moving mouth', xlabel='Time (s)', ylabel='Cumulative intake')
    a.legend(frameon=False, loc='lower right', fontsize=8)
    a = axes[1, 1]
    names = ['sugar_cut', *[f'set_{i}' for i in range(1, 7)]]
    for offset, result in zip((-.18, .18), eon, strict=True):
        values = [result['metrics'][name]['mn9_window_spikes'] for name in names]
        a.bar(np.arange(len(names)) + offset, values, width=.32, label=f"Seed {result['seed']}")
    a.set_xticks(range(len(names)), ['Cut', *[str(i) for i in range(1, 7)]])
    a.set(title='Frozen Eon response to StN candidate groups', xlabel='StN candidate group', ylabel='MN9 spikes (190 ms)')
    a.legend(frameon=False, loc='upper right', fontsize=8)
    for a in axes.flat:
        a.spines[['top', 'right']].set_visible(False)
        a.grid(axis='y', alpha=.18)
    fig.savefig(directory / 'FEEDING.png', dpi=155)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
