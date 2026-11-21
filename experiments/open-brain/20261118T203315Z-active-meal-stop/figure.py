"""Render the frozen-readout qualification from its saved recordings."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root, folders):
    fig, axes = plt.subplots(4, 2, figsize=(12, 10.5), layout='constrained', sharex='col')
    colors = ('#247d6e', '#b85539', '#274c93', '#857a99')
    for col, folder in enumerate(folders):
        path = root / folder
        metadata = json.loads((path / 'RESULT.json').read_text())
        names = metadata['names']
        controller = metadata.get('controller') or {'stop': .63, 'restart': .53}
        with np.load(path / 'TRACE.npz', allow_pickle=False) as trace:
            t = np.arange(1, len(trace['command']) + 1) * .01
            level = trace['control_level'] if 'control_level' in trace else trace['predicted_crop']
            for j, (name, color) in enumerate(zip(names, colors, strict=True)):
                axes[0, col].plot(t, level[:, j], color=color, label=name, lw=1.35)
                axes[1, col].step(t, trace['command'][:, j] + j * 1.3, where='post', color=color, lw=1.35)
                axes[2, col].plot(t, np.cumsum(trace['ingested'][:, j]), color=color, lw=1.65)
                axes[3, col].plot(t, trace['crop_before'][:, j], color=color, lw=1.55)
        axes[0, col].set_title(f"Seed {metadata['seed']}", loc='left')
        axes[0, col].legend(ncol=2, frameon=False, fontsize=8, loc='upper left')
        axes[0, col].axhline(controller['stop'], color='black', lw=.7, ls='--')
        axes[0, col].axhline(controller['restart'], color='black', lw=.7, ls=':')
        axes[0, col].set_ylabel('StN synaptic input' if metadata.get('readout') == 'stn_first_synapse'
                                else 'Decoded stretch')
        axes[1, col].set_yticks(np.arange(4) * 1.3 + .5, names)
        axes[1, col].set_ylabel('Feeding command')
        axes[2, col].set_ylabel('Cumulative intake (units)')
        axes[3, col].set_ylabel('Crop capacity used')
        axes[3, col].set_xlabel('Time (s)')
        axes[3, col].axhline(1., color='black', lw=.7, ls='--')
        axes[3, col].set_ylim(0., 1.04)
        for a in axes[:, col]:
            a.axvline(.620, color='#274c93', lw=.8, ls=':')
            a.set_xlim(0., 1.2)
            a.spines[['top', 'right']].set_visible(False)
            a.grid(axis='y', alpha=.18)
    fig.savefig(root / f'{folders[0]}_{folders[1]}.png', dpi=150)


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2:])
