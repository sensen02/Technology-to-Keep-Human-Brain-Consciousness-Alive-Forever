"""Figure: per-phase J / foot support / minimum COM for R0,R1,R2 at both neural timesteps."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common as C

TAG_STYLE = {'R0': dict(color='#c0392b', marker='o', label='R0 current readout'),
             'R1': dict(color='#2471a3', marker='s', label='R1 fitted readout'),
             'R2': dict(color='#1e8449', marker='^', label='R2 fitted + RMS gain')}


def main():
    s = C.js(C.O / 'SUMMARY.json')
    agg = s['per_condition']
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    for row, (dt, tag) in enumerate(C.DTS):
        phases = None
        for readout in C.TAGS:
            a = agg[f'{readout}__{tag}']
            x = [p['phase'] for p in a['per_phase']]
            phases = x
            st = TAG_STYLE[readout]
            J = [p['J'] for p in a['per_phase']]
            fs = [p['foot_support_mean'] for p in a['per_phase']]
            cz = [p['com_z_min'] for p in a['per_phase']]
            dp = [p['D_pass_count'] for p in a['per_phase']]
            for col, y in enumerate([J, fs, cz, dp]):
                ax = axes[row, col]
                ax.plot(x, y, **st, lw=1.6, ms=8)
                ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel(f'{tag}  ({dt * 1000:g} ms)\nJ  (lower is better)')
        axes[row, 1].axhline(1.000012295922673, ls='--', c='k', lw=1,
                             label='assisted reference foot support')
        axes[row, 1].axhline(0.900012295922673, ls=':', c='k', lw=1, label='D3 lower band')
        axes[row, 2].axhline(1.2374996556699485, ls='--', c='k', lw=1, label='assisted COM mean')
        axes[row, 2].axhline(1.2374996556699485 - 0.10, ls=':', c='k', lw=1, label='D5 lower band')
        axes[row, 2].axhline(0.8 * 1.2072528224791905, ls='-.', c='k', lw=1, label='D8 absolute floor')
        axes[row, 3].axhline(9, ls=':', c='k', lw=1, label='all 9 gates')
        for col, t in enumerate(['J', 'foot support mean', 'min COM height', 'D gates passed']):
            axes[row, col].set_title(t)
            axes[row, col].set_xlabel('release phase (s)')
            axes[row, col].set_xticks(phases)
        axes[row, 1].legend(fontsize=7, loc='lower left')
        axes[row, 2].legend(fontsize=7, loc='lower left')
        axes[row, 3].legend(fontsize=7, loc='lower left')
        axes[row, 3].set_ylim(-0.3, 9.5)
    axes[0, 0].legend(fontsize=9, loc='upper left')
    fig.suptitle('S5 readout-interface correction on sealed neural warm states — '
                 'per-phase outcomes (not a mean; no timestep is selected)', fontsize=13)
    fig.text(0.01, 0.005,
             'Offline ENGINEERING fitting of the readout interface, NOT biological calibration. '
             'Neurons still generate the command. No walking / autonomous-support / '
             'biological-validity / absolute-force claim.',
             fontsize=9, color='#555')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = C.O / 'FIGURE_readout_fit.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('FIG', out, out.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
