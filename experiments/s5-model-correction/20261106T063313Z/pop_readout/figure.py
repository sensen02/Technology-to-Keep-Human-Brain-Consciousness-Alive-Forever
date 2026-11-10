"""Figure: per phase x neural timestep, J / foot support / min COM / D-gate count, all readouts."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common as C

STYLE = {'R0': dict(color='#c0392b', marker='o', label='R0  current 204-cell readout (control)'),
         'P1': dict(color='#1f4e9c', marker='s', label='P1  ridge, all 12000 cells, no constant'),
         'P2': dict(color='#1e8449', marker='^', label='P2  ridge, all 12000, cell means removed'),
         'P3': dict(color='#8e44ad', marker='D', label='P3  ridge, 204 motor cells only (control)')}
ASSIST_FOOT = 1.000012295922673
ASSIST_COM = 1.2374996556699485
H0 = 1.2072528224791905


def main():
    s = C.js(C.O / 'SUMMARY.json')
    agg = s['per_condition']
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))
    for row, (dt, tag) in enumerate(C.DTS):
        phases = None
        for readout in C.TAGS:
            a = agg[f'{readout}__{tag}']
            x = [p['phase'] for p in a['per_phase']]
            phases = x
            st = STYLE[readout]
            for col, key in enumerate(['J', 'foot_support_mean', 'com_z_min', 'D_pass_count']):
                y = [p[key] for p in a['per_phase']]
                axes[row, col].plot(x, y, lw=1.7, ms=8, **st)
                axes[row, col].grid(alpha=0.3)
        axes[row, 0].set_ylabel('%s  (%g ms neural dt)\nJ  (lower is better)'
                               % (tag, dt * 1000))
        for col in (1, 2, 3):
            axes[row, col].set_ylabel('')
        axes[row, 1].axhline(ASSIST_FOOT, ls='--', c='k', lw=1, label='assisted foot support')
        axes[row, 1].axhline(ASSIST_FOOT - 0.10, ls=':', c='k', lw=1, label='D3 lower band')
        axes[row, 2].axhline(ASSIST_COM, ls='--', c='k', lw=1, label='assisted COM mean')
        axes[row, 2].axhline(ASSIST_COM - 0.10, ls=':', c='k', lw=1, label='D5 lower band')
        axes[row, 2].axhline(0.8 * H0, ls='-.', c='k', lw=1, label='D8 absolute floor')
        axes[row, 3].axhline(9, ls=':', c='k', lw=1, label='all 9 D checks')
        for col, title in enumerate(['J (lower is better)', 'foot support mean',
                                     'minimum COM height', 'D checks passed (of 9)']):
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('release phase (s)')
            axes[row, col].set_xticks(phases)
        for col in (1, 2, 3):
            axes[row, col].legend(fontsize=7, loc='lower left')
        axes[row, 3].set_ylim(-0.4, 9.6)
    axes[0, 0].legend(fontsize=9, loc='upper left')
    fig.suptitle('S5 whole-population readout test on sealed neural warm states — '
                 'per-phase, per-timestep outcomes (no mean, no timestep selected)', fontsize=13)
    fig.text(0.01, 0.005,
             'OFFLINE ENGINEERING FITTING of the readout interface, NOT biological calibration. '
             'Neurons still generate the command. No walking / autonomous-support / '
             'biological-validity / absolute-force claim. One conclusion per timestep.',
             fontsize=9, color='#555')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = C.O / 'FIGURE_pop_readout.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('FIG', out, out.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
