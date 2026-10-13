"""Summary figure of the FlyBody branch: the closure comparison across the three bodies.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  This file reads the branch artifacts and draws them;
it runs no episode and decides nothing.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BODY = HERE / 'outputs/model_branches/flybody_body'
LOOP = HERE / 'outputs/model_branches/flybody_loop'
KINDS = ('canonical', 'flybody_servo', 'flybody_torque')
LABEL = {'canonical': 'canonical FlyMimic/Miller body',
         'flybody_servo': 'FlyBody, published servos kept',
         'flybody_torque': 'FlyBody, servos -> torque motors'}
COLOR = {'canonical': 'tab:blue', 'flybody_servo': 'tab:green', 'flybody_torque': 'tab:red'}


def main():
    doc = json.loads((LOOP/'flybody_closure_200ms.json').read_text())
    measured = doc['measured']
    tibia = json.loads((LOOP/'flybody_closure_200ms_tibiasign_measured.json').read_text())['measured']
    # The closure archive stores each arm's arrays as a dict, so it is loaded with allow_pickle;
    # it is this branch's own file, written by run_flybody_loop.py in this same directory.
    traces = np.load(LOOP/'flybody_closure_200ms.npz', allow_pickle=True)
    translate = json.loads((LOOP/'flybody_translation.json').read_text())['measured']
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    for kind in KINDS:
        for arm, style in (('normal', '-'), ('spike_output_silence', '--')):
            records = measured['configurations'][kind]['arms'][arm]['records']
            t = [r['time_end_s']*1e3 for r in records]
            axes[0].plot(t, [r['height_mm'] for r in records], style, color=COLOR[kind], lw=1.7,
                         label=f'{LABEL[kind]} | ' +
                               ('connectome driving' if arm == 'normal' else 'spikes silenced'))
    axes[0].set(xlabel='time (ms)', ylabel='body height (mm)',
                title='the same connectome, the same decoder, three bodies\n'
                      '(height = thorax height for the canonical body, centre-of-mass height for FlyBody)')
    axes[0].grid(alpha=.3)
    axes[0].legend(fontsize=6.5)

    xs = np.arange(len(KINDS))
    width = .38
    for i, arm in enumerate(('normal', 'spike_output_silence')):
        vals = [200.0 if measured['configurations'][k]['arms'][arm]['escape_time_s'] is None
                else measured['configurations'][k]['arms'][arm]['escape_time_s']*1e3 for k in KINDS]
        bars = axes[1].bar(xs + (i-.5)*width, vals, width,
                           color=['tab:orange', 'tab:grey'][i],
                           label='connectome driving' if i == 0 else 'spikes silenced')
        for b, v, k in zip(bars, vals, KINDS):
            never = measured['configurations'][k]['arms'][arm]['escape_time_s'] is None
            axes[1].text(b.get_x()+b.get_width()/2, v+3, 'never escaped' if never else f'{v:.1f} ms',
                         ha='center', fontsize=7)
    for i, kind in enumerate(('flybody_servo', 'flybody_torque')):
        e = tibia['configurations'][kind]['arms']['normal']['escape_time_s']
        axes[1].plot([xs[KINDS.index(kind)] + .5*width], [200.0 if e is None else e*1e3], 'k*', ms=10,
                     label='same, with the hind-leg tibia sign measured' if i == 0 else None)
    axes[1].axhline(200, color='k', ls=':', lw=1)
    axes[1].set(xticks=xs, xticklabels=[LABEL[k].replace(', ', '\n') for k in KINDS], ylim=(0, 232),
                ylabel='escape time (ms)',
                title='when the body first falls below 90% of its seated height\n'
                      '(the 200 ms horizon; "never escaped" is the dotted line)')
    axes[1].legend(fontsize=7, loc='upper left')
    axes[1].grid(alpha=.3, axis='y')

    arms = ('no_external_touch', 'sensory_cut', 'motor_cut', 'spike_output_silence', 'replay')
    diffs = []
    for arm in arms:
        a = traces['flybody_servo_normal'][()]['trajectory']
        b = traces[f'flybody_servo_{arm}'][()]['trajectory']
        diffs.append(float(np.max(np.abs(a-b))))
    bars = axes[2].bar(range(len(arms)), [1.0]*len(arms),
                       color=['tab:red' if d == 0 else 'tab:green' for d in diffs])
    axes[2].set(xticks=range(len(arms)),
                xticklabels=[a.replace('_', '\n') for a in arms], ylim=(0, 1.45),
                ylabel='does the arm change the body at all?',
                title='is the loop still causally in the chain?\nFlyBody with its published servos')
    for i, (bar, d) in enumerate(zip(bars, diffs)):
        text = 'IDENTICAL\nto the driving arm' if d == 0 else f'differs by\n{d:.1f} (log units)'
        axes[2].text(bar.get_x()+bar.get_width()/2, 1.02, text, ha='center', fontsize=7,
                     color='darkred' if d == 0 else 'darkgreen')
    axes[2].text(.02, .06, 'green = the loop is in the causal chain for that cut\n'
                           'red = no measurable effect: the LF tactile probe has no modelled\n'
                           'motor path, and replay reproduces the driving arm by construction',
                 transform=axes[2].transAxes, fontsize=7)
    axes[2].set_xlabel(f"{translate['segment_projection']['n_muscles']} canonical muscles projected "
                       f"onto {len(translate['compiled_drive_matrix'])} FlyBody hinges; the transferred "
                       f"tibia sign is verified on {translate['tibia_direction_check']['n_consistent']} "
                       'of 6 legs', fontsize=7)

    fig.suptitle('Body-switch experiment: the same sensory -> connectome -> motor -> body loop on the '
                 'canonical body and on the published FlyBody model '
                 '(manual research prototype, not a product)', fontsize=10)
    fig.tight_layout()
    path = LOOP/'flybody_branch_summary.png'
    fig.savefig(path, dpi=130)
    print(path)
    return path


if __name__ == '__main__':
    main()
