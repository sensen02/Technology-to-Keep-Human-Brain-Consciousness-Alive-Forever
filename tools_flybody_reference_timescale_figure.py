"""Figure for the reference and time-scale audits.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  Reads the two artifacts and draws them; runs nothing.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/model_branches/flybody_loop'


def main():
    ts = json.loads((OUT/'flybody_timescale_audit.json').read_text())['measured']
    rf = json.loads((OUT/'flybody_reference_audit.json').read_text())['measured']
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))

    ax = axes[0]
    com = ts['com_response']
    t = np.array(com['trace_times_s'])*1e3
    ax.plot(t, np.array(com['trace_deviation_mm']), color='k', lw=2,
            label='whole body: centre-of-mass height')
    ax.axhline(0, color='k', ls=':', lw=1)
    for hinge, colour in (('coxa_T1_left', 'tab:blue'), ('femur_T3_right', 'tab:green'),
                          ('tibia_T1_left', 'tab:orange')):
        row = ts['joint_time_constants'][hinge]
        jt = np.array(row['trace_times_s'])*1e3
        jv = np.array(row['trace_deviation_rad'])
        ax.plot(jt, jv*15.0, color=colour, lw=1.2,
                label=f"{hinge}: 1/e {row['tau_one_over_e_s']*1e3:.0f} ms, "
                      f"keeps {abs(row['residual_at_end']):.4f} rad")
    ax.text(150, .035, 'the three thin curves are joint angles\n(0.1 rad displacement, scaled to this '
                       'axis)\nevery one of the 36 hinges decays in 7-27 ms\nand then KEEPS a static '
                       'droop: none returns to neutral', fontsize=7,
            bbox=dict(fc='white', ec='0.7', alpha=.9))
    ax.set(xlabel='time (ms)', ylabel='deviation from the relaxed state (mm, joints scaled)',
           title="the body's own passive response, no neural drive\n"
                 "a 0.05 mm displacement is never recovered: it overshoots, then drifts away")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    ax = axes[1]
    runs = rf['attribution']['runs']
    currents = runs['none_ablated']['channel_currents']
    shares = currents['per_channel']['share_of_total_current']
    order = sorted(shares, key=lambda k: -shares[k])
    ax.barh([c.replace('_', '\n') for c in order], [shares[c] for c in order], color='tab:blue',
            alpha=.75, label='share of the total input current')
    effect = [runs[f'ablate_{c}']['drive_fraction_of_baseline'] for c in order]
    ax.barh([c.replace('_', '\n') for c in order], [-v for v in effect], color='tab:red', alpha=.75,
            label='drive left after ablating it (right axis, shown as a negative bar)')
    for i, (c, e) in enumerate(zip(order, effect)):
        ax.text(-e/2, i, f'{e:.3f}', va='center', ha='center', fontsize=7, color='white')
    ax.set(xlabel='share of total input current  (the red bars are "drive remaining", negative for '
                  'readability)',
           title='where the drive comes from: ablating the contact-force\nchannel silences it '
                 'completely; the tibia-deviation channel carries the harm')
    ax.legend(fontsize=7)
    ax.grid(alpha=.3, axis='x')

    ax = axes[2]
    runs = ts['closure_at_each_tau']
    taus = [r['filter_tau_s']*1e3 for r in runs.values()]
    heights = [r['height_mean_last_50ms_mm'] for r in runs.values()]
    stand = [r['standing']['standing_under_this_projects_criteria'] for r in runs.values()]
    sourced = ts['tau_sourced_s']*1e3
    ax.plot(taus, heights, 'o-', color='tab:green')
    for x, y, s_ in zip(taus, heights, stand):
        dx = -14 if abs(x-12.75) < 1e-6 else 0
        ax.annotate('STANDING' if s_ else 'not standing', (x, y), fontsize=7,
                    rotation=0 if abs(x-12.75) < 1e-6 else 90,
                    xytext=(dx, -46 if s_ else 6), textcoords='offset points', ha='center')
    ax.axvline(sourced, color='tab:red', ls='--', lw=1.5)
    ax.text(sourced+.6, min(heights)-.003, f'the BODY\'s own time constant\n{sourced:.2f} ms '
                                           '(13.0 ms hinge 1/e, 12.5 ms damping/gain)',
            fontsize=7, color='tab:red')
    ax.axhline(1.1610881388659853, color='k', ls=':', lw=1)
    ax.set(xscale='log', xticks=[5, 10, 12.75, 20, 40], xticklabels=['5', '10', '12.75', '20', '40'],
           xlabel="the decoder's filter tau (ms)", ylabel='mean height over the last 50 ms (mm)',
           title='the sourced time scale does not restore the stance\n'
                 '(the working tau has no source)')
    ax.grid(alpha=.3)

    fig.suptitle('The two requested paths: where the drive comes from, and what time scale the body '
                 'itself asks for (manual research prototype, not a product)', fontsize=11)
    fig.tight_layout()
    path = OUT/'flybody_reference_timescale.png'
    fig.savefig(path, dpi=125)
    print(path)
    return path


if __name__ == '__main__':
    main()
