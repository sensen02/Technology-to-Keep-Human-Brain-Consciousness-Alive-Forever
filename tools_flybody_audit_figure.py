"""Figure for the FlyBody drive audit: amplitude, offset, reference, bandwidth and recovery.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  Reads the audit artifact and draws it; runs nothing.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/model_branches/flybody_loop'
COLOUR = {('flybody_servo', 'declared'): 'tab:green',
          ('flybody_torque', 'declared'): 'tab:red',
          ('flybody_torque', 'measured'): 'tab:purple'}


def figure(doc, path):
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    for ax in axes.flat:
        ax.title.set_fontsize(9)

    ax = axes[0, 0]
    for key, row in doc['configurations'].items():
        kind, sign = key.split('|')
        c = COLOUR[(kind, sign)]
        g = [p['drive_fraction'] for p in row['gain_curve']]
        e = [200.0 if p['escape_time_s'] is None else p['escape_time_s']*1e3
             for p in row['gain_curve']]
        ax.plot(g, e, 'o-', color=c, label=f'{kind} / tibia {sign}')
        silence = row['escape_time_silenced']
        ax.axhline(200.0 if silence is None else silence*1e3, color=c, ls='--', lw=1)
    ax.axhline(200, color='k', ls=':', lw=1)
    ax.set(xlabel='drive amplitude (fraction of half the declared control range)',
           ylabel='first crossing of 90% height (ms)', ylim=(0, 215),
           title='Q1 amplitude\nwhen the 90% height line is first crossed\n'
                 '(dashed = spikes silenced)')
    ax.legend(fontsize=6)
    ax.grid(alpha=.3)

    ax = axes[0, 1]
    names, vals, cols = [], [], []
    for key, row in doc['held_configuration'].items():
        for run, r in row['runs'].items():
            verdict = 'STANDING' if r['standing']['standing_under_this_projects_criteria'] else 'no'
            names.append(f"{run.replace('_', ' ')}\\n{key.split('|')[0].replace('flybody_', '')}"
                         f"\\n{verdict}")
            vals.append(r['height_mean_last_50ms_mm'])
            cols.append('tab:blue' if 'frozen' in run else
                        ('tab:grey' if 'silence' in run else
                         ('tab:purple' if 'brace' in run else 'tab:orange')))
    ax.bar(range(len(names)), vals, color=cols)
    ax.axhline(doc['published_neutral_height_mm'], color='k', ls=':', lw=1)
    ax.text(.01, doc['published_neutral_height_mm']*1.01, 'published neutral height',
            transform=ax.get_yaxis_transform(), fontsize=6)
    ax.set(xticks=range(len(names)), xticklabels=names, ylim=(0, 2.0),
           ylabel='mean height over the last 50 ms (mm)',
           title="Q4 mechanism: height alone cannot tell\na stance from a launched body\n"
                 "(STANDING = the five-part checklist)")
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.grid(alpha=.3, axis='y')

    ax = axes[0, 2]
    for key, row in doc['bandwidth_scan'].items():
        taus = [r['filter_tau_s']*1e3 for r in row['runs'].values()]
        hs = [r['height_mean_last_50ms_mm'] for r in row['runs'].values()]
        stand = [r['standing']['standing_under_this_projects_criteria']
                 for r in row['runs'].values()]
        ax.plot(taus, hs, 'o-', color='tab:green', label=key)
        for x, y, s_ in zip(taus, hs, stand):
            ax.annotate('STANDING' if s_ else 'no', (x, y), fontsize=6, rotation=90,
                        xytext=(0, -40 if s_ else 5), textcoords='offset points', ha='center')
    ax.axhline(doc['published_neutral_height_mm'], color='k', ls=':', lw=1)
    ax.set(xscale='log', xlabel="decoder's own filter tau (ms); 5 ms in every primary run",
           ylabel='mean height over the last 50 ms (mm)',
           xticks=[2.5, 5, 10, 20, 40], xticklabels=['2.5', '5\\n(primary)', '10', '20', '40'],
           ylim=(.93, 1.16),
           title='the harm scales with the drive BANDWIDTH\n'
                 'slowing the same drive restores the stance')
    ax.minorticks_off()
    ax.legend(fontsize=7, title='5 ms is the primary translation', title_fontsize=6)
    ax.grid(alpha=.3)

    ax = axes[1, 0]
    for key, row in doc['push_recovery'].items():
        for run, r in row['runs'].items():
            t_ms = [t*1e3 for t in r['height_series_times_s']]
            ax.plot(t_ms, r['height_series_mm'], lw=1.5,
                    label=f"{run.replace('_', ' ')}: "
                          f"{'recovers' if r['returned_to_band'] else 'does not recover'}"
                          f"{' (standing)' if r['standing']['standing_under_this_projects_criteria'] else ''}")
        push_start = row['runs']['normal']['push']['start_s']*1e3
        ax.axvline(push_start, color='k', ls=':', lw=1)
        ax.text(push_start+3, 1.19, 'identical push\\n(50% body weight, 5 ms)', fontsize=7)
    ax.set(xlabel='time (ms)', ylabel='height (mm)',
           title='identical perturbation (50% body weight, 5 ms, at 150 ms)\n'
                 'which arms get their stance back?')
    ax.legend(fontsize=6.5, loc='lower left')
    ax.grid(alpha=.3)

    ax = axes[1, 1]
    for key, row in doc['configurations'].items():
        kind, sign = key.split('|')
        per = row['posture_reference']['declared_translation']['per_hinge']
        hinges = sorted(per)
        c = COLOUR[(kind, sign)]
        ax.scatter([per[h]['deviation_std_rad'] for h in hinges],
                   [abs(per[h]['drive_dc']) for h in hinges], s=26, color=c,
                   label=f'{kind}/{sign}: static part')
        ax.scatter([per[h]['deviation_std_rad'] for h in hinges],
                   [per[h]['drive_ac_std'] for h in hinges], s=26, color=c, alpha=.35, marker='x',
                   label=f'{kind}/{sign}: fluctuation')
    ax.set(xlabel='hinge deviation from the neutral pose, standard deviation (rad)', yscale='log',
           ylabel='drive magnitude (dimensionless)',
           title='Q2 offset: a large static part\nAND a large fluctuation')
    ax.legend(fontsize=6)
    ax.grid(alpha=.3)

    ax = axes[1, 2]
    for key, row in doc['configurations'].items():
        kind, sign = key.split('|')
        per = row['posture_reference']['declared_translation']['per_hinge']
        values = [per[h]['correlation_drive_vs_deviation'] for h in sorted(per)
                  if per[h]['correlation_drive_vs_deviation'] is not None]
        med = float(np.median(values))
        ax.hist(values, bins=np.linspace(-1.05, 1.05, 22), alpha=.55, color=COLOUR[(kind, sign)],
                label=f'{kind}/{sign}: median r={med:+.2f}, n={len(values)}')
    ax.axvline(0, color='k', lw=1)
    ax.axvspan(-1.05, 0, color='tab:blue', alpha=.07)
    ax.text(-1.0, 1.0, 'a posture reference\nwould have to live here', fontsize=7)
    ax.set(xlabel="in-phase correlation between a hinge's drive and its own deviation",
           ylabel='hinges',
           title='Q3 reference: does the connectome know\nwhere neutral is?  '
                 '(all configurations sit on the reinforcing side)')
    ax.legend(fontsize=6)
    ax.grid(alpha=.3)

    fig.suptitle('Translation-layer diagnostic of the connectome-to-FlyBody drive: amplitude, '
                 'offset, reference, bandwidth, recovery (manual research prototype, not a product)',
                 fontsize=11)
    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    fig.savefig(path, dpi=120)
    return path


def main():
    doc = json.loads((OUT/'flybody_drive_audit.json').read_text())['measured']
    path = figure(doc, OUT/'flybody_drive_audit.png')
    print(path)
    return path


if __name__ == '__main__':
    main()
