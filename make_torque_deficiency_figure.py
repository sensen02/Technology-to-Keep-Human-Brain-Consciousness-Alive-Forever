"""Figure for the joint torque deficiency audit."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / 'outputs/model_branches/torque_deficiency'


def main():
    import json
    doc = json.loads((OUT / 'torque_deficiency_results.json').read_text())
    with np.load(OUT / 'torque_deficiency_traces.npz') as z:
        t = z['time_s'] * 1000
        required, actual, deficit = z['required'], z['actual'], z['deficit']
        names = [str(x) for x in z['joint_names']]
        weight = float(z['weight'])
    coverage = np.abs(actual).sum(axis=1) / np.abs(required).sum(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(t, coverage, lw=1.6, color='#4477aa')
    axes[0].axhline(1.0, color='k', ls='--', lw=1, label='actuator = hold requirement')
    for ms in (24, 95):
        axes[0].axvline(ms, color='#cc3311', ls=':', lw=1)
    axes[0].set(title='Joint torque coverage', xlabel='Time (ms)',
                ylabel='sum |actual| / sum |required|', ylim=(0, 1.6))
    axes[0].legend(fontsize=8)
    for name, colour in (('joint_LFTibia_pitch', '#4477aa'), ('joint_RFTibia_pitch', '#66ccee'),
                         ('joint_LHCoxa_yaw', '#ee6677'), ('joint_RHCoxa_yaw', '#aa3377')):
        if name in names:
            axes[1].plot(t, deficit[:, names.index(name)] / weight, lw=1.4, label=name.replace('joint_', ''))
    axes[1].axhline(0, color='k', lw=.8)
    axes[1].set(title='Signed deficit / body weight', xlabel='Time (ms)',
                ylabel='(required - actual) / weight', ylim=(-0.5, 0.5))
    axes[1].legend(fontsize=7)
    i0 = 0
    order = np.argsort(-np.abs(deficit[i0]))
    top = order[:8]
    x = np.arange(len(top))
    axes[2].bar(x - .2, required[i0][top], .4, label='required hold torque', color='#4477aa')
    axes[2].bar(x + .2, actual[i0][top], .4, label='actual actuator torque', color='#cc3311')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([names[j].replace('joint_', '') for j in top], rotation=45, ha='right', fontsize=7)
    axes[2].axhline(0, color='k', lw=.8)
    axes[2].set(title='At t=0 the tibia joints are unsupported', ylabel='Torque (mN*mm = model units)')
    axes[2].legend(fontsize=8)
    fig.text(.5, .01, 'Required is the per-instant static-hold torque given the actual contact forces; '
                      'a deficit is a coordination shortfall, not proof the posture is unachievable.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'torque_deficiency.png'
    fig.savefig(p, dpi=135, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
