"""Figure: the 63.8% figure is a passive-state number; support reaches weight by ~10 ms."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / 'outputs/model_branches/passive_vs_actuated'


def main():
    doc = json.loads((OUT / 'passive_vs_actuated.json').read_text())
    with np.load(OUT / 'passive_vs_actuated.npz') as z:
        t = z['time_s'] * 1000
        ratio = z['support_ratio']
        act = z['mean_activation']
    tl = doc['timeline']
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    axes[0].plot(t, ratio, lw=1.5, color='#4477aa')
    axes[0].axhline(1.0, color='k', ls='--', lw=1, label='body weight')
    axes[0].axvline(tl['time_first_support_above_weight_s'] * 1000, color='#cc3311', ls=':', lw=1.2,
                    label=f"support reaches weight at {tl['time_first_support_above_weight_s']*1000:.2f} ms")
    axes[0].set(xlim=(0, 200), ylim=(0, 2.4), xlabel='Time (ms)',
                ylabel='Ground support / body weight', title='Support is passive-limited only at t=0')
    axes[0].legend(fontsize=8)
    t_act = np.linspace(0, t[-1], act.size)
    axes[1].plot(t_act, act, lw=1.5, color='#117733')
    axes[1].axvline(tl['time_first_tibia_activation_s'] * 1000, color='#cc3311', ls=':', lw=1.2,
                    label=f"activation starts at {tl['time_first_tibia_activation_s']*1000:.2f} ms")
    axes[1].set(xlim=(0, 50), xlabel='Time (ms)', ylabel='Mean muscle activation',
                title='Activation builds within the first milliseconds')
    axes[1].legend(fontsize=8)
    scans = doc['static_scan_reframing']
    axes[2].bar(['deployed\npassive', '2-param scan\nbest', 'symmetric search\nbest'],
                [tl['support_ratio_at_0'], scans['two_parameter_scan_max_support_ratio'],
                 scans['symmetric_search_best_support_ratio']],
                color=['#4477aa', '#ee8866', '#aa3377'])
    axes[2].axhline(1.0, color='k', ls='--', lw=1, label='body weight')
    for i, v in enumerate([tl['support_ratio_at_0'], scans['two_parameter_scan_max_support_ratio'],
                           scans['symmetric_search_best_support_ratio']]):
        axes[2].text(i, v + .02, f'{v:.3f}', ha='center', fontsize=9)
    axes[2].set(ylim=(0, 1.25), ylabel='Support / weight',
                title='Static scans are act=0: passive only')
    axes[2].legend(fontsize=8)
    fig.text(.5, .01, 'Correction: 63.8% describes the PASSIVE stance at the declared zero-activation initial state. '
                      'It must not be read as "the stance cannot carry the body".', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'passive_vs_actuated.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
