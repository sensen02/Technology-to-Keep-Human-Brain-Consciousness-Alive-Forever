"""Figure for the perturbation experiment; reads the measured artifact, changes nothing."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'outputs/neural_muscle_closure'


def main():
    d = json.loads((OUT / 'neural_muscle_perturbation.json').read_text())['measured']
    rows = d['rows']
    fr = [r['weight_fraction'] for r in rows]
    arms = d['arms']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    for arm in arms:
        ax[0].plot(fr, [1000*r['arms'][arm]['escape_time_s'] for r in rows], 'o-', label=arm)
    ax[0].axhline(1000*rows[0]['arms']['spike_output_silence']['escape_time_s'],
                  ls=':', c='gray', lw=1)
    ax[0].set(title='Escape time vs perturbation\n(passive arms are flat: metric is blunt)',
              xlabel='push as fraction of body weight', ylabel='time to 90% initial height (ms)')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)

    adv = [r['advantage_normal_minus_silence_ms'] for r in rows]
    ax[1].plot(fr, adv, 's-', color='crimson')
    ax[1].annotate(f'{adv[-1]/adv[0]:.2f}x baseline\n(H19 rejected: expected >1.2x)',
                   xy=(fr[-1], adv[-1]), xytext=(fr[0]+0.05, adv[-1]+0.8), fontsize=9,
                   arrowprops=dict(arrowstyle='->', lw=1))
    ax[1].set(title='Normal minus silence advantage\nDECREASES with disturbance, not feedback',
              xlabel='push as fraction of body weight', ylabel='advantage (ms)')
    ax[1].grid(alpha=.25)

    for arm in arms:
        xs = [r['arms'][arm]['x_span_model'] for r in rows]
        ax[2].plot(fr, xs, 'o-', label=arm)
    ax[2].set(title='Probe witness: lateral span\n(force was applied; applied window fires once)',
              xlabel='push as fraction of body weight', ylabel='thorax x span (model length)')
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.25)

    s = d['probe_efficacy']['summary']
    fig.suptitle('External perturbation vs the closure: a measured negative on corrective feedback '
                 f'(force applied={s["push_force_was_applied"]}, '
                 f'lateral change={s["lateral_motion_changed_from_no_push"]}, '
                 f'passive metric sensitive={s["escape_time_metric_sensitive_in_passive_arms"]})',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT / 'neural_muscle_perturbation.png', dpi=160)
    print('wrote', OUT / 'neural_muscle_perturbation.png')


if __name__ == '__main__':
    main()
