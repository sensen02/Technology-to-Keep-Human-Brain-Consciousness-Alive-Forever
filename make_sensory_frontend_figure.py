"""Figure: sensory front-end drive balance and evidence coverage."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / 'outputs/model_branches/sensory_frontend_balance'
EVIDENCE = {'claw_position': 0.98, 'hook_direction': 0.978, 'club_vibration': 0.92,
            'hairplate_angle': 0.329, 'campaniform_strain': 0.20,
            'bristle_touch': 0.0, 'multidendritic_noxious': 0.0}


def main():
    doc = json.loads((OUT / 'sensory_frontend_balance.json').read_text())
    ch = doc['per_channel']
    names = sorted(ch, key=lambda k: ch[k]['share_of_total_input'] or 0.0)
    share = [(ch[k]['share_of_total_input'] or 0.0) * 100 for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9))
    bars = axes[0].barh(names, share, color=['#cc3311' if EVIDENCE[k] < 0.35 else '#4477aa' for k in names])
    for b, k, s in zip(bars, names, share):
        axes[0].text(min(s + 1.5, 84), b.get_y() + b.get_height() / 2,
                     f'{s:.1f}%   (verified NT {EVIDENCE[k]*100:.0f}%)', va='center', fontsize=8)
    axes[0].set(xlim=(0, 108), xlabel='Share of total delivered sensory input current (%)',
                title='Where the front-end drive actually goes')
    vals = [ch[k]['value_max'] for k in names]
    bars = axes[1].barh(names, vals, color=['#cc3311' if EVIDENCE[k] < 0.35 else '#4477aa' for k in names])
    axes[1].axvline(1.0, color='k', ls='--', lw=1, label='declared normaliser = 1.0')
    for b, v in zip(bars, vals):
        axes[1].text(v + .08, b.get_y() + b.get_height() / 2, f'{v:.2f}', va='center', fontsize=8)
    axes[1].set(xlim=(0, 7.8), xlabel='Maximum normalized channel value in the episode',
                title='Channel gains are not on a common scale')
    axes[1].legend(fontsize=8)
    fig.text(.5, .01, '88% of the input current goes to a channel with no verified transmitter and therefore no '
                      'propagating synapses; the best-evidenced channels are exactly zero at t=0.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'sensory_frontend_balance.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
