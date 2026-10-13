"""Figure: injected versus propagable sensory input current per channel."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / 'outputs/model_branches/propagable_input'


def main():
    doc = json.loads((OUT / 'propagable_input.json').read_text())
    ch = doc['per_channel']
    names = sorted(ch, key=lambda k: ch[k]['injected_current_nA'])
    inj = [ch[k]['injected_current_nA'] for k in names]
    prop = [ch[k]['propagable_current_nA'] for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    y = np.arange(len(names))
    axes[0].barh(y, inj, color='#bbbbbb', label='injected (all cells)')
    axes[0].barh(y, prop, color='#4477aa', label='propagable (verified cells only)')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    for i, (a, b) in enumerate(zip(inj, prop)):
        axes[0].text(a + 8, i, f'{b:.1f} / {a:.1f} nA', va='center', fontsize=8)
    axes[0].set(xlim=(0, max(inj) * 1.32), xlabel='Mean absolute injected current, summed over 2001 samples (nA)',
                title='Injected vs propagable sensory input')
    axes[0].legend(fontsize=8)
    frac = [(ch[k]['propagable_fraction_of_channel'] or 0.0) * 100 for k in names]
    bars = axes[1].barh(y, frac, color=['#cc3311' if f < 40 else '#4477aa' for f in frac])
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([k for k in names])
    for b, f, k in zip(bars, frac, names):
        axes[1].text(min(f + 2, 88), b.get_y() + b.get_height() / 2,
                     f'{f:.0f}%  ({ch[k]["verified_cells"]}/{ch[k]["cells"]} cells verified)', va='center', fontsize=8)
    axes[1].set(xlim=(0, 118), xlabel='Share of the channel current that can propagate (%)',
                title='Evidence-weighted drive per channel')
    fig.text(.5, .01, f'Total: only {doc["totals"]["propagable_share"]*100:.2f}% of the injected sensory current lands on '
                      f'cells with a verified transmitter ({doc["totals"]["verified_sensory_cells"]} of '
                      f'{doc["totals"]["sensory_cells"]} cells).', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'propagable_input.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
