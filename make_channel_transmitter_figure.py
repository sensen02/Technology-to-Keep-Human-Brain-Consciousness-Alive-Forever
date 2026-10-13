"""Figure: verified-transmitter coverage per sensory channel of the published run."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / 'outputs/model_branches/channel_transmitter_audit'


def main():
    doc = json.loads((OUT / 'channel_transmitter_audit.json').read_text())
    ch = doc['per_channel']
    names = sorted(ch, key=lambda k: ch[k]['verified_fraction'] or 0.0)
    frac = [(ch[k]['verified_fraction'] or 0.0) * 100 for k in names]
    counts = [ch[k]['tier_cells'] for k in names]
    verified = [ch[k]['verified_nonempty'] for k in names]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    bars = axes[0].barh(names, frac, color=['#cc3311' if f < 35 else '#4477aa' for f in frac])
    for b, k, f, v, n in zip(bars, names, frac, verified, counts):
        axes[0].text(min(f + 1.5, 96), b.get_y() + b.get_height() / 2, f'{v}/{n}  ({f:.0f}%)',
                     va='center', fontsize=8)
    axes[0].set(xlim=(0, 118), xlabel='Cells with a Verified NT label (%)',
                title='Verified transmitter coverage per channel\n(published run, tier cells)')
    bars = axes[1].barh(names, counts, color='#777777')
    for b, n, v in zip(bars, counts, verified):
        axes[1].text(n + 30, b.get_y() + b.get_height() / 2, f'{n} cells, {v} verified', va='center', fontsize=8)
    axes[1].set(xlim=(0, max(counts) * 1.5), xlabel='Channel size (tier cells)',
                title='Channel size is unrelated to evidence')
    fig.text(.5, .01, 'Empty Verified NT is absence of a verified label, not evidence of no transmitter. '
                      'Contact/load channels are the least evidenced of all.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'channel_transmitter_audit.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
