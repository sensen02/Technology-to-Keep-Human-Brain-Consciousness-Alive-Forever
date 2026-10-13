"""Figure: motor output census — uncommandable, silent and driven actuators."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / 'outputs/model_branches/motor_output_census'
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']


def main():
    doc = json.loads((OUT / 'motor_output_census.json').read_text())
    census = doc['census']
    s = doc['summary']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    axes[0].bar(['no neural pool\n(uncommandable)', 'pool present\nbut silent', 'driven'],
                [s['uncommandable_no_pool'], s['silent_pool_no_spikes'], s['driven']],
                color=['#cc3311', '#ee8866', '#4477aa'])
    for i, v in enumerate([s['uncommandable_no_pool'], s['silent_pool_no_spikes'], s['driven']]):
        axes[0].text(i, v + 1, str(v), ha='center', fontsize=11)
    axes[0].set(ylabel='Actuators (of 90)', title='Structural output capacity')
    axes[0].grid(alpha=.2, axis='y')
    width = .38
    x = np.arange(len(LEGS))
    flex, ext = [], []
    for leg in LEGS:
        f = sum(c['pool_spikes_total'] for c in census if c['leg'] == leg and c['joint_family'] == 'trochanter'
                and c['direction'] == 'flexor')
        e = sum(c['pool_spikes_total'] for c in census if c['leg'] == leg and c['joint_family'] == 'trochanter'
                and c['direction'] == 'extensor')
        flex.append(f)
        ext.append(e)
    axes[1].bar(x - width / 2, flex, width, label='trochanter flexor', color='#4477aa')
    axes[1].bar(x + width / 2, ext, width, label='trochanter extensor', color='#cc6677')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LEGS)
    axes[1].set(ylabel='Pool spikes over 200 ms', xlabel='Leg',
                title='Antagonist drive: RM has no flexor drive at all')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2, axis='y')
    fig.text(.5, .01, f'{s["uncommandable_no_pool"]} of 90 actuators have no neural pool (all coxa promotor/remotor bundles); '
                      f'{s["pool_spikes_total"]} spikes were produced by {s["pool_cells_total"]} pool cells in 200 ms.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'motor_output_census.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
