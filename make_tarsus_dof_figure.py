"""Figure: extended tarsus-DOF model versus the published 25 us baseline."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/model_branches/tarsus_dof'


def main():
    doc = json.loads((OUT / 'tarsus_dof_results_v4.json').read_text())
    with np.load(OUT / 'tarsus_dof_traces_v4.npz') as z, \
         np.load(HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz') as b:
        t = np.arange(z['tarsus_normal_heights_mm'].shape[0]) * 0.0001 * 1000
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
        axes[0].plot(t, z['tarsus_normal_heights_mm'][:, 2], color='#cc3311',
                     label='extended (v4): +6 tarsus DOF, anatomical axis')
        axes[0].plot(t, b['25us_normal_heights_mm'][:, 2], color='#4477aa', label='published 25us baseline')
        axes[0].set(xlabel='Time (ms)', ylabel='COM height (mm)', title='Extended body falls faster')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=.2)
        tset = [f'{leg}' for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')]
        pools = [doc['extension']['pool_detail'][f'{leg}Tarsus_depressor']['cells'] for leg in tset]
        pools_l = [doc['extension']['pool_detail'][f'{leg}Tarsus_levator']['cells'] for leg in tset]
        x = np.arange(len(tset))
        axes[1].bar(x - .2, pools, .4, label='tarsus depressor pool', color='#117733')
        axes[1].bar(x + .2, pools_l, .4, label='tarsus levator pool', color='#88ccee')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tset)
        axes[1].set(ylabel='Annotated motor neurons', xlabel='Leg',
                    title='New actuators driven by real annotated pools')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=.2, axis='y')
        base = doc['published_25us_baseline']['escape_time_s'] * 1000
        new = doc['branch']['escape_time_s'] * 1000
        axes[2].bar(['baseline\n(25us)', 'v1 declared\naxis', 'v2 flybody\nrange', 'v3 + flexion\nspringref',
                     'v4 anatomical\naxis (mCT)'],
                    [base, 8.75, 8.75, 8.75, new],
                    color=['#4477aa', '#ee8866', '#ddcc77', '#aa3377', '#cc3311'])
        for i, v in enumerate([base, 8.75, 8.75, 8.75, new]):
            axes[2].text(i, v + .3, f'{v:.2f} ms', ha='center', fontsize=9)
        axes[2].set(ylabel='10% thorax-height escape time (ms)',
                    title='Negative result: the free DOF adds a collapse mode')
        axes[2].grid(alpha=.2, axis='y')
        fig.text(.5, .01, 'Four independent parameterisations of the added joint all degrade the stance identically, including an axis '
                          'derived from published micro-CT condyle positions.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, 1))
        p = OUT / 'tarsus_dof_comparison.png'
        fig.savefig(p, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(p)


if __name__ == '__main__':
    main()
