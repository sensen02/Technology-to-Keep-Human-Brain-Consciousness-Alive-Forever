"""Figure: declared-reference branch versus the published 25 us baseline."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/model_branches/declared_reference'


def main():
    import run_declared_reference_branch as run
    doc = json.loads((OUT / 'declared_reference_results.json').read_text())
    with np.load(OUT / 'declared_reference_traces.npz') as z, \
         np.load(HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz') as b:
        t = z['declared_normal_heights_mm']
        tt = np.arange(t.shape[0]) * .0001 * 1000
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        axes[0].plot(tt, t[:, 2], label='declared zero + measured normaliser', color='#cc3311')
        axes[0].plot(tt, b['25us_normal_heights_mm'][:, 2], label='published 25us baseline', color='#4477aa')
        axes[0].set(xlabel='Time (ms)', ylabel='COM height (mm)', title='COM height')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=.2)
        base_escape = json.loads((HERE / 'outputs/model_branches/fine_clock/fine_clock_results.json').read_text())['episodes']['25us_normal']['escape_time_s']
        vals = [base_escape * 1000, doc['declared_branch']['escape_time_s'] * 1000]
        axes[1].bar(['baseline\n(25us)', 'declared\nreference'], vals, color=['#4477aa', '#cc3311'])
        for i, v in enumerate(vals):
            axes[1].text(i, v + .3, f'{v:.2f} ms', ha='center', fontsize=9)
        axes[1].set(ylabel='10% thorax-height escape time (ms)', title='Both fall; declared branch falls sooner')
        axes[1].grid(alpha=.2, axis='y')
        fig.text(.5, .01, 'Declared assumptions changed, nothing tuned. Negative result: a traceable zero and a measured '
                          'force normaliser do not by themselves produce standing.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, 1))
        p = OUT / 'declared_reference.png'
        fig.savefig(p, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(p)


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    main()
