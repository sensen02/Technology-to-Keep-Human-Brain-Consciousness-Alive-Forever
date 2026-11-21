"""Independent count audit and comparison chart for virtual local block."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SORT = HERE.parent / '20261120T051324Z-neck-spike-sorting'
sys.path.insert(0, str(SORT))
from evaluate import reference


def main():
    result = json.loads((HERE / 'RESULT.json').read_text())
    for p, h in result['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    truth = Counter((h, t, 'down' if h < t else 'up') for h, t, _ in reference(11701, 'intact'))
    with np.load(HERE / 'PREDICTIONS.npz', allow_pickle=False) as z:
        assert set(z.files) == {'algorithm'} | {
            f'{case}_{direction}'
            for case in ('same_noise', 'independent_noise_1', 'independent_noise_2', 'independent_noise_3')
            for direction in ('down', 'up')}
        for case, rows in result['scores'].items():
            calls = Counter()
            if case == 'algorithm':
                calls.update((int(h), int(t), 'down' if h < t else 'up')
                             for h, t, _ in z['algorithm'])
            else:
                for direction in ('down', 'up'):
                    calls.update((int(h), int(t), direction)
                                 for h, t, _ in z[f'{case}_{direction}'])
            hits = calls & truth
            for direction in ('down', 'up'):
                row = rows[direction]
                assert row['actual'] == sum(v for key, v in truth.items() if key[2] == direction)
                assert row['called'] == sum(v for key, v in calls.items() if key[2] == direction)
                assert row['correct'] == sum(v for key, v in hits.items() if key[2] == direction)
                assert np.isclose(row['precision'], row['correct']/row['called'])
                assert np.isclose(row['recall'], row['correct']/row['actual'])
            assert rows['pooled']['correct'] == sum(v['correct'] for k, v in rows.items() if k != 'pooled')
    algorithm = result['scores']['algorithm']
    ideal = result['scores']['same_noise']
    indep = [result['scores'][f'independent_noise_{i}'] for i in range(1, 4)]
    summary = {'algorithm_pooled': algorithm['pooled'],
               'ideal_pooled': ideal['pooled'],
               'independent_pooled_correct': [s['pooled']['correct'] for s in indep],
               'independent_pooled_called': [s['pooled']['called'] for s in indep],
               'independent_pooled_precision_mean': float(np.mean([s['pooled']['precision'] for s in indep])),
               'independent_pooled_recall_mean': float(np.mean([s['pooled']['recall'] for s in indep])),
               'independent_pooled_f1_mean': float(np.mean([s['pooled']['f1'] for s in indep])),
               'source_hashes_verified': True, 'counts_recomputed_from_predictions': True}
    (HERE / 'AUDIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    names = ['Down', 'Up', 'Combined']
    keys = ['down', 'up', 'pooled']
    x = np.arange(3)
    width = .25
    colors = ['#2563a8', '#c15b32', '#528c62']
    for axis, metric in zip(axes, ('recall', 'precision')):
        series = [np.array([algorithm[k][metric] for k in keys]),
                  np.array([ideal[k][metric] for k in keys]),
                  np.array([np.mean([s[k][metric] for s in indep]) for k in keys])]
        for j, (values, color, name) in enumerate(zip(series, colors,
                                                       ('Current algorithm', 'Virtual block, shared noise',
                                                        'Virtual block, independent noise'))):
            axis.bar(x + (j-1)*width, values*100, width, label=name, color=color)
        axis.set_xticks(x, names)
        axis.set_ylim(0, 105)
        axis.set_ylabel(('Actual events found (%)' if metric == 'recall' else 'Correct among calls (%)'))
        axis.set_title('Detection coverage' if metric == 'recall' else 'Direction and timing precision')
        axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('11701 holdout: frozen voltage sorter vs idealized outside-bank spike block')
    fig.savefig(HERE / 'COMPARISON.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
