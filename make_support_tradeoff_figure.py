"""Plot normalized compact rows extracted from verified tradeoff artifact."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'outputs/workspace_recheck'


def render(rows):
    poses = list(dict.fromkeys(r['pose'] for r in rows))
    fig, axes = plt.subplots(1, len(poses), figsize=(12, 5), squeeze=False)
    for ax, pose in zip(axes[0], poses):
        for condition, marker in [('covered74', 'o'), ('all90', 's')]:
            selected = sorted([r for r in rows if r['pose']==pose and r['condition']==condition], key=lambda r:r['weight'])
            ax.plot([r['angular_ratio'] for r in selected], [r['support'] for r in selected], marker=marker, label=condition)
            for r in selected:
                ax.annotate(str(r['weight']), (r['angular_ratio'], r['support']), xytext=(4,4), textcoords='offset points', fontsize=8)
        ax.axvline(1.1, color='gray', ls='--', label='angular screen 1.1x')
        ax.axhline(.99, color='black', ls=':', label='99% weight support')
        ax.set_title(pose, fontsize=10)
        ax.set_xlabel('Angular acceleration RMS / original same-condition baseline')
        ax.set_ylabel('Vertical ground force / weight')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed-pose support tradeoff: labels are COM objective weights')
    fig.text(.5,.02,'Engineering screens only; static snapshots are not standing or neural success. Canonical physics unchanged.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94))
    path=OUT/'support_tradeoff.png'
    fig.savefig(path,dpi=140,bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    render(json.loads((OUT/'support_tradeoff.json').read_text())['rows'])
