"""Plot the compact verified force-audit summary, not mixed generalized norms."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/workspace_recheck'


def render(summary):
    rows = summary['rows']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [r['pose'] + '\n' + r['condition'] for r in rows]
    x = range(len(rows))
    axes[0].bar(x, [r['support_fraction'] for r in rows], color='steelblue')
    axes[0].axhline(1, color='black', ls='--', label='weight balance, vertical only')
    axes[0].set_ylabel('World vertical ground force / body weight')
    axes[0].set_title('Same-pose support comparisons')
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [r['com_vertical_mm_s2'] for r in rows], color='darkorange')
    axes[1].axhline(0, color='black', lw=1)
    axes[1].set_ylabel('Whole-body COM vertical acceleration (mm/s²)')
    axes[1].set_title('Mass-weighted COM Jacobian check')
    for ax in axes:
        ax.set_xticks(list(x), labels, rotation=35, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Ground support and COM acceleration: matched initial-state snapshots')
    fig.text(.5, .01, 'Native physics unchanged. Not neural success, standing proof or causal ablation. Research prototype.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    target = OUT / 'support_force_audit.png'
    fig.savefig(target, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    source = OUT / 'support_force_audit.json'
    doc = json.loads(source.read_text())
    if doc['status'] != 'MEASURED_WITHIN_DECLARED_TOLERANCES':
        raise RuntimeError('Refuse to plot a force audit that failed physical bookkeeping checks')
    rows = []
    aliases = {'round37_explicit_limit_repair': 'pose A (repaired round37)',
               'small_offset_01': 'pose B (offset 01)'}
    for pose in doc['poses']:
        for s in pose['snapshots']:
            rows.append({'pose': aliases.get(pose['label'], pose['label']),
                         'condition': s['condition'], 'support_fraction': s['support_over_weight'],
                         'com_vertical_mm_s2': s['COM_acceleration_mm_s2'][2]})
    render({'rows': rows})
