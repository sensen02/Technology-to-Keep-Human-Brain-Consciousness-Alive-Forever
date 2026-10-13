"""Plot only recorded dimensionless search results; no hard-coded outcome values."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/workspace_recheck'


def main():
    source = OUT / 'dimensionless_support_search.json'
    doc = json.loads(source.read_text())
    rows = [r for r in doc['rows'] if r['status'] == 'EVALUATED']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    if rows:
        x = list(range(len(rows)))
        for cond, shift, color in [('covered74', -.18, 'steelblue'), ('all90', .18, 'darkorange')]:
            values = [r[cond]['residual']['sensitivity_L_mm']['1.0']['all']['rms'] for r in rows]
            axes[0].bar([v+shift for v in x], values, width=.36, color=color, label=cond)
        axes[0].set_xticks(x, [r['label'] for r in rows], rotation=35, ha='right', fontsize=8)
        axes[0].legend()
    else:
        axes[0].text(.1, .5, 'No candidate passed the declared gates.', transform=axes[0].transAxes)
    axes[0].set_title('Bounded local search: evaluated candidates')
    axes[0].set_ylabel('Dimensionless RMS (L = 1 mm engineering choice)')
    axes[0].grid(axis='y', alpha=.2)
    axes[1].axis('off')
    notes = ['WHAT WAS ACTUALLY TESTED', '',
             'Translation: a/g; angular coordinates: angular a * L/g.',
             'L is declared, not biological calibration.',
             'All excluded muscles retain native steady-state floors.',
             'Canonical XML and physical parameters unchanged.', '',
             f"Evaluated: {len(rows)} / {len(doc['rows'])} proposed poses.",
             f"Per-solve max_nfev: {doc['method']['max_nfev']} (bounded, not a convergence proof).", '']
    for cond, result in doc.get('best', {}).items():
        dyn = result['integration_50ms']
        notes.extend([cond + ' best: ' + result['label'],
                      '  50 ms constant-command mechanical check:',
                      f"  root height {dyn['initial_root_height_mm']:.4f} -> {dyn['final_root_height_mm']:.4f} mm",
                      f"  warnings {dyn['warning_count']}; nonfoot contacts max {dyn['max_nonfoot_floor_contacts']}", ''])
    notes += ['Independent muscle commands relax neural-pool constraints.',
              'No neural-control, standing, gait or infeasibility claim.',
              'Hand-built research prototype; not a finished product.']
    axes[1].text(0, 1, '\n'.join(notes), va='top', fontsize=10, linespacing=1.4,
                 transform=axes[1].transAxes)
    fig.suptitle('Corrected support audit: dimensionally explicit, bounded evidence', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    path = OUT / 'dimensionless_support_search.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
