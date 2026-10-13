"""Figure for the load-support audit: per-leg initial load and support-over-weight over time."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/model_branches/support_audit'


def main():
    import json
    doc = json.loads((OUT / 'support_audit_results.json').read_text())
    weight = doc['weight_and_mass']['weight_model_units']
    ideal = doc['weight_and_mass']['weight_per_leg_model_units']
    detail = doc['initial_leg_detail']['canonical_100us']
    legs = list(detail)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    vals = [detail[leg]['vertical_force'] for leg in legs]
    bars = axes[0].bar(legs, vals, color=['#4477aa' if v > 0 else '#cc3311' for v in vals])
    axes[0].axhline(ideal, color='k', ls='--', lw=1, label=f'balanced share = weight/6 = {ideal:.3f}')
    axes[0].axhline(0, color='gray', lw=.8)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    axes[0].set(title='Initial vertical load per leg (rm is airborne)', ylabel='Vertical force (mN = model units)',
                xlabel='Leg', ylim=(0, max(ideal * 1.45, max(vals) * 1.2)))
    axes[0].legend(fontsize=8)
    with np.load(OUT / 'support_audit_traces.npz') as z:
        for us, style in ((100, '-'), (25, '-')):
            t = z[f'{us}us_time_s'] * 1000
            axes[1].plot(t, z[f'{us}us_grf_z_B'] / weight, style, lw=1.6, label=f'{us}us body step')
    axes[1].axhline(1.0, color='k', ls='--', lw=1, label='support = body weight')
    axes[1].axhline(0.5, color='#cc3311', ls=':', lw=1, label='half weight')
    axes[1].set(title='Total ground reaction force / body weight', xlabel='Time (ms)',
                ylabel='Support / weight (dimensionless)', ylim=(0, 2.5))
    axes[1].legend(fontsize=8)
    fig.text(.5, .01, 'Ratio is independent of mass-unit interpretation. Initial support is 63.8% of weight; '
                      'the deficit is not absence of force but an unloaded leg and a collapsing stance.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'support_audit.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
