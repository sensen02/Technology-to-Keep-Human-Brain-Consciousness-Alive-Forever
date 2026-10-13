"""Figure: model-versus-annotation structural gaps."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / 'outputs/model_branches/model_annotation_gap'


def main():
    doc = json.loads((OUT / 'model_annotation_gap.json').read_text())
    cov = doc['annotation']['leg_muscle_targets']
    unmodelled = doc['reverse_gap']['targets']
    totals = [sum(v.values()) for t, v in ((t, cov[t]) for t in unmodelled)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].barh([t.replace('_muscle', '') for t in unmodelled], totals, color='#cc3311')
    for i, v in enumerate(totals):
        axes[0].text(v + .4, i, str(v), va='center', fontsize=9)
    axes[0].set(xlim=(0, max(totals) * 1.25),
                xlabel='Annotated leg motor neurons (all legs/sides)',
                title='Annotated muscles with NO actuator in the model')
    reasons = doc['forward_gap']['by_reason']
    labels = [k.replace('target ', '') for k in reasons]
    vals = [reasons[k] for k in reasons]
    axes[1].barh(labels, vals, color=['#4477aa', '#ee8866', '#aa3377'][:len(labels)])
    for i, v in enumerate(vals):
        axes[1].text(v + .15, i, str(v), va='center', fontsize=9)
    axes[1].set(xlim=(0, max(vals) * 1.3), xlabel='Actuators (of 90)',
                title='Modelled muscles with NO annotated pool on their leg')
    fig.text(.5, .01, 'Per leg the model articulates coxa (3), trochanter (3) and tibia (1) — there is no femur or '
                      'tarsus joint. Neither gap may be closed by mirroring cells across legs or sides.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    p = OUT / 'model_annotation_gap.png'
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
