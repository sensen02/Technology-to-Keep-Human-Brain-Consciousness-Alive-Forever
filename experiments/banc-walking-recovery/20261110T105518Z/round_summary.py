"""Bounded final round comparison with genuine physical outcomes only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from identity import HERE,atomic_json,sha256


def main():
    root=HERE.parent
    variants=[('Parent recovery','20261110T094846Z/EVAL_FF_RECOVERY'),
      ('Old-data budget','20261110T102045Z/EVAL_BUDGET_CONTROL'),
      ('Half-student data','20261110T102045Z/EVAL_HALF_RECOVERY'),
      ('Student bursts','20261110T103413Z/EVAL_FF_BURST'),
      ('Paired feedback','20261110T104706Z/EVAL_FF_FEEDBACK'),
      ('Inhibitory ANN','20261110T105518Z/EVAL_FF_INHIBITORY')]
    results=[]
    for name,directory in variants:
        p=root/directory/'RESULT.json'
        if not p.exists():continue
        d=json.loads(p.read_text());results.append(dict(name=name,path=str(p),sha256=sha256(p),
          metrics=d['autonomous'],cycles=d['autonomous_cycles']['complete_per_leg'],transition_exact=d['transition']['all_exact']))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    panels=[('forward_signed_native_per_s','Signed forward (native/s)',.05),('com_height_min','Minimum COM height',1.2),
      ('nonfoot_force_max_native','Peak nonfoot force / W',.2),('slip_ratio_max','Maximum per-leg slip ratio',.5)]
    for ax,(key,title,gate) in zip(axes.flat,panels):
        values=[r['metrics'][key] for r in results]
        if key.startswith('nonfoot'):values=np.asarray(values)/24.446354755878005
        ax.bar([r['name'] for r in results],values,color=['#287c69' if r['metrics']['all_gates'] else '#af493c' for r in results])
        ax.axhline(gate,color='#b00020',ls='--');ax.set_title(title);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Actual continuous neural takeover: assistance removed, original engineering gates retained')
    path=HERE/'ROUND3_ALL_RESULTS.png';fig.savefig(path,dpi=140);plt.close(fig)
    atomic_json(HERE/'ROUND3_ALL_RESULTS.json',dict(results=results,
      scope='controlled candidates with declared budget/architecture differences; no MSE or algorithmteacher result treated as neuralwalking'))
    print(json.dumps([dict(name=r['name'],COM=r['metrics']['com_height_min'],gates=r['metrics']['all_gates'],cycles=r['cycles']) for r in results]),flush=True)

if __name__=='__main__':main()
