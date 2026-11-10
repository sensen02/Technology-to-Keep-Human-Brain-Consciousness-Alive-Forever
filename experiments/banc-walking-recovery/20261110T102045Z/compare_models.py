"""Compare matched BC-budget control and half-student recovery actual results."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from identity import HERE,atomic_json


def main():
    conditions=[('Parent recovery',HERE.parent/'20261110T094846Z/EVAL_FF_RECOVERY'),
                ('Old-data budget control',HERE/'EVAL_BUDGET_CONTROL'),
                ('Half-student data',HERE/'EVAL_HALF_RECOVERY')]
    results=[]
    for name,directory in conditions:
        p=directory/'RESULT.json'
        if not p.exists():continue
        d=json.loads(p.read_text())
        results.append(dict(name=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
          autonomous=d['autonomous'],cycles=d['autonomous_cycles']['complete_per_leg'],transition_exact=d['transition']['all_exact']))
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax,key,title,gate in [(axes[0],'com_height_min','Minimum COM height',1.2),
          (axes[1],'nonfoot_force_max_native','Peak nonfoot force / W',.2),
          (axes[2],'forward_signed_native_per_s','Signed forward (native/s)',.05)]:
        values=[r['autonomous'][key] for r in results]
        if key.startswith('nonfoot'):values=np.array(values)/24.446354755878005
        ax.bar([r['name'] for r in results],values,color=['#287c69' if r['autonomous']['all_gates'] else '#af493c' for r in results])
        ax.axhline(gate,color='#b00020',ls='--');ax.set_title(title);ax.tick_params(axis='x',rotation=22);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Actual continuous neural takeover: equal additional training budget, different recovery data')
    path=HERE/'ROUND3_COMPARISON.png';fig.savefig(path,dpi=150);plt.close(fig)
    atomic_json(HERE/'ROUND3_RESULTS.json',dict(results=results,scope='sameparent/rawmodel/teacher/physics/1000steps/seed; originalfullacceptance remainsseparate'))
    print(json.dumps([{k:r[k] for k in ['name','transition_exact','cycles']}|{'gates':r['autonomous']['all_gates'],'COM':r['autonomous']['com_height_min']} for r in results]),flush=True)

if __name__=='__main__':main()
