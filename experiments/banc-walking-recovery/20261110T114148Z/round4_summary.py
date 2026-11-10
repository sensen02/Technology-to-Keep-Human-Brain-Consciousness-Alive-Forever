"""Saved-outcome comparison; body support is not autonomous walking."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from identity import HERE,atomic_json,sha256


def main():
    root=HERE.parent
    variants=[('Error-input ANN','20261110T105518Z/EVAL_FF_INHIBITORY'),
      ('Target-input FF','20261110T111240Z/EVAL_FF_TARGET'),
      ('Zero-rate hold','20261110T111240Z/EVAL_ZERO_MATCHED_RETRY'),
      ('Slow GRU','20261110T112326Z/EVAL_SLOW_GRU'),
      ('Latent-loss FF','20261110T112848Z/EVAL_FF_LATENT'),
      ('FF exact5000','20261110T113454Z/EVAL_FF_TARGET_5000'),
      ('GRU role-supervised','20261110T114148Z/EVAL_GRU_ROLE_AUX')]
    results=[]
    for name,folder in variants:
        path=root/folder/'RESULT.json'
        if not path.exists():continue
        d=json.loads(path.read_text());m=d['autonomous']
        body=all(m['gates'][g] for g in ['support','nonfoot','com'])
        results.append(dict(name=name,control=d['control'],path=str(path),sha256=sha256(path),metrics=m,
          body_support_gates_only=body,transition_exact=d['transition']['all_exact'],cycles=d['autonomous_cycles']['complete_per_leg']))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    panels=[('com_height_min','Minimum COM height',1.2),('nonfoot_force_max_native','Nonfoot peak / W',.2),
      ('forward_signed_native_per_s','SIGNED forward (native/s)',.05),('slip_ratio_max','Max per-leg slip ratio',.5)]
    for ax,(key,title,gate) in zip(axes.flat,panels):
        values=[r['metrics'][key] for r in results]
        if key.startswith('nonfoot'):values=np.asarray(values)/24.446354755878005
        if key=='slip_ratio_max':
            values=[np.nan if r['control']=='zero' else v for r,v in zip(results,values)]
        colors=['#777777' if r['control']=='zero' else ('#347b9b' if r['body_support_gates_only'] else '#af493c') for r in results]
        ax.bar([r['name'] for r in results],values,color=colors)
        ax.axhline(gate,color='#b00020',ls='--');ax.set_title(title);ax.tick_params(axis='x',rotation=28);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Actual uninterrupted takeover: blue = body support retained ONLY; gray = actuator hold; walking gates separate')
    fig.savefig(HERE/'ROUND4_COMPARISON.png',dpi=140);plt.close(fig)
    atomic_json(HERE/'ROUND4_RESULTS.json',dict(results=results,scope='derived engineering gates, original fullautonomous acceptance not established; zero-output support cannot be attributed to ANN'))
    print(json.dumps([dict(name=r['name'],body_support=r['body_support_gates_only'],all_gates=r['metrics']['all_gates'],forward=r['metrics']['forward_signed_native_per_s'],COM=r['metrics']['com_height_min']) for r in results]),flush=True)

if __name__=='__main__':main()
