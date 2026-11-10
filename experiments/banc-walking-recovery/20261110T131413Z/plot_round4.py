"""Actual six-leg neural gait progress; all acceptance gates retained."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from identity import HERE,atomic_json,sha256


def main():
    root=HERE.parent
    items=[('NN role300','20261110T114148Z/EVAL_GRU_ROLE_AUX'),
      ('NN exact1000','20261110T120531Z/EVAL_GRU_ROLE_1000'),
      ('Foot-speed link cut','20261110T120951Z/EVAL_FOOT_VELOCITY_ZERO'),
      ('Input synapse pruning','20261110T123957Z/EVAL_PRUNED_GRU'),
      ('Physical PPO3','20261110T122634Z/EVAL_PPO_SLIP_3U'),
      ('NN exact1500','20261110T125712Z/EVAL_ROLE_1500'),
      ('NN exact2000','20261110T125712Z/EVAL_ROLE_2000')]
    rows=[]
    for label,directory in items:
        p=root/directory/'RESULT.json'
        if not p.exists():continue
        d=json.loads(p.read_text());rows.append(dict(label=label,path=str(p),sha256=sha256(p),autonomous=d['autonomous'],
            cycles=d['autonomous_cycles']['complete_per_leg'] if 'autonomous_cycles' in d else d.get('cycles',{})))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    panels=[('slip_ratio_max','Maximum loaded foot slip ratio',.5),('forward_signed_native_per_s','Signed forward native/s',.05),
      ('com_height_min','Minimum COM height',1.2),('nonfoot_force_max_native','Nonfoot peak / W',.2)]
    for ax,(key,title,gate) in zip(axes.flat,panels):
        values=[r['autonomous'].get(key,r['autonomous'].get('nonfoot_force_max_W',0)) for r in rows]
        if key=='nonfoot_force_max_native':values=np.array(values)/24.446354755878005
        ax.bar([r['label'] for r in rows],values,color=['#287c69' if r['autonomous'].get('all_gates',False) else '#b38336' for r in rows])
        ax.axhline(gate,color='#b00020',ls='--');ax.set_title(title);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Actual pure neural handover: six-leg forward gait emerges; slip gate remains 0.5')
    fig.savefig(HERE/'ROUND4_GAIT_PROGRESS.png',dpi=140);plt.close(fig)
    atomic_json(HERE/'ROUND4_GAIT_RESULTS.json',dict(results=rows,
      scope='declaredtraining/parameterinterventions separate; fulloriginalnativeacceptance stillrequired forwholegoal'))
    print([(r['label'],r['autonomous']['slip_ratio_max'],r['autonomous'].get('all_gates')) for r in rows])

if __name__=='__main__':main()
