"""Teacher success and neural failures are distinct conclusions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from identity import HERE,atomic_json
from physics import W


def main():
    teacher=json.loads((HERE/'teacher_native_replay/RESULT.json').read_text())
    names=['Algorithm teacher','FF neural takeover','GRU neural takeover','Zero-rate hold']
    rows=[teacher]
    for name in ['EVAL_FF238_HANDOVER','EVAL_GRU238_HANDOVER','EVAL_ZERO_RATE_CONTROL']:
        rows.append(json.loads((HERE/name/'RESULT.json').read_text())['autonomous'])
    colors=['#287c69','#af493c','#b17c42','#808080']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    panels=[('forward_signed_native_per_s','Signed forward (native length/s)',.05),
            ('com_height_min','Minimum COM height (native)',1.2),
            ('nonfoot_force_max_native','Peak nonfoot force / body weight',.2),
            ('slip_ratio_max','Maximum per-leg slip ratio (lower better)',.5)]
    for ax,(key,label,gate) in zip(axes.flat,panels):
        vals=[r[key]/W if key=='nonfoot_force_max_native' else r[key] for r in rows]
        if key=='slip_ratio_max':vals[-1]=np.nan
        ax.bar(names,vals,color=colors)
        ax.axhline(gate,color='#b00020',linestyle='--',label='Derived engineering gate')
        ax.set_title(label);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8)
        if key=='slip_ratio_max':ax.text(3,.1,'N/A\nno forward',ha='center',va='bottom')
    fig.suptitle('Same physical body: algorithm teacher passes; observation-only neural takeover NOT yet stable')
    path=HERE/'TEACHER_NEURAL_RESULTS.png';fig.savefig(path,dpi=150);plt.close(fig)
    atomic_json(HERE/'TEACHER_NEURAL_RESULTS.json',dict(labels=names,results=rows,
        scope='teacher10s and neural/zero9.6s diagnostics; not causal architecture comparison; no final original acceptance claim'))
    print(str(path))

if __name__=='__main__':main()
