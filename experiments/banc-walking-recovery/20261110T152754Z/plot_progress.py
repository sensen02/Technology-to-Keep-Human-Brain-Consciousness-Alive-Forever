"""Compare frozen actual same-window ANN results; no simulation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery')
OUT=Path(__file__).resolve().parent
RUN=ROOT/'20261110T152714Z'
SOURCES=[(300,ROOT/'20261110T114148Z/EVAL_GRU_ROLE_AUX'),(1000,ROOT/'20261110T120531Z/EVAL_GRU_ROLE_1000'),(1500,ROOT/'20261110T125712Z/EVAL_ROLE_1500'),(2000,ROOT/'20261110T125712Z/EVAL_ROLE_2000'),(2500,RUN/'EVAL_ROLE_2500'),(3000,RUN/'EVAL_ROLE_3000')]

def main():
    rows=[]
    for step,path in SOURCES:
        result=json.loads((path/'RESULT.json').read_text());a=result['autonomous']
        rows.append(dict(step=step,path=str(path),slip=a['slip_ratio_max'],forward=a['forward_signed_native_per_s'],COM=a['com_height_min'],nonfoot=a['nonfoot_force_max_native'],support=a['foot_support_mean'],gates=a['gates'],all_gates=a['all_gates'],cycles=result['autonomous_cycles']['complete_per_leg']))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    steps=[r['step'] for r in rows]
    for ax,key,title,limit,good in [(axes[0,0],'slip','Max per-leg loaded slip (9.6s pure ANN)',.5,'below'),(axes[0,1],'forward','Signed forward speed (native length/s)',.05,'above'),(axes[1,0],'COM','Minimum COM height (native length)',1.2,'above')]:
        values=[r[key] for r in rows];ax.plot(steps,values,'o-',color='#136f9b');ax.axhline(limit,color='#a33b29',linestyle='--',label=f'Gate: {good} {limit}')
        for x,y in zip(steps,values):ax.annotate(f'{y:.5f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(title);ax.set_xlabel('Global BC updates');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];ax.axis('off');latest=rows[-1]
    text=['ACTUAL PHYSICS / SAME BODY & GATES','1s zero settle +4.8s teacher +9.6s pure ANN','Full body:71 opaque meshes;42 finite PD motors','No adhesion / no root writes / no extra forces','',f'Latest3000: '+('engineering PASS' if latest['all_gates'] else 'engineering FAIL'),f'Slip {latest["slip"]:.9f}; forward {latest["forward"]:.6f}',f'Nonfoot max {latest["nonfoot"]:.6f}; support {latest["support"]:.6f}','Cycle counts: '+' '.join(f'{k}{v}' for k,v in latest['cycles'].items()),'','Original native acceptance: separate audit','Substitute ANN / derived body; not connectome']
    ax.text(.02,.98,'\n'.join(text),va='top',fontsize=10,linespacing=1.65)
    fig.suptitle('Full-body neural walking: fixed continuation, all failures retained',fontsize=15)
    path=OUT/'ROUND5_GAIT_PROGRESS.png';fig.savefig(path,dpi=160);plt.close(fig)
    (OUT/'ROUND5_GAIT_RESULTS.json').write_text(json.dumps(dict(rows=rows,plot=str(path)),indent=2)+'\n')
    print(json.dumps(dict(plot=str(path),rows=len(rows),latest_slip=latest['slip'],latest_engineering=latest['all_gates'])))

if __name__=='__main__':main()
