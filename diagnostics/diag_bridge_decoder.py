"""Plot deterministic decoder causality tests, not real neural recordings."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.embodied.neural_motor_bridge import SpikeIntentBridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=ROOT/'outputs/neural_motor_bridge'
out.mkdir(parents=True,exist_ok=True)
sp=np.zeros((2000,4),dtype=bool)
sp[300:1400:10,:2]=True
sp[300:1400:30,2:]=True
traces={}
for name in ('intact','swap','silence','cut','sham'):
    bridge=SpikeIntentBridge(4,[0,1],[2,3],provenance='SYNTHETIC_SPIKES_ENGINEERING_CAUSAL_TEST')
    rows=[]
    for k in range(len(sp)):
        s=sp[k:k+1].copy()
        if name=='swap':s=s[:,::-1]
        if name=='silence':s[:]=False
        r=bridge.advance(s,spike_dt_s=.001,locomotion_permission=.8,connected=name!='cut')
        rows.append(r)
    traces[name]=rows
assert np.allclose([r['turn'] for r in traces['intact']],[-r['turn'] for r in traces['swap']])
assert all(r['turn']==0 for r in traces['silence'])
assert all(r['forward']==r['turn']==0 for r in traces['cut'])
assert traces['intact']==traces['sham']
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
t=np.arange(1,len(sp)+1)*.001
for name,rows in traces.items():
    axes[0].plot(t,[r['turn'] for r in rows],label=name)
    axes[1].plot(t,[r['forward'] for r in rows],label=name)
axes[0].set_ylabel('Engineered turn intent')
axes[1].set_ylabel('Explicit external drive')
axes[1].set_xlabel('Time (s)')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.suptitle('Spike decoder intervention tests — synthetic spikes, NOT identified neuron recordings')
fig.tight_layout()
fig.savefig(out/'decoder_causality.png',dpi=140)
(out/'decoder_causality.json').write_text(json.dumps({'status':'PASS','provenance':'deterministic synthetic binary spikes; engineered mapping; not biological validation','traces':traces},indent=2))
print('PASS: swap sign, silence steering, cut entire drive, sham identity; outputs',out)
