"""Independent saved-array motion/support comparison, no simulation or optimization."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z');BASE=OUT/'learning/evaluation';W=24.446354755878005
summary={};fig,ax=plt.subplots(3,2,figsize=(13,10),layout='constrained')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
for col,name in enumerate(['selected','zero']):
    p=BASE/name;r=json.loads((p/'RESULT.json').read_text());b=load(p/'BODY.npz');n=load(p/'NATIVE.npz')['rows'];s=load(p/'SLIP.npz')['rows'];pol=load(p/'POLICY.npz');proto=json.loads((p/'PROTOCOL.json').read_text());duration=float(b['time'][-1]-b['time'][0]);head=np.array(proto['initial_heading_xy']);com=b['com'];disp=com[-1]-com[0];path=float(np.linalg.norm(np.diff(com[:,:2],axis=0),axis=1).sum());pathspeed=path/duration
    speeds=s[:,7:13].sum(0)/np.maximum(s[:,1:7].sum(0),1e-15);counts=(n[:,1:7]>.05*W).sum(1)
    stats=dict(duration_s=duration,forward_COM_displacement=float(disp[:2]@head),lateral_COM_displacement=float(disp[:2]@np.array([-head[1],head[0]])),COM_pathlength=path,COM_path_mean_speed=pathspeed,net_horizontal_displacement=float(np.linalg.norm(disp[:2])),material_slip_loadweighted_speed=speeds.tolist(),material_slip_to_pathspeed=(speeds/max(pathspeed,1e-12)).tolist(),mean_foot_support_weight=float(n[:,1:7].sum(1).mean()/W),min_foot_support_weight=float(n[:,1:7].sum(1).min()/W),nonfoot_force_max=float(n[:,7].max()),COM_z_min=float(n[:,10].min()),min_bearing_legs=int(counts.min()),mean_abs_rate=float(np.abs(pol['predicted_rate']).mean()),full_checkpoint_pass=bool(r['physical_gate']['passed']),reward=r['reward'],theta_nonzero=int(np.count_nonzero(load(p/'ACTOR.npz')['theta'])),neural_voltage_range_mV=r['neural_voltage_range_mV'],data_sha256={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['BODY.npz','NATIVE.npz','SLIP.npz','POLICY.npz']})
    summary[name]=stats;t=b['time']-b['time'][0];nt=n[::80,0]-b['time'][0]
    ax[0,col].plot(t,(com[:,:2]-com[0,:2])@head,label='forward');ax[0,col].plot(t,(com[:,:2]-com[0,:2])@np.array([-head[1],head[0]]),label='lateral');ax[0,col].set_title(name+' | privileged evaluation only');ax[0,col].set_ylabel('COM displacement (native)');ax[0,col].legend()
    ax[1,col].plot(nt,n[::80,1:7].sum(1)/W,label='foot/W');ax[1,col].plot(nt,n[::80,7]/W,label='nonfoot/W');ax[1,col].set_ylabel('Ground load');ax[1,col].legend()
    ax[2,col].plot(t,com[:,2],label='COM z');ax[2,col].set_xlabel('Release time (s)');ax[2,col].set_ylabel('COM height (native)');ax[2,col].legend()
fig.suptitle('Frozen5s validation after1s training search | support and reward are not walking acceptance')
fig.savefig(OUT/'PHYSICAL_EVALUATION.png',dpi=150);plt.close(fig)
summary['limits']='Saved arrays only; no physicalsteps. Material slip at0.5ms, supportfullnative12.5us. Low-speed zero baseline ratios have small denominator. Initialreference checkpoint isnotautonomousstartup.'
with (OUT/'PHYSICAL_COMPARISON.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
