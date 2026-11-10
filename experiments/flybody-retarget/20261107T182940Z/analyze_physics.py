"""Compare saved physical failures; scratch forward only, no steps or tuning."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco as mj
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z')
fig,ax=plt.subplots(2,3,figsize=(16,9));summary={};calls=0
for col,tag in enumerate(['teacherderived_physics','teacherderived_slow4_physics']):
    p=OUT/tag;result=json.loads((p/'RESULT.json').read_text());prot=json.loads((p/'PROTOCOL.json').read_text());ver=json.loads((p/'VERIFICATION.json').read_text())
    def load(name):
        with np.load(p/name) as z:return {k:z[k] for k in z.files}
    b=load('BODY.npz');cmd=load('COMMANDS.npz')['rows'];n=load('NATIVE.npz')['rows'];init=load('INITIAL.npz')
    m=mj.MjModel.from_binary_path(str(p/'MODEL.mjb'));d=mj.MjData(m);nonfoot={};tilt=[];planes=m.geom_type==mj.mjtGeom.mjGEOM_PLANE
    assert ver['status']=='REPLAY_PASS' and max(ver['max_errors'].values())==0
    for i,t in enumerate(b['time']):
        d.time=t
        for key in ['qpos','qvel','ctrl','qacc_warmstart']:getattr(d,key)[:]=b[key][i]
        mj.mj_forward(m,d);calls+=1;tilt.append(float(np.arccos(np.clip(d.xmat[m.body('Thorax').id].reshape(3,3)[2,2],-1,1))))
        for ci,c in enumerate(d.contact):
            ga,gb=int(c.geom1),int(c.geom2)
            if not(planes[ga] or planes[gb]):continue
            g=gb if planes[ga] else ga;name=m.geom(g).name or str(g)
            if 'Tars' in name and name[:2] in ['LF','LM','LH','RF','RM','RH']:continue
            f=np.zeros(6);mj.mj_contactForce(m,d,ci,f);fz=float((c.frame.reshape(3,3).T@f[:3])[2])*(1 if planes[ga] else -1)
            if fz<=0:continue
            row=nonfoot.setdefault(name,dict(first_s=float(t),last_s=float(t),sample_contacts=0,peak_native=0.))
            row['last_s']=float(t);row['sample_contacts']+=1;row['peak_native']=max(row['peak_native'],fz)
    error=b['qpos'][1:,init['qa']]-cmd[:,43:85];mean=np.sqrt(np.mean(error**2,axis=0));worst=int(mean.argmax())
    summary[tag]=dict(nonfoot_1ms_scratch_contacts=nonfoot,joint_tracking_RMS_rad=float(np.sqrt(np.mean(error**2))),worst_joint_RMS_rad=float(mean[worst]),worst_joint=m.joint(int(m.actuator_trnid[worst,0])).name,max_tilt_rad=max(tilt),max_command_rate_rad_s=float(np.abs(cmd[:,1:43]).max()),torque_saturated_entries=ver['recovered_saturated_entries'],native_steps=result['native_steps'],all_replay_errors_zero=True)
    label='original tempo' if col==0 else 'source-time scale4'
    ax[0,0].plot(b['time'],b['com'][:,2],label=label);ax[0,1].plot(n[::40,0],n[::40,1:7].sum(1)/24.446354755878005,label=label);ax[0,2].plot(n[::40,0],n[::40,7]/24.446354755878005,label=label)
    ax[1,0].plot(b['time'],tilt,label=label);ax[1,1].plot(b['time'][1:],np.sqrt(np.mean(error**2,axis=1)),label=label)
    ax[1,2].plot(b['com'][:,0],b['com'][:,1],label=label)
for a,title,ylabel in zip(ax.flat,['COM height','Claw/tarsal support','Nonfoot contact','Thorax tilt','Finite-PD joint tracking error','Actual COM path'],['current native','body weight','body weight','radians','RMS radians','native y']):
    a.set_title(title);a.set_ylabel(ylabel);a.set_xlabel('physical time(s)' if title!='Actual COM path' else 'native x');a.grid(alpha=.25);a.legend()
fig.suptitle('Both physical candidates FAIL; time scale4 is engineering retiming, not neural locomotion')
fig.tight_layout();fig.savefig(OUT/'PHYSICS_COMPARISON.png');plt.close(fig)
report=dict(episodes=summary,diagnostic_phase='1ms scratch forward contact attribution; complete native forces in episode NATIVE.npz',mj_forward_calls=calls,mj_step_calls=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (OUT/'PHYSICS_DIAGNOSIS.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
