"""Bounded fixed-state audit of retained muscle repairs; no physics/neural steps."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
import engine.canonical_physics as cp
OUT=Path('/home/sensen/Desktop/engine修复验收/s4-critical-fix-regression/20261028T020000Z')
INITIAL=Path('/home/sensen/Desktop/engine修复验收/_历史归档/actual-neural-matched-20261018T031000Z/initial_runtime_state.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(cp.__file__),Path(cp.CANONICAL_XML),INITIAL]
    hashes={str(p):sha(p) for p in sources}
    config=dict(purpose='Fixed-state regression check, not support/standing test',states=['model_keyframe0','declared_neutral','latest_runtime_initial'],model_loads=1,forward=6,body_steps=0,neural_steps=0,inputs_sha256=hashes)
    f=OUT/'FIXED_STATE_PREREGISTRATION.json';assert not f.exists();f.write_text(json.dumps(config,indent=2)+'\n')
    m=cp.load_model();names=[m.actuator(i).name for i in range(m.nu)];assert m.nu==90
    d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);key=d.qpos.copy();neutral=key.copy()
    angles=(-.11,.35,.5,-.1,-2.8,0.,2.)
    labels=('Coxa_yaw','Coxa_pitch','Coxa_roll','Trochanter_yaw','Trochanter_pitch','Trochanter_roll','Tibia_pitch')
    for leg in ('LF','RF','LM','RM','LH','RH'):
        for label,v in zip(labels,angles):neutral[m.joint('joint_'+leg+label).qposadr[0]]=v
    rows={}
    for label,q in [('model_keyframe0',key),('declared_neutral',neutral),('latest_runtime_initial',np.load(INITIAL)['qpos'])]:
        d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d);f0=d.actuator_force.copy()
        d.act[:]=1;mujoco.mj_forward(m,d);f1=d.actuator_force.copy();active=f1-f0;lr=m.actuator_lengthrange;frac=(d.actuator_length-lr[:,0])/(lr[:,1]-lr[:,0])
        total_dead=np.flatnonzero(abs(f1)<1e-9);active_dead=np.flatnonzero(abs(active)<1e-9)
        A=np.zeros((m.nu,m.nv))
        for i in range(m.nu):
            start=int(d.moment_rowadr[i]);n=int(d.moment_rownnz[i]);A[i,d.moment_colind[start:start+n]]=d.actuator_moment[start:start+n]
        legs=[]
        for j in range(m.njnt):
            if int(m.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_HINGE):continue
            col=A[:,int(m.jnt_dofadr[j])];has_pos=bool(np.any(col>1e-9));has_neg=bool(np.any(col< -1e-9));legs.append(dict(joint=m.joint(j).name,positive_arm=has_pos,negative_arm=has_neg,bidirectional=has_pos and has_neg))
        rows[label]=dict(total_force_dead_count=len(total_dead),total_force_dead=[names[i] for i in total_dead],active_force_dead_count=len(active_dead),active_force_dead=[names[i] for i in active_dead],outside_band_count=int(((frac<0)|(frac>1)).sum()),bidirectional_joint_count=sum(x['bidirectional'] for x in legs),joint_arm_rows=legs,muscles=[dict(name=names[i],length=float(d.actuator_length[i]),fraction=float(frac[i]),force_at_act0=float(f0[i]),force_at_act1=float(f1[i]),active_delta=float(active[i])) for i in range(m.nu)])
    assert all(sha(Path(k))==v for k,v in hashes.items())
    result=dict(states=rows,actual_model_load=1,actual_forward=6,body_steps=0,neural_steps=0,inputs_unchanged=True,exit_code=0,note='Act0/act1 are offline probes only, never preload production episodes. Bidirectional moment arm is not a proof of sufficient force or support.')
    (OUT/'FIXED_STATE_REPAIR_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('total_force_dead_count','active_force_dead_count','outside_band_count','bidirectional_joint_count')} for k,v in rows.items()},indent=2))
if __name__=='__main__':main()
