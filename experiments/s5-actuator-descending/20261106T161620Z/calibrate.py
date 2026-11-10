"""Declared assisted actuator calibration. Pulses are NOT neural control."""
import json, hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from flygym.compose.fly.musculoskeletal import _load_mjcf
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/calibration')
P=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z')
DT=0.0000125; CDT=.001; JOINT='joint_LFTibia_pitch'
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 s=_load_mjcf(P/'MODEL_JOINT_TORQUE.xml'); orig=s.compile(); d=mj.MjData(orig)
 mj.mj_resetDataKeyframe(orig,d,0); mj.mj_forward(orig,d)
 body=orig.body('Thorax').id
 s.add_equality(name='diagnostic_thorax_fixture',type=mj.mjtEq.mjEQ_WELD,objtype=mj.mjtObj.mjOBJ_BODY,name1='Thorax',data=[0,0,0,*d.xpos[body],*d.xquat[body],1],solref=[.002,1])
 for j in orig.actuator_trnid[:,0]:
  name=orig.joint(j).name
  if name!=JOINT:
   s.add_equality(name='fixture_'+name,type=mj.mjtEq.mjEQ_JOINT,objtype=mj.mjtObj.mjOBJ_JOINT,name1=name,data=[float(d.qpos[orig.jnt_qposadr[j]]),0,0,0,0,0,0,0,0,0,0],solref=[.002,1])
 m=s.compile(); m.opt.gravity[:]=0
 assert m.nmesh==71 and m.nu==42 and m.neq==42
 return m

def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'RESULTS.json').exists(),'Do not overwrite completed calibration'
 mapping=json.loads((P/'POOL_TO_JOINT.json').read_text())
 pools=[x for x in mapping['muscle_to_joint'] if x['joint']==JOINT]
 assert {x['sign'] for x in pools}=={-1,1}
 protocol=dict(importance='核心必需',joint=JOINT,pools=pools,gravity=0,physics_dt=DT,command_dt=CDT,duration=.1,
  arms=['torque','incremental','muscle'],pulse=[-.2,0,.2],pulse_duration=.03,
  neural=False,assistance='Thorax weld + 41 other joint equalities; gravity zero; finite solver constraints',
  parameters='kp=limit/0.2rad; damping=limit*0.005s/rad; vmax=2rad/s; activation_tau=0.01s; muscle passive kp=0.1*kp',
  checks='finite,no solver warnings,torque bounded; positive pulse displacement > zero and negative < zero; no autonomous-support claim',
  sources={str(p):sha(p) for p in [Path(__file__),P/'MODEL_JOINT_TORQUE.xml',P/'POOL_TO_JOINT.json']})
 dump(O/'PREREGISTRATION.json',protocol)
 m=build(); j=m.joint(JOINT).id; qa=m.jnt_qposadr[j]; va=m.jnt_dofadr[j]
 a=int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]); lim=float(m.actuator_ctrlrange[a,1]); kp=lim/.2; damp=lim*.005
 results=[]
 for arm in protocol['arms']:
  for pulse in protocol['pulse']:
   d=mj.MjData(m); mj.mj_resetDataKeyframe(m,d,0); mj.mj_forward(m,d)
   q0=float(d.qpos[qa]); ref=q0; activation=np.zeros(2); rows=[]; control=0.
   for step in range(round(.1/DT)):
    t=step*DT; u=pulse if t<.03 else 0.
    if step%round(CDT/DT)==0:
     ref=float(np.clip(ref+CDT*2*u,*m.jnt_range[j]))
    if arm=='torque': control=lim*u
    elif arm=='incremental': control=kp*(ref-d.qpos[qa])-damp*d.qvel[va]
    else:
     target=np.array([max(u,0),max(-u,0)])
     activation=target+(activation-target)*np.exp(-DT/.01)
     control=lim*(activation[0]-activation[1])+.1*kp*(q0-d.qpos[qa])-damp*d.qvel[va]
    d.ctrl[:]=0; d.ctrl[a]=np.clip(control,-lim,lim)
    mj.mj_step(m,d)
    rows.append([d.time,d.qpos[qa],d.qvel[va],d.ctrl[a],ref,*activation,float(np.linalg.norm(d.qfrc_constraint)),float(np.max(np.abs(d.efc_force))) if d.nefc else 0])
   arr=np.array(rows); tag=arm+('_neg' if pulse<0 else '_pos' if pulse>0 else '_zero')
   np.savez_compressed(O/(tag+'.npz'),rows=arr)
   results.append(dict(arm=arm,pulse=pulse,q0=q0,displacement=float(arr[-1,1]-q0),finite=bool(np.isfinite(arr).all()),max_torque=float(abs(arr[:,3]).max()),limit=lim,warnings=int(d.warning.number.sum()),file=str(O/(tag+'.npz'))))
 checks={}
 for arm in protocol['arms']:
  r=[x for x in results if x['arm']==arm]
  checks[arm+'_direction']=r[0]['displacement']<r[1]['displacement']<r[2]['displacement']
 checks['finite_bounded_warning_free']=all(x['finite'] and x['max_torque']<=x['limit'] and x['warnings']==0 for x in results)
 dump(O/'RESULTS.json',dict(runs=results,checks=checks,passed=all(checks.values()),physics_runs=9,neural_runs=0,importance='核心必需'))
 print(json.dumps(dict(checks=checks,output=str(O)),ensure_ascii=False))
if __name__=='__main__': main()
