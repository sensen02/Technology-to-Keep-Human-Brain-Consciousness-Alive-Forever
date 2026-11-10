"""Engineering antagonistic tensile actuators on preserved model tendon paths."""
import json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from flygym.compose.fly.musculoskeletal import _load_mjcf
P=Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/tendon_muscle')
NAMES=['LFTibia_flex_93434_tendon','LFTibia_extensor_93932_tendon'];JOINT='joint_LFTibia_pitch';DT=.0000125

def build():
 s=_load_mjcf(P/'MODEL_JOINT_TORQUE.xml');orig=s.compile();d=mj.MjData(orig);mj.mj_resetDataKeyframe(orig,d,0);mj.mj_forward(orig,d)
 body=orig.body('Thorax').id;relpos=-d.xmat[body].reshape(3,3).T@d.xpos[body];quat=d.xquat[body].copy();quat[1:]*=-1
 s.add_equality(name='fixture_thorax',type=mj.mjtEq.mjEQ_WELD,objtype=mj.mjtObj.mjOBJ_BODY,name1='Thorax',data=[0,0,0,*relpos,*quat,1],solref=[.002,1])
 for j in orig.actuator_trnid[:,0]:
  if orig.joint(j).name!=JOINT:
   s.add_equality(name='fixture_'+orig.joint(j).name,type=mj.mjtEq.mjEQ_JOINT,objtype=mj.mjtObj.mjOBJ_JOINT,name1=orig.joint(j).name,data=[float(d.qpos[orig.jnt_qposadr[j]]-orig.qpos0[orig.jnt_qposadr[j]]),0,0,0,0,0,0,0,0,0,0],solref=[.002,1])
 for name in NAMES:
  s.add_actuator(name='engineering_'+name,trntype=mj.mjtTrn.mjTRN_TENDON,target=name,gaintype=mj.mjtGain.mjGAIN_FIXED,gainprm=[1,0,0,0,0,0,0,0,0,0],biastype=mj.mjtBias.mjBIAS_NONE,dyntype=mj.mjtDyn.mjDYN_NONE,gear=[1,0,0,0,0,0])
 for key in s.keys:key.ctrl=list(key.ctrl)+[0.,0.]
 m=s.compile();m.opt.gravity[:]=0;m.opt.disableflags|=int(mj.mjtDisableBit.mjDSBL_CONTACT)
 assert m.nmesh==71 and m.nu==44 and m.neq==42
 d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d)
 assert abs(d.efc_pos[d.efc_type==mj.mjtConstraint.mjCNSTR_EQUALITY]).max()<1e-9
 return m

def force_law(activation,length,velocity,l0,fmax):
 lam=length/l0;v=velocity/(10*l0)
 fl=np.exp(-((lam-1)/.25)**2)
 fv=np.where(v<0,np.maximum(0,1+v)/(1+4*np.maximum(-v,0)),1+.5*np.maximum(v,0)/(np.maximum(v,0)+.25))
 passive=.05*(np.maximum(lam-1,0)/.1)**2
 viscous=.02*np.maximum(v,0)
 return np.clip(fmax*(activation*fl*fv+passive+viscous),0,fmax),fl,fv,passive

def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];tids=[m.tendon(n).id for n in NAMES];d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d);q0=float(d.qpos[qa]);l0=d.ten_length[tids].copy()
 # Finite-difference geometry scan before trials; original tendons, no invented moment arms.
 slopes=[]
 for q in np.linspace(*m.jnt_range[j],41):
  lengths=[]
  for offset in [-1e-6,1e-6]:
   d.qpos[qa]=q+offset;mj.mj_forward(m,d);lengths.append(d.ten_length[tids].copy())
  slopes.append((lengths[1]-lengths[0])/2e-6)
 slopes=np.array(slopes);limit=float(m.actuator_ctrlrange[6,1]);fmax=.01*limit/np.max(abs(slopes),axis=0)
 assert np.all(fmax>0)
 protocol=dict(importance='核心必需',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=hashlib.sha256((P/'MODEL_JOINT_TORQUE.xml').read_bytes()).hexdigest(),tendons=NAMES,l0=l0.tolist(),fmax=fmax.tolist(),max_moment_arm=np.max(abs(slopes),axis=0).tolist(),force_budget='each tendon <=.01*original joint torque limit over scanned range; engineering small-signal calibration, not biological strength',physics_dt=DT,activation_tau_on_s=.01,activation_tau_off_s=.03,duration_s=.2,input_on_s=.03,conditions={'zero':[0,0],'flex':[.2,0],'ext':[0,.2],'both':[.2,.2]},assistance='original full body and tendon paths; root/other joints fixed, no gravity/contact; old42 actuator commands zero',criterion='pulse-end flex angle>zero>ext, zero drift<1e-6rad, within joint range, tensile nonnegative force, bounded force and finite warning-free; not neural support')
 (O/'PREREGISTRATION.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');records=[]
 for name,command in protocol['conditions'].items():
  d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d);activation=np.zeros(2);rows=[]
  for step in range(round(.2/DT)):
   target=np.array(command) if step*DT<.03 else np.zeros(2);tau=np.where(target>activation,.01,.03);activation=target+(activation-target)*np.exp(-DT/tau)
   mj.mj_forward(m,d);length=d.ten_length[tids].copy();velocity=d.ten_velocity[tids].copy();force,fl,fv,passive=force_law(activation,length,velocity,l0,fmax)
   d.ctrl[:]=0;d.ctrl[-2:]=-force
   # Transmission supplies all generalized torques; no qfrc_applied or posture correction.
   mj.mj_step(m,d)
   rows.append([d.time,d.qpos[qa],d.qvel[va],*activation,*length,*velocity,*force,*fl,*fv,*passive,float(np.linalg.norm(d.qfrc_constraint))])
  arr=np.array(rows);np.savez_compressed(O/(name+'.npz'),rows=arr)
  records.append(dict(condition=name,pulse_end_shift=float(arr[2399,1]-q0),final_shift=float(arr[-1,1]-q0),range_ok=bool((arr[:,1]>=m.jnt_range[j,0]).all() and (arr[:,1]<=m.jnt_range[j,1]).all()),force_bounded=bool((arr[:,9:11]>=0).all() and (arr[:,9:11]<=fmax).all()),finite=bool(np.isfinite(arr).all()),warnings=int(d.warning.number.sum())))
 zero,flex,ext,both=records;checks=dict(direction=flex['pulse_end_shift']>zero['pulse_end_shift']>ext['pulse_end_shift'],zero_drift=abs(zero['final_shift'])<1e-6,all_range=all(x['range_ok'] for x in records),bounded_finite=all(x['force_bounded'] and x['finite'] and x['warnings']==0 for x in records))
 (O/'SUMMARY.json').write_text(json.dumps(dict(importance='核心必需',records=records,checks=checks,passed=all(checks.values()),physics_runs=4,neural_runs=0),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(records=records,checks=checks)))
if __name__=='__main__':main()
