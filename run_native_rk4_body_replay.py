"""Explicit derived RK4 body-only replay, no neural controller or pose correction."""
from pathlib import Path
import json,time,hashlib,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
RK=Path('/home/sensen/Desktop/engine修复验收/S1-activation-rk4-20261014T230203')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import mujoco
 from engine import canonical_physics as cp
 from flygym.compose.fly import musculoskeletal as mus
 from run_s2_fixed_stimulus import dense_moment
 out=Path('/home/sensen/Desktop/engine修复验收')/('S1-native-rk4-body-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 r=dict(status='RUNNING',scope='DERIVED_ZERO_FLOOR_NATIVE_RK4_FIXED_COMMAND_BODY_ONLY_NOT_STANDING',cases={},steps=0,explicit_forward_calls=0,callback_calls=0)
 posepath=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz';paths=[Path(__file__),ROOT/'S1_NATIVE_RK4_BODY_REPLAY_PROTOCOL.zh-CN.md',Path(cp.__file__),Path(mus.__file__),ROOT/'run_s2_fixed_stimulus.py',RK/'rk4.npz',RK/'report.json',posepath];hashes={str(p):sha(p) for p in paths};saved={}
 try:
  assert mujoco.get_mjcb_control() is None,'preexisting control callback'
  with np.load(RK/'rk4.npz') as z:commands=z['commands'].copy();reference={k:z[k].copy() for k in z.files if k!='commands'}
  with np.load(posepath) as z:pose=z['projected_seated_qpos'].copy()
  spec=mus._load_mjcf(str(cp.CANONICAL_XML));assets=[cp.CANONICAL_XML]+[Path(x.file).resolve() for x in spec.meshes];lib=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));hashes.update({str(p):sha(p) for p in assets+[lib]});assert sha(cp.CANONICAL_XML)==cp.CANONICAL_XML_SHA256
  m=cp.load_model();assert (m.nq,m.nv,m.na,m.nu)==(49,48,90,90)
  checks=dict(muscle=bool(np.all(m.actuator_dyntype==int(mujoco.mjtDyn.mjDYN_MUSCLE))),ordered_act=bool(np.array_equal(m.actuator_actadr,np.arange(90)) and np.all(m.actuator_actnum==1)),activation_unlimited=not bool(m.actuator_actlimited.any()),actearly_false=not bool(m.actuator_actearly.any()),no_history=m.nhistory==0,no_plugins=m.nplugin==0,no_sleep=not bool(m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)),no_disabled_group=m.opt.disableactuator==0,no_disabled_actuation=not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)),no_actuator_gravcomp=not bool(m.jnt_actgravcomp.any()),no_joint_force_limit=not bool(m.jnt_actfrclimited.any()),dynprm=bool(np.array_equal(m.actuator_dynprm[:,:3],np.tile([.0001,.0004,0],(90,1)))))
  for name in ('mjcb_act_dyn','mjcb_act_gain','mjcb_act_bias'):
   getter=getattr(mujoco,'get_'+name);checks['no_'+name]=getter() is None
  r['preflight']=checks
  if not all(checks.values()):raise RuntimeError('preflight failed')
  # Snapshot every numeric model ndarray and reject changes other than declared branch.
  before={k:getattr(m,k).copy() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)};native_ranges=m.actuator_ctrlrange.copy();m.actuator_ctrlrange[:,0]=0;m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
  assert all(np.array_equal(v,getattr(m,k)) for k,v in before.items() if k!='actuator_ctrlrange')
  r.update(native_ctrlrange=native_ranges.tolist(),derived_ctrlrange=m.actuator_ctrlrange.tolist(),branch_changes=['actuator_ctrlrange lower=0','integrator=RK4','timestep=25us or12.5us'],environment=dict(mujoco=mujoco.__version__,numpy=np.__version__),model_arrays_unchanged_except_ctrlrange=True)
  with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in paths[:5]:z.write(p,p.name)
  fields=('qpos','qvel','act','ctrl','actuator_force','qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint','qfrc_applied','xfrc_applied')
  for name,factor,hook in [('baseline',4,True),('repeat',4,True),('hookoff',4,False),('refined',8,True)]:
   dt=.0001/factor;m.opt.timestep=dt;assert .005/dt==50*factor and .0005/dt==5*factor
   d=mujoco.MjData(m);d.qpos[:]=pose;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0
   records={k:[] for k in fields+('moment','time_s','ncon','contact_signature','warnings')};logs=[];phase='initial'
   def callback(model,data):
    logs.append((phase,float(data.time),data.act.copy(),data.ctrl.copy(),data.qpos.copy(),data.qvel.copy()));r['callback_calls']+=1
   mujoco.set_mjcb_control(callback if hook else None)
   r['explicit_forward_calls']+=1;mujoco.mj_forward(m,d)
   for i in range(500*factor):
    d.ctrl[:]=commands[i//(50*factor)];d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;phase='stage';start=len(logs);t0=float(d.time)
    r['steps']+=1;mujoco.mj_step(m,d)
    if hook:
     stage=logs[start:];assert len(stage)==4;assert np.allclose([x[1] for x in stage],t0+np.array([0,.5,.5,1])*dt,rtol=0,atol=1e-12);assert all(np.array_equal(x[3],commands[i//(50*factor)]) for x in stage)
    phase='endpoint';r['explicit_forward_calls']+=1;mujoco.mj_forward(m,d)
    if d.warning.number.any():raise RuntimeError('native warning')
    moment=dense_moment(m,d);assert np.allclose(moment.T@d.actuator_force,d.qfrc_actuator,rtol=1e-10,atol=1e-10)
    for k in fields:records[k].append(getattr(d,k).copy())
    records['moment'].append(moment);records['time_s'].append(float(d.time));records['ncon'].append(d.ncon);records['warnings'].append(d.warning.number.copy());signature=sorted((int(c.geom[0]),int(c.geom[1]),int(c.dim),int(c.exclude)) for c in d.contact);records['contact_signature'].append(json.dumps(signature,separators=(',',':')))
   a={k:np.asarray(v) for k,v in records.items()};a['commands']=commands.copy()
   if logs:
    a.update(callback_phase=np.array([x[0] for x in logs]),callback_time_s=np.array([x[1] for x in logs]),callback_act=np.array([x[2] for x in logs]),callback_ctrl=np.array([x[3] for x in logs]),callback_qpos=np.array([x[4] for x in logs]),callback_qvel=np.array([x[5] for x in logs]))
   mujoco.set_mjcb_control(None);np.savez_compressed(out/(name+'.npz'),**a)
   entry=dict(steps=500*factor,dt_s=dt,hook=hook,callbacks=len(logs),stage_time_and_hold_checks=True,finite=all(np.isfinite(v).all() for v in a.values() if v.dtype.kind not in 'US'),external_loads_zero=not bool(a['qfrc_applied'].any() or a['xfrc_applied'].any()),activation_vs_classical_max_error=float(abs(a['act']-reference[f'factor{factor}_trace']).max()))
   err=a['act']-reference[f'factor{factor}_reference'];entry.update(activation_reference_max_error=float(abs(err).max()),activation_reference_rms_error=float(np.sqrt(np.mean(err**2))),activation_min=float(a['act'].min()),activation_max=float(a['act'].max()))
   if hook:entry.update(stage_activation_min=float(a['callback_act'].min()),stage_activation_max=float(a['callback_act'].max()))
   r['cases'][name]=entry;dump(out/'report.json',r)
   assert entry['finite'] and entry['external_loads_zero'] and entry['activation_vs_classical_max_error']<=1e-10 and entry['activation_reference_max_error']<=.01 and entry['activation_reference_rms_error']<=.001
   assert a['act'].min()>=-1e-12 and a['act'].max()<=1+1e-12
   if hook:assert a['callback_act'].min()>=-1e-12 and a['callback_act'].max()<=1+1e-12
   if name=='repeat':entry['all_arrays_exact']=all(np.array_equal(v,saved['baseline'][k]) for k,v in a.items());assert entry['all_arrays_exact']
   if name=='hookoff':entry['state_exact_hook_equivalence']=all(np.array_equal(a[k],saved['baseline'][k]) for k in ('qpos','qvel','act'));assert entry['state_exact_hook_equivalence']
   saved[name]=a;dump(out/'report.json',r)
  b=saved['baseline'];f=saved['refined'];fq=f['qpos'][1::2];fv=f['qvel'][1::2];rot=[]
  for q1,q2 in zip(b['qpos'],fq):v=np.zeros(m.nv);mujoco.mj_differentiatePos(m,v,1.,q1,q2);rot.append(v[3:6])
  r['body_differences_only_not_convergence']=dict(root_translation_max_model_length=float(abs(b['qpos'][:,:3]-fq[:,:3]).max()),root_rotation_tangent_max_rad=float(abs(np.array(rot)).max()),scalar_joint_max_rad=float(abs(b['qpos'][:,7:]-fq[:,7:]).max()),root_linear_velocity_max_model_length_s=float(abs(b['qvel'][:,:3]-fv[:,:3]).max()),angular_velocity_max_rad_s=float(abs(b['qvel'][:,3:]-fv[:,3:]).max()),contact_signature_disagree_samples=int(np.sum(b['contact_signature']!=f['contact_signature'][1::2])))
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
  for n in ('baseline','refined'):
   a=saved[n];axes[0].plot(a['time_s']*1000,a['act'][:,14],label=n);axes[1].plot(a['time_s']*1000,a['qpos'][:,13],label=n)
  axes[0].set_ylabel('activation14');axes[1].set_ylabel('LF tibia angle(rad)');axes[1].set_xlabel('time(ms)');axes[0].legend();axes[1].legend();fig.suptitle('Native RK4 fixed-command body replay — NOT standing/convergence');fig.tight_layout();fig.savefig(out/'native-replay.png');plt.close(fig)
  r['status']='PASS_NATIVE_ACTIVATION_AND_TIMING_NOT_BODY_CONVERGENCE'
 except Exception as e:r.update(status='FAIL_STOP',error=type(e).__name__+': '+str(e))
 finally:
  mujoco.set_mjcb_control(None);r['callback_restored_none']=mujoco.get_mjcb_control() is None;r['hashes_before']=hashes;r['hashes_after']={p:sha(p) for p in hashes};r['hashes_unchanged']=r['hashes_before']==r['hashes_after'];r['dynamics_forward_evaluations']=4*r['steps']+r['explicit_forward_calls'];dump(out/'report.json',r)
 print(out);print(r['status']);print(r.get('error',''));return 0 if r['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
