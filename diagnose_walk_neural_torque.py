"""Saved-state torque capacity / decoder-latency diagnostic. Never integrates or controls a live fly."""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-torque-diagnostic/20261028T060000Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 def conv(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)
 p.write_text(json.dumps(x,indent=2,default=conv)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);start=time.time();counts=dict(model_load=0,forward=0,body_steps=0,neural_steps=0)
 sources=[S/x for x in ['MODEL.xml','BODY.npz','MAPPING.json','HANDOVER_BEFORE.npz','NEURAL_RECORDS.json']]+[Path(__file__)]
 hashes={str(x):sha(x) for x in sources};dump(O/'PREREGISTRATION.json',dict(scope='Offline capacity and neural latency only; optimization never used as runtime motor command',sources=hashes,sample_times=[.90,.95,.999,1.,1.005,1.01,1.015,1.03,1.05,1.1,1.5],probe_activations=[0.,.1,1.],units='native torque, absolute physiology NOT_ESTABLISHED'))
 with np.load(S/'BODY.npz') as f:b={k:f[k] for k in f.files}
 mapping=json.loads((S/'MAPPING.json').read_text());records=json.loads((S/'NEURAL_RECORDS.json').read_text())
 m=_load_mjcf(str(S/'MODEL.xml')).compile();counts['model_load']+=1
 jids=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE];va=np.array([m.jnt_dofadr[j] for j in jids]);assert len(va)==42
 names=[m.joint(j).name for j in jids];groups={}
 for row in mapping['rows']:
  if row['local_indices']:groups.setdefault(tuple(row['pool_root_ids']),[]).append(row['actuator_index'])
 G=np.zeros((90,len(groups)))
 for j,ix in enumerate(groups.values()):G[ix,j]=1
 assert len(groups)==66 and int(np.count_nonzero(G.sum(1)==0))==16
 def state(i,activation=None):
  d=mujoco.MjData(m);d.qpos[:]=b['qpos'][i];d.qvel[:]=b['qvel'][i];d.act[:]=b['activation'][i] if activation is None else activation;d.ctrl[:]=b['command'][i];mujoco.mj_forward(m,d);counts['forward']+=1;return d
 def moments(d):
  mat=np.zeros((m.nu,m.nv))
  for r in range(m.nu):
   a=int(d.moment_rowadr[r]);n=int(d.moment_rownnz[r]);mat[r,d.moment_colind[a:a+n]]=d.actuator_moment[a:a+n]
  np.testing.assert_allclose(mat.T@d.actuator_force,d.qfrc_actuator,rtol=1e-10,atol=1e-10)
  return mat
 samples=[];full=[]
 for requested in [.90,.95,.999,1.,1.005,1.01,1.015,1.03,1.05,1.1,1.5]:
  i=int(np.argmin(abs(b['time']-requested)));actual=state(i);zero=state(i,0.);tenth=state(i,.1);one=state(i,1.);mom=moments(actual);A=mom.T@(np.diag(one.actuator_force-zero.actuator_force));A=A[va]
  linearerr=float(np.max(abs((tenth.qfrc_actuator-zero.qfrc_actuator)[va]-.1*(one.qfrc_actuator-zero.qfrc_actuator)[va])))
  assist=b['assist'][i];target=actual.qfrc_actuator[va]+assist[va];increment=target-zero.qfrc_actuator[va]
  fit=lsq_linear(A@G,increment,bounds=(0.,1.),tol=1e-10,max_iter=500);fit90=G@fit.x;pred=zero.qfrc_actuator[va]+A@fit90
  free=lsq_linear(A,increment,bounds=(0.,1.),tol=1e-10,max_iter=500)
  detail=dict(requested=requested,time=float(b['time'][i]),saved_index=i,required_torque=target,actual_muscle_torque=actual.qfrc_actuator[va],assist_torque=assist[va],zero_activation_torque=zero.qfrc_actuator[va],active_capacity_matrix=A,pool_activation_fit=fit.x,actuator_activation_fit=fit90,fitted_torque=pred,actual_activation=b['activation'][i],actual_command=b['command'][i],gainprm=m.actuator_gainprm,biasprm=m.actuator_biasprm)
  full.append(detail);den=max(np.linalg.norm(increment),1e-12)
  leg=[]
  for label in ['LF','LM','LH','RF','RM','RH']:
   jx=[k for k,n in enumerate(names) if n.startswith('joint_'+label)];leg.append(dict(leg=label,target_norm=float(np.linalg.norm(target[jx])),actual_muscle_norm=float(np.linalg.norm(actual.qfrc_actuator[va[jx]])),assist_norm=float(np.linalg.norm(assist[va[jx]])),pool_fit_relative_residual=float(np.linalg.norm((pred-target)[jx])/max(np.linalg.norm(increment[jx]),1e-12)),saturated_pool_count=int(np.sum(fit.x>=.999))))
  samples.append(dict(time=float(b['time'][i]),assist_norm=float(np.linalg.norm(assist[va])),actual_muscle_torque_norm=float(np.linalg.norm(actual.qfrc_actuator[va])),required_torque_norm=float(np.linalg.norm(target)),all90_active_force_nonzero=int(np.count_nonzero(abs(one.actuator_force-zero.actuator_force)>1e-10)),active_force_abs_max=float(abs(one.actuator_force-zero.actuator_force).max()),activation_max=float(b['activation'][i].max()),command_max=float(b['command'][i].max()),command_mean=float(b['command'][i].mean()),active_linear_probe_error=linearerr,pool_fit_relative_residual=float(np.linalg.norm(pred-target)/den),free90_fit_relative_residual=float(np.linalg.norm(A@free.x-increment)/den),pool_fit_activation_max=float(fit.x.max()),pool_fit_saturated_count=int(np.sum(fit.x>.999)),pool_fit_status=int(fit.status),legs=leg))
 first_cmd=np.flatnonzero((b['time']>=1-1e-9)&np.any(b['command']>0,axis=1));firstact=np.flatnonzero((b['time']>=1-1e-9)&(b['activation'].max(1)>.001))
 latency=dict(first_command_absolute_s=None if not len(first_cmd) else float(b['time'][first_cmd[0]]),first_activation_above_001_s=None if not len(firstact) else float(b['time'][firstact[0]]),intervals=[dict(start=r['absolute_start_s'],motor_spikes=r['motor_spikes'],command_max=float(np.max(r['applied_command'])),decoded_max=float(np.max(r['decoded_next_command']))) for r in records[:12]],dynamics_parameters=m.actuator_dynprm.tolist(),activation_dynamic_type=m.actuator_dyntype.tolist(),native_ctrlrange=m.actuator_ctrlrange.tolist(),n_neural_intervals=len(records),total_motor_spike_events=sum(r['motor_spikes'] for r in records))
 dump(O/'CAPACITY.json',samples);dump(O/'LATENCY.json',latency);dump(O/'DETAIL.json',dict(joint_names=names,samples=full,pool_groups=[dict(root_ids=k,actuator_indices=v) for k,v in groups.items()]))
 dump(O/'EXECUTION.json',dict(counts=counts,exit_code=0,runtime_s=time.time()-start,source_unchanged=all(sha(Path(k))==v for k,v in hashes.items()),runtime_optimization_applied=False));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(samples=samples,latency=latency['intervals'][:5],firstcommand=latency['first_command_absolute_s'],firstactivation=latency['first_activation_above_001_s'],dynprm=m.actuator_dynprm[0].tolist(),counts=counts),indent=2),flush=True)
if __name__=='__main__':main()
