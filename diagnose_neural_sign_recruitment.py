"""Offline native muscle antagonism/sensory-sign analysis; never integrate or control a fly."""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
S=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-sign-and-recruitment/20261028T067000Z')
P=Path(__file__).parent
LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);started=time.time();sources=[S/x for x in ['MODEL.xml','BODY.npz','MAPPING.json','NEURAL_RECORDS.json']]+[Path(__file__),P/'engine/embodied/sensory_channels.py',P/'run_neural_muscle_closure.py'];hashes={str(x):sha(x) for x in sources}
 dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,scope='Saved-state FK only. Actual/zero activation torque decomposition and instantaneous removal of each active shared pool at10ms. No optimization motor command, no integration, no sign flips, no model changes.',times_after_handover_ms=[0,7,10,15,20,50],units='native; absolute physiology NOT_ESTABLISHED'))
 with np.load(S/'BODY.npz') as z:b={k:z[k] for k in z.files}
 mapping=json.loads((S/'MAPPING.json').read_text());records=json.loads((S/'NEURAL_RECORDS.json').read_text());groups={}
 for r in mapping['rows']:
  if r['local_indices']:groups.setdefault(tuple(r['pool_root_ids']),[]).append(r['actuator_index'])
 assert len(groups)==66
 m=_load_mjcf(str(S/'MODEL.xml')).compile();counts=dict(model_load=1,forward=0,body_steps=0,neural_steps=0)
 hinges=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE];va=np.array([m.jnt_dofadr[j] for j in hinges]);qa=np.array([m.jnt_qposadr[j] for j in hinges]);names=[m.joint(j).name for j in hinges];weight=m.body_mass.sum()*abs(m.opt.gravity[2])
 def state(i,acts):
  d=mujoco.MjData(m);d.qpos[:]=b['qpos'][i];d.qvel[:]=b['qvel'][i];d.act[:]=acts;d.ctrl[:]=b['command'][i];d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;mujoco.mj_forward(m,d);counts['forward']+=1;return d
 def moment(d):
  a=np.zeros((m.nu,m.nv))
  for r in range(m.nu):
   st=int(d.moment_rowadr[r]);n=int(d.moment_rownnz[r]);a[r,d.moment_colind[st:st+n]]=d.actuator_moment[st:st+n]
  np.testing.assert_allclose(a.T@d.actuator_force,d.qfrc_actuator,rtol=1e-10,atol=1e-10);return a
 def reaction(d):
  foot=0.;non=0.
  for k in range(d.ncon):
   co=d.contact[k];a,c=int(co.geom1),int(co.geom2);pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[c]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(pa or pb):continue
   g=c if pa else a;f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);fz=((co.frame.reshape(3,3).T@f[:3])*(1 if pa else -1))[2]
   if any(m.geom(g).name.startswith(l) for l in LEGS) and 'Tars' in m.geom(g).name:foot+=fz
   else:non+=fz
  return dict(foot_fz=float(foot),foot_load_ratio=float(foot/weight),nonfoot_fz=float(non),root_qacc=d.qacc[:6].copy())
 samples=[];detail=[]
 for ms in [0,7,10,15,20,50]:
  i=int(np.argmin(abs(b['time']-(1+ms/1000))));d=state(i,b['activation'][i]);zero=state(i,np.zeros(m.na));a=moment(d);activeforce=d.actuator_force-zero.actuator_force;terms=a[:,va]*activeforce[:,None];net=terms.sum(0);absum=abs(terms).sum(0);pos=np.maximum(terms,0).sum(0);neg=np.minimum(terms,0).sum(0)
  np.testing.assert_allclose(net,(d.qfrc_actuator-zero.qfrc_actuator)[va],rtol=0,atol=1e-10)
  legrows=[]
  for leg in LEGS:
   j=np.array([k for k,n in enumerate(names) if n.startswith('joint_'+leg)]);legrows.append(dict(leg=leg,net_active_torque_norm=float(np.linalg.norm(net[j])),sum_abs_contributions=float(absum[j].sum()),net_abs_sum=float(abs(net[j]).sum()),cancellation_fraction=0. if absum[j].sum()==0 else float(1-abs(net[j]).sum()/absum[j].sum()),joint_positive_torque=pos[j],joint_negative_torque=neg[j]))
  cocontracted=(pos>1e-6)&(neg<-1e-6)
  samples.append(dict(relative_ms=ms,actual_saved_time=float(b['time'][i]),motor_activated_actuators=int(np.sum(b['activation'][i]>.001)),commanded_actuators=int(np.sum(b['command'][i]>0)),active_pool_count=sum(bool(np.any(b['activation'][i,ids]>.001)) for ids in groups.values()),joint_antagonist_cocontraction_count=int(cocontracted.sum()),active_torque_norm=float(np.linalg.norm(net)),sum_abs_torque_contributions=float(absum.sum()),cancellation_fraction=0. if absum.sum()==0 else float(1-abs(net).sum()/absum.sum()),legs=legrows,reaction_actual=reaction(d),reaction_zero_activation=reaction(zero)))
  detail.append(dict(relative_ms=ms,joint_names=names,actuator_names=[m.actuator(j).name for j in range(m.nu)],actuator_moment=a[:,va],actual_activation=b['activation'][i],actual_command=b['command'][i],active_force=activeforce,individual_active_joint_torque=terms,net_active_joint_torque=net,positive_joint_torque=pos,negative_joint_torque=neg))
 # Same-state instantaneous deletion, never applied to a trajectory; no claim of causal behavioral benefit.
 idx=int(np.argmin(abs(b['time']-1.010)));base=state(idx,b['activation'][idx]);base_reaction=reaction(base);drops=[]
 for roots,ids in groups.items():
  if not np.any(b['activation'][idx,ids]>.001):continue
  acts=b['activation'][idx].copy();acts[ids]=0.;dd=state(idx,acts);rr=reaction(dd)
  drops.append(dict(pool_root_ids=list(roots),actuator_indices=ids,actuator_names=[m.actuator(j).name for j in ids],activation_before=b['activation'][idx,ids],foot_load_delta=float(rr['foot_load_ratio']-base_reaction['foot_load_ratio']),root_z_acceleration_delta=float(rr['root_qacc'][2]-base_reaction['root_qacc'][2]),root_angular_acceleration_norm_delta=float(np.linalg.norm(rr['root_qacc'][3:])-np.linalg.norm(base_reaction['root_qacc'][3:])),scope='Instantaneous deletion of activation, not fixed spike-gain trial or support prediction. Residual activation kinetics not represented.'))
 signs=[]
 for leg in ['lf','lm','lh','rf','rm','rh']:
  channels={}
  for ch in ['claw_position','hook_direction','hairplate_angle','campaniform_strain','bristle_touch']:
   vals=np.array([r['channels']['per_leg'][leg][ch]['value'] for r in records if r['relative_start_s']<.020]);r0=records[0]['channels']['per_leg'][leg][ch];channels[ch]=dict(min=float(vals.min()),max=float(vals.max()),negative_windows=int(np.sum(vals<0)),positive_windows=int(np.sum(vals>0)),n_cells=r0['n_cells'],signed=r0['signed'],current_nA_min=float(vals.min()*.3),current_nA_max=float(vals.max()*.3))
  signs.append(dict(leg=leg,channels=channels))
 dump(O/'ACTUAL_TORQUE_SUMMARY.json',samples);dump(O/'ACTUAL_TORQUE_DETAIL.json',detail);dump(O/'INSTANTANEOUS_POOL_DELETION.json',dict(relative_ms=10,baseline=base_reaction,pools=sorted(drops,key=lambda r:r['foot_load_delta'],reverse=True)));dump(O/'SENSORY_SIGN_AUDIT.json',dict(first20ms=signs,claims=['Native signed tibia deviation/velocity and mean of seven raw joint deviations, no anatomical left/right sign normalization','No rectification: negative values become negative current into every cell in channel','Dataset function labels do not specify per-cell flexion/extension preferred direction','Contact strain channel counts differ by leg; software is not an experimentally validated tuning map'],no_runtime_change=True))
 dump(O/'EXECUTION.json',dict(counts=counts,exit_code=0,runtime_s=time.time()-started,source_unchanged=all(sha(Path(k))==v for k,v in hashes.items()),no_runtime_control=True));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(samples=[{k:v for k,v in x.items() if k!='legs'} for x in samples],best_pool_deletions=sorted(drops,key=lambda r:r['foot_load_delta'],reverse=True)[:3],counts=counts),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x,indent=2),flush=True)
if __name__=='__main__':main()
