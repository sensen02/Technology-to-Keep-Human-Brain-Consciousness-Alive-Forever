"""Zero-integration snapshot oracle queries. All qpos writes are diagnostic restores.
Post targets are CORRECTIVE_EXAMPLES, never claimed physically executed teacher PASS.
"""
from pathlib import Path
import argparse,sys,json,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
ATTEMPT=ROOT.parent.parent/'attempt02';ACTUAL=ROOT.parent/'left_hind_stance_refit/EVAL_right'
TOL=1e-10

def sha256(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('CODE_ONLY until parent resource release')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});out=ROOT/'ORACLE'
 if out.exists():raise ValueError('do not overwrite or blindly restart oracle')
 out.mkdir();sys.path.insert(0,str(ATTEMPT));from turning_teacher import pin_baseline,make_teacher
 pin_baseline();import physics,identity,mujoco as mj,static_teacher,student_wave,wave_reference
 if sha256(wave_reference.DEFAULT_REFERENCE)!=wave_reference.REFERENCE_SHA256:raise ValueError('fixed measured reference data SHA mismatch')
 t=arrays(ACTUAL/'trace_1ms.npz');d=arrays(ACTUAL/'neural_decisions.npz');g=arrays(ACTUAL/'geometry_1ms.npz');reference=json.loads((ACTUAL/'IDENTITY.json').read_text());actual_result=json.loads((ACTUAL/'RESULT.json').read_text())
 expected_parent='ef768c788c94aeecbb49f0bd1ccb5c424333af36ec514941b72e49ab0d4917ff'
 if actual_result['checkpoint_sha256']!=expected_parent or sha256(actual_result['checkpoint'])!=expected_parent or actual_result['identity_sha256']!=reference['identity_sha256']:raise ValueError('fixed ef768 actual source identity required')
 for key,name in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz')]:
  if sha256(ACTUAL/name)!=actual_result[key]['sha256']:raise ValueError('actual source artifact changed '+name)
 inputs=[ACTUAL/n for n in ('trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','IDENTITY.json','RESULT.json')]+[Path(actual_result['checkpoint']),Path(wave_reference.DEFAULT_REFERENCE)]
 sources=[ROOT/'PREREG.json',ROOT/'oracle.py',ATTEMPT/'turning_teacher.py']
 before={str(x):sha256(x) for x in inputs+sources};step_calls=[0];original={n:getattr(mj,n) for n in ('mj_step','mj_step1','mj_step2')}
 def forbidden(*a,**k):step_calls[0]+=1;raise RuntimeError('oracle forbids every integration call')
 for n in original:setattr(mj,n,forbidden)
 prefix=[];qdes=[];restored=[];oracle_actions=[];warm_errors=[];postcounts=dict(unsupported=0,unbacked=0,hypothetical=0,rateclip=0);max_geometry=0.;restores=0;teacher=None;c=None;manifest=None;error=None
 def dump(name,obj):identity.atomic_json(out/name,obj)
 try:
  c=physics.Core(adhesion=False);identity.verify_runtime(reference,c)
  if c.plant.nq!=49 or c.plant.nv!=48 or c.plant.nu!=42 or c.plant.na!=0:raise ValueError('full finitePD49/48/42 and no missing activation state required')
  manifest=identity.capture(c,dict(kind='ZERO_INTEGRATION_SNAPSHOT_RESTORATION_CORRECTIVE_ORACLE',qpos_writes='private diagnostic restores, NOT physicalrootwrites0',mj_step_allowed=False,teacher='speed1 yaw-.06 prime4.8; fullprivateIKhistory',actual=str(ACTUAL),tolerance=TOL),extra_paths=inputs+sources);dump('IDENTITY.json',manifest)
  head=c.plant.body('Head').id
  def restore(i):
   nonlocal restores
   c.d.qpos[:]=t['qpos'][i];c.d.qvel[:]=t['qvel'][i];c.d.ctrl[:]=t['actualctrl'][i];c.d.time=float(t['time'][i]);c.d.xfrc_applied[:]=0.;c.d.qfrc_applied[:]=0.;c.d.qacc_warmstart[:]=0.;c.actuator.target_rad[:]=t['target'][i];c.requested_rate=t['actual_requested_action'][i].copy()*2.
   if i>=39:c.buf=t['target'][i-39:i+1].copy()
   c._padz_prev=t['padz'][i].copy();c._pad_rate=np.zeros(6) if i==0 else (t['padz'][i]-t['padz'][i-1])/.001;c.t=i+1;mj.mj_forward(c.plant,c.d);restores+=1
   if not np.array_equal(c.d.qpos,t['qpos'][i]) or not np.array_equal(c.d.qvel,t['qvel'][i]) or not np.array_equal(c.d.ctrl,t['actualctrl'][i]) or not np.array_equal(c.actuator.target_rad,t['target'][i]):raise ValueError('snapshot not restored exactly')
  def geometry(i):
   m=c.measure();rot=c.d.xmat[c.th].reshape(3,3);yaw=float(np.arctan2(rot[1,0],rot[0,0]));value=dict(foot=m['foot'].copy(),padz=m['padz'].copy(),com=c.d.subtree_com[c.th].copy(),head_xyz=c.d.xpos[head].copy(),head_rotation=c.d.xmat[head].reshape(3,3).copy(),thorax_yaw=yaw)
   angle_delta=yaw-float(g['thorax_yaw'][i]);yaw_error=abs(float(np.arctan2(np.sin(angle_delta),np.cos(angle_delta))))
   errors=[np.max(np.abs(value['foot']-t['relativefoot'][i])),np.max(np.abs(value['padz']-t['padz'][i])),np.max(np.abs(value['com']-t['com'][i])),np.max(np.abs(value['head_xyz']-g['head_xyz'][i])),np.max(np.abs(value['head_rotation']-g['head_rotation'][i])),yaw_error]
   return value,float(max(errors))
  for i in range(5800):
   restore(i);v,e=geometry(i);prefix.append(v);max_geometry=max(max_geometry,e)
   if e>TOL:raise ValueError('prefix geometry tolerance failed at1ms row'+str(i))
  restore(999);teacher=make_teacher(c,speed_scale=1.,yaw_rate=-.06,prime_s=4.8);initial=dict(q_des=teacher.q_des.copy(),initial_qpos=teacher.initial_qpos.copy(),initial_rot=teacher.initial_rot.copy(),initial_origin=teacher.initial_origin.copy(),rest=teacher.rest.copy(),rest_pad_bottom=teacher.rest_pad_bottom.copy(),qa=teacher.qa.copy(),va=teacher.va.copy())
  if not np.array_equal(teacher.q_des,t['qpos'][999]) or not np.array_equal(teacher.initial_qpos,t['qpos'][999]):raise ValueError('private qdes must initialize actual1s qpos')
  np.savez_compressed(out/'TEACHER_INITIAL.npz',**initial)
  with (out/'POST_DIAGNOSTICS.jsonl').open('x') as diagnostics:
   for k in range(2880):
    i=999+5*k;restore(i)
    if abs(float(c.d.time)-float(d['time'][k]))>=1e-8:raise ValueError('preaction time not matched')
    snapshot=dict(qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),ctrl=c.d.ctrl.copy(),target=c.actuator.target_rad.copy(),time=np.asarray(c.d.time))
    action,diag=teacher.action(c,k*.005)
    def finite_diag(value):
     if isinstance(value,dict):return all(finite_diag(v) for v in value.values())
     if isinstance(value,(list,tuple)):return all(finite_diag(v) for v in value)
     if isinstance(value,(float,np.floating)):return bool(np.isfinite(value))
     return True
    if not finite_diag(diag):raise ValueError('nonfinite oracle diagnostic')
    if not np.isfinite(action).all() or np.max(np.abs(action))>1.+1e-12 or not np.isfinite(teacher.q_des).all():raise ValueError('nonfinite/unbounded oracle')
    if not all(np.array_equal(a,b) for a,b in [(snapshot['qpos'],c.d.qpos),(snapshot['qvel'],c.d.qvel),(snapshot['ctrl'],c.d.ctrl),(snapshot['target'],c.actuator.target_rad),(snapshot['time'],np.asarray(c.d.time))]):raise ValueError('teacher wrote actual snapshot or integrated label')
    qdes.append(teacher.q_des.copy());restored.append(snapshot);oracle_actions.append(action.copy())
    if k<960:
     err=float(np.max(np.abs(action-d['actually_applied_action'][k])));warm_errors.append(err)
     if err>TOL:raise ValueError('warm teacher action mismatch k'+str(k)+' max'+str(err))
     if k==959:
      dump('PREFIX_AUDIT.json',dict(all_passed=True,warm_actions=960,max_warm_action_error=max(warm_errors),max_geometry_error=max_geometry,full_prefix_endpoints=5800,tolerance=TOL,not_claimed_bitexact=True,private_qdes_initial_actual_exact=True,original_private_qdes_chain_not_saved=True,Core_constructions=1,mj_step_calls=step_calls[0],snapshot_restore_writes=restores))
      np.savez_compressed(out/'PREFIX_GEOMETRY.npz',**{key:np.asarray([row[key] for row in prefix]) for key in prefix[0]})
    else:
     postcounts['unsupported']+=int(diag['unsupported']);postcounts['unbacked']+=int(bool(diag['unbacked_support_indices']));postcounts['hypothetical']+=int(diag['support_is_hypothetical']);postcounts['rateclip']+=int(diag['rate_clip_count']>0);diagnostics.write(json.dumps(dict(k=k,label_kind='CORRECTIVE_EXAMPLES',**diag))+'\n')
  for path,h in before.items():
   if sha256(path)!=h:raise ValueError('oracle input/source changed '+path)
  identity.verify_runtime(manifest,c);identity.verify(manifest)
  labels=np.asarray(oracle_actions);np.savez_compressed(out/'CORRECTIVE_EXAMPLES.npz',X238=d['X238'],command2=d['command'],Y42=labels,actual_time=d['time'],post_prime=np.arange(2880)>=960)
  np.savez_compressed(out/'ORACLE_REPLAY.npz',private_q_des=np.asarray(qdes),warm_action_error=np.asarray(warm_errors),restored_qpos=np.asarray([r['qpos'] for r in restored]),restored_qvel=np.asarray([r['qvel'] for r in restored]),restored_ctrl=np.asarray([r['ctrl'] for r in restored]),restored_target=np.asarray([r['target'] for r in restored]),qa=teacher.qa,va=teacher.va)
  artifacts={str(p):sha256(p) for p in (out/'TEACHER_INITIAL.npz',out/'PREFIX_GEOMETRY.npz',out/'PREFIX_AUDIT.json',out/'POST_DIAGNOSTICS.jsonl',out/'CORRECTIVE_EXAMPLES.npz',out/'ORACLE_REPLAY.npz')}
  dump('RESULT.json',dict(status='UNEXECUTED_CORRECTIVE_ORACLE_QUERIES_NOT_QUALIFIED_TEACHER_PHYSICS',all_passed=True,Core_constructions=1,mj_step_calls=step_calls[0],physics_integration_steps=0,private_snapshot_restore_writes=restores,max_warm_action_error=max(warm_errors),max_prefix_geometry_error=max_geometry,tolerance=TOL,post_labels=1920,post_diagnostic_counts=postcounts,post_diagnostic_rates={k:v/1920 for k,v in postcounts.items()},finite_all=True,input_source_hashes=before,artifacts=artifacts,identity_sha256=manifest['identity_sha256'],unrecorded_solver_warmstart='zeroed eachsnapshot; uniformpref labels do not consume constraint force solution',snapshot_time_not_a_new_physical_trajectory=True))
  print(json.dumps(dict(all_passed=True,max_warm_action_error=max(warm_errors),max_geometry_error=max_geometry,post_diagnostic_rates={k:v/1920 for k,v in postcounts.items()},mj_step_calls=step_calls[0])),flush=True)
 except BaseException as exc:
  error=exc;dump('FAILURE.json',dict(error=repr(exc),Core_constructions=0 if c is None else 1,mj_step_calls=step_calls[0],private_snapshot_restore_writes=restores,completed_teacher_queries=len(oracle_actions),warm_queries=len(warm_errors),max_warm_error=None if not warm_errors else max(warm_errors),max_geometry_error=max_geometry,postlabel_publication=(out/'CORRECTIVE_EXAMPLES.npz').exists(),training_allowed=False))
 finally:
  for n,f in original.items():setattr(mj,n,f)
 if error is not None:raise error
if __name__=='__main__':main()
