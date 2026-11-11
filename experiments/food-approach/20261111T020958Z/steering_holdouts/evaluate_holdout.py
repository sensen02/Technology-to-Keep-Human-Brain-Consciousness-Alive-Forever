"""Prepared sixcase physical qualifier. Main guarded BEFORE backend imports."""
from pathlib import Path
import argparse,json,importlib.util,sys
import numpy as np
from contract import HERE,CASES,command_at,pulse,yaw_score,digest,validate_launch
ROOT=HERE.parent
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')

def jsonable(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
 return v

def model_loader():
 spec=importlib.util.spec_from_file_location('_holdout_actual_modelio',ROOT/'train/model_io.py');io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io);return io

def parser():
 p=argparse.ArgumentParser();p.add_argument('--launch-authorized',action='store_true');p.add_argument('--qualification',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--profile',choices=CASES,required=True);p.add_argument('--out',required=True);p.add_argument('--run-plan',required=True);return p

def evaluate(a):
 qualification=validate_launch(a)
 ready=json.loads((HERE/'OFFLINE_CHECKS.json').read_text())
 if not ready['all_passed']:raise ValueError('offline preparation checks required')
 validate_source(ready['source_hashes'])
 import torch
 torch.set_num_threads(1)
 sys.path.insert(0,str(BASE));import student_wave as sw;import identity;import mujoco as mj
 io=model_loader();model,payload=io.load_checked(a.checkpoint,a.sha256,require_final=True)
 if payload['curve'][-1]['step']!=5000:raise ValueError('fixed5000only')
 sw.teacher_preflight();sw.student_preflight()
 for entry in qualification['conditions'].values():identity.verify(json.loads((Path(entry['result']).parent/'IDENTITY.json').read_text()))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 paths=[HERE/n for n in ('contract.py','evaluate_holdout.py','verify_saved.py','pipeline.py','PREREG_DRAFT.json','check_prepared.py','OFFLINE_CHECKS.json')]+[ROOT/'PREREG.json',Path(a.run_plan),Path(a.qualification),Path(a.checkpoint),ROOT/'train/model_io.py']
 hashes={str(p):digest(p) for p in paths}
 c=None;trace=None;rows=[];angles=[];boundaries={};failure=None;result=None
 def dump(name,value):identity.atomic_json(out/name,jsonable(value))
 def flat(state):return torch.cat(tuple(state),-1)
 try:
  c=sw.Core(adhesion=False);sw.load_student(BASE/'BC_GRU_ROLE_3000/checkpoint_002500.pt',c)
  mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
  if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('71opaque meshes required')
  ident=identity.capture(c,dict(kind='sixcase neural steering qualification',cli=vars(a),source_hashes=hashes,qualification_sha256=digest(a.qualification)),extra_paths=paths);dump('IDENTITY.json',ident)
  np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type,geom_rgba=c.plant.geom_rgba)
  trace=sw.Trace(full=True)
  def tick(action,source):
   trace.tick(c,action,source);r=c.d.xmat[c.th].reshape(3,3);angles.append(float(np.arctan2(r[1,0],r[0,0])))
  motorpulse,seed=pulse(a.profile)
  if seed is not None:
   for _ in range(20):tick(motorpulse,1)
  for _ in range(1000):tick(np.zeros(42),0)
  teacher=sw.StaticTeacher(c,sw.teacher_config(3));state=model.initial_state(1);previous=np.zeros(42);events=[];observer=None;calls=0
  for k in range(2880):
   warm=k<960
   if k in (960,1920):
    before=sw.continuity_snapshot(c,previous,flat(state))
    if k==960:observer=sw.tracker(c)
    after=sw.continuity_snapshot(c,previous,flat(state));exact={name:bool(np.array_equal(before[name],after[name])) for name in before}
    if not all(exact.values()):raise ValueError('physical/neuralboundary reset')
    boundaries[str(k)]=dict(all_exact=True,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),time_s=float(c.d.time));dump('BOUNDARIES.json',boundaries)
   x,_=sw.observations(c,previous,(2880-k)*.005);u=command_at(a.profile,k);hb=flat(state).clone()
   with torch.no_grad():mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
   policy=mean[0].tanh().numpy()
   if warm:action,_=teacher.action(c,k*.005);calls+=1;source=2
   else:action=policy;source=3
   rows.append(dict(time=float(c.d.time),X238=x,command=u,hidden_before=hb[0].numpy(),hidden_after=flat(state)[0].numpy(),mean=mean[0].numpy(),policy_action=policy,actually_applied_action=action.copy(),control_source=source))
   for _ in range(5):
    tick(action,source)
    if not warm:events.extend(observer.measure(c)['events'])
   previous=action.copy()
  auto=trace.summary(c,3);cycles=sw.cycle_report(observer,events);yaw=yaw_score(a.profile,angles)
  criteria=dict(original_five=auto['all_gates'],all_six=cycles['all_legs_completed'],strict_nonfoot_zero=auto['nonfoot_force_max_native']==0.,yaw=yaw['passed'])
  identity.verify(ident);identity.verify_runtime(ident,c);validate_source(hashes);validate_launch_no_output(a);io.load_checked(a.checkpoint,a.sha256)
  result=dict(status='FIXED_STEERING_ENGINEERING_ONLY_NOT_FOOD_OR_NATIVE_CERT',engineering_pass=bool(all(criteria.values())),criteria=criteria,profile=a.profile,checkpoint=a.checkpoint,checkpoint_sha256=a.sha256,qualification=a.qualification,qualification_sha256=digest(a.qualification),autonomous=auto,autonomous_cycles=cycles,yaw=yaw,boundaries=boundaries,pulse=dict(seed=seed,action=motorpulse.tolist(),duration_ms=0 if seed is None else 20),source_hashes=hashes,teacher_calls=calls,teacher_calls_after_handover=0,physical_steps=len(trace.rec)*80,identity_sha256=ident['identity_sha256'],food_success=False)
 except BaseException as exc:failure=exc
 finally:
  artifacts={}
  if trace is not None and trace.rows:artifacts['trace']=trace.save(out/'trace_1ms.npz')
  if rows:
   path=out/'neural_decisions.npz';np.savez_compressed(path,**{key:np.asarray([row[key] for row in rows]) for key in rows[0]},yaw_after_each_1ms=np.asarray(angles));artifacts['neural_decisions']=dict(path=str(path),sha256=digest(path),rows=len(rows))
  if failure:dump('FAILURE.json',dict(error=repr(failure),completed_endpoints=0 if trace is None else len(trace.rec),attempted_decisions=len(rows),artifacts=artifacts,engineering_pass=False))
  elif result:result.update(artifacts);dump('RESULT.json',result)
 if failure:raise failure
 print(json.dumps(dict(profile=a.profile,engineering_pass=result['engineering_pass'],gates=auto['gates'],yaw=yaw)))
 return result

def validate_source(hashes):
 for p,h in hashes.items():
  if digest(p)!=h:raise ValueError('capturedsource drift '+p)
def validate_launch_no_output(a):
 from contract import verify_qualification
 return verify_qualification(a.qualification,a.checkpoint,a.sha256)
if __name__=='__main__':evaluate(parser().parse_args())
