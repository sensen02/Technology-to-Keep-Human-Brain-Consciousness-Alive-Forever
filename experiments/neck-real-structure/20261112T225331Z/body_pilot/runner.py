"""Parent-only six-case actual brain->source neck->body physical execution, no camera."""
import argparse,time
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS,required=True);p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no model or physics');return
 cpu1();f=verify()
 if read(ROOT/'PREFLIGHT.json')['all_passed'] is not True:raise ValueError('Actual0Core preflight required')
 bridge,motor,bp,mp=load_policies();sw,identity=pin();sw.teacher_preflight();sw.student_preflight()
 import numpy as np
 import torch
 import mujoco as mj
 peak=module('_bodypilot_native_peak',PEAK_SOURCE);out=ROOT/'trials'/a.condition;out.mkdir(parents=True,exist_ok=False);sources=dict(f['source_hashes']);sources[str(ROOT/'FROZEN.json')]=sha(ROOT/'FROZEN.json');sources[str(ROOT/'PREFLIGHT.json')]=sha(ROOT/'PREFLIGHT.json')
 c=None;tr=None;recorder=None;loop=None;rows=[];geometry=[];error=None;result=None;start=time.monotonic()
 try:
  recorder=peak.PeakRecorder();c=sw.Core(adhesion=False,recorder=recorder);mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
  if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('Complete71opaque physical body required')
  ident=identity.capture(c,dict(kind='ARTIFICIAL_BRAIN_SOURCE_NECK_BODY_PILOT_V1',condition=a.condition,evaluation_source_hashes=sources,state_layout=[128,64,64,16],state_total=272,port_buffers_separate=True,brain_sha256=BRIDGE_SHA,motor_sha256=MOTOR_SHA,neck_sha256=BINDING_SHA),extra_paths=list(sources));identity.atomic_json(out/'IDENTITY.json',ident)
  np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type,geom_rgba=c.plant.geom_rgba)
  head=c.plant.body('Head').id;save(out/'INITIAL_STATE.json',dict(time_s=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),head_xyz=c.d.xpos[head].copy(),head_rotation=c.d.xmat[head].reshape(3,3).copy()))
  tr=sw.Trace(full=True);loop=BrainNeckLoop(bridge,a.condition);motorstate=motor.initial_state(1);previous=np.zeros(42);teacher=None;teacher_calls=0;observer=None;events=[];transition=None;ms=0
  for k in range(3080):
   now=k*5
   if now!=ms or abs(float(c.d.time)-ms*.001)>1e-8:raise ValueError('Physical integer neural clock drift')
   settle=now<1000;warm=now<5800
   if now==1000:teacher=sw.StaticTeacher(c,sw.teacher_config(3))
   if now==5800:
    before=sw.continuity_snapshot(c,previous,flat(motorstate,loop.brain));queuebefore=json.dumps(loop.snapshot(),sort_keys=True);observer=sw.tracker(c);events=[];after=sw.continuity_snapshot(c,previous,flat(motorstate,loop.brain));queueafter=json.dumps(loop.snapshot(),sort_keys=True);exact={key:bool(np.array_equal(before[key],after[key])) for key in before};exact['port_queue_and_held_continuous']=queuebefore==queueafter
    if not all(exact.values()):raise ValueError('Body/neural/queue handover reset')
    transition=dict(time_s=float(c.d.time),all_exact=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),queue_before_sha256=hashlib.sha256(queuebefore.encode()).hexdigest(),queue_after_sha256=hashlib.sha256(queueafter.encode()).hexdigest(),no_reset=True,state_layout=[128,64,64,16],queue_retained=True,warmup_real_AN_retained=True);save(out/'HANDOVER.json',transition)
   x,_=sw.observations(c,previous,(15400-now)*.001);mb=torch.cat(motorstate,-1)[0].numpy().copy();fullbefore=flat(motorstate,loop.brain)[0].numpy().copy();neural=loop.decision(x,now)
   with torch.no_grad():
    if settle:mean=np.zeros(42);policy=np.zeros(42)
    else:
     means,motorstate=motor.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(neural['command'][None],dtype=torch.float64),motorstate);mean=means[0].numpy().copy();policy=means[0].tanh().numpy().copy()
   if settle:action=np.zeros(42);source=sw.SOURCE['settle']
   elif warm:action,_=teacher.action(c,(now-1000)*.001);teacher_calls+=1;source=sw.SOURCE['teacher']
   else:action=policy.copy();source=sw.SOURCE['student']
   rows.append(dict(time=float(c.d.time),neural_index=k,physical_step_ms=now,X238=x.copy(),**neural,hidden_before=mb,hidden_after=torch.cat(motorstate,-1)[0].numpy().copy(),full_hidden_before=fullbefore,full_hidden_after=flat(motorstate,loop.brain)[0].numpy().copy(),mean=mean,policy_action=policy,actually_applied_action=action.copy(),control_source=source))
   for _ in range(5):
    tr.tick(c,action,source);ms+=1;loop.advance();rot=c.d.xmat[c.th].reshape(3,3);geometry.append(dict(time=float(c.d.time),head_xyz=c.d.xpos[head].copy(),head_rotation=c.d.xmat[head].reshape(3,3).copy(),thorax_yaw=float(np.arctan2(rot[1,0],rot[0,0])),control_source=source))
    if not warm:events.extend(observer.measure(c)['events'])
   previous=action.copy()
   if k%480==0:print(json.dumps(dict(condition=a.condition,decision=k,time=float(c.d.time),wall_s=time.monotonic()-start)),flush=True)
  auto=tr.summary(c,sw.SOURCE['student']);cycles=sw.cycle_report(observer,events);native=recorder.close();recorder=None;yaw=np.unwrap([r['thorax_yaw'] for r in geometry]);queue=loop.finalize();save(out/'SOURCE_QUEUE.json',queue)
  result=dict(kind='ARTIFICIAL_BRAIN_SOURCE_NECK_BODY_PILOT_V1',condition=a.condition,brain_checkpoint=str(BRIDGE),brain_checkpoint_sha256=BRIDGE_SHA,checkpoint=str(MOTOR),checkpoint_sha256=MOTOR_SHA,neck_source_binding_sha256=BINDING_SHA,state_layout=[128,64,64,16],state_total=272,neural_records=3080,AN1_DN2_held_buffers_separate=True,autonomous=auto,autonomous_cycles=cycles,native=native,engineering_pass=auto['all_gates'],functional_stable_body=stable(auto,cycles,native),yaw_change_rad=float(yaw[-1]-yaw[5799]),transition=transition,physical_steps=len(tr.rec)*80,teacher_calls=teacher_calls,teacher_calls_after_handover=0,extra_forces=False,mesh_count=71,all_mesh_opaque=True,source_hashes=sources,identity_sha256=ident['identity_sha256'],cue_is_experimental_instruction_only=True,visual_or_odor_input=False,neural_queue_final=queue['queue_final'],lowlevel_autonomous_walking_not_attributed_to_brain=True,new_whole_neck_function_verified=False,wall_s=time.monotonic()-start)
  identity.verify_runtime(ident,c);identity.verify(ident);verify()
 except BaseException as exc:error=exc
 finally:
  artifacts={}
  if tr is not None and tr.rows:artifacts['trace']=tr.save(out/'trace_1ms.npz')
  for filename,records,key in [('neural_decisions.npz',rows,'neural_decisions'),('geometry_1ms.npz',geometry,'geometry')]:
   if records:
    path=out/filename;np.savez_compressed(path,**{key:np.asarray([row[key] for row in records]) for key in records[0]});artifacts[key]=dict(path=str(path),sha256=sha(path),rows=len(records))
  if recorder is not None:
   try:save(out/'PARTIAL_NATIVE.json',recorder.close())
   except BaseException as recerror:save(out/'NATIVE_CLOSE_ERROR.json',dict(error=repr(recerror)))
  for name,key in [('INITIAL_STATE.json','initial_state'),('runtime_geometry.npz','runtime_geometry'),('SOURCE_QUEUE.json','source_queue')]:
   if (out/name).exists():artifacts[key]=dict(path=str(out/name),sha256=sha(out/name))
  if error is not None:save(out/'FAILURE.json',dict(error=repr(error),artifacts=artifacts,physical_endpoints=0 if tr is None else len(tr.rec),neural_decisions=len(rows)))
  elif result is not None:result.update(artifacts);save(out/'RESULT.json',result)
 if error is not None:raise error
 print(json.dumps(dict(condition=a.condition,stable=result['functional_stable_body'],yaw_deg=float(np.degrees(result['yaw_change_rad'])),slip=auto['slip_ratio_max'],source_queue_steps=result['neural_queue_final']['actual_steps'])),flush=True)
if __name__=='__main__':main()
