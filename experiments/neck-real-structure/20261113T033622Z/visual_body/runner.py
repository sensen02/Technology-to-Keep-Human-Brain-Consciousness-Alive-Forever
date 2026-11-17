"""Parent-only complete actual headRGB->brain->source neck->motor physical trial."""
import argparse,time
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS,required=True);p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no model/physics');return
 cpu1();f=verify()
 if read(ROOT/'PREFLIGHT.json')['all_passed'] is not True:raise ValueError('Actual zeroCore preflight required')
 visual,bridge,motor,vp,bp,mp=load_policies();sw,identity=pin();sw.teacher_preflight();sw.student_preflight()
 import numpy as np
 import torch
 import mujoco as mj
 cg=camera_runtime();peak=module('_neckvision_native_peak',PEAK_SOURCE);out=ROOT/'trials'/a.condition;out.mkdir(parents=True,exist_ok=False);binding=read(ROOT/'INPUT_BINDINGS.json');sources=dict(f['source_hashes']);sources[str(ROOT/'FROZEN.json')]=sha(ROOT/'FROZEN.json');sources[str(ROOT/'PREFLIGHT.json')]=sha(ROOT/'PREFLIGHT.json')
 c=None;tr=None;stream=None;recorder=None;rows=[];geometry=[];error=None;result=None;start=time.monotonic();loop=None
 try:
  recorder=peak.PeakRecorder();c=sw.Core(adhesion=False,recorder=recorder);mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
  if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('Full71opaque body')
  ident=identity.capture(c,dict(kind=KIND,condition=a.condition,camera_condition=camera_condition(a.condition),evaluation_source_hashes=sources,state_layout=[128,64,64,32,16],state_total=304,feature_cache16_separate=True,port_buffers_separate=True,visual_sha256=VISION_SHA,brain_sha256=binding['brain_sha256'],motor_sha256=MOTOR_SHA,neck_sha256=BINDING_SHA),extra_paths=list(sources));identity.atomic_json(out/'IDENTITY.json',ident)
  np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type,geom_rgba=c.plant.geom_rgba)
  tr=sw.Trace(full=True);stream=cg['CameraStream'](c,camera_condition(a.condition));loop=BrainNeckLoop(bridge,a.condition);motorstate=motor.initial_state(1);visionstate=visual.initial_state(1);held=visual.head.weight.new_zeros((1,16));previous=np.zeros(42);last_capture=-1;ms=0;teacher=None;teacher_calls=0;observer=None;events=[];transition=None
  save(out/'INITIAL_STATE.json',dict(time_s=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),head_xyz=c.d.xpos[stream.head].copy(),head_rotation=c.d.xmat[stream.head].reshape(3,3).copy()));stream.capture(0)
  for k in range(3080):
   now=k*5
   if ms!=now or abs(float(c.d.time)-now*.001)>1e-8:raise ValueError('Physical/neural clock drift')
   settle=now<1000;warm=now<5800
   if now==1000:teacher=sw.StaticTeacher(c,sw.teacher_config(3))
   if now==5800:
    before=sw.continuity_snapshot(c,previous,flat(motorstate,visionstate,loop.brain));beforeq=json.dumps(loop.snapshot(),sort_keys=True);beforecache=held[0].numpy().copy();stream.anchor();observer=sw.tracker(c);events=[];after=sw.continuity_snapshot(c,previous,flat(motorstate,visionstate,loop.brain));afterq=json.dumps(loop.snapshot(),sort_keys=True);exact={key:bool(np.array_equal(before[key],after[key])) for key in before};exact.update(source_queue_held_continuous=beforeq==afterq,feature_cache_continuous=np.array_equal(beforecache,held[0].numpy()))
    if not all(exact.values()):raise ValueError('Handover reset')
    transition=dict(time_s=float(c.d.time),all_exact=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),queue_before_sha256=hashlib.sha256(beforeq.encode()).hexdigest(),queue_after_sha256=hashlib.sha256(afterq.encode()).hexdigest(),no_reset=True,state_layout=[128,64,64,32,16],state_total=304,camera_queue_retained=True,source_queue_retained=True,feature_cache_retained=True,warmup_real_AN_retained=True);save(out/'HANDOVER.json',transition)
   obs=actor_observation(stream,now*.001);valid=obs['valid'];fid=-1 if not valid else int(round(obs['capture_time_s']/.020));updated=valid and fid!=last_capture;vb=visionstate[0].numpy().copy();mb=torch.cat(motorstate,-1)[0].numpy().copy();fullbefore=flat(motorstate,visionstate,loop.brain)[0].numpy().copy()
   with torch.no_grad():
    if updated:held=visual.encode(torch.from_numpy(obs['image'][None,None]))[0];last_capture=fid
    if warm:vr=held.new_zeros(1);latent=vr.clone()
    else:
     if not valid:raise ValueError('ANN requires genuine alreadyreleased RGB')
     vr,latent,visionstate=visual.step(held,visionstate)
   x,_=sw.observations(c,previous,(15400-now)*.001);neural=loop.decision(x,vr,now)
   with torch.no_grad():
    if settle:mean=np.zeros(42);policy=np.zeros(42)
    else:means,motorstate=motor.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(neural['command'][None],dtype=torch.float64),motorstate);mean=means[0].numpy().copy();policy=means[0].tanh().numpy().copy()
   if settle:action=np.zeros(42);source=sw.SOURCE['settle']
   elif warm:action,_=teacher.action(c,(now-1000)*.001);teacher_calls+=1;source=sw.SOURCE['teacher']
   else:action=policy.copy();source=sw.SOURCE['student']
   rows.append(dict(time=float(c.d.time),neural_index=k,physical_step_ms=now,X238=x.copy(),held_features=held[0].numpy().copy(),encoder_cache_updated=updated,image_capture_id=fid,image_capture_time_s=-1. if not valid else obs['capture_time_s'],image_release_time_s=-1. if not valid else obs['release_time_s'],image_valid=valid,raw_released_image_capture_id=obs['raw_capture_id'] if valid else -1,raw_released_capture_time_s=-1. if not valid else obs['raw_capture_time_s'],raw_released_release_time_s=-1. if not valid else obs['raw_release_time_s'],actor_frame_frozen=obs['actor_frame_frozen'],used_stimulus_code=-1 if not valid else ('off','left','right','center').index(stream.rows[fid]['stimulus']),vision_hidden_before=vb,vision_hidden_after=visionstate[0].numpy().copy(),visual_raw_turn=float(vr[0]),visual_latent=float(latent[0]),hidden_before=mb,hidden_after=torch.cat(motorstate,-1)[0].numpy().copy(),full_hidden_before=fullbefore,full_hidden_after=flat(motorstate,visionstate,loop.brain)[0].numpy().copy(),**neural,mean=mean,policy_action=policy,actually_applied_action=action.copy(),control_source=source))
   for _ in range(5):
    tr.tick(c,action,source);ms+=1;loop.advance();hr=c.d.xmat[stream.head].reshape(3,3);rot=c.d.xmat[c.th].reshape(3,3);geometry.append(dict(time=float(c.d.time),head_xyz=c.d.xpos[stream.head].copy(),head_rotation=hr.copy(),thorax_yaw=float(np.arctan2(rot[1,0],rot[0,0])),control_source=source,source_anchor_valid=stream.centers is not None,target_centers_xyz=np.zeros((3,3)) if stream.centers is None else np.asarray([stream.centers[n] for n in ('left','right','center')])))
    if ms%20==0 and ms<15400:stream.capture(ms)
    if not warm:events.extend(observer.measure(c)['events'])
   previous=action.copy()
   if k%480==0:print(json.dumps(dict(condition=a.condition,decision=k,time=float(c.d.time),wall_s=time.monotonic()-start)),flush=True)
  auto=tr.summary(c,sw.SOURCE['student']);cycles=sw.cycle_report(observer,events);native=recorder.close();recorder=None;yaw=np.unwrap([g['thorax_yaw'] for g in geometry]);queue=loop.finalize();save(out/'SOURCE_QUEUE.json',queue)
  result=dict(kind=KIND,condition=a.condition,camera_condition=camera_condition(a.condition),stage='ACTUAL_RGB_NECK_BODY',visual_checkpoint=str(VISION),visual_checkpoint_sha256=VISION_SHA,brain_checkpoint=str(BRIDGE),brain_checkpoint_sha256=binding['brain_sha256'],checkpoint=str(MOTOR),checkpoint_sha256=MOTOR_SHA,neck_source_binding_sha256=BINDING_SHA,state_layout=[128,64,64,32,16],state_total=304,neural_records=3080,feature_cache16_separate=True,port_buffers_separate=True,autonomous=auto,autonomous_cycles=cycles,native=native,engineering_pass=auto['all_gates'],functional_stable_body=stable(auto,cycles,native),yaw_change_rad=float(yaw[-1]-yaw[5799]),transition=transition,physical_steps=len(tr.rec)*80,teacher_calls=teacher_calls,teacher_calls_after_handover=0,extra_forces=False,mesh_count=71,all_mesh_opaque=True,source_hashes=sources,identity_sha256=ident['identity_sha256'],source_kind='ACTUAL_MOVING_HEAD_WORLD_FIXED_RENDER_ONLY_FAR_LANDMARK',olfaction_connected=False,capture_count=len(stream.rows),camera_state_reset=False,neural_queue_final=queue['queue_final'],no_visual_motor_bypass=True,brain_side_input='deliveredAN1+actualRGBderivedcontinuousvisualrawturn1',lowlevel_basewalking_belongs_OTHER=True,whole_neck_or_native_function_verified=False,wall_s=time.monotonic()-start)
  identity.verify_runtime(ident,c);identity.verify(ident);verify()
 except BaseException as exc:error=exc
 finally:
  artifacts={}
  if tr is not None and tr.rows:artifacts['trace']=tr.save(out/'trace_1ms.npz')
  for name,records,key in [('neural_decisions.npz',rows,'neural_decisions'),('geometry_1ms.npz',geometry,'geometry')]:
   if records:
    path=out/name;np.savez_compressed(path,**{k:np.asarray([row[k] for row in records]) for k in records[0]});artifacts[key]=dict(path=str(path),sha256=sha(path),rows=len(records))
  if stream is not None:
   if stream.images:
    path=out/'capture_rgb.npz';np.savez_compressed(path,rgb=np.stack(stream.images));artifacts['capture_rgb']=dict(path=str(path),sha256=sha(path),rows=len(stream.images));save(out/'CAPTURES.json',stream.rows);artifacts['captures_metadata']=dict(path=str(out/'CAPTURES.json'),sha256=sha(out/'CAPTURES.json'))
   try:stream.close();artifacts.update(stream.additional_artifacts)
   except BaseException as close_error:
    if error is None:error=close_error
    else:save(out/'CAMERA_CLOSE_ERROR.json',dict(error=repr(close_error)))
  if recorder is not None:
   try:save(out/'PARTIAL_NATIVE.json',recorder.close())
   except BaseException as recerror:save(out/'NATIVE_CLOSE_ERROR.json',dict(error=repr(recerror)))
  for name,key in [('INITIAL_STATE.json','initial_state'),('runtime_geometry.npz','runtime_geometry'),('SOURCE_QUEUE.json','source_queue')]:
   if (out/name).exists():artifacts[key]=dict(path=str(out/name),sha256=sha(out/name))
  if error is not None:save(out/'FAILURE.json',dict(error=repr(error),artifacts=artifacts,physical_endpoints=0 if tr is None else len(tr.rec),neural_records=len(rows)))
  elif result is not None:result.update(artifacts);save(out/'RESULT.json',result)
 if error is not None:raise error
 print(json.dumps(dict(condition=a.condition,stable=result['functional_stable_body'],yaw_deg=float(np.degrees(result['yaw_change_rad'])),slip=result['autonomous']['slip_ratio_max'],captures=result['capture_count'])),flush=True)
if __name__=='__main__':main()
