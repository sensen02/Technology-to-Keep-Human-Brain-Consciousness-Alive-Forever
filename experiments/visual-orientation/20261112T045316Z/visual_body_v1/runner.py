"""Direct real moving-head vision physical loop; parent-only, no training."""
import argparse,time
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS,required=True);p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no model or physics');return
 cpu1();frozen=verify()
 if read(ROOT/'PREFLIGHT.json')['all_passed'] is not True:raise ValueError('actual visualbody zeroCore preflight required')
 visual,motor,vp,mp=load_policies();sw,identity=pin();sw.teacher_preflight();sw.student_preflight()
 import numpy as np
 import torch
 import mujoco as mj
 peak=module('_visual_native_peak_only',PEAK_SOURCE);out=ROOT/'trials'/args.condition;out.mkdir(parents=True,exist_ok=False)
 sources=dict(frozen['source_hashes']);sources[str(ROOT/'FROZEN.json')]=sha(ROOT/'FROZEN.json');sources[str(ROOT/'PREFLIGHT.json')]=sha(ROOT/'PREFLIGHT.json')
 c=None;tr=None;stream=None;rows=[];geometry=[];error=None;result=None;recorder=None;start=time.monotonic();binding=read(ROOT/'INPUT_BINDINGS.json')
 try:
  recorder=peak.PeakRecorder();c=sw.Core(adhesion=False,recorder=recorder);mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
  if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('71opaque body required')
  ident=identity.capture(c,dict(kind='VISUAL_BODY_WORLD_LANDMARK_V1',condition=args.condition,evaluation_source_hashes=sources,state_layout=[128,64,64,32],state_total=288,feature_cache16_separate=True,visual_sha256=binding['visual_sha256'],motor_sha256=MOTOR_SHA),extra_paths=list(sources));identity.atomic_json(out/'IDENTITY.json',ident)
  np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type,geom_rgba=c.plant.geom_rgba)
  tr=sw.Trace(full=True);stream=CameraStream(c,args.condition);save(out/'INITIAL_STATE.json',dict(time_s=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),head_xyz=c.d.xpos[stream.head].copy(),head_rotation=c.d.xmat[stream.head].reshape(3,3).copy()));stream.capture(0);ms=0;previous=np.zeros(42);motorstate=motor.initial_state(1);visionstate=visual.initial_state(1);held=None;last_capture_id=-1
  def tick(action,source):
   nonlocal ms
   tr.tick(c,action,source);ms+=1;hr=c.d.xmat[stream.head].reshape(3,3);rot=c.d.xmat[c.th].reshape(3,3)
   geometry.append(dict(time=float(c.d.time),head_xyz=c.d.xpos[stream.head].copy(),head_rotation=hr.copy(),thorax_yaw=float(np.arctan2(rot[1,0],rot[0,0])),control_source=source,source_anchor_valid=stream.centers is not None,target_centers_xyz=np.zeros((3,3)) if stream.centers is None else np.asarray([stream.centers[x] for x in ('left','right','center')])))
   if ms%20==0 and ms<15400:stream.capture(ms)
  for _ in range(1000):tick(np.zeros(42),sw.SOURCE['settle'])
  teacher=sw.StaticTeacher(c,sw.teacher_config(3));teacher_calls=0;transition=None;observer=None;events=[]
  for k in range(2880):
   warm=k<960;now=1.+k*.005
   if k==960:
    before=sw.continuity_snapshot(c,previous,flatten(motorstate,visionstate));stream.anchor();observer=sw.tracker(c);events=[];after=sw.continuity_snapshot(c,previous,flatten(motorstate,visionstate))
    exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
    if not all(exact.values()):raise ValueError('continuous visionbody handover failed')
    transition=dict(time_s=float(c.d.time),all_exact=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),no_reset=True,state_dims=[128,64,64,32],camera_queue_retained=True,vision_hidden_initial_preserved=True);save(out/'HANDOVER.json',transition)
   obs=stream.queue.query(now)
   if not obs['valid']:raise ValueError('warmup must supply actual prior released RGB before neural decisions')
   fid=int(round(obs['capture_time_s']/.020));updated=fid!=last_capture_id
   with torch.no_grad():
    if updated:held=visual.encode(torch.from_numpy(obs['image'][None,None]))[0];last_capture_id=fid
    vb=visionstate[0].numpy().copy();mb=torch.cat(motorstate,-1)[0].numpy().copy();fullbefore=flatten(motorstate,visionstate)[0].numpy().copy()
    if warm:turn=held.new_zeros(1);latent=turn.clone()
    else:turn,latent,visionstate=visual.step(held,visionstate)
    command=project(turn);x,_=sw.observations(c,previous,(2880-k)*.005);mean,motorstate=motor.step(torch.tensor(x[None],dtype=torch.float64),command,motorstate);policy=mean[0].tanh().numpy().copy()
   if warm:action,_=teacher.action(c,k*.005);teacher_calls+=1;source=sw.SOURCE['teacher']
   else:action=policy.copy();source=sw.SOURCE['student']
   rows.append(dict(time=float(c.d.time),X238=x.copy(),held_features=held[0].numpy().copy(),encoder_cache_updated=updated,image_capture_id=fid,image_capture_time_s=obs['capture_time_s'],image_release_time_s=obs['release_time_s'],image_valid=obs['valid'],used_stimulus_code=('off','left','right','center').index(stream.rows[fid]['stimulus']),vision_hidden_before=vb,vision_hidden_after=visionstate[0].numpy().copy(),hidden_before=mb,hidden_after=torch.cat(motorstate,-1)[0].numpy().copy(),full_hidden_before=fullbefore,full_hidden_after=flatten(motorstate,visionstate)[0].numpy().copy(),raw_turn=float(turn[0]),latent=float(latent[0]),command=command[0].numpy().copy(),mean=mean[0].numpy().copy(),policy_action=policy,actually_applied_action=action.copy(),control_source=source))
   for _ in range(5):
    tick(action,source)
    if not warm:events.extend(observer.measure(c)['events'])
   previous=action.copy()
   if k%480==0:print(json.dumps(dict(condition=args.condition,decision=k,time=float(c.d.time),wall_s=time.monotonic()-start)),flush=True)
  auto=tr.summary(c,sw.SOURCE['student']);cycles=sw.cycle_report(observer,events);yaw=np.unwrap([r['thorax_yaw'] for r in geometry]);native=recorder.close()
  result=dict(kind='VISUAL_BODY_WORLD_LANDMARK_V1',condition=args.condition,stage='VISUAL_BODY',visual_checkpoint=binding['visual_checkpoint'],visual_checkpoint_sha256=binding['visual_sha256'],checkpoint=str(MOTOR),checkpoint_sha256=MOTOR_SHA,state_layout=[128,64,64,32],state_total=288,feature_cache16_separate=True,autonomous=auto,autonomous_cycles=cycles,native=native,engineering_pass=auto['all_gates'],functional_stable_body=stable(auto,cycles,native),functional_pass=None,yaw_change_rad=float(yaw[-1]-yaw[5799]),transition=transition,physical_steps=len(tr.rec)*80,teacher_calls=teacher_calls,teacher_calls_after_handover=0,extra_forces=False,mesh_count=71,all_mesh_opaque=True,source_hashes=sources,identity_sha256=ident['identity_sha256'],source_kind='ACTUAL_MOVING_HEAD_WORLD_FIXED_RENDER_ONLY_LANDMARK',olfaction_connected=False,food_success=False,capture_count=len(stream.rows),camera_state_reset=False,wall_s=time.monotonic()-start)
  identity.verify_runtime(ident,c);identity.verify(ident);verify()
  for path,h in sources.items():
   if sha(path)!=h:raise ValueError('source drift '+path)
 except BaseException as exc:error=exc
 finally:
  artifacts={}
  if tr is not None and tr.rows:artifacts['trace']=tr.save(out/'trace_1ms.npz')
  for filename,records,key in [('neural_decisions.npz',rows,'neural_decisions'),('geometry_1ms.npz',geometry,'geometry')]:
   if records:
    path=out/filename;np.savez_compressed(path,**{key:np.asarray([row[key] for row in records]) for key in records[0]});artifacts[key]=dict(path=str(path),sha256=sha(path),rows=len(records))
  if stream is not None:
   if stream.images:
    path=out/'capture_rgb.npz';np.savez_compressed(path,rgb=np.stack(stream.images));artifacts['capture_rgb']=dict(path=str(path),sha256=sha(path),rows=len(stream.images));save(out/'CAPTURES.json',stream.rows);artifacts['captures_metadata']=dict(path=str(out/'CAPTURES.json'),sha256=sha(out/'CAPTURES.json'))
   stream.close()
  if (out/'INITIAL_STATE.json').exists():artifacts['initial_state']=dict(path=str(out/'INITIAL_STATE.json'),sha256=sha(out/'INITIAL_STATE.json'))
  if (out/'runtime_geometry.npz').exists():artifacts['runtime_geometry']=dict(path=str(out/'runtime_geometry.npz'),sha256=sha(out/'runtime_geometry.npz'))
  if error is not None:save(out/'FAILURE.json',dict(error=repr(error),artifacts=artifacts,physical_endpoints=0 if tr is None else len(tr.rec),neural_decisions=len(rows)))
  elif result is not None:result.update(artifacts);save(out/'RESULT.json',result)
 if error is not None:raise error
 print(json.dumps(dict(condition=args.condition,stable=result['functional_stable_body'],yaw_deg=float(np.degrees(result['yaw_change_rad'])),slip=auto['slip_ratio_max'],captures=result['capture_count'])),flush=True)
if __name__=='__main__':main()
