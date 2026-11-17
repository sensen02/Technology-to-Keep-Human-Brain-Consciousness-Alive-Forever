"""One complete actual A/B episode. No actual qpos/qvel writes or handover reset."""
import argparse,time,traceback
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--trial',choices=IDS,required=True);args=p.parse_args();cpu1();frozen=verify_frozen()
 if not read(ROOT/'PREFLIGHT.json')['all_passed']:raise ValueError('preflight required')
 stage='A' if args.trial in A_IDS else 'B'
 if stage=='B':
  gate=read(ROOT/'A_GATE.json')
  if not gate['all_A_integrity_and_functional_pass']:raise ValueError('A gate failed; B forbidden')
 motor,behavior,mp,bp=load_policies();sw,identity=pin();sw.teacher_preflight();sw.student_preflight()
 out=ROOT/'trials'/args.trial;out.mkdir(parents=True,exist_ok=False)
 source_hashes=dict(frozen['source_hashes']);source_hashes[str(ROOT/'FROZEN.json')]=sha(ROOT/'FROZEN.json');source_hashes[str(ROOT/'PREFLIGHT.json')]=sha(ROOT/'PREFLIGHT.json')
 if stage=='B':source_hashes[str(ROOT/'A_GATE.json')]=sha(ROOT/'A_GATE.json')
 c=None;tr=None;recorder=None;rows=[];geometry=[];result=None;error=None;started=time.monotonic()
 try:
  import mujoco as mj
  recorder=PeakRecorder();c=sw.Core(adhesion=False,recorder=recorder)
  mesh=c.plant.geom_type==mj.mjtGeom.mjGEOM_MESH
  if mesh.sum()!=71 or not np.all(c.plant.geom_rgba[mesh,3]==1.):raise ValueError('complete opaque71 required')
  ident=identity.capture(c,dict(kind='STIMULUS_AB_288_V1',condition=args.trial,stage=stage,motor_sha256=MOTOR_SHA,behavior_sha256=SENSOR_SHA,evaluation_source_hashes=source_hashes,state_layout=[128,64,64,32],state_total=288,receptor_filter_separate=True,experimental_visual_admission=str(ACCEPT),old_strict_motor_qualified=False),extra_paths=list(source_hashes))
  identity.atomic_json(out/'IDENTITY.json',ident)
  np.savez_compressed(out/'runtime_geometry.npz',geom_type=c.plant.geom_type,geom_rgba=c.plant.geom_rgba)
  head=c.plant.body('Head').id;audit=read(GEOMETRY);local=np.asarray(audit['sensor_offsets']['local_xyz'])
  if head!=audit['head']['body_id']:raise ValueError('head ID mismatch')
  tr=sw.Trace(full=True)
  def tick(action,source):
   tr.tick(c,action,source);rot=c.d.xmat[c.th].reshape(3,3);hr=c.d.xmat[head].reshape(3,3);pos=c.d.xpos[head].copy();sensors=pos+local@hr.T
   geometry.append(dict(time=float(c.d.time),thorax_yaw=float(np.arctan2(rot[1,0],rot[0,0])),head_yaw=float(np.arctan2(hr[1,0],hr[0,0])),head_xyz=pos,head_rotation=hr.copy(),sensor_xyz=sensors,sensor_midpoint_xyz=sensors.mean(0),control_source=int(source)))
  for _ in range(1000):tick(np.zeros(42),sw.SOURCE['settle'])
  teacher=sw.StaticTeacher(c,sw.teacher_config(3));ms=motor.initial_state(1);bs=behavior.initial_state(1);filter=ClampFilter();previous=np.zeros(42);events=[];observer=None;handover=None;teacher_calls=0
  for k in range(2880):
   warm=k<960;x,_=sw.observations(c,previous,(2880-k)*.005);elapsed=(k-960)*.005
   if k==960:
    before=sw.continuity_snapshot(c,previous,fullflat(ms,bs));observer=sw.tracker(c);events=[]
    after=sw.continuity_snapshot(c,previous,fullflat(ms,bs));exact={key:np.array_equal(before[key],after[key]) for key in before}
    if not all(exact.values()):raise ValueError('handover discontinuity')
    handover=dict(time_s=float(c.d.time),all_exact=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),no_reset=True,state_dims=[128,64,64,32],motor_state_dims=[128,64,64],filter_state_before=filter.state.tolist(),pending_before=filter.pending.tolist())
    identity.atomic_json(out/'HANDOVER.json',jsonable(handover))
   mb=flat(ms)[0].detach().numpy().copy();bb=bs[0].detach().numpy().copy();fb=fullflat(ms,bs)[0].detach().numpy().copy()
   sensory={key:np.zeros(2) for key in ('receptors','concentration','filter_before','filter_after','pending_before','pending_after')}
   with torch.no_grad():
    if warm:raw=torch.zeros((1,2),dtype=torch.float64);command=project(raw)
    elif stage=='A':raw=torch.tensor([[0.,probe_turn(args.trial,elapsed)]],dtype=torch.float64);command=project(raw)
    else:
     sensory=filter.advance(concentration(args.trial,elapsed));raw,bs=behavior.step(torch.tensor(sensory['receptors'][None],dtype=torch.float64),bs);command=project(raw)
    mean,ms=motor.step(torch.tensor(x[None],dtype=torch.float64),command,ms)
   policy=mean[0].tanh().numpy().copy()
   if warm:action,_=teacher.action(c,k*.005);teacher_calls+=1;code=sw.SOURCE['teacher']
   else:action=policy.copy();code=sw.SOURCE['student']
   row=dict(time=float(c.d.time),X238=x.copy(),command=command[0].numpy().copy(),raw_command=raw[0].numpy().copy(),raw_turn=float(raw[0,1]),hidden_before=mb,hidden_after=flat(ms)[0].numpy().copy(),behavior_hidden_before=bb,behavior_hidden_after=bs[0].numpy().copy(),full_hidden_before=fb,full_hidden_after=fullflat(ms,bs)[0].numpy().copy(),mean=mean[0].numpy().copy(),policy_action=policy,actually_applied_action=action.copy(),control_source=code,**sensory)
   rows.append(row)
   for _ in range(5):
    tick(action,code)
    if not warm:events.extend(observer.measure(c)['events'])
   previous=action.copy()
   if k%480==0:print(json.dumps(dict(trial=args.trial,decision=k,physical_time=float(c.d.time),wall_s=time.monotonic()-started)),flush=True)
  auto=tr.summary(c,sw.SOURCE['student']);cycles=sw.cycle_report(observer,events);g={key:np.asarray([row[key] for row in geometry]) for key in geometry[0]};yy=np.unwrap(g['thorax_yaw']);yaw=float(yy[-1]-yy[5799]);native=recorder.close();flags=strict_flags(auto,cycles,native,yaw)
  functional=evaluate_A(args.trial,auto,cycles,native,g) if stage=='A' else dict(stable_body=basic_functional(auto,cycles,native),stimulus_comparison='PENDING matched B_noodor independent audit')
  result=dict(status='EXPERIMENTAL_STIMULUS_PHYSICS_NOT_FORMAL_MOTOR_OR_FOOD_QUALIFICATION',kind='STIMULUS_AB_288_V1',condition=args.trial,stage=stage,checkpoint=str(MOTOR),checkpoint_sha256=MOTOR_SHA,behavior_checkpoint=str(SENSOR),behavior_checkpoint_sha256=SENSOR_SHA,autonomous=auto,autonomous_cycles=cycles,yaw_change_rad=yaw,native=native,criteria=flags,engineering_pass=bool(auto['all_gates']),strict_endpoint_qualified=flags['eligible'],endpoint30_applicable_to_stimulus=False,old_motor_engineering_qualified=False,old_motor_qualification_unchanged=True,original_motor_flags_unchanged=True,experimental_user_visual_admission=True,transition=handover,state_layout=[128,64,64,32],motor_state_layout=[128,64,64],state_total=288,filter_state_separate=True,functional=functional,functional_pass=functional.get('functional_pass',None),physical_steps=len(tr.rec)*80,teacher_calls=teacher_calls,teacher_calls_after_handover=0,extra_forces=False,mesh_count=71,all_mesh_opaque=True,source_hashes=source_hashes,identity_sha256=ident['identity_sha256'],food_success=False,source_kind='NONE_HIGH_LEVEL_MOTOR_PROBE' if stage=='A' else 'CONCENTRATION_CLAMP_NOT_SPATIAL_NAVIGATION',wall_s=time.monotonic()-started)
  identity.verify_runtime(ident,c)
  for path,h in source_hashes.items():
   if sha(path)!=h:raise ValueError('input/source changed '+path)
 except BaseException as exc:error=exc
 finally:
  artifacts={}
  if tr is not None and tr.rows:artifacts['trace']=tr.save(out/'trace_1ms.npz')
  for name,records,key in [('neural_decisions',rows,'neural_decisions'),('geometry_1ms',geometry,'geometry')]:
   if records:
    path=out/(name+'.npz');np.savez_compressed(path,**{k:np.asarray([v[k] for v in records]) for k in records[0]});artifacts[key]=dict(path=str(path),sha256=sha(path),rows=len(records))
  if (out/'runtime_geometry.npz').exists():artifacts['runtime_geometry']=dict(path=str(out/'runtime_geometry.npz'),sha256=sha(out/'runtime_geometry.npz'))
  if recorder is not None:artifacts['native']=recorder.close()
  if error is None and result is not None:result.update(artifacts);save_new(out/'RESULT.json',result)
  else:save_new(out/'FAILURE.json',dict(error=repr(error),completed_endpoints=0 if tr is None else len(tr.rec),decisions=len(rows),artifacts=artifacts))
 if error is not None:raise error
 print(json.dumps(dict(trial=args.trial,functional=functional,yaw_deg=np.degrees(yaw),slip=auto['slip_ratio_max'],cycles=cycles['complete_per_leg'],native_nonfoot=native['post_nonfoot_peak_native'])),flush=True)
if __name__=='__main__':main()
