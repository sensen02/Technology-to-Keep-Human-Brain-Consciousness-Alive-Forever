"""Independent saved actual RGB -> batch1 CNN -> GRU/motor and metrics replay. No Core."""
import argparse,hashlib
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS,required=True);p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:print('CODE_ONLY no model/arrays');return
 cpu1();verify();visual,motor,vp,mp=load_policies();sw,identity=pin()
 import numpy as np
 import torch
 out=ROOT/'trials'/args.condition;r=read(out/'RESULT.json');ident=read(out/'IDENTITY.json');identity.verify(ident);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('visualbody independent audit '+k)
 check('identity_bound',r['identity_sha256']==ident['identity_sha256'] and r['source_hashes']==ident['config']['evaluation_source_hashes'])
 check('fixedpolicy_identity',r['checkpoint_sha256']==MOTOR_SHA and r['visual_checkpoint_sha256']==read(ROOT/'INPUT_BINDINGS.json')['visual_sha256'])
 for path,h in r['source_hashes'].items():check('source:'+path,sha(path)==h)
 for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata'),('INITIAL_STATE.json','initial_state')]:check('hash:'+name,sha(out/name)==r[key]['sha256'])
 t=arrays(out/'trace_1ms.npz');d=arrays(out/'neural_decisions.npz');g=arrays(out/'geometry_1ms.npz');rgb=arrays(out/'capture_rgb.npz')['rgb'];captures=read(out/'CAPTURES.json');compiled=arrays(out/'runtime_geometry.npz')
 check('all_saved_finite',all(np.isfinite(v).all() for group in (t,d,g,compiled) for v in group.values()))
 check('fullbody15400',t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48));check('capture770uint8',rgb.shape==(770,2,32,32,3) and rgb.dtype==np.uint8 and len(captures)==770 and r['capture_count']==770)
 check('clock1ms',np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8);check('clock5ms',np.max(abs(d['time']-(1+np.arange(2880)*.005)))<1e-8)
 check('full288chain',d['full_hidden_before'].shape==d['full_hidden_after'].shape==(2880,288) and np.array_equal(d['full_hidden_before'][0],np.zeros(288)) and np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]))
 check('components256plus32',all(np.array_equal(d['full_hidden_'+s],np.concatenate((d['hidden_'+s],d['vision_hidden_'+s]),axis=1)) for s in ('before','after')))
 sources=np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)]
 check('control_sources',np.array_equal(t['control_source'],sources) and np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)]))
 check('actualheldactions',np.array_equal(t['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0)) and np.array_equal(t['action'],t['actual_requested_action']))
 check('ANNdirect42outputs',np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]))
 check('zero_settle_noextra',np.array_equal(t['action'][:1000],np.zeros((1000,42))) and all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')))
 check('observed_previous_action',np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]) and np.array_equal(d['X238'][0,195:237],np.zeros(42)))
 check('no_actor_clock',np.array_equal(d['X238'][:,-1],np.full(2880,.075)) and visual.identity()['input']=='actualRGB2x32x32x3only')
 camera=module('_visual_audit_camera',VIS/'camera.py');spec=read(VIS/'PREREG_CAMERA_PROBE.json')['eyes'];audit=read(GEOMETRY)
 initial=read(out/'INITIAL_STATE.json');check('capture0_initial_snapshot',initial['time_s']==0 and np.array_equal(initial['head_xyz'],captures[0]['actual_head_position']) and np.array_equal(initial['head_rotation'],captures[0]['actual_head_rotation']) and hashlib.sha256(np.asarray(initial['qpos'],dtype=np.float64).tobytes()).hexdigest()==captures[0]['physical_before_sha256']['qpos'] and hashlib.sha256(np.asarray(initial['qvel'],dtype=np.float64).tobytes()).hexdigest()==captures[0]['physical_before_sha256']['qvel'])
 caphead=captures[290];centers=anchor_landmarks(np.asarray(caphead['actual_head_position']),np.asarray(caphead['actual_head_rotation']),camera,spec,audit)
 for i,cap in enumerate(captures):
  check('capture_clock'+str(i),cap['capture_id']==i and cap['capture_time_s']==i*.020 and cap['release_time_s']==i*.020+.020 and abs(cap['actual_physical_time_s']-i*.020)<1e-8)
  check('capture_RGB'+str(i),hashlib.sha256(rgb[i].tobytes()).hexdigest()==cap['rgb_sha256'])
  check('render_readonly'+str(i),cap['render_does_not_modify_physical_data'] is True and cap['contact_pipeline_unchanged'] is True and cap['physical_before_sha256']==cap['physical_after_sha256'])
  expected=frame_target(args.condition,i*.020-5.8) if i>290 else 'off'
  check('capture_world_visibility'+str(i),cap['stimulus']==expected)
  if expected!='off':check('capture_fixedworldcenter'+str(i),np.array_equal(cap['target_center_xyz'],centers[expected]))
  if i>0:
   ix=i*20-1;check('capture_actualhead'+str(i),np.array_equal(cap['actual_head_position'],g['head_xyz'][ix]) and np.array_equal(cap['actual_head_rotation'],g['head_rotation'][ix]))
  frames=camera.eye_frames(np.asarray(cap['actual_head_position']),np.asarray(cap['actual_head_rotation']),audit,spec)
  check('capture_actualopticalbasis'+str(i),all(np.array_equal(frames[j][key],cap['camera_metadata'][j][key]) for j in range(2) for key in ('pos','forward','up')))
 querytimes=1+np.arange(2880)*.005;expectedids=np.floor((np.arange(2880)*5+1000-20)/20).astype(int)
 check('20ms_delayed_frame_ids',np.array_equal(d['image_capture_id'],expectedids) and np.all(d['image_valid']))
 check('image_release_no_future',np.max(np.abs(d['image_capture_time_s']-expectedids*.020))<1e-10 and np.max(np.abs(d['image_release_time_s']-(expectedids*.020+.020)))<1e-10 and np.all(d['image_release_time_s']<=querytimes+1e-10))
 check('encoder_only_new_frame',np.array_equal(d['encoder_cache_updated'],np.r_[True,expectedids[1:]!=expectedids[:-1]]))
 ms=motor.initial_state(1);vs=visual.initial_state(1);last=-1;features=None;means=[];mh=[];vh=[];turns=[];latents=[];commands=[];actions=[];ff=[]
 with torch.no_grad():
  for k,fid in enumerate(expectedids):
   if fid!=last:features=visual.encode(torch.from_numpy(rgb[fid][None,None]))[0];last=fid
   ff.append(features[0].numpy().copy())
   if k<960:turn=features.new_zeros(1);latent=turn.clone()
   else:turn,latent,vs=visual.step(features,vs)
   cmd=project(turn);mean,ms=motor.step(torch.tensor(d['X238'][k:k+1],dtype=torch.float64),cmd,ms)
   means.append(mean[0].numpy().copy());mh.append(torch.cat(ms,-1)[0].numpy().copy());vh.append(vs[0].numpy().copy());turns.append(float(turn[0]));latents.append(float(latent[0]));commands.append(cmd[0].numpy().copy());actions.append(mean[0].tanh().numpy().copy())
 for name,value,key in [('CNNcache',ff,'held_features'),('visionturn',turns,'raw_turn'),('visionlatent',latents,'latent'),('visionmemory',vh,'vision_hidden_after'),('motorhidden',mh,'hidden_after'),('motormeans',means,'mean'),('projectedcommands',commands,'command'),('torqueactions',actions,'policy_action')]:check('actual_RGB_online_'+name+'_exact',np.array_equal(np.asarray(value),d[key]))
 check('warm_vision_heldzero',np.max(abs(d['vision_hidden_after'][:960]))==0)
 hand=read(out/'HANDOVER.json');check('handover_same',hand==r['transition'] and hand['all_exact'] and hand['no_reset'] and hand['camera_queue_retained'] and hand['before_sha256']==hand['after_sha256'])
 boundary={'qpos':t['qpos'][5799],'qvel':t['qvel'][5799],'ctrl':t['actualctrl'][5799],'target':t['target'][5799],'requested_rate':t['actual_requested_action'][5799]*2.,'previous_actual_action':d['actually_applied_action'][959],'time':np.asarray(t['time'][5799]),'hidden':d['full_hidden_before'][960:961]}
 for key,value in boundary.items():check('handover_boundary:'+key,hashlib.sha256(value.tobytes()).hexdigest()==hand['before_sha256'][key])
 check('no_teacher_afterhandover',r['teacher_calls']==960 and r['teacher_calls_after_handover']==0)
 rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,15400)];auto=sw.summarise(rec,768000,initial_heading=np.asarray(r['autonomous']['initial_heading_diagnostics']['heading_world_xy']));auto['window']=dict(first_sample_s=float(t['time'][5800]),last_sample_s=float(t['time'][-1]),sample_count=9600)
 check('original_metrics_exact',jsonable(auto)==r['autonomous'])
 observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
 for i in range(5800,15400):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
 cycles=sw.cycle_report(observer,events);check('cycles_recomputed_exact',jsonable(cycles)==r['autonomous_cycles']);check('body_stability_scored_separate',stable(auto,cycles,r['native'])==r['functional_stable_body'] and r['engineering_pass']==auto['all_gates'])
 yaw=np.unwrap(g['thorax_yaw']);check('actualyaw_exact',float(yaw[-1]-yaw[5799])==r['yaw_change_rad'])
 check('nativefull_complete',r['native']['actual_native_steps']==1232000 and r['native']['post_native_steps']==768000 and r['physical_steps']==1232000)
 baseline=read(NEUTRAL/'IDENTITY.json');check('unchanged_physical_compile',ident['compiled_arrays']==baseline['compiled_arrays'] and ident['physics']==baseline['physics'] and ident['actuator']==baseline['actuator'])
 check('71_opaque',np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1))
 oldt=arrays(NEUTRAL/'trace_1ms.npz');oldd=arrays(NEUTRAL/'neural_decisions.npz')
 for key in t:
  if key in oldt:check('unchanged_physics_prime:'+key,np.array_equal(t[key][:5800],oldt[key][:5800]))
 for key in ('X238','mean','policy_action','actually_applied_action','hidden_before','hidden_after'):check('unchanged_motor_prime:'+key,np.array_equal(d[key][:960],oldd[key][:960]))
 identity.verify(ident);verify();save(out/'VERIFY.json',dict(integrity_all_pass=True,checks=checks,condition=args.condition,run=str(out),result_sha256=sha(out/'RESULT.json'),visual_checkpoint_sha256=r['visual_checkpoint_sha256'],checkpoint_sha256=MOTOR_SHA,physics_steps=0,Core_constructions=0,offline_RGB_CNN_GRU_motor_replay=True,engineering_pass=r['engineering_pass'],functional_stable_body=r['functional_stable_body'],state_total=288))
 print(json.dumps(dict(condition=args.condition,checks=len(checks),integrity_all_pass=True)))
if __name__=='__main__':main()
