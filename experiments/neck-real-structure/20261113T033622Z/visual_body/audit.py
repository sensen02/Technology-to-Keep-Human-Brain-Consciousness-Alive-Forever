"""Fresh-process savedactualRGB/obs->vision32->sourcequeue->brain16->motor replay, zeroCore."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS,required=True);p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no numerical replay');return
 cpu1();verify();visual,bridge,motor,vp,bp,mp=load_policies();sw,identity=pin()
 import numpy as np
 import torch
 out=ROOT/'trials'/a.condition;r=read(out/'RESULT.json');ident=read(out/'IDENTITY.json');identity.verify(ident);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Visualneck independent audit '+k)
 check('identity_bound',r['kind']==KIND and r['condition']==a.condition and r['camera_condition']==camera_condition(a.condition) and r['identity_sha256']==ident['identity_sha256'] and r['source_hashes']==ident['config']['evaluation_source_hashes'])
 check('fixed_three_policy_and_source_identity',r['visual_checkpoint_sha256']==VISION_SHA and r['brain_checkpoint_sha256']==read(ROOT/'INPUT_BINDINGS.json')['brain_sha256'] and r['checkpoint_sha256']==MOTOR_SHA and r['neck_source_binding_sha256']==BINDING_SHA)
 for path,h in r['source_hashes'].items():check('source:'+path,sha(path)==h)
 for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata'),('SOURCE_QUEUE.json','source_queue'),('INITIAL_STATE.json','initial_state')]:check('hash:'+name,sha(out/name)==r[key]['sha256'])
 t=arrays(out/'trace_1ms.npz');d=arrays(out/'neural_decisions.npz');g=arrays(out/'geometry_1ms.npz');rgb=arrays(out/'capture_rgb.npz')['rgb'];captures=read(out/'CAPTURES.json');compiled=arrays(out/'runtime_geometry.npz');savedqueue=read(out/'SOURCE_QUEUE.json')
 check('all_saved_numeric_finite',all(np.isfinite(v).all() for group in (t,d,g,compiled) for v in group.values()))
 check('complete304chain3080body15400',t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and d['X238'].shape==(3080,238) and d['full_hidden_before'].shape==d['full_hidden_after'].shape==(3080,304) and np.array_equal(d['full_hidden_before'][0],np.zeros(304)) and np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]))
 check('components256plus32plus16',all(np.array_equal(d['full_hidden_'+s],np.concatenate((d['hidden_'+s],d['vision_hidden_'+s],d['brain_hidden_'+s]),1)) for s in ('before','after')))
 check('actual_clocks',np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8 and np.array_equal(d['physical_step_ms'],np.arange(3080)*5))
 check('capture770uint8',rgb.shape==(770,2,32,32,3) and rgb.dtype==np.uint8 and len(captures)==770 and r['capture_count']==770)
 check('actual_phase_sources',np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)]) and np.array_equal(d['control_source'],np.r_[np.zeros(200),np.full(960,2),np.full(1920,3)]))
 check('actual_action_hold',np.array_equal(t['action'],np.repeat(d['actually_applied_action'],5,axis=0)) and np.array_equal(t['action'],t['actual_requested_action']))
 check('ANN_policy_directly_applied',np.array_equal(d['actually_applied_action'][1160:],d['policy_action'][1160:]))
 check('warm_brains_DN_and_motorcmd_zero',np.max(abs(d['vision_hidden_after'][:1160]))==np.max(abs(d['brain_hidden_after'][:1160]))==np.max(abs(d['DN_encoded'][:1160]))==np.max(abs(d['command'][:1160]))==0)
 check('settle_actual_action_motorstate_zero',np.max(abs(d['actually_applied_action'][:200]))==np.max(abs(d['hidden_after'][:200]))==0)
 check('actualbody_previous_action_and_targetspeed_excluded_frombrain',np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]) and np.array_equal(d['X238'][:,237],np.full(3080,.075)) and bridge.allowed_raw_indices.tolist()==list(range(0,42))+list(range(84,126))+list(range(168,195)))
 check('brain_input_is_current_neural_visualscalar',np.array_equal(d['visual_brain_input'],d['visual_raw_turn']))
 cg=camera_runtime();camera=module('_visualneck_audit_optics',VIS/'camera.py');spec=read(VIS/'PREREG_CAMERA_PROBE.json')['eyes'];geom=read(PROJECT/'experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json');centers=cg['anchor_landmarks'](np.asarray(captures[290]['actual_head_position']),np.asarray(captures[290]['actual_head_rotation']),camera,spec,geom)
 initial=read(out/'INITIAL_STATE.json');check('capture0_actual_initial_head',initial['time_s']==0 and np.array_equal(initial['head_xyz'],captures[0]['actual_head_position']) and np.array_equal(initial['head_rotation'],captures[0]['actual_head_rotation']))
 for i,cap in enumerate(captures):
  check('capture_clock'+str(i),cap['capture_id']==i and cap['capture_time_s']==i*.02 and cap['release_time_s']==i*.02+.02 and abs(cap['actual_physical_time_s']-i*.02)<1e-8)
  check('capture_RGBhash'+str(i),hashlib.sha256(rgb[i].tobytes()).hexdigest()==cap['rgb_sha256'])
  check('render_no_physical_effect'+str(i),cap['render_does_not_modify_physical_data'] and cap['contact_pipeline_unchanged'] and cap['physical_before_sha256']==cap['physical_after_sha256'])
  expected=cg['frame_target'](camera_condition(a.condition),i*.02-5.8) if i>290 else 'off';check('metadata_stimulus_rendereronly'+str(i),cap['stimulus']==expected)
  if expected!='off':check('source_fixed_farworld_center'+str(i),np.array_equal(cap['target_center_xyz'],centers[expected]))
  if i:check('capture_actual_live_head'+str(i),np.array_equal(cap['actual_head_position'],g['head_xyz'][i*20-1]) and np.array_equal(cap['actual_head_rotation'],g['head_rotation'][i*20-1]))
  frames=camera.eye_frames(np.asarray(cap['actual_head_position']),np.asarray(cap['actual_head_rotation']),geom,spec);check('actual_eye_opticalbasis'+str(i),all(np.array_equal(frames[j][key],cap['camera_metadata'][j][key]) for j in range(2) for key in ('pos','forward','up')))
 class Actor:pass
 actor=Actor();actor.queue=camera.PendingVision();actor.condition=camera_condition(a.condition);actor.frozen_actor=None;actor.actor_query=types.MethodType(cg['CameraStream'].actor_query,actor);actor.queue.capture(0.,rgb[0]);loop=BrainNeckLoop(bridge,a.condition);motorstate=motor.initial_state(1);vs=visual.initial_state(1);held=visual.head.weight.new_zeros((1,16));last=-1;replayed=[]
 with torch.no_grad():
  for k in range(3080):
   ms=k*5;before=flat(motorstate,vs,loop.brain)[0].numpy().copy();obs=actor_observation(actor,ms*.001);fid=-1 if not obs['valid'] else int(round(obs['capture_time_s']/.02));updated=obs['valid'] and fid!=last
   if updated:held=visual.encode(torch.from_numpy(rgb[fid][None,None]))[0];last=fid
   if ms<5800:vr=held.new_zeros(1);latent=vr.clone()
   else:vr,latent,vs=visual.step(held,vs)
   n=loop.decision(d['X238'][k],vr,ms)
   if ms<1000:mean=np.zeros(42);policy=np.zeros(42)
   else:means,motorstate=motor.step(torch.tensor(d['X238'][k:k+1],dtype=torch.float64),torch.tensor(n['command'][None],dtype=torch.float64),motorstate);mean=means[0].numpy().copy();policy=means[0].tanh().numpy().copy()
   row=dict(full_hidden_before=before,full_hidden_after=flat(motorstate,vs,loop.brain)[0].numpy().copy(),hidden_after=torch.cat(motorstate,-1)[0].numpy().copy(),vision_hidden_after=vs[0].numpy().copy(),held_features=held[0].numpy().copy(),visual_raw_turn=float(vr[0]),visual_latent=float(latent[0]),encoder_cache_updated=updated,image_capture_id=fid,image_valid=obs['valid'],raw_released_image_capture_id=obs['raw_capture_id'],actor_frame_frozen=obs['actor_frame_frozen'],mean=mean,policy_action=policy,**n);replayed.append(row)
   for tick in range(ms+1,ms+6):
    loop.advance()
    if tick%20==0 and tick<15400:actor.queue.capture(tick*.001,rgb[tick//20])
 for name in replayed[0]:check('actual_RGB_source_neural_motor_replay:'+name,np.array_equal(np.asarray([x[name] for x in replayed]),d[name]))
 proof=loop.finalize();check('all_source_queue_consumptions_exact',jsonable(proof)==savedqueue);check('genuine_images_beforeANN',np.all(d['image_valid'][4:]) and not d['image_valid'][:4].any() and np.array_equal(d['image_capture_id'][:4],np.full(4,-1)))
 # Frozen legacy controls consume only1000ms onward immutable views: no on-disk mutation.
 controls={key:value[200:] for key,value in d.items()};cg['audit_controls'](check,out,dict(r,condition=camera_condition(a.condition)),controls,rgb,captures)
 hand=read(out/'HANDOVER.json');check('continuous_handover',hand==r['transition'] and hand['all_exact'] and hand['no_reset'] and hand['source_queue_retained'] and hand['feature_cache_retained'] and hand['before_sha256']==hand['after_sha256'])
 rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,15400)];auto=sw.summarise(rec,768000,initial_heading=np.asarray(r['autonomous']['initial_heading_diagnostics']['heading_world_xy']));auto['window']=dict(first_sample_s=float(t['time'][5800]),last_sample_s=float(t['time'][-1]),sample_count=9600);check('original_metrics_exact',jsonable(auto)==r['autonomous'])
 observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
 for i in range(5800,15400):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
 cycles=sw.cycle_report(observer,events);check('cycles_recomputed_exact',jsonable(cycles)==r['autonomous_cycles']);check('body_stability_scored_separate',stable(auto,cycles,r['native'])==r['functional_stable_body'] and r['engineering_pass']==auto['all_gates']);check('native_full_duration',r['native']['actual_native_steps']==1232000 and r['native']['post_native_steps']==768000 and r['physical_steps']==1232000)
 yaw=np.unwrap(g['thorax_yaw']);check('actualyaw_exact',float(yaw[-1]-yaw[5799])==r['yaw_change_rad']);baseline=read(NEUTRAL/'IDENTITY.json');check('physical_compile_unchanged',ident['compiled_arrays']==baseline['compiled_arrays'] and ident['physics']==baseline['physics'] and ident['actuator']==baseline['actuator']);check('71_opaque',np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1))
 oldt=arrays(NEUTRAL/'trace_1ms.npz');oldd=arrays(NEUTRAL/'neural_decisions.npz')
 for key in t:
  if key in oldt:check('unchanged_physical_warmup:'+key,np.array_equal(t[key][:5800],oldt[key][:5800]))
 for key in ('X238','mean','policy_action','actually_applied_action','hidden_before','hidden_after'):check('unchanged_motor_prime:'+key,np.array_equal(d[key][200:1160],oldd[key][:960]))
 check('no_teacher_afterhandover',r['teacher_calls']==960 and r['teacher_calls_after_handover']==0);identity.verify(ident);verify();save(out/'VERIFY.json',dict(integrity_all_pass=True,checks=checks,condition=a.condition,run=str(out),result_sha256=sha(out/'RESULT.json'),visual_checkpoint_sha256=VISION_SHA,brain_checkpoint_sha256=r['brain_checkpoint_sha256'],checkpoint_sha256=MOTOR_SHA,neck_source_binding_sha256=BINDING_SHA,saved_actual_RGB_source_queue_brain_motor_replay=True,offline_RGB_CNN_GRU_motor_replay=True,Core_constructions=0,physics_steps=0,state_total=304,engineering_pass=r['engineering_pass'],functional_stable_body=r['functional_stable_body']))
 print(json.dumps(dict(condition=a.condition,integrity_all_pass=True,checks=len(checks))),flush=True)
if __name__=='__main__':main()
