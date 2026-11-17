"""Fresh actual RGB/headfield/pending/sourcequeue/fourNN replay; no Core construction."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS,required=True);p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no numericalreplay');return
 cpu1();verify();visual,odor,bridge,motor,vp,op,bp,mp=load_policies();sw,identity=pin()
 import numpy as np
 import torch
 folder=ROOT/'trials'/a.condition;r=read(folder/'RESULT.json');ident=read(folder/'IDENTITY.json');identity.verify(ident);checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Sensorybodyaudit '+k)
 check('actualidentity',r['kind']==KIND and r['condition']==a.condition and r['identity_sha256']==ident['identity_sha256']);check('allfourpolicyids',r['visual_checkpoint_sha256']==VISION_SHA and r['odor_checkpoint_sha256']==ODOR_SHA and r['neck_source_binding_sha256']==BINDING_SHA and r['checkpoint_sha256']==MOTOR_SHA and r['brain_checkpoint_sha256']==read(ROOT/'INPUT_BINDINGS.json')['brain_sha256'])
 for path,h in r['source_hashes'].items():check('source:'+path,sha(path)==h)
 for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('capture_rgb.npz','capture_rgb'),('CAPTURES.json','captures_metadata'),('SOURCE_QUEUE.json','source_queue'),('INITIAL_STATE.json','initial_state'),('runtime_geometry.npz','runtime_geometry')]:check('artifact:'+name,sha(folder/name)==r[key]['sha256'])
 t=arrays(folder/'trace_1ms.npz');d=arrays(folder/'neural_decisions.npz');geom=arrays(folder/'geometry_1ms.npz');compiled=arrays(folder/'runtime_geometry.npz');rgb=arrays(folder/'capture_rgb.npz')['rgb'];caps=read(folder/'CAPTURES.json');initial=read(folder/'INITIAL_STATE.json');savedqueue=read(folder/'SOURCE_QUEUE.json')
 check('fullshape336_3080_15400_770',d['full_hidden_before'].shape==d['full_hidden_after'].shape==(3080,336) and d['X238'].shape==(3080,238) and t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and rgb.shape==(770,2,32,32,3) and len(caps)==770)
 check('allnumericfinite',all(np.isfinite(v).all() for data in (t,d,geom,compiled) for v in data.values()));check('physical_neural_clocks',np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8 and np.array_equal(d['physical_step_ms'],np.arange(3080)*5));check('336statechain',np.array_equal(d['full_hidden_before'][0],np.zeros(336)) and np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]))
 for suffix in ('before','after'):check('336components_'+suffix,np.array_equal(d['full_hidden_'+suffix],np.concatenate((d['hidden_'+suffix],d['vision_hidden_'+suffix],d['odor_hidden_'+suffix],d['brain_hidden_'+suffix]),1)))
 check('phase_sources',np.array_equal(t['control_source'],np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)]));check('policy_action_only_ANN',np.array_equal(d['actually_applied_action'][1160:],d['policy_action'][1160:]) and np.array_equal(t['action'],np.repeat(d['actually_applied_action'],5,axis=0)));check('allthree_brain_warmzero',all(np.max(abs(d[k][:1160]))==0 for k in ('vision_hidden_after','odor_hidden_after','brain_hidden_after','DN_encoded','command')));check('no_direct_body_to_brain',bridge.allowed_raw_indices.tolist()==list(range(42))+list(range(84,126))+list(range(168,195)) and bridge.brain.input_size==3)
 cg=camera_runtime();camera=module('_sensoryaudit_actualcamera',VIS/'camera.py');spec=read(VIS/'PREREG_CAMERA_PROBE.json')['eyes'];headproof=read(HEAD_AUDIT)
 for i,cap in enumerate(caps):
  check('rgbsha_'+str(i),cap['capture_id']==i and hashlib.sha256(rgb[i].tobytes()).hexdigest()==cap['rgb_sha256'] and abs(cap['capture_time_s']-i*.02)<1e-10 and abs(cap['release_time_s']-(i*.02+.02))<1e-10)
  check('renderphysicalunchanged_'+str(i),cap['render_does_not_modify_physical_data'] and cap['physical_before_sha256']==cap['physical_after_sha256'] and cap['contact_pipeline_unchanged'])
  frames=camera.eye_frames(np.asarray(cap['actual_head_position']),np.asarray(cap['actual_head_rotation']),headproof,spec);check('actualeyegeometry_'+str(i),all(np.array_equal(frames[j][key],cap['camera_metadata'][j][key]) for j in range(2) for key in ('pos','forward','up')))
 loop=BrainNeckLoop(bridge,a.condition);field=OdorField(a.condition);queue=camera.PendingVision();queue.capture(0.,rgb[0]);vh=visual.initial_state(1);oh=odor.initial_state(1);mh=motor.initial_state(1);held=visual.head.weight.new_zeros((1,16));last=-1;replayed=[]
 with torch.no_grad():
  for k in range(3080):
   ms=k*5;head=np.asarray(initial['head_xyz']) if k==0 else geom['head_xyz'][ms-1];rot=np.asarray(initial['head_rotation']) if k==0 else geom['head_rotation'][ms-1]
   if ms==5800:field.anchor(head,rot)
   ss=field.sample(head,rot,ms);before=flat(mh,vh,oh,loop.brain)[0].numpy().copy();obs=queue.query(ms*.001);fid=-1 if not obs['valid'] else int(round(obs['capture_time_s']/.02));updated=obs['valid'] and fid!=last
   if updated:held=visual.encode(torch.from_numpy(rgb[fid][None,None]))[0];last=fid
   if ms<5800:vr=held.new_zeros(1);vl=vr.clone();orr=vr.clone();ol=vr.clone()
   else:vr,vl,vh=visual.step(held,vh);meanodor,oh=odor.mean_step(torch.tensor(ss['actor_receptors'][None],dtype=torch.float64),oh);ol=meanodor[:,0];orr=ol.tanh()
   n=loop.decision(d['X238'][k],vr,orr,ms)
   if ms<1000:mean=np.zeros(42);policy=np.zeros(42)
   else:means,mh=motor.step(torch.tensor(d['X238'][k:k+1],dtype=torch.float64),torch.tensor(n['command'][None],dtype=torch.float64),mh);mean=means[0].numpy().copy();policy=means[0].tanh().numpy().copy()
   replayed.append(dict(full_hidden_before=before,full_hidden_after=flat(mh,vh,oh,loop.brain)[0].numpy().copy(),hidden_after=torch.cat(mh,-1)[0].numpy().copy(),vision_hidden_after=vh[0].numpy().copy(),odor_hidden_after=oh[0].numpy().copy(),visual_raw_turn=float(vr[0]),visual_latent=float(vl[0]),odor_raw_turn=float(orr[0]),odor_latent=float(ol[0]),held_features=held[0].numpy().copy(),image_capture_id=fid,image_valid=obs['valid'],encoder_cache_updated=updated,mean=mean,policy_action=policy,**ss,**n))
   for tick in range(ms+1,ms+6):
    loop.advance()
    if tick%20==0 and tick<15400:queue.capture(tick*.001,rgb[tick//20])
 for key in replayed[0]:check('actualRGB_field_receptor_queue_4NN_replay:'+key,np.array_equal(np.asarray([row[key] for row in replayed]),d[key]))
 check('sourcequeueall_events_exact',jsonable(loop.finalize())==savedqueue);check('filter_endstate_exact',jsonable(field.snapshot())==r['odor_end_state']);check('actualproxies_all1ms',np.array_equal(np.asarray([sensors(h,rr) for h,rr in zip(geom['head_xyz'],geom['head_rotation'])]),geom['sensor_xyz']));check('fixedworldodoranchor',np.array_equal(geom['odor_source_xy'][5799:],np.broadcast_to(field.center,(9601,2))))
 rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,15400)];auto=sw.summarise(rec,768000,initial_heading=np.asarray(r['autonomous']['initial_heading_diagnostics']['heading_world_xy']));auto['window']=dict(first_sample_s=float(t['time'][5800]),last_sample_s=float(t['time'][-1]),sample_count=9600);check('originalmetrics_exact',jsonable(auto)==r['autonomous']);observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
 for i in range(5800,15400):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
 cycles=sw.cycle_report(observer,events);check('cyclesexact',jsonable(cycles)==r['autonomous_cycles']);check('nativefull_duration',r['native']['actual_native_steps']==1232000 and r['native']['post_native_steps']==768000 and r['physical_steps']==1232000);check('stabilityscoredseparate',stable(auto,cycles,r['native'])==r['functional_stable_body']);check('all71opaque',np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1))
 hand=read(folder/'HANDOVER.json');check('handovercontinuous',hand==r['transition'] and hand['all_exact'] and hand['no_reset'] and hand['before_sha256']==hand['after_sha256']);oldt=arrays(NEUTRAL/'trace_1ms.npz');oldd=arrays(NEUTRAL/'neural_decisions.npz')
 for key in t:
  if key in oldt:check('physicalwarmunchanged:'+key,np.array_equal(t[key][:5800],oldt[key][:5800]))
 for key in ('X238','mean','policy_action','actually_applied_action','hidden_before','hidden_after'):check('motorprimeunchanged:'+key,np.array_equal(d[key][200:1160],oldd[key][:960]))
 check('no_teacher_afterhandover',r['teacher_calls']==960 and r['teacher_calls_after_handover']==0);identity.verify(ident);verify();save(folder/'VERIFY.json',dict(integrity_all_pass=True,checks=checks,condition=a.condition,result_sha256=sha(folder/'RESULT.json'),brain_checkpoint_sha256=r['brain_checkpoint_sha256'],visual_checkpoint_sha256=VISION_SHA,odor_checkpoint_sha256=ODOR_SHA,checkpoint_sha256=MOTOR_SHA,source_hashes=r['source_hashes'],same_actualRGB_receptors_X_sourcequeue_full336_replay=True,environment_headfield_exact=True,state_total=336,Core_constructions=0,new_physics_steps=0));print(json.dumps(dict(condition=a.condition,integrity_all_pass=True,checks=len(checks))))
if __name__=='__main__':main()
