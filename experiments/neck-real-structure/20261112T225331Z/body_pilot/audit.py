"""Fresh-process saved actualobs->sourcequeue->bridge->motor replay and metric audit;0Core."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=IDS,required=True);p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no numerical replay');return
 cpu1();verify();bridge,motor,bp,mp=load_policies();sw,identity=pin()
 import numpy as np
 import torch
 out=ROOT/'trials'/a.condition;r=read(out/'RESULT.json');ident=read(out/'IDENTITY.json');identity.verify(ident);checks={}
 def check(name,v):
  checks[name]=bool(v)
  if not checks[name]:raise ValueError('Actual source/body independent audit '+name)
 check('identity_and_conditions',r['condition']==a.condition and r['identity_sha256']==ident['identity_sha256'] and r['source_hashes']==ident['config']['evaluation_source_hashes'])
 check('exact_model_and_source_bindings',r['brain_checkpoint_sha256']==BRIDGE_SHA and r['checkpoint_sha256']==MOTOR_SHA and r['neck_source_binding_sha256']==BINDING_SHA)
 for path,h in r['source_hashes'].items():check('source:'+path,sha(path)==h)
 for name,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('geometry_1ms.npz','geometry'),('runtime_geometry.npz','runtime_geometry'),('INITIAL_STATE.json','initial_state'),('SOURCE_QUEUE.json','source_queue')]:check('saved_hash:'+name,sha(out/name)==r[key]['sha256'])
 t=arrays(out/'trace_1ms.npz');d=arrays(out/'neural_decisions.npz');g=arrays(out/'geometry_1ms.npz');compiled=arrays(out/'runtime_geometry.npz');queue=read(out/'SOURCE_QUEUE.json')
 check('all_numeric_finite',all(np.isfinite(v).all() for data in (t,d,g,compiled) for v in data.values()))
 check('complete_body15400_neural3080',t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and d['X238'].shape==(3080,238) and r['neural_records']==3080)
 check('actual_clocks',np.max(abs(t['time']-np.arange(1,15401)*.001))<1e-8 and np.max(abs(d['time']-np.arange(3080)*.005))<1e-8 and np.array_equal(d['physical_step_ms'],np.arange(3080)*5))
 check('272_chain',d['full_hidden_before'].shape==d['full_hidden_after'].shape==(3080,272) and np.array_equal(d['full_hidden_before'][0],np.zeros(272)) and np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]))
 check('brain16_motor256_components',all(np.array_equal(d['full_hidden_'+suffix],np.concatenate((d['hidden_'+suffix],d['brain_hidden_'+suffix]),axis=1)) for suffix in ('before','after')))
 sources=np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)];check('actual_phase_sources',np.array_equal(t['control_source'],sources) and np.array_equal(d['control_source'],np.r_[np.zeros(200),np.full(960,2),np.full(1920,3)]))
 check('applied_action_hold',np.array_equal(t['action'],np.repeat(d['actually_applied_action'],5,axis=0)) and np.array_equal(t['action'],t['actual_requested_action']))
 check('ANN_direct_policy_output',np.array_equal(d['actually_applied_action'][1160:],d['policy_action'][1160:]))
 check('settle_zero_motor_hidden',np.max(abs(t['action'][:1000]))==0 and np.max(abs(d['hidden_after'][:200]))==0)
 check('warm_brain_state_and_DN_zero',np.max(abs(d['brain_hidden_after'][:1160]))==0 and np.max(abs(d['DN_encoded'][:1160]))==0 and np.max(abs(d['command'][:1160]))==0)
 check('body_obs_previous_action',np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]) and np.max(abs(d['X238'][0,195:237]))==0)
 check('constant_local_targetspeed_not_brain_input',np.array_equal(d['X238'][:,237],np.full(3080,.075)) and bridge.identity()['allowed_raw_indices']==list(range(0,42))+list(range(84,126))+list(range(168,195)))
 check('experimental_instruction_schedule',np.array_equal(d['cue'],np.asarray([cue(a.condition,int(ms)) for ms in d['physical_step_ms']])))
 check('no_external_forces',all(np.max(abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')) and r['extra_forces'] is False)
 ms=motor.initial_state(1);loop=BrainNeckLoop(bridge,a.condition);collected={};expectedmeans=[];expectedpolicy=[];expectedmh=[]
 for k,x in enumerate(d['X238']):
  record=loop.decision(x,k*5)
  for key,value in record.items():collected.setdefault(key,[]).append(value)
  with torch.no_grad():
   if k<200:mean=np.zeros(42);policy=np.zeros(42)
   else:
    mean_t,ms=motor.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(record['command'][None],dtype=torch.float64),ms);mean=mean_t[0].numpy().copy();policy=mean_t[0].tanh().numpy().copy()
  expectedmeans.append(mean);expectedpolicy.append(policy);expectedmh.append(torch.cat(ms,-1)[0].numpy().copy())
  for tick in range(5):loop.advance()
 for key,values in collected.items():check('actual_source_neural_replay_exact:'+key,np.array_equal(np.asarray(values),d[key]))
 for name,value,key in [('motor_mean',expectedmeans,'mean'),('motor_state',expectedmh,'hidden_after'),('motor_action',expectedpolicy,'policy_action')]:check('actual_body_decoder_motor_replay_exact:'+name,np.array_equal(np.asarray(value),d[key]))
 replay=loop.finalize();check('all_source_queue_and_consumption_exact',jsonable(replay)==queue and jsonable(replay['queue_final'])==r['neural_queue_final'])
 check('first_ANN_actual_warm_AN_not_reset',d['AN_capture_step'][1160]==5795 and d['AN_release_step'][1160]==5797 and np.array_equal(d['AN_held'][1160],d['AN_encoded'][1159]) and np.max(abs(d['DN_held'][1160]))==0)
 if a.condition in ('P_left_DN_cut','P_left_AN_cut'):
  key='DN' if a.condition=='P_left_DN_cut' else 'AN';i=1480;check('cut_atANN1point6_no_state_reset',np.array_equal(d[key+'_cut'],np.arange(3080)>=i) and np.max(abs(d[key+'_held'][i:]))==0 and all(e['step']==7400 for e in queue['queue_journal'] if e['event']=='cut'));check('cut_not_direct_turn_gate',np.array_equal(d['raw_turn'][i:],d['shadow_decoder_turn'][i:]))
 else:check('normal_no_cuts',not np.any(d['AN_cut']) and not np.any(d['DN_cut']))
 hand=read(out/'HANDOVER.json');check('continuous_handover',hand==r['transition'] and hand['all_exact'] and hand['no_reset'] and hand['queue_retained'] and hand['warmup_real_AN_retained'] and hand['before_sha256']==hand['after_sha256'] and hand['queue_before_sha256']==hand['queue_after_sha256'])
 boundaries={'qpos':t['qpos'][5799],'qvel':t['qvel'][5799],'ctrl':t['actualctrl'][5799],'target':t['target'][5799],'requested_rate':t['actual_requested_action'][5799]*2.,'previous_actual_action':d['actually_applied_action'][1159],'time':np.asarray(t['time'][5799]),'hidden':d['full_hidden_before'][1160:1161]}
 for key,value in boundaries.items():check('actual_handover_boundary:'+key,hashlib.sha256(value.tobytes()).hexdigest()==hand['before_sha256'][key])
 rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,15400)];auto=sw.summarise(rec,768000,initial_heading=np.asarray(r['autonomous']['initial_heading_diagnostics']['heading_world_xy']));auto['window']=dict(first_sample_s=float(t['time'][5800]),last_sample_s=float(t['time'][-1]),sample_count=9600);check('original_metrics_exact',jsonable(auto)==r['autonomous'])
 observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
 for i in range(5800,15400):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
 cycles=sw.cycle_report(observer,events);check('cycles_recomputed_exact',jsonable(cycles)==r['autonomous_cycles']);check('body_stable_original_gates_separate',stable(auto,cycles,r['native'])==r['functional_stable_body'] and r['engineering_pass']==auto['all_gates'])
 yaw=np.unwrap(g['thorax_yaw']);check('yaw_exact',float(yaw[-1]-yaw[5799])==r['yaw_change_rad']);check('native_full_duration',r['native']['actual_native_steps']==r['physical_steps']==1232000 and r['native']['post_native_steps']==768000 and queue['queue_final']['actual_steps']==15400)
 original=read(NEUTRAL/'IDENTITY.json');check('unchanged_physics_compile',ident['compiled_arrays']==original['compiled_arrays'] and ident['physics']==original['physics'] and ident['actuator']==original['actuator']);check('71opaque',np.sum(compiled['geom_type']==7)==71 and np.all(compiled['geom_rgba'][compiled['geom_type']==7,3]==1))
 oldt=arrays(NEUTRAL/'trace_1ms.npz');oldd=arrays(NEUTRAL/'neural_decisions.npz')
 for key in t:
  if key in oldt:check('unchanged_physical_warmup:'+key,np.array_equal(t[key][:5800],oldt[key][:5800]))
 for key in ('X238','mean','policy_action','actually_applied_action','hidden_before','hidden_after'):check('unchanged_motor_warmup:'+key,np.array_equal(d[key][200:1160],oldd[key][:960]))
 check('no_teacher_after_handover',r['teacher_calls']==960 and r['teacher_calls_after_handover']==0);identity.verify(ident);verify();save(out/'VERIFY.json',dict(integrity_all_pass=True,checks=checks,condition=a.condition,run=str(out),result_sha256=sha(out/'RESULT.json'),brain_checkpoint_sha256=BRIDGE_SHA,checkpoint_sha256=MOTOR_SHA,neck_source_binding_sha256=BINDING_SHA,source_hashes=r['source_hashes'],saved_source_queue_bridge_motor_replay=True,Core_constructions=0,physics_steps=0,engineering_pass=r['engineering_pass'],functional_stable_body=r['functional_stable_body'],new_whole_neck_function_verified=False))
 print(json.dumps(dict(condition=a.condition,integrity_all_pass=True,checks=len(checks))),flush=True)
if __name__=='__main__':main()
