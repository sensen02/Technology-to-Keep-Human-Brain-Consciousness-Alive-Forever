"""Fresh-process saved-data/model replay audit: no Core, optimizer or integration."""
import argparse,ast
from runtime import *
def verify(trial):
 cpu1();verify_frozen();motor,behavior,mp,bp=load_policies();sw,identity=pin()
 out=ROOT/'trials'/trial;r=read(out/'RESULT.json');ident=read(out/'IDENTITY.json');checks={}
 def check(name,v):
  checks[name]=bool(v)
  if not checks[name]:raise ValueError('independent audit failed '+name)
 identity.verify(ident);check('identity_bound',r['identity_sha256']==ident['identity_sha256']);check('bound_source_map',r['source_hashes']==ident['config']['evaluation_source_hashes'])
 check('motor_checkpoint_bound',r['checkpoint']==str(MOTOR) and r['checkpoint_sha256']==MOTOR_SHA and sha(MOTOR)==MOTOR_SHA)
 check('sensor_checkpoint_bound',r['behavior_checkpoint']==str(SENSOR) and r['behavior_checkpoint_sha256']==SENSOR_SHA and sha(SENSOR)==SENSOR_SHA)
 for p,h in r['source_hashes'].items():check('source:'+p,sha(p)==h)
 for k,n in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:check('hash:'+n,sha(out/n)==r[k]['sha256'])
 t=arrays(out/'trace_1ms.npz');d=arrays(out/'neural_decisions.npz');g=arrays(out/'geometry_1ms.npz');compiled=arrays(out/'runtime_geometry.npz')
 check('all_finite',all(np.isfinite(v).all() for group in (t,d,g,compiled) for v in group.values()))
 check('full15400',t['qpos'].shape==(15400,49) and t['qvel'].shape==(15400,48) and g['thorax_yaw'].shape==(15400,))
 check('full2880_motor256_behavior32',d['X238'].shape==(2880,238) and d['hidden_before'].shape==d['hidden_after'].shape==(2880,256) and d['behavior_hidden_before'].shape==d['behavior_hidden_after'].shape==(2880,32) and d['full_hidden_before'].shape==d['full_hidden_after'].shape==(2880,288))
 check('layout288',r['state_total']==288 and r['state_layout']==[128,64,64,32] and r['filter_state_separate'] is True)
 check('full288_hidden_chain',np.array_equal(d['full_hidden_before'][0],np.zeros(288)) and np.array_equal(d['full_hidden_before'][1:],d['full_hidden_after'][:-1]))
 check('motor256_hidden_chain',np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1]))
 check('full_hidden_components_exact',np.array_equal(d['full_hidden_before'],np.concatenate([d['hidden_before'],d['behavior_hidden_before']],axis=1)) and np.array_equal(d['full_hidden_after'],np.concatenate([d['hidden_after'],d['behavior_hidden_after']],axis=1)))
 check('clock1ms',np.max(np.abs(t['time']-np.arange(1,15401)*.001))<1e-8)
 check('clock5ms',np.max(np.abs(d['time']-(1+np.arange(2880)*.005)))<1e-8)
 sources=np.r_[np.zeros(1000),np.full(4800,2),np.full(9600,3)]
 check('sources_full',np.array_equal(t['control_source'],sources) and np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)]))
 check('held_actions',np.array_equal(t['action'][1000:],np.repeat(d['actually_applied_action'],5,axis=0)))
 check('ANN_direct_outputs',np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]) and np.array_equal(t['action'],t['actual_requested_action']))
 check('actions_bounded',np.max(np.abs(t['action']))<=1 and np.array_equal(t['action'][:1000],np.zeros((1000,42))))
 check('noextra_forces',all(np.max(np.abs(t[k]))==0 for k in ('external_body_force_max','external_generalized_force_max')) and r['extra_forces'] is False)
 check('actual_observation_history',np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]) and np.array_equal(d['X238'][0,195:237],np.zeros(42)))
 check('actor238_no_clock',np.array_equal(d['X238'][:,-1],np.full(2880,.075)) and motor.identity()['inputs']==238 and behavior.identity()['inputs']=='actual local receptor[left,right] only')
 ms=motor.initial_state(1);bs=behavior.initial_state(1);filter=ClampFilter();means=[];production_actions=[];mh=[];bh=[];raws=[];commands=[];sensory={k:[] for k in ('receptors','concentration','filter_before','filter_after','pending_before','pending_after')}
 with torch.no_grad():
  for k in range(2880):
   if k<960:raw=torch.zeros((1,2),dtype=torch.float64);s={key:np.zeros(2) for key in sensory}
   elif trial in A_IDS:raw=torch.tensor([[0.,probe_turn(trial,(k-960)*.005)]],dtype=torch.float64);s={key:np.zeros(2) for key in sensory}
   else:s=filter.advance(concentration(trial,(k-960)*.005));raw,bs=behavior.step(torch.tensor(s['receptors'][None],dtype=torch.float64),bs)
   u=project(raw);mean,ms=motor.step(torch.tensor(d['X238'][k:k+1],dtype=torch.float64),u,ms)
   means.append(mean[0].numpy().copy());production_actions.append(mean[0].tanh().numpy().copy());mh.append(flat(ms)[0].numpy().copy());bh.append(bs[0].numpy().copy());raws.append(raw[0].numpy().copy());commands.append(u[0].numpy().copy())
   for key in sensory:sensory[key].append(s[key])
 check('actual2880_motor_outputs_exact',np.array_equal(np.asarray(means),d['mean']))
 check('actual2880_torch_actions_exact',np.array_equal(np.asarray(production_actions),d['policy_action']))
 check('actual2880_motor_memory_exact',np.array_equal(np.asarray(mh),d['hidden_after']))
 check('actual2880_behavior_outputs_exact',np.array_equal(np.asarray(raws),d['raw_command']) and np.array_equal(np.asarray(raws)[:,1],d['raw_turn']) and np.array_equal(np.asarray(bh),d['behavior_hidden_after']))
 check('neural_relu_coupling_exact',np.array_equal(np.asarray(commands),d['command']))
 check('causal_receptor_filter_exact',all(np.array_equal(np.asarray(sensory[key]),d[key]) for key in sensory))
 check('one_decision_pending_delay',np.array_equal(d['receptors'][960],np.zeros(2)) and np.array_equal(d['receptors'][961:],d['pending_after'][960:-1]))
 check('no_behavior_prime_advance',np.max(np.abs(d['behavior_hidden_after'][:960]))==0)
 h=read(out/'HANDOVER.json');check('handover_exact',h==r['transition'] and h['all_exact'] and all(h['exact'].values()) and h['no_reset'] and h['before_sha256']==h['after_sha256'] and h['state_dims']==[128,64,64,32])
 boundary={'qpos':t['qpos'][5799],'qvel':t['qvel'][5799],'ctrl':t['actualctrl'][5799],'target':t['target'][5799],'requested_rate':t['actual_requested_action'][5799]*2,'previous_actual_action':d['actually_applied_action'][959],'time':np.asarray(t['time'][5799]),'hidden':d['full_hidden_before'][960:961]}
 for k,v in boundary.items():check('handover_boundary:'+k,hashlib.sha256(v.tobytes()).hexdigest()==h['before_sha256'][k])
 check('no_teacher_posthandover',r['teacher_calls']==960 and r['teacher_calls_after_handover']==0)
 mesh=compiled['geom_type']==7;check('71_opaque',mesh.sum()==71 and np.all(compiled['geom_rgba'][mesh,3]==1))
 for k in ('geom_type','geom_rgba'):check('runtime_array:'+k,hashlib.sha256(compiled[k].tobytes()).hexdigest()==ident['compiled_arrays'][k]['sha256'])
 local=np.asarray(read(GEOMETRY)['sensor_offsets']['local_xyz']);actual_sensors=np.asarray([pos+local@rot.T for pos,rot in zip(g['head_xyz'],g['head_rotation'])])
 check('actual_sensor_geometry_exact',np.array_equal(actual_sensors,g['sensor_xyz']) and np.array_equal(actual_sensors.mean(1),g['sensor_midpoint_xyz']))
 check('geometry_clock_source',np.array_equal(g['time'],t['time']) and np.array_equal(g['control_source'],sources))
 check('head_rotation_proper',np.max(np.abs(np.linalg.det(g['head_rotation'])-1))<1e-10 and np.max(np.abs(np.einsum('tji,tjk->tik',g['head_rotation'],g['head_rotation'])-np.eye(3)))<1e-10)
 q=t['qpos'][:,3:7];qy=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2));check('yaw_from_actualqpos',np.max(np.abs(np.angle(np.exp(1j*(qy-g['thorax_yaw'])))))<1e-10)
 rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(5800,15400)]
 auto=sw.summarise(rec,768000,initial_heading=np.asarray(r['autonomous']['initial_heading_diagnostics']['heading_world_xy']));auto['window']=dict(first_sample_s=float(t['time'][5800]),last_sample_s=float(t['time'][-1]),sample_count=9600)
 check('whole_metrics_exact',jsonable(auto)==r['autonomous'])
 observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][5799]));events=[]
 for i in range(5800,15400):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
 cycles=sw.cycle_report(observer,events);check('metrics_cycles_recomputed',jsonable(cycles)==r['autonomous_cycles'])
 yaw=np.unwrap(g['thorax_yaw']);delta=float(yaw[-1]-yaw[5799]);check('actual_yaw_exact',delta==r['yaw_change_rad'])
 check('strict_flags_preserved',strict_flags(auto,cycles,r['native'],delta)==r['criteria'] and r['engineering_pass']==auto['all_gates'] and r['strict_endpoint_qualified']==r['criteria']['eligible'] and r['endpoint30_applicable_to_stimulus'] is False and r['old_motor_engineering_qualified'] is False and r['old_motor_qualification_unchanged'] is True)
 if trial in A_IDS:check('A_functional_recomputed',evaluate_A(trial,auto,cycles,r['native'],g)==r['functional'])
 else:check('B_stable_body_recomputed',r['functional']['stable_body']==basic_functional(auto,cycles,r['native']))
 native=r['native'];check('native_complete_and_noextra',native['actual_native_steps']==1232000 and native['post_native_steps']==768000 and r['physical_steps']==1232000 and native['extra_force_max']==0)
 check('native_peak_not_less_than_endpoints',native['post_nonfoot_peak_native']+1e-10>=np.max(t['nonfoot'][5800:]))
 baseline=read(MOTOR_DIR/'EVAL_neutral/IDENTITY.json');check('original_compiled_physics_unchanged',ident['compiled_arrays']==baseline['compiled_arrays'] and ident['physics']==baseline['physics'] and ident['actuator']==baseline['actuator'] and ident['adhesion']['enabled'] is False)
 original=arrays(MOTOR_DIR/'EVAL_neutral/trace_1ms.npz');oldneural=arrays(MOTOR_DIR/'EVAL_neutral/neural_decisions.npz')
 for key in t:
  if key in original:check('unchanged5800_prefix:'+key,np.array_equal(t[key][:5800],original[key][:5800]))
 for key in ('X238','mean','policy_action','actually_applied_action','control_source','hidden_before','hidden_after'):check('unchanged960_motorprime:'+key,np.array_equal(d[key][:960],oldneural[key][:960]))
 # New source and inherited core tick contain no physical pose writes or resets.
 tree=ast.parse((ROOT/'runner.py').read_text());writes=[n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(s in ast.unparse(n).split('=')[0] for s in ('.qpos','.qvel'))];resets=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('reset_from',)]
 check('actual_runner_no_pose_writes_or_reset_from',not writes and not resets)
 identity.verify(ident);verify_frozen()
 result=dict(integrity_all_pass=all(checks.values()),condition=trial,run=str(out),stage=r['stage'],checkpoint_sha256=MOTOR_SHA,behavior_checkpoint_sha256=SENSOR_SHA,result_sha256=sha(out/'RESULT.json'),engineering_pass=r['engineering_pass'],strict_endpoint_qualified=r['strict_endpoint_qualified'],functional_pass=r['functional_pass'],state_total=288,state_layout=[128,64,64,32],checks=checks,physical_integration_steps=0,Core_constructions=0,optimizer_updates=0,scope='Saved array and exact actual neural/filter replay; native summaries bound, not reconstructible full native forces or D1-D8')
 save_new(out/'VERIFY.json',result);print(json.dumps(dict(trial=trial,integrity_all_pass=True,checks=len(checks),functional=r['functional'],yaw_deg=np.degrees(delta),slip=auto['slip_ratio_max'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trial',choices=IDS,required=True);a=p.parse_args();verify(a.trial)
