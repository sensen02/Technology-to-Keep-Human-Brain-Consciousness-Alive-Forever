"""Independent once-loaded endpoint/model replay; no Core construction or mj_step."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from contract import HERE,command_at,pulse,yaw_score,digest,verify_qualification
from evaluate_holdout import BASE,model_loader,jsonable,validate_source

def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(run):
 import torch
 torch.set_num_threads(1);sys.path.insert(0,str(BASE));import student_wave as sw;import identity
 run=Path(run);r=json.loads((run/'RESULT.json').read_text());ident=json.loads((run/'IDENTITY.json').read_text());identity.verify(ident);validate_source(r['source_hashes']);verify_qualification(r['qualification'],r['checkpoint'],r['checkpoint_sha256'])
 checks={}
 def check(name,value):
  checks[name]=bool(value)
  if not value:raise ValueError('failed: '+name)
 t=arrays(run/'trace_1ms.npz');d=arrays(run/'neural_decisions.npz');g=arrays(run/'runtime_geometry.npz');case=r['profile'];shift=20 if case.startswith('pulse') else 0;n=15400+shift
 check('trace_digest',digest(run/'trace_1ms.npz')==r['trace']['sha256']);check('neural_digest',digest(run/'neural_decisions.npz')==r['neural_decisions']['sha256'])
 check('identity_binding',r['identity_sha256']==ident['identity_sha256'] and r['checkpoint_sha256']==ident['config']['cli']['sha256'])
 check('complete_shapes',t['qpos'].shape==(n,49) and t['qvel'].shape==(n,48) and d['X238'].shape==(2880,238) and d['hidden_after'].shape==(2880,192))
 check('finite_arrays',all(np.isfinite(v).all() for v in t.values()) and all(np.isfinite(v).all() for v in d.values()))
 check('clock_complete',np.max(np.abs(t['time']-np.arange(1,n+1)*.001))<1e-8 and np.max(np.abs(d['time']-(1.+shift*.001+np.arange(2880)*.005)))<1e-8)
 sources=np.r_[np.ones(shift,dtype=int),np.zeros(1000,dtype=int),np.full(4800,2),np.full(9600,3)]
 check('phase_counts',np.array_equal(t['control_source'],sources) and np.array_equal(d['control_source'],np.r_[np.full(960,2),np.full(1920,3)]))
 action,seed=pulse(case)
 check('pulse_actual',np.array_equal(t['action'][:shift],np.tile(action,(shift,1))) and r['pulse']['seed']==seed)
 check('zero_settle',np.array_equal(t['action'][shift:shift+1000],np.zeros((1000,42))))
 check('full_command_profile',np.array_equal(d['command'],np.array([command_at(case,k) for k in range(2880)])))
 check('hidden_chain',np.array_equal(d['hidden_before'][0],np.zeros(192)) and np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1]))
 check('action_hold',np.array_equal(t['action'][shift+1000:],np.repeat(d['actually_applied_action'],5,axis=0)))
 check('direct_actor_afterwithdrawal',np.array_equal(d['policy_action'][960:],d['actually_applied_action'][960:]))
 check('previous_action_inputs',np.array_equal(d['X238'][0,195:237],np.zeros(42)) and np.array_equal(d['X238'][1:,195:237],d['actually_applied_action'][:-1]))
 check('zero_external',all(np.max(np.abs(t[key]))==0 for key in ('external_body_force_max','external_generalized_force_max')))
 check('bounded_actions',np.max(np.abs(t['action']))<=1 and np.array_equal(t['action'],t['actual_requested_action']))
 check('teacher_only_prefix',r['teacher_calls']==960 and r['teacher_calls_after_handover']==0)
 io=model_loader();m,p=io.load_checked(r['checkpoint'],r['checkpoint_sha256']);state=m.initial_state(1);means=[];states=[];acts=[]
 with torch.no_grad():
  for x,u in zip(d['X238'],d['command']):
   mean,state=m.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state);means.append(mean[0].numpy());states.append(torch.cat(state,-1)[0].numpy());acts.append(mean[0].tanh().numpy())
 check('actual2880_checkpoint_means',np.array_equal(np.asarray(means),d['mean']));check('actual2880_dual_memory',np.array_equal(np.asarray(states),d['hidden_after']));check('actual2880_actions',np.array_equal(np.asarray(acts),d['policy_action']))
 start=5800+shift;rec=[(float(t['time'][i]),t['com'][i],t['loads'][i],float(t['nonfoot'][i]),t['speed'][i]) for i in range(start,n)]
 auto=sw.summarise(rec,768000,initial_heading=np.asarray(r['autonomous']['initial_heading_diagnostics']['heading_world_xy']));auto['window']=dict(first_sample_s=float(t['time'][start]),last_sample_s=float(t['time'][-1]),sample_count=9600)
 check('full9600_summary',jsonable(auto)==r['autonomous'])
 observer=sw.CycleTracker(sw.CycleConfig(timeout=4.)).reset(start_time=float(t['time'][start-1]));events=[]
 for i in range(start,n):events.extend(observer.update(dict(time=float(t['time'][i]),loads=t['loads'][i],padz=t['padz'][i],relativefoot=t['relativefoot'][i]))['events'])
 cycles=sw.cycle_report(observer,events);check('full_cycle_events',jsonable(cycles)==r['autonomous_cycles'])
 yaw=yaw_score(case,d['yaw_after_each_1ms']);check('actual_yaw_criteria',yaw==r['yaw'])
 criteria=dict(original_five=auto['all_gates'],all_six=cycles['all_legs_completed'],strict_nonfoot_zero=auto['nonfoot_force_max_native']==0.,yaw=yaw['passed']);check('pass_recomputed',criteria==r['criteria'] and r['engineering_pass']==bool(all(criteria.values())))
 b=json.loads((run/'BOUNDARIES.json').read_text());check('handover_switch_snapshots',b==r['boundaries'] and set(b)=={'960','1920'} and all(x['all_exact'] and x['before_sha256']==x['after_sha256'] for x in b.values()))
 for key in ('geom_type','geom_rgba'):
  import hashlib
  check('compiled_'+key,hashlib.sha256(g[key].tobytes()).hexdigest()==ident['compiled_arrays'][key]['sha256'])
 mesh=g['geom_type']==7;check('all71_opaque',mesh.sum()==71 and np.all(g['geom_rgba'][mesh,3]==1.))
 check('actual_steps',r['physical_steps']==n*80)
 import hashlib
 for decision in (960,1920):
  i=shift+1000+decision*5-1
  derived=dict(qpos=t['qpos'][i],qvel=t['qvel'][i],ctrl=t['actualctrl'][i],target=t['target'][i],requested_rate=t['actual_requested_action'][i]*2.,previous_actual_action=d['actually_applied_action'][decision-1],time=np.asarray(t['time'][i]),hidden=d['hidden_before'][decision:decision+1])
  for key,value in derived.items():check('derived_boundary_'+str(decision)+'_'+key,hashlib.sha256(value.tobytes()).hexdigest()==b[str(decision)]['before_sha256'][key])
 if shift==0:
  q=json.loads(Path(r['qualification']).read_text());reference=Path(q['conditions']['neutral']['result']).parent
  oldt=arrays(reference/'trace_1ms.npz');oldd=arrays(reference/'neural_decisions.npz')
  for key in t:check('matched_prefix_'+key,np.array_equal(t[key][:5800],oldt[key][:5800]))
  for key in ('X238','hidden_before','hidden_after','actually_applied_action','control_source'):check('matched_neural_prime_'+key,np.array_equal(d[key][:960],oldd[key][:960]))
 identity.verify(ident);validate_source(r['source_hashes'])
 return dict(integrity_all_pass=True,checks=checks,run=str(run.resolve()),profile=case,checkpoint_sha256=r['checkpoint_sha256'],engineering_pass=r['engineering_pass'],physical_integration_steps=0,scope='saved actual checkpoint/endpoint replay, not full native certification or food success')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();report=verify(a.run)
 with Path(a.out).open('x') as f:json.dump(jsonable(report),f,indent=2);f.write('\n')
 print(json.dumps(dict(integrity_all_pass=True,checks=len(report['checks']))))
