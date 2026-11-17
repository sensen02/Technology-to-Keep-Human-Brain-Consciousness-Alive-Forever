"""One zero-Core actual component/input-boundary preflight before all physical trials."""
import ast,time
from runtime import *
def main():
 cpu1();started=time.monotonic();motor,behavior,mp,bp=load_policies();checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('preflight failed '+k)
 check('motor_actual_SHA',sha(MOTOR)==MOTOR_SHA);check('sensor_actual_SHA',sha(SENSOR)==SENSOR_SHA)
 check('all_weights_frozen',all(not p.requires_grad for m in (motor,behavior) for p in m.parameters()))
 check('sensor_scope_NOT_deployed',bp['deployment_qualified'] is False and bp['scope_tag']=='SENSOR_PRETRAIN_ONLY')
 # Both sampling implementations must describe the same local sites exactly.
 geometry=read(GEOMETRY);offset=geometry['sensor_offsets'];local=np.asarray(offset['local_xyz']);aabb=geometry['head']['local_mesh_aabb'];f=aabb['max'][0]+.1*aabb['extent'][0];l=max(abs(aabb['min'][1]),abs(aabb['max'][1]))+.1*aabb['extent'][1]
 check('head_sampling_offsets_exact',np.array_equal(local,np.array([[f,l,0.],[f,-l,0.]])) and f==offset['forward_offset'] and l==offset['lateral_offset'])
 pos=np.asarray(geometry['head']['xpos']);rot=np.asarray(geometry['head']['xmat']);sensors=(pos+local@rot.T)[:,:2];mid=sensors.mean(0);forward=rot[:2,0].copy();forward/=np.linalg.norm(forward);left=np.array([-forward[1],forward[0]]);center=mid+.6*forward+.6*np.tan(.12)*left;C=1.25*np.exp(-.5*np.sum(((sensors-center)/.5)**2,axis=1))
 check('clamp_left_stronger',C[0]>C[1]>0)
 save_new(ROOT/'CLAMP_LEVELS.json',dict(kind='STATIC_AUDITED_GEOMETRY_CONCENTRATION_CLAMP',left_pair=C.tolist(),right_pair=C[::-1].tolist(),equal_pair=np.full(2,C.mean()).tolist(),derivation=dict(angle_rad=.12,forward_offset=.6,sigma=.5,amplitude=1.25,source_center_xy=center.tolist(),sensor_xy=sensors.tolist(),geometry=str(GEOMETRY),geometry_sha256=sha(GEOMETRY)),noise_std=0.,K=1.,tau_s=.05,dt_s=.005,note='Engineering clamp levels only. Actual moving head geometry recorded eachms but does not alter clamped concentrations. No spatial navigation claim.'))
 raw=torch.tensor([[0.,-.75],[0.,0.],[0.,.75]],dtype=torch.float64);u=project(raw)
 check('neural_relu_projection',torch.equal(u,torch.tensor([[0.,-.75],[0.,0.],[.25,.75]],dtype=torch.float64)))
 z=torch.zeros((1,2),dtype=torch.float64);ms=motor.initial_state(1);bs=behavior.initial_state(1);check('full288_initial',fullflat(ms,bs).shape==(1,288) and torch.count_nonzero(fullflat(ms,bs))==0)
 # Test new9.6s offline actor determinism with continuous pulse/withdrawal memory, without physics.
 state1=behavior.initial_state(1);state2=behavior.initial_state(1);sensor1=ClampFilter();sensor2=ClampFilter();delay=[];turn=[]
 with torch.no_grad():
  for k in range(1920):
   x=k*.005;ss=sensor1.advance(concentration('B_left_pulse',x));tt=sensor2.advance(concentration('B_left_pulse',x))
   cmd1,state1=behavior.step(torch.tensor(ss['receptors'][None],dtype=torch.float64),state1);cmd2,state2=behavior.step(torch.tensor(tt['receptors'][None],dtype=torch.float64),state2)
   if not torch.equal(cmd1,cmd2) or not torch.equal(state1,state2):raise ValueError('actual behavior replay mismatch')
   delay.append(ss['receptors']);turn.append(float(cmd1[0,1]))
 check('actual_behavior_repeat_exact_full9_6s',True)
 check('pending_firstzero',np.array_equal(delay[0],np.zeros(2)) and np.array_equal(delay[120],np.zeros(2)))
 check('response_only_after_interval',np.any(np.asarray(delay)[122:]>0) and np.max(np.abs(np.asarray(delay)[:121]))==0)
 # Actor uses body238, no clock source or teacher inputs: compare base reference neutral once.
 reference=arrays(MOTOR_DIR/'EVAL_neutral/neural_decisions.npz');ms=motor.initial_state(1);means=[];hidden=[]
 with torch.no_grad():
  for x in reference['X238']:
   mean,ms=motor.step(torch.tensor(x[None],dtype=torch.float64),z,ms);means.append(mean[0].numpy().copy());hidden.append(flat(ms)[0].numpy().copy())
 check('neutral2880_original_means_exact',np.array_equal(np.asarray(means),reference['mean']))
 check('neutral2880_256_hidden_exact',np.array_equal(np.asarray(hidden),reference['hidden_after']))
 check('probe_start_delay',all(probe_turn(trial,0)==0 and probe_turn(trial,.6)==0 for trial in A_IDS))
 check('three_fixed_probes',probe_turn('A_left_half',9.5)==.5 and probe_turn('A_right_half',9.5)==-.5 and probe_turn('A_switch',9.5)==-.5)
 check('all_B_finite_nonnegative',all(np.isfinite(concentration(trial,x)).all() and (concentration(trial,x)>=0).all() for trial in B_IDS for x in np.arange(1920)*.005))
 rejected=False
 try:behavior.step(torch.zeros((1,3),dtype=torch.float64),behavior.initial_state(1))
 except ValueError:rejected=True
 check('behavior_rejects_privileged_third_input',rejected)
 paths=source_paths(mp,bp)
 for filename in ('runner.py','audit.py','orchestrate.py','runtime.py','preflight.py','summarize_B.py'):ast.parse((ROOT/filename).read_text())
 check('actual_executables_AST',True)
 # Snapshot only static executables and immutable inputs; NEVER mutable journals or visual dir.
 check('dynamic_files_excluded',not any(p.name in ('CURRENT_STATE.json','JOBS.json','REPORT.zh-CN.md') and p.parent==ROOT for p in paths))
 hashes={str(p):sha(p) for p in paths};save_new(ROOT/'FROZEN.json',dict(kind='STIMULUS_AB_288_INPUTS_FROZEN_BEFORE_ANY_PHYSICS',source_hashes=hashes,motor=MOTOR_SHA,behavior=SENSOR_SHA,Core_constructions=0,physics_steps=0))
 save_new(ROOT/'PREFLIGHT.json',dict(all_passed=all(checks.values()),checks=checks,Core_constructions=0,physics_steps=0,optimizer_updates=0,seconds=time.monotonic()-started,clamp_left=C.tolist(),offline_pulse_turn_peak=float(np.max(turn)),all_sources_sha256=sha(ROOT/'FROZEN.json')))
 print(json.dumps(dict(all_passed=True,checks=len(checks),seconds=time.monotonic()-started,clamp=C.tolist(),offline_turn_peak=max(turn)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
