"""Parent-only actual0Core neural/source queue ABI preflight before physical pilot."""
import argparse
from runtime import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');a=p.parse_args()
 if not a.parent_resource_released:print('CODE_ONLY no numerical execution');return
 cpu1();verify();sentinel=object();prior=sys.modules.get('model',sentinel);bridge,motor,bp,mp=load_policies()
 import numpy as np
 import torch
 checks={}
 def check(name,v):
  checks[name]=bool(v)
  if not checks[name]:raise ValueError('Bodypilot0Core preflight '+name)
 check('model_alias_restored',sys.modules.get('model',sentinel) is prior);check('no_physical_backend',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')))
 check('actual_final200_bridge',bp['step']==200 and bp['format_version']=='ENGINEERING_SOURCE_NECK_OFFLINE_CALIBRATION_V1');check('all_weights_frozen',all(not p.requires_grad for m in (bridge,motor) for p in m.parameters()))
 check('brain16_motor256_full272',flat(motor.initial_state(1),torch.zeros((1,16),dtype=torch.float64)).shape==(1,272));check('AN_mask111_no237',bridge.allowed_raw_indices.tolist()==list(range(0,42))+list(range(84,126))+list(range(168,195)))
 with np.load(NEUTRAL/'neural_decisions.npz',allow_pickle=False) as z:x=z['X238'][0].copy()
 loop=BrainNeckLoop(bridge,'P_empty');first=loop.decision(x,0)
 for _ in range(5):loop.advance()
 second=loop.decision(x,5)
 check('actual_source_N2M1_order',loop.transport.ports=={'DN':['720575941433155799','720575941566493282'],'AN':['720575941351919408']})
 check('actual_AN_encode_warmup_publishes2ms_consumes5ms',second['AN_capture_step']==0 and second['AN_release_step']==2 and np.array_equal(second['AN_held'],first['AN_encoded']))
 check('warmup_brain_DN_heldzero_commandzero',np.max(abs(second['brain_hidden_after']))==np.max(abs(second['DN_encoded']))==np.max(abs(second['command']))==0)
 with torch.no_grad():
  expected=float(bridge.decoder(torch.zeros((1,2),dtype=torch.float64))[0,0]);dn,h=bridge.brain_step(torch.tensor(second['AN_held'][None],dtype=torch.float64),torch.ones((1,1),dtype=torch.float64),torch.zeros((1,16),dtype=torch.float64));decoded=bridge.decoder(dn);cmd=project(decoded[:,0]);means,states=motor.step(torch.tensor(x[None],dtype=torch.float64),cmd,motor.initial_state(1))
 check('shadow_decoder_zero_not_forced_zero',second['shadow_decoder_turn']==expected);check('actual_neural_motor_shapes',dn.shape==(1,2) and h.shape==(1,16) and cmd.shape==(1,2) and means.shape==(1,42) and torch.cat(states,-1).shape==(1,256))
 check('explicit_frozen_motor_command_ABI',torch.equal(cmd[:,0:1],torch.relu(cmd[:,1:2])/3.));check('experimental_cue_only_brain_schedule',cue('P_left_pulse',6400)==1 and cue('P_right_pulse',6400)==-1 and cue('P_left_cuezero',6400)==0 and cue('P_left_pulse',9400)==0)
 check('actual_transport_sources_unchanged',loop.transport.verify_final()['source_files_unchanged']);check('preflight_no_Core_or_physics',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')))
 verify();save(ROOT/'PREFLIGHT.json',dict(all_passed=True,checks=checks,Core_constructions=0,physics_steps=0,model_loads=2,actual_brain_parameters_frozen=True,neural_records_declared=3080,state_total=272,first_warm_AN_capture_step=0,first_warm_AN_release_step=2,first_warm_AN_consumption_step=5,zeroDN_learned_decoder_turn=expected,physical_handover_history_new_validation=True,sourcefreeze_sha256=sha(ROOT/'FROZEN.json')));print(json.dumps(dict(all_passed=True,checks=len(checks),Core_constructions=0,physics_steps=0,zeroDN_decoder_turn=expected)))
if __name__=='__main__':main()
