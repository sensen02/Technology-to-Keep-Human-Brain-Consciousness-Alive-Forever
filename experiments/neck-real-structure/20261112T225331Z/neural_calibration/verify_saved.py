"""Fresh parent process actual sourcequeue1ms and same1920 production checkpoint audit."""
from common import *
def main(step):
 torch,np=backend();path=HERE/'TRAIN'/('checkpoint_'+str(step).zfill(6)+'.pt');m,p=load_checkpoint(path,sha(path),final=False);checks={};errors={}
 from data import arrays,tensors
 from model import NeuralBridge,python_rollout,make_scan,objective
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Saved verifier '+k)
 def close(k,l,r):
  error=float((l-r).abs().max().detach());errors[k]=error;check(k,error<=1e-10)
 d=arrays('train');x,cue,y=tensors(d);j=p['replay_index'];check('replay_uses_directional_train_sequence',j==2)
 with torch.no_grad():
  actual=python_rollout(m,x[:,j:j+1],cue[:,j:j+1]);scan=make_scan(m)(x,cue)
  for k,a,b in zip(('turn','DN','AN','recon','brainhidden'),actual,p['replay']):close('saved_batch1_'+k,a,b)
  close('saved_batch4_turn',scan[0],p['train_prediction']);bp,b,digest=binding();queue,proof=real_queue_rollout(m,x[:,j:j+1],cue[:,j:j+1],bp,digest)
  for key,a,b in zip(('turn','DN','AN','recon','brainhidden'),queue,actual):close('sourcequeue_vs_actual_'+key,a,b)
 check('all1920_sourcequeue_replayed',proof['actual_steps']==9600);check('all_actual_consumers_delayed',all(v['queue_release_step']==v['capture_step']+2 and v['neural_consume_step']==v['capture_step']+5 for v in proof['neural_consumption_events']))
 check('step_and_curve',p['step']==step and len(p['curve'])==step+1);check('same_scope_no_fullneck_training',p['scoped_engineering_only'] and p['full_neck_training_allowed'] is False and p['new_walking_verified'] is False)
 torch.manual_seed(20261112);initial=NeuralBridge(torch.from_numpy(d['mean']),torch.from_numpy(d['std']));check('RNG_state_exact',torch.equal(torch.get_rng_state(),p['torch_rng_state']))
 if step==0:check('actual_initial_parameters_seed_exact',all(torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items()))
 else:check('actual_parameter_updates_nonzero',any(not torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items() if k in dict(initial.named_parameters())))
 check('initial_decoder_hold_is_zero_vector',torch.equal(actual[0][0],m.decoder(x.new_zeros((1,2)))))
 check('brain16_actual_state_shape',actual[4].shape==(1920,1,16));check('no_motor_backend',not any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave')))
 f=frozen();save(HERE/'TRAIN'/('VERIFY_'+str(step).zfill(6)+'.json'),dict(all_passed=True,checks=checks,numerical_errors=errors,step=step,sha256=sha(path),identity=m.identity(),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],source_binding_sha256=digest,actual_production_batch1_and4=True,actual_queue1ms_replay=True,neural_consumption_events_count=len(proof['neural_consumption_events']),physics_steps=0,motor_model_loads=0,additional_training_updates=0));print(json.dumps(dict(step=step,all_passed=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-calibration')
