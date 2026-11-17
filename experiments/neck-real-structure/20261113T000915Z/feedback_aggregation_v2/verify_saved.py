"""Fresh checkpoint verifier on fullhistory originalarchitecture andactualsourcequeue."""
from common import *
def main(step):
 torch,np=backend();path=HERE/'TRAIN'/('checkpoint_'+str(step).zfill(6)+'.pt');m,p=load_checkpoint(path,sha(path),final=False)
 from data import arrays,tensors
 from model import python_rollout,make_scan
 checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Savedaggregation '+k)
 def close(k,l,r):
  error=float((l-r).abs().max().detach());errors[k]=error;check(k,error<=1e-10)
 d=arrays('train');x,cue,y,w=tensors(d);j=p['replay_index'];check('actual_onpolicy_left_replay_index',j==5)
 with torch.no_grad():
  actual=python_rollout(m,x[:,j:j+1],cue[:,j:j+1]);scan=make_scan(m)(x,cue)
  for key,a,b in zip(('turn','DN','AN','recon','brainhidden'),actual,p['replay']):close('saved_batch1_'+key,a,b)
  close('saved_batch7_turn',scan[0],p['train_prediction']);bp,b,digest=binding();queue,proof=real_queue_rollout(m,x[:,j:j+1],cue[:,j:j+1],bp,digest)
  for key,a,b in zip(('turn','DN','AN','recon','brainhidden'),queue,actual):close('actual_sourcequeue_'+key,a,b)
 check('full1920_actualqueue9600steps',proof['actual_steps']==9600);check('queue_release_and_consume_causal',len(proof['neural_consumption_events'])==3838 and all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in proof['neural_consumption_events']))
 torch.manual_seed(20261112);initial,parent=initialize();check('actual_rng_exact',torch.equal(torch.get_rng_state(),p['torch_rng_state']))
 if step==0:check('all_initial_parent_tensors_exact',all(torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items()))
 else:check('actual_learned_parameter_changes',any(not torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items() if k in dict(initial.named_parameters())))
 check('same_parent_normalizer_andmask',all(torch.equal(p['state_dict'][k],parent['state_dict'][k]) for k in ('observation_mean','observation_std','allowed_raw_indices')));check('fixedstepcurvebatch7',p['step']==step and len(p['curve'])==step+1 and p['batch_size']==7);check('no_false_wholeneck_or_newwalking',p['full_neck_training_allowed'] is False and p['new_walking_verified'] is False);check('not_physical_backend',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')))
 f=frozen();save(HERE/'TRAIN'/('VERIFY_'+str(step).zfill(6)+'.json'),dict(all_passed=True,checks=checks,numerical_errors=errors,step=step,sha256=sha(path),identity=m.identity(),source_binding_sha256=digest,parent_sha256=PARENT_SHA,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],actual_sourcequeue1ms_replay=True,production_batch1_and7=True,additional_training_updates=0,physics_steps=0));print(json.dumps(dict(step=step,all_passed=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-aggregation')
