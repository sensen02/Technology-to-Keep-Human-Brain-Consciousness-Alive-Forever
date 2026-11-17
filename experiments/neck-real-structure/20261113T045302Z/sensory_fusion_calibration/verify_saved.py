"""Fresh-process saved0/200/400 actualsourcequeue replay and frozen lineages;0Core."""
from common import *
def main(step):
 torch,np=backend();path=HERE/'TRAIN'/('checkpoint_'+str(step).zfill(6)+'.pt');m,p=load_checkpoint(path,sha(path),final=False)
 from data import arrays,tensors
 from model import python_rollout,make_scan
 checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Savedsensoryfusion '+k)
 def close(k,l,r):
  e=float((l-r).abs().max().detach());errors[k]=e;check(k,e<=1e-10)
 d=arrays('train');x,u,y,w=tensors(d);j=p['replay_index'];check('declared_actualpaired_replay_index',j==20);fixed=frozen_tensors(m)
 with torch.no_grad():
  actual=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);batch=make_scan(m)(x,u)
  for key,l,r in zip(('turn','DN','AN','recon','brainhidden'),actual,p['replay']):close('saved_batch1_'+key,l,r)
  close('saved_effective27_turn',batch[0],p['train_prediction']);bp,b,digest=binding();queue,proof=real_queue_rollout(m,x[:,j:j+1],u[:,j:j+1],bp,digest)
  for key,l,r in zip(('turn','DN','AN','recon','brainhidden'),actual,queue):close('sourcequeue9600_'+key,l,r)
 check('full9600sourceclock',proof['actual_steps']==9600);check('3838separate_release_consume_events',len(proof['neural_consumption_events'])==3838 and all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in proof['neural_consumption_events']))
 torch.manual_seed(20261113);initial,parent,old=initialize();check('RNG_exact_initialize_path',torch.equal(torch.get_rng_state(),p['torch_rng_state']));check('ten_trainable_names',p['trainable_names']==[k for k,v in trainable(initial)] and len(p['trainable_names'])==10)
 if step==0:check('initial_fullstate_exact_columncopy',all(torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items()));check('initialzero_odor_column',bool((p['state_dict']['brain.weight_ih'][:,2]==0).all()))
 else:check('actual_parameter_updates',any(not torch.equal(v,p['state_dict'][k]) for k,v in trainable(initial)))
 for k,v in frozen_tensors(initial).items():check('frozen_parent_'+k,torch.equal(v,p['state_dict'][k]) and hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()==p['frozen_tensor_hashes'][k])
 check('identity336_not304',p['identity']['total_recurrent_state_with_motor']==336 and len(p['identity']['brain_input'])==3);check('effective27_micro9_accum3',p['effective_batch']==27 and p['microbatch']==9 and p['accumulation']==3);check('no_runtime_fusionrule',p['runtime_fusion_rule'] is False);check_frozen(m,fixed);f=frozen();save(HERE/'TRAIN'/('VERIFY_'+str(step).zfill(6)+'.json'),dict(all_passed=True,checks=checks,numerical_errors=errors,step=step,sha256=sha(path),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,source_binding_sha256=digest,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],actual_sourcequeue1ms_replay=True,production_batch1_and27=True,frozen_parent_tensors_exact=True,additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(step=step,all_passed=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
