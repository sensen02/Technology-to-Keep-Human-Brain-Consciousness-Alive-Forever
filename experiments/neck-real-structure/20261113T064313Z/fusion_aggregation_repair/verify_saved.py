"""Fresh full1920production and actualsourcequeue/strictparentfrozenstate audit."""
from common import *
def main(step):
 torch,np=backend();path=HERE/'TRAIN'/('checkpoint_'+str(step).zfill(6)+'.pt');m,p=load_checkpoint(path,sha(path),final=False)
 from data import arrays,tensors
 from model import make_scan,python_rollout
 checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Savedsensoryagg '+k)
 def close(k,l,r):
  e=float((l-r).abs().max().detach());errors[k]=e;check(k,e<=1e-10)
 d=arrays('train');x,u,y,w=tensors(d);j=p['replay_index'];check('realFleftpaired_replay',j==9)
 with torch.no_grad():
  actual=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);scan=make_scan(m)(x,u)
  for k,a,b in zip(('turn','DN','AN','reconstruction','brainhidden'),actual,p['replay']):close('savedbatch1_'+k,a,b)
  close('savedbatch24_turn',scan[0],p['train_prediction']);bp,b,digest=binding();queue,proof=real_queue_rollout(m,x[:,j:j+1],u[:,j:j+1],bp,digest)
  for k,a,b in zip(('turn','DN','AN','reconstruction','brainhidden'),queue,actual):close('actual1msqueue_'+k,a,b)
 check('all1920actualqueue9600',proof['actual_steps']==9600);check('all3838causalconsumers',len(proof['neural_consumption_events'])==3838 and all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in proof['neural_consumption_events']))
 torch.manual_seed(20261113);initial,parent=initialize();check('initialize_RNG_exact',torch.equal(torch.get_rng_state(),p['torch_rng_state']));check('ten_trainable_names',p['trainable_names']==[k for k,v in trainable(initial)] and len(p['trainable_names'])==10)
 if step==0:check('allinitialdb25tensors_exact',all(torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items()))
 else:check('trainedparameters_changed',any(not torch.equal(v,p['state_dict'][k]) for k,v in trainable(initial)))
 for k,v in frozen_tensors(initial).items():check('frozenparent_'+k,torch.equal(v,p['state_dict'][k]) and hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()==p['frozen_tensor_hashes'][k])
 check('exact336identity',m.identity()['total_recurrent_state_with_motor']==336 and m.brain.input_size==3);check('no_runtimefusion_formula',p['runtime_fusion_rule'] is False);check('24batchcurve_step',p['effective_batch']==24 and p['step']==step and len(p['curve'])==step+1);f=frozen();save(HERE/'TRAIN'/('VERIFY_'+str(step).zfill(6)+'.json'),dict(all_passed=True,checks=checks,numerical_errors=errors,step=step,sha256=sha(path),parent_sha256=PARENT_SHA,odor_sha256=odor_binding()['sha256'],visual_sha256=VISION_SHA,source_binding_sha256=digest,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],all_frozen_tensors_exact=True,actual_sourcequeue1ms_replay=True,additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(step=step,all_passed=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
