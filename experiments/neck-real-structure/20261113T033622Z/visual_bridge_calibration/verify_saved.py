"""Fresh final0/150/300 checkpoint production and actualneckqueue verification, zero Core."""
from common import *
def main(step):
 torch,np=backend();path=HERE/'TRAIN'/('checkpoint_'+str(step).zfill(6)+'.pt');m,p=load_checkpoint(path,sha(path),final=False)
 from data import arrays,tensors
 from model import python_rollout,make_scan
 checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Savedvisualbridge '+k)
 def close(k,l,r):
  e=float((l-r).abs().max().detach());errors[k]=e;check(k,e<=1e-10)
 d=arrays('train');x,u,y,w=tensors(d);j=p['replay_index'];check('actual_RGB_left_replay_index',j==2);fixed=frozen_tensors(m)
 with torch.no_grad():
  actual=python_rollout(m,x[:,j:j+1],u[:,j:j+1]);scan=make_scan(m)(x,u)
  for key,a,b in zip(('turn','DN','AN','reconstruction','brainhidden'),actual,p['replay']):close('saved_batch1_'+key,a,b)
  close('saved_batch9_turn',scan[0],p['train_prediction']);bp,b,digest=binding();queue,proof=real_queue_rollout(m,x[:,j:j+1],u[:,j:j+1],bp,digest)
  for key,a,b in zip(('turn','DN','AN','reconstruction','brainhidden'),actual,queue):close('actual_sourcequeue_'+key,a,b)
 check('all1920_actualsourcequeue9600',proof['actual_steps']==9600);check('3838no_future_neuralconsumers',len(proof['neural_consumption_events'])==3838 and all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in proof['neural_consumption_events']))
 torch.manual_seed(20261113);initial,parent=initialize();check('RNG_same_initialize_doubleconstruction',torch.equal(torch.get_rng_state(),p['torch_rng_state']));check('trainable_names_exact',p['trainable_names']==[k for k,v in trainable(initial)] and len(p['trainable_names'])==10)
 if step==0:check('initial4153allstate_exact',all(torch.equal(v,p['state_dict'][k]) for k,v in initial.state_dict().items()))
 else:check('actual_trainable_weights_changed',any(not torch.equal(v,p['state_dict'][k]) for k,v in trainable(initial)))
 for k,v in frozen_tensors(initial).items():check('frozen_parent_tensor_'+k,torch.equal(v,p['state_dict'][k]) and hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()==p['frozen_tensor_hashes'][k])
 check_frozen(m,fixed);check('same304_identity',p['identity']['total_recurrent_state_with_motor']==304 and p['identity']['brain_input']==['deliveredAN1','frozen_actual_RGB_visual_rawturn1']);check('strictsavedstepcurve',p['step']==step and len(p['curve'])==step+1);check('no_body_backend',not any(k in sys.modules for k in ('physics','mujoco','student_wave','flygym')))
 f=frozen();save(HERE/'TRAIN'/('VERIFY_'+str(step).zfill(6)+'.json'),dict(all_passed=True,checks=checks,numerical_errors=errors,step=step,sha256=sha(path),parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,source_binding_sha256=digest,source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],actual_sourcequeue1ms_replay=True,production_batch1_and9=True,all_frozen_tensors_exact=True,trainable_parameter_tensors=10,additional_training_updates=0,new_physics_steps=0));print(json.dumps(dict(step=step,all_passed=True,checks=len(checks),max_error=max(errors.values()))))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
