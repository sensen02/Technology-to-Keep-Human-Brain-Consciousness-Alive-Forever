"""Parent-launched real loss/gradient/Adam preflight, no physical backend."""
from copy import deepcopy
from common import *
from data import training_data
from engine import make,production,objective

def main():
 cpu1();guard()
 if not (HERE/'DATASET.json').exists():
  from generate_data import main as generate
  generate()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('preflightalreadyfrozen; donotrestart')
 torch.manual_seed(20261111);np.random.seed(20261111);m,parent=initialize();a=deepcopy(m);b=deepcopy(m);data=training_data();idx=[0,8,16,20];x=data['receptors'][:,idx];y=data['targets'][:,idx];w=data['weights'][:,idx];checks={};errors={}
 def close(name,a,b):
  err=float((a-b).abs().max());errors[name]=err;checks[name]=err<=1e-12
 p,l,h=production(a,x);q,z,k=make(b)(x);close('production_scan_turn',p,q);close('production_scan_latent',l,z);close('production_scan_hidden',h,k)
 loss1=objective(p,l,y,w);loss2=objective(q,z,y,w);close('actual_loss',loss1,loss2)
 manual=(((p-y).square()+.01*(l-torch.atanh(y.clamp(-.95,.95))).square())*w).sum(0).mean();close('loss_manual',loss1,manual)
 loss1.backward();loss2.backward()
 for (name,pa),(other,pb) in zip(a.named_parameters(),b.named_parameters()):
  assert name==other;close('gradient_'+name,pa.grad,pb.grad)
 checks['all8_finite_nonzero_grad']=len(list(a.parameters()))==8 and all(torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in a.parameters())
 oa=torch.optim.Adam(a.parameters(),lr=1e-4);ob=torch.optim.Adam(b.parameters(),lr=1e-4);torch.nn.utils.clip_grad_norm_(a.parameters(),1.);torch.nn.utils.clip_grad_norm_(b.parameters(),1.);oa.step();ob.step()
 for name,v in a.state_dict().items():close('unit_model_'+name,v,b.state_dict()[name])
 for i,(sa,sb) in enumerate(zip(oa.state.values(),ob.state.values())):
  for key in sa:close('unit_Adam_'+str(i)+'_'+key,sa[key],sb[key])
 checks['fresh8Adam_step1']=len(oa.state)==8 and all(float(v['step'])==1 for v in oa.state.values())
 checks['parent_all_weights_exact']=exact(m.state_dict(),parent['state_dict'])
 checks['no_runtime_gating_or_reset']=m.identity()==parent['identity'] and m.identity()['inputs']=='actual local receptor[left,right] only'
 checks['data_weights_per_episode']=float((data['weights'].sum(0)-1).abs().max())<1e-12
 checks['fixed24train12validation']=read(HERE/'DATASET.json')['splits']['train']['episodes']==24 and read(HERE/'DATASET.json')['splits']['validation']['episodes']==12
 checks['motor_raw_SHA_unchanged_no_model_load']=sha(MOTOR)==MOTOR_SHA
 checks['no_Core_or_physical_imports']=not any(k in sys.modules for k in ('physics','student_wave','mujoco','flygym'))
 paths=source_paths()+[HERE/'DATASET.json',HERE/'train.npz',HERE/'validation.npz',MOTOR]
 hashes={str(p):sha(p) for p in paths}
 assert all(checks.values()),checks
 save(HERE/'FROZEN.json',dict(kind=FORMAT,source_hashes=hashes,original_parent_source_hashes=parent['config']['source_hashes'],original_parent_data_hashes=parent['config']['data_hashes'],before_training=True))
 save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,max_errors=errors,tolerance=1e-12,full1920steps=True,batch_size=4,temporary_unit_updates=1,actual_optimizer_updates=0,Core_constructions=0,motor_model_loads=0,sourcehash_sha256=sha(HERE/'FROZEN.json')))
 print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),Core_constructions=0,motor_model_loads=0)))
if __name__=='__main__':main()
