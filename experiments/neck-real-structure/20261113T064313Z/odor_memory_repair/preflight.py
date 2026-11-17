"""Parent full16 vs two micro8 loss/grad/Adam, exactsourceclass andnoreset preflight."""
from common import *
def main():
 torch,np=backend()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('Inspectpriorpreflight')
 from data import main as prepare,arrays,tensors
 from model import production,make_scan,objective,accumulated_backward
 from copy import deepcopy
 import types
 prepare();d=arrays();z,y,w=tensors(d);torch.manual_seed(20261113);m,parent=initialize();a=deepcopy(m);reference=anchors(m);checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Odormemorypreflight '+k)
 def close(k,l,r):
  error=float((l-r).abs().max().detach());errors[k]=error;check(k,error<=1e-10)
 check('all66f_initialtensors_exact',all(torch.equal(v,parent['state_dict'][k]) for k,v in m.state_dict().items()));check('exact8trainable',len(parameters(m))==8 and all(p.requires_grad for p in parameters(m)));check('same_original_identity_ALPHA',m.identity()==parent['identity'] and m.VERSION=='receptors2_slow_gru32_turnonly_v1' and m.ALPHA==-__import__('math').expm1(-.005/.1));check('actual13replays_3aug_prefixchecks',read(HERE/'PARENT_REPLAY.json')['all_bitexact'] and read(HERE/'PARENT_REPLAY.json')['actual_cases']==13 and all(read(HERE/'PARENT_REPLAY.json')['augmentation_checks'].values()));check('full16x1920_shape',z.shape==(1920,16,2) and y.shape==w.shape==(1920,16,1));check('onlyreceptors2_input',m.encoder.in_features==2 and m.gru.hidden_size==32 and m.head.out_features==1)
 # Exercise real strictinitialize under hostile basename sentinels, then require restore.
 sentinel=object();previous={k:sys.modules.get(k,sentinel) for k in ('common','model','steering_behavior')};fakes={k:types.ModuleType('_odor_preflight_fake_'+k) for k in previous};beforepath=list(sys.path)
 try:
  sys.modules.update(fakes);probe,pp=initialize();check('initialize_nested_aliases_restored',all(sys.modules.get(k) is v for k,v in fakes.items()) and sys.path==beforepath);check('initialize_under_sentinels_same66f',all(torch.equal(v,parent['state_dict'][k]) for k,v in probe.state_dict().items()))
 finally:
  sys.path[:]=beforepath
  for k,v in previous.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 del probe
 for j in range(16):check('weightsum_'+str(j),abs(float(w[:,j].sum())-1)<1e-12)
 # Complete parentproduction graph thenfree before micro8 comparison; labels never input.
 full=production(m,z);scan=make_scan(a)
 with torch.no_grad():
  compare=scan(z)
  for key,l,r in zip(('turn','latent','h32'),full,compare):close('Python_JIT_full16_'+key,l,r)
 loss,terms=objective(full,y,w,m,reference);manual=((full[0]-y).square()*w).sum(0).mean()+.001*sum((p-q).square().sum() for p,q in zip(m.parameters(),reference))/sum(v.numel() for v in reference);close('independent_manual_loss',loss,manual);lossvalue=float(loss.detach());loss.backward();gradients={k:p.grad.detach().clone() for k,p in m.named_parameters()};check('full8gradientsfinite',all(torch.isfinite(v).all() for v in gradients.values()));del full,compare,loss,terms,manual
 micro_loss,micro_terms=accumulated_backward(scan,a,z,y,w,reference);close('full16_micro8x2_loss',z.new_tensor(lossvalue),z.new_tensor(micro_loss));norms={}
 for k,p in a.named_parameters():check('microfinite_'+k,p.grad is not None and torch.isfinite(p.grad).all());close('full16_micro8_gradient_'+k,gradients[k],p.grad);norms[k]=float(p.grad.norm())
 torch.nn.utils.clip_grad_norm_(m.parameters(),1.);torch.nn.utils.clip_grad_norm_(a.parameters(),1.);o1=torch.optim.Adam(m.parameters(),lr=1e-4);o2=torch.optim.Adam(a.parameters(),lr=1e-4);o1.step();o2.step()
 for (k,p),(other,q) in zip(m.named_parameters(),a.named_parameters()):
  check('parameterorder_'+k,k==other);close('Adamparam_'+k,p,q)
  for key in ('exp_avg','exp_avg_sq','step'):close('Adamstate_'+key+'_'+k,o1.state[p][key],o2.state[q][key])
 check('oneAdamstep_aftertwoaccumulations',len(o1.state)==len(o2.state)==8 and all(float(v['step'])==1 for o in (o1,o2) for v in o.state.values()));check('no_othermodality_orphysicalmodel_loaded',not any(k in sys.modules for k in ('mujoco','physics','student_wave','flygym')) and not any('visual' in getattr(mod,'__file__','') and getattr(mod,'__file__','').endswith('/model.py') for mod in tuple(sys.modules.values()) if mod is not None and isinstance(getattr(mod,'__file__',None),str)))
 import resource
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.;check('RSSunder1800_no_batchchange',rss<1800.);ds=read(HERE/'DATASET.json');sh={str(p):sha(p) for p in sources()};sh.update(ds['source_hashes']);dh={v['path']:v['sha256'] for v in ds['outputs'].values()};dh[str(HERE/'DATASET.json')]=sha(HERE/'DATASET.json');verify(sh);verify(dh);save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,numerical_errors=errors,maximum_error=max(errors.values()),parameter_gradient_norms=norms,peak_RSS_MiB=rss,effective_batch=16,microbatch=8,accumulation=2,history_steps=1920,source_class_identity=m.identity(),training_updates=0,Core_constructions=0,new_physics_steps=0));sh[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json');save(HERE/'FROZEN.json',dict(source_hashes=sh,data_hashes=dh,parent_sha256=PARENT_SHA,scope='Learned sensorymemory repair only; no actualbodyvalidation',full_neck_training_allowed=False));print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),RSS_MiB=rss,effective16_micro8=True)))
if __name__=='__main__':raise SystemExit('Use parent explicitflag run.py')
