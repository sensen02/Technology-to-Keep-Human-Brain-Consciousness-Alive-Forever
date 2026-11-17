"""Parent real B9 full-history RGB/loss/gradient/freeze/queue preflight, zero Core."""
from common import *
def main():
 torch,np=backend()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('No blind preflight repeat')
 from data import main as prepare,arrays,tensors
 from model import python_rollout,make_scan,objective
 from copy import deepcopy
 prepare();d=arrays('train');x,u,y,w=tensors(d);torch.manual_seed(20261113);m,parent=initialize();a=deepcopy(m);ref=anchors(parent);fixed=frozen_tensors(m);checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Visualbridgepreflight '+k)
 def close(k,l,r):
  e=float((l-r).abs().max().detach());errors[k]=e;check(k,e<=1e-10)
 check('all4153initialtensors_exact',all(torch.equal(v,parent['state_dict'][k]) for k,v in m.state_dict().items()));check('onlyten_trainable',len(trainable(m))==10 and all(k.startswith(PREFIXES) for k,v in trainable(m)));check('frozen_eight_parameters',len([p for p in m.parameters() if not p.requires_grad])==8)
 check('B9full1920',x.shape==(1920,9,111) and u.shape==y.shape==w.shape==(1920,9,1));check('causaldelayed_target',torch.equal(y[0],torch.zeros_like(y[0])) and torch.equal(y[1:],u[:-1]));check('continuous_neural_visual_range',bool(torch.isfinite(u).all() and (u.abs()<=1).all()));check('no_aux_rgb_fabricated',all(not z['actual_paired_RGB'] and z.get('no_visual_image_or_hidden_state_created') for z in read(HERE/'DATASET.json')['rows'] if z['condition'] in AUX))
 check('RGB_replay_exact',read(HERE/'VISUAL_REPLAY.json')['all_bitexact']);check('same111mask',m.allowed_raw_indices.tolist()==MASK and len(MASK)==111)
 raw=torch.zeros((2,238),dtype=torch.float64,requires_grad=True);body=raw.index_select(1,m.allowed_raw_indices);an=m.feedback(m.normalize(body));g=torch.autograd.grad(an.sum(),raw)[0];excluded=sorted(set(range(238))-set(MASK));check('forbidden_raw_indices_gradient_zero',bool((g[:,excluded]==0).all()) and 237 in excluded)
 check('neuralbrain_AN1_visual1',m.brain.input_size==2 and m.brain.hidden_size==16);check('decoder_DNonly',m.decoder[0].in_features==2 and m.decoder[2].out_features==1)
 for j in range(9):check('actionmass_'+str(j),abs(float(w[:,j].sum())-1)<1e-12)
 o1=python_rollout(m,x,u);o2=make_scan(a)(x,u)
 for k,l,r in zip(('turn','DN','AN','frozen_reconstruction','brainhidden'),o1,o2):close('production_scan_'+k,l,r)
 l1,t1=objective(o1,y,w,m,ref);l2,t2=objective(o2,y,w,a,ref);close('objective',l1,l2);parameters=dict(a.named_parameters());manual=((o2[0]-y).square()*w).sum(0).mean()+.1*a.decoder(x.new_zeros((9,2))).square().mean()+.001*o2[1].square().mean()+.001*sum((parameters[k]-v).square().sum() for k,v in ref.items())/sum(v.numel() for v in ref.values());close('independent_manual_loss',l2,manual)
 l1.backward();l2.backward();grads={}
 for (name,p),(other,q) in zip(trainable(m),trainable(a)):
  check('gradientfinite_'+name,name==other and p.grad is not None and q.grad is not None and torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all());close('gradient_'+name,p.grad,q.grad);grads[name]=float(p.grad.norm())
 check('all_frozen_gradients_absent',all(p.grad is None for p in m.parameters() if not p.requires_grad));torch.nn.utils.clip_grad_norm_([v for k,v in trainable(m)],1.);torch.nn.utils.clip_grad_norm_([v for k,v in trainable(a)],1.);op1=torch.optim.Adam([v for k,v in trainable(m)],lr=3e-4);op2=torch.optim.Adam([v for k,v in trainable(a)],lr=3e-4);op1.step();op2.step()
 for (k,p),(other,q) in zip(trainable(m),trainable(a)):
  close('Adamparameter_'+k,p,q)
  for key in ('exp_avg','exp_avg_sq','step'):close('Adamstate_'+key+'_'+k,op1.state[p][key],op2.state[q][key])
 check('exact10Adamcounter1',len(op1.state)==10 and all(float(v['step'])==1 for v in op1.state.values()));check_frozen(m,fixed);check('frozen_feedback_recon_normalizer_unchanged',True);del o1,o2,l1,l2,manual,a,op1,op2
 bp,b,digest=binding()
 with torch.no_grad():
  expect=python_rollout(m,x[:,2:3],u[:,2:3]);actual,proof=real_queue_rollout(m,x[:,2:3],u[:,2:3],bp,digest)
  for key,l,r in zip(('turn','DN','AN','reconstruction','brainhidden'),expect,actual):close('actualsourcequeue_'+key,l,r)
 check('actual9600sourcequeue',proof['actual_steps']==9600);check('3838causal_consume_events',len(proof['neural_consumption_events'])==3838 and all(e['queue_release_step']==e['capture_step']+2 and e['neural_consume_step']==e['capture_step']+5 for e in proof['neural_consumption_events']));save(HERE/'PREFLIGHT_QUEUE.json',proof)
 import resource
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.;check('RSS_under1800_no_batchchange',rss<1800.);ds=read(HERE/'DATASET.json');sh={str(p):sha(p) for p in sources()};sh.update(ds['source_hashes']);dh={v['path']:v['sha256'] for v in ds['outputs'].values()};dh[str(HERE/'DATASET.json')]=sha(HERE/'DATASET.json');verify(sh);verify(dh)
 save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,numerical_errors=errors,maximum_error=max(errors.values()),trainable_gradient_norms=grads,peak_RSS_MiB=rss,production_batch=9,history_steps=1920,trainable_parameter_tensors=10,visual_actualRGB_replay=True,source_binding_sha256=digest,parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,training_updates=0,Core_constructions=0,new_physics_steps=0));sh[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json');sh[str(HERE/'PREFLIGHT_QUEUE.json')]=sha(HERE/'PREFLIGHT_QUEUE.json');save(HERE/'FROZEN.json',dict(source_hashes=sh,data_hashes=dh,parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,source_binding_sha256=digest,full_neck_training_allowed=False,scope='ActualRGB neural scalar distillation only, no new body validation'))
 print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),RSS_MiB=rss,trainable10=True)))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
