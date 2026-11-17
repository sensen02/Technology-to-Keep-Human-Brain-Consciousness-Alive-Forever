"""Parent actualfullB27vs3micro9 gradient/Adam preflight; no batch fallback or physics."""
from common import *
def main():
 torch,np=backend()
 if (HERE/'FROZEN.json').exists():raise FileExistsError('Inspect priorpreflight')
 from data import main as prepare,arrays,tensors
 from model import python_rollout,make_scan,objective,accumulated_backward
 from copy import deepcopy
 prepare();d=arrays('train');x,u,y,w=tensors(d);torch.manual_seed(20261113);m,parent,old=initialize();a=deepcopy(m);fixed=frozen_tensors(m);ref=anchors(m);checks={};errors={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Fusionpreflight '+k)
 def close(k,l,r,tol=1e-10):
  error=float((l-r).abs().max().detach());errors[k]=error;check(k,error<=tol)
 check('actual27full1920',x.shape==(1920,27,111) and u.shape==(1920,27,2) and y.shape==w.shape==(1920,27,1));check('10trainable_frozen8',len(trainable(m))==10 and sum(not p.requires_grad for p in m.parameters())==8);check('source_replays_all_exact',read(HERE/'SENSORY_REPLAY.json')['all_bitexact']);check('parentcolumns_exact',torch.equal(m.brain.weight_ih[:,:2],parent['state_dict']['brain.weight_ih']));check('odorcolumn_exact_zero',bool((m.brain.weight_ih[:,2]==0).all()));check('allother_parent_tensors_exact',all(k=='brain.weight_ih' or torch.equal(v,parent['state_dict'][k]) for k,v in m.state_dict().items()));check('same111mask_norm',m.allowed_raw_indices.tolist()==MASK and torch.equal(m.observation_mean,parent['state_dict']['observation_mean']) and torch.equal(m.observation_std,parent['state_dict']['observation_std']))
 for j in range(27):check('weightmass_'+str(j),abs(float(w[:,j].sum())-1)<1e-12)
 raw=torch.zeros((1,238),dtype=torch.float64,requires_grad=True);an=m.feedback(m.normalize(raw.index_select(1,m.allowed_raw_indices)));g=torch.autograd.grad(an.sum(),raw)[0];forbidden=sorted(set(range(238))-set(MASK));check('forbidden_raw_action_target_speed_zero_gradient',bool((g[:,forbidden]==0).all()) and 237 in forbidden)
 with torch.no_grad():
  oldmod=module('_fusion_parent_parity_rollout',PROJECT/'experiments/neck-real-structure/20261112T225331Z/neural_calibration/model.py');expected=oldmod.python_rollout(old,x[:,:1],u[:,:1,:1]);pure=u[:,:1].clone();pure[:,:,1]=0;new=python_rollout(m,x[:,:1],pure);arbitrary=python_rollout(m,x[:,:1],u[:,:1]);jit=make_scan(a)(x[:,:1],u[:,:1])
  for k,left,right,anyodor,scanout in zip(('turn','DN','AN','recon','brainhidden'),expected,new,arbitrary,jit):close('parent_visual_parity_'+k,left,right,1e-12);close('initial_any_odor_no_effect_'+k,right,anyodor,1e-12);close('python_jit_batch1_'+k,anyodor,scanout)
 check('braininput3bodydecoderDNonly',m.brain.input_size==3 and m.brain.hidden_size==16 and m.decoder[0].in_features==2 and m.decoder[2].out_features==1)
 # Compute the independent full27 Python graph, then release it BEFORE accumulation graphs.
 full=python_rollout(m,x,u);loss,terms=objective(full,y,w,m,ref);manual=((full[0]-y).square()*w).sum(0).mean()+.1*m.decoder(x.new_zeros((27,2))).square().mean()+.001*full[1].square().mean()+.001*sum((dict(m.named_parameters())[k]-v).square().sum() for k,v in ref.items())/sum(v.numel() for v in ref.values());close('full27_independent_manual_objective',loss,manual);loss_value=float(loss.detach());loss.backward();full_grads={k:p.grad.detach().clone() for k,p in trainable(m)};check('full27_all10_finite_gradients',all(bool(torch.isfinite(v).all()) for v in full_grads.values()));del full,loss,terms,manual
 scan=make_scan(a);accum_loss,accum_terms=accumulated_backward(scan,a,x,u,y,w,ref);close('full27_vs_micro9_accumulated_loss',x.new_tensor(loss_value),x.new_tensor(accum_loss))
 norms={}
 for k,p in trainable(a):check('micro9_finitegrad_'+k,p.grad is not None and torch.isfinite(p.grad).all());close('full27_vs_accumulated_gradient_'+k,full_grads[k],p.grad);norms[k]=float(p.grad.norm())
 check('all_frozen_gradients_absent',all(p.grad is None for model in (m,a) for p in model.parameters() if not p.requires_grad));torch.nn.utils.clip_grad_norm_([p for k,p in trainable(m)],1.);torch.nn.utils.clip_grad_norm_([p for k,p in trainable(a)],1.);o1=torch.optim.Adam([p for k,p in trainable(m)],lr=3e-4);o2=torch.optim.Adam([p for k,p in trainable(a)],lr=3e-4);o1.step();o2.step()
 for (k,p),(other,q) in zip(trainable(m),trainable(a)):
  check('parameter_order_'+k,k==other);close('full27_vs_accum_Adamparam_'+k,p,q)
  for name in ('exp_avg','exp_avg_sq','step'):close('full27_vs_accum_Adamstate_'+name+'_'+k,o1.state[p][name],o2.state[q][name])
 check('one_Adam_step_after_three_microbatches',len(o1.state)==len(o2.state)==10 and all(float(v['step'])==1 for optimizer in (o1,o2) for v in optimizer.state.values()));check_frozen(m,fixed);check_frozen(a,fixed);check('frozen_feedback_recon_norm_exact',True);del full_grads,a,scan,o1,o2
 bp,b,digest=binding()
 with torch.no_grad():
  expected=python_rollout(m,x[:,20:21],u[:,20:21]);actual,proof=real_queue_rollout(m,x[:,20:21],u[:,20:21],bp,digest)
  for k,l,r in zip(('turn','DN','AN','recon','brainhidden'),expected,actual):close('actual1ms_sourcequeue_'+k,l,r)
 check('actual9600ticks3838consumers',proof['actual_steps']==9600 and len(proof['neural_consumption_events'])==3838 and all(v['queue_release_step']==v['capture_step']+2 and v['neural_consume_step']==v['capture_step']+5 for v in proof['neural_consumption_events']));save(HERE/'PREFLIGHT_QUEUE.json',proof)
 import resource
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.;check('RSS_under1800_without_effective_batchchange',rss<1800.);ds=read(HERE/'DATASET.json');sh={str(p):sha(p) for p in sources()};sh.update(ds['source_hashes']);dh={v['path']:v['sha256'] for v in ds['outputs'].values()};dh[str(HERE/'DATASET.json')]=sha(HERE/'DATASET.json');verify(sh);verify(dh)
 save(HERE/'PREFLIGHT.json',dict(all_passed=True,checks=checks,numerical_errors=errors,maximum_error=max(errors.values()),gradient_norms=norms,peak_RSS_MiB=rss,effective_batch=27,microbatch=9,accumulation=3,history_steps=1920,actual_sensory_replays=True,parent_odor_column_zero_parity_verified=True,source_binding_sha256=digest,training_updates=0,Core_constructions=0,new_physics_steps=0));sh[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json');sh[str(HERE/'PREFLIGHT_QUEUE.json')]=sha(HERE/'PREFLIGHT_QUEUE.json');save(HERE/'FROZEN.json',dict(source_hashes=sh,data_hashes=dh,parent_sha256=PARENT_SHA,visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,source_binding_sha256=digest,full_neck_training_allowed=False,scope='Parent-authorized source sensory fusion neural calibration only'))
 print(json.dumps(dict(all_passed=True,checks=len(checks),max_error=max(errors.values()),RSS_MiB=rss,effective27_micro9=True)))
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
