"""Full production/cache correctness and initial LH teacher-state diagnostics.
Call only after parent release; load mask used exclusively in loss.
"""
from copy import deepcopy
import torch,numpy as np
from model_io import *
def run_checks(m,c):
 checks={};errors={}
 def close(n,a,b):
  errors[n]=float((a-b).abs().max());checks[n]=errors[n]<=1e-12
 a=deepcopy(m);b=deepcopy(m);initial=frozen(m);pa,sa=a.sequence(c['x'],c['u']);pb=cached_forward(b,c);cached_initial=pb.detach().clone();parent,p=old.load_checked(PARENT,PARENT_SHA)
 close('production_cache_means',pa,pb);close('residual_hidden',sa[1],c['hidden']);close('base_hidden',sa[0],c['base_states']);checks['initial_allmeans_parent_exact']=torch.equal(pa,p['replay']['means'])
 loss(pa,c).backward();loss(pb,c).backward()
 for name in ('lh_weight','lh_bias'):close('gradient_'+name,getattr(a.head,name).grad,getattr(b.head,name).grad)
 gradient=torch.cat([q.grad.reshape(-1) for q in b.adapter_parameters()]);checks['actual_LH_gradient_finite_nonzero']=bool(torch.isfinite(gradient).all() and gradient.abs().max()>0)
 checks['outsideLH_no_grad']=all(not q.requires_grad and q.grad is None for n,q in a.named_parameters() if n not in ('head.lh_weight','head.lh_bias'))
 oa=torch.optim.Adam(a.adapter_parameters(),lr=.001);ob=torch.optim.Adam(b.adapter_parameters(),lr=.001);checks['freshAdam_empty']=len(oa.state)==len(ob.state)==0;oa.step();ob.step()
 for n,v in a.state_dict().items():close('unit_model_'+n,v,b.state_dict()[n])
 for i,(u,v) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in u:close('unit_Adam_'+str(i)+'_'+k,u[k],v[k])
 checks['all_frozen_weights_exact']=exact(frozen(a),initial) and exact(frozen(b),initial)
 other=list(range(7))+list(range(14,42))
 with torch.no_grad():updated,_=a.sequence(c['x'],c['u']);cached_updated=cached_forward(b,c)
 checks['unit_update_actual_nonLH_outputs_exact']=torch.equal(updated[:,:,other],c['means'][:,:,other]);checks['unit_update_cached_nonLH_outputs_exact']=torch.equal(cached_updated[:,:,other],cached_initial[:,:,other]);checks.update({'neutral_'+k:v for k,v in neutral(a).items()})
 checks['mapping_LH1_rows7to14']=c['mapping']['LH_contact_index']==1 and c['mapping']['LH_output_rows']==list(range(7,14));checks['preaction_indices']=int(c['preaction_indices'][960])==5799 and c['time_error']<1e-8;checks['455_only']=sum(q.numel() for q in m.parameters() if q.requires_grad)==455
 pred=c['means'][960:,:,7:14].tanh();target=c['y'][960:,:,7:14];s=torch.from_numpy(c['stance'][960:]);e=(pred-target).square()[:,0]
 diag=dict(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,LH_plain_action_MSE=float(e.mean()),LH_stance_action_MSE=float(e[s].mean()),LH_nonstance_action_MSE=float(e[~s].mean()),LH_stance_error_fraction=float(e[s].sum()/e.sum()),stance_fraction=float(s.double().mean()),LH_weighted_action_MSE=float(loss(c['means'],c)),actual_LH_gradient_norm=float(gradient.norm()),actual_LH_gradient_abs_max=float(gradient.abs().max()),actual_LH_gradient_finite_nonzero=checks['actual_LH_gradient_finite_nonzero'],effective_parameters=455,preaction_time_max_error=c['time_error'],mapping=c['mapping'],physics_steps=0,optimizer_updates=0)
 return dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,tolerance=1e-12,diagnostic=diag)
