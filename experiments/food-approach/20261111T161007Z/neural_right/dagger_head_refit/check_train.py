"""Actual per-trajectory production/dense-assembly/cache/gradient/unitAdam preflight."""
from copy import deepcopy
import torch
from model_io import *
def run_checks(m,c):
 checks={};errors={}
 def close(n,a,b):
  errors[n]=float((a-b).abs().max());checks[n]=errors[n]<=1e-12
 for j,e in enumerate(c['episodes']):
  checks[e['name']+'_initial_means_parent_bitexact']=torch.equal(c['means'][:,j:j+1],e['parent_means']);checks[e['name']+'_initial_basehidden_parent_bitexact']=torch.equal(c['base_states'][:,j:j+1],e['parent_base_states']);checks[e['name']+'_initial_residualhidden_parent_bitexact']=torch.equal(c['hidden'][:,j:j+1],e['parent_residual_states'])
 a=deepcopy(m);b=deepcopy(m);production=[];states=[];base=frozen(m)
 for e in c['episodes']:
  p,h=a.sequence(e['x'],e['u']);production.append(p);states.append(h)
 pa=torch.cat(production,1);pb=cached_forward(b,c);close('production_cached_means',pa,pb);close('actual_full_residual_history',torch.cat([h[1] for h in states],1),c['hidden']);close('actual_full_base_history',torch.cat([h[0] for h in states],1),c['base_states'])
 la=objective(pa,c);lb=objective(pb,c);la.backward();lb.backward()
 for n in ('weight','bias'):close('gradient_head_'+n,getattr(a.head,n).grad,getattr(b.head,n).grad)
 checks['frozen_no_grad']=all(not q.requires_grad and q.grad is None for n,q in a.named_parameters() if not n.startswith('head.'))
 oa=torch.optim.Adam(a.adapter_parameters(),lr=.0003);ob=torch.optim.Adam(b.adapter_parameters(),lr=.0003);checks['freshAdam_empty']=len(oa.state)==len(ob.state)==0;oa.step();ob.step()
 for n,v in a.state_dict().items():close('unit_model_'+n,v,b.state_dict()[n])
 for i,(u,v) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in u:close('unit_Adam_'+str(i)+'_'+k,u[k],v[k])
 checks['all_frozen_unchanged']=exact(frozen(a),base) and exact(frozen(b),base);checks['head2730']=sum(q.numel() for q in m.parameters() if q.requires_grad)==2730;checks['two_equal2880']=len(c['episodes'])==2 and c['y'].shape==(2880,2,42);checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 return dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,tolerance=1e-12,initial_equal_objective=float(la.detach()),initial_episode_MSE={e['name']:float((pa[960:,j].detach().tanh()-c['y'][960:,j]).square().mean()) for j,e in enumerate(c['episodes'])},physics_steps=0)
