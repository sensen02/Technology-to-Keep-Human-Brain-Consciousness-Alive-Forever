"""Full valid sequence/output/state/adapter-gradient equivalence; no persistent training."""
import json,time
from copy import deepcopy
import numpy as np
import torch
from model_io import HERE,OLD,load_parent,sha256,exact
from data import prepare,padded,sequential,batch,neutral
from engine import make

def main():
 torch.set_num_threads(1);start=time.monotonic();model,parent=load_parent()
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.001);opt.load_state_dict(parent['optimizer'])
 checks=dict(parent8Adamstates=len(opt.state)==8,all_parent_counters500=all(float(s['step'])==500 for s in opt.state.values()),parent_model_exact=exact(model.state_dict(),parent['state_dict']),parent_optimizer_exact=exact(opt.state_dict(),parent['optimizer']))
 episodes,hashes=prepare(model);p=padded(episodes);engine=make(model)
 checks.update({'neutral_before_'+k:v for k,v in neutral(model).items()})
 max_mean=max_state=0.
 with torch.no_grad():
  bm,bs=batch(model,p);sm,ss=engine(p['features'],p['gate'],p['means'],p['valid'])
  for j,e in enumerate(episodes):
   prod,ph=model.sequence(e['x'][:,None],e['u'][:,None]);sq,sh=sequential(model,e)
   max_mean=max(max_mean,float((prod[:,0]-bm[:e['length'],j]).abs().max()),float((sq-sm[:e['length'],j]).abs().max()))
   max_state=max(max_state,float((ph[1][:,0]-bs[:e['length'],j]).abs().max()),float((sh-ss[:e['length'],j]).abs().max()))
   if e['length']<2880:checks['padding_freezes_'+e['name']]=torch.equal(ss[e['length']-1,j].expand_as(ss[e['length']:,j]),ss[e['length']:,j])
 checks['full_production_means_tol1e12']=max_mean<=1e-12;checks['full_dual_history_tol1e12']=max_state<=1e-12
 # Original sequential objective, one contiguous graph per episode, all parameters.
 model.zero_grad(set_to_none=True)
 for e in episodes:
  q,_=sequential(model,e);loss=(q[960:].tanh()-e['y'][960:]).square().mean()/4;loss.backward()
 ref=[v.grad.clone() for v in model.adapter_parameters()]
 model.zero_grad(set_to_none=True);tick=time.monotonic()
 pred,_=engine(p['features'],p['gate'],p['means'],p['valid']);loss=((pred.tanh()-p['targets']).square()*p['weights']).sum();loss.backward()
 scan_update_s=time.monotonic()-tick
 err=max(float((v.grad-g).abs().max()) for v,g in zip(model.adapter_parameters(),ref))
 checks['all8_adapter_gradients_tol1e12']=err<=1e-12
 checks['finite_nonzero_adapter_gradient']=all(torch.isfinite(v.grad).all() and v.grad.abs().max()>0 for v in model.adapter_parameters())
 checks['frozen_base_no_grad']=all(not v.requires_grad and v.grad is None for v in model.base.parameters())
 # Temporary update verifies cached engine uses the live original parameter objects.
 before=deepcopy(model.state_dict());opt.param_groups[0]['lr']=.0003;opt.step()
 checks['adapter_actual_update']=any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if not k.startswith('base.'))
 checks['base_all_weights_unchanged']=all(torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith('base.'))
 checks.update({'neutral_after_'+k:v for k,v in neutral(model).items()})
 source_names=['PREREG.json','model_io.py','data.py','engine.py','check_train.py']
 sources={str(HERE/n):sha256(HERE/n) for n in source_names}
 report=dict(all_passed=all(checks.values()),checks=checks,full_mean_max_error=max_mean,full_state_max_error=max_state,all_adapter_gradient_max_error=err,batch_forward_backward_s=scan_update_s,wall_s=time.monotonic()-start,source_hashes=sources,data_hashes=hashes,optimizer_temporary_unit_updates=1,physical_steps=0)
 (HERE/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
 if not report['all_passed']:raise ValueError('training preflight failed')

if __name__=='__main__':main()
