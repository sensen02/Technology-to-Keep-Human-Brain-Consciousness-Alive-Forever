"""Offline implementation verification, two discarded optimizer unit updates."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from model_io import HERE,ROOT,NonlinearMotor,load_parent,save,load_checked,exact,sha256
from data import prepare,cached,invariance

def main():
 torch.set_num_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 parent,_=load_parent();m=NonlinearMotor(parent);e,dh=prepare(m);checks={}
 checks['parent197608']=sum(p.numel() for p in m.parent.parameters())==197608
 checks['new_parameters22378']=sum(p.numel() for p in m.adapter_parameters())==22378
 checks['zero_output']=bool((m.output.weight==0).all() and (m.output.bias==0).all())
 checks.update({'initial_'+k:v for k,v in invariance(m,('neutral','left','right')).items()})
 before={k:v.clone() for k,v in m.parent.state_dict().items()};opt=torch.optim.Adam(m.adapter_parameters(),lr=.001)
 initial=float((cached(m,e)[960:].tanh()-e['y'][960:]).square().mean())
 for unit in range(2):
  opt.zero_grad(set_to_none=True);loss=(cached(m,e)[960:].tanh()-e['y'][960:]).square().mean();loss.backward()
  if unit==0:checks['first_encoder_grad_expected_zero']=bool(m.encoder.weight.grad.abs().max()==0 and m.encoder.bias.grad.abs().max()==0)
  else:checks['second_all_new_gradients_finite_nonzero']=all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in m.adapter_parameters())
  opt.step()
 checks['parent_no_grad']=all(not p.requires_grad and p.grad is None for p in m.parent.parameters())
 checks['all_parent_weights_unchanged']=exact(before,m.parent.state_dict())
 checks.update({'after_update_'+k:v for k,v in invariance(m).items()})
 mean,state=m.sequence(e['x'][:,None],e['u'][:,None]);v=cached(m,e)
 oe=float((mean[:,0]-v).abs().max().detach());checks['production_cache_output_tol']=oe<=1e-12
 checks['full_dual_state_exact']=torch.equal(state[0][:,0],e['bh']) and torch.equal(state[1][:,0],e['rh'])
 m.zero_grad(set_to_none=True);((mean[960:,0].tanh()-e['y'][960:]).square().mean()).backward();grads=[p.grad.clone() for p in m.adapter_parameters()]
 m.zero_grad(set_to_none=True);((v[960:].tanh()-e['y'][960:]).square().mean()).backward();ge=max(float((p.grad-g).abs().max()) for p,g in zip(m.adapter_parameters(),grads));checks['all_gradient_tol']=ge<=1e-12
 xx=e['x'][1200].clone().requires_grad_(True);hh=(e['bh'][1199:1200],e['rh'][1199:1200]);jac=torch.autograd.functional.jacobian(lambda row:m.step(row[None],e['u'][1200:1201],hh)[0][0],xx)
 checks['target_negative_jacobian_exact']=torch.equal(jac[:,126:168],-torch.diag(m.base.gain()))
 sources={str(HERE/n):sha256(HERE/n) for n in ('model_io.py','data.py','check_train.py','PREREG.json')};sources[str(ROOT/'PREREG.json')]=sha256(ROOT/'PREREG.json')
 config=dict(kind='fixed_nonlinear_positive_readout5000',physical_eval_only_step=5000,source_hashes=sources,data_hashes=dh)
 rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());curve=[dict(step=0,loss=initial),dict(step=1,loss=float(loss.detach()))]
 with tempfile.TemporaryDirectory(prefix='nonlinear-unit-',dir=HERE) as tmp:
  path=Path(tmp)/'unit.pt';save(path,m,opt,config,curve,rng);p=torch.load(path,map_location='cpu',weights_only=False)
  parent2,_=load_parent();clone=NonlinearMotor(parent2);clone.load_state_dict(p['state_dict']);op=torch.optim.Adam(clone.adapter_parameters(),lr=.001);op.load_state_dict(p['optimizer'])
  checks['weights_optimizer_rng_curve_saved_exact']=exact(clone.state_dict(),m.state_dict()) and exact(op.state_dict(),opt.state_dict()) and exact(p['rng'],rng) and exact(p['curve'],curve)
  with torch.no_grad():checks['clone_left_inference_exact']=torch.equal(cached(m,e),cached(clone,e))
 report=dict(all_passed=all(checks.values()),checks=checks,initial_loss=initial,output_max_error=oe,gradient_max_error=ge,source_hashes=sources,data_hashes=dh,physical_steps=0,discarded_optimizer_unit_steps=2,persistent_training_updates=0)
 (HERE/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),output_error=oe,gradient_error=ge)))
 if not report['all_passed']:raise ValueError('nonlinear preflight failed')
if __name__=='__main__':main()
