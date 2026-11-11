"""Read-only conditional gate before any weighted-loss optimizer update."""
import json
import numpy as np
import torch
from model_io import HERE,ROOT,load_parent,sha256
from data import prepare,cached,objective


def main():
 torch.set_num_threads(1);model,p=load_parent();e,data_hashes,alignment=prepare(model)
 predicted=cached(model,e)
 err=(predicted[960:].tanh()-e['y'][960:]).square()
 mask=e['loaded_mask'][960:]
 total=float(err.sum().detach());rh_loaded=float(err[mask,28:35].sum().detach())
 fraction=rh_loaded/total if total>0 else 0.
 model.zero_grad(set_to_none=True);plain=objective(predicted,e,False);plain.backward()
 g0=torch.cat([p.grad.flatten().clone() for p in model.adapter_parameters()])
 model.zero_grad(set_to_none=True);weighted=objective(cached(model,e),e,True);weighted.backward()
 g1=torch.cat([p.grad.flatten().clone() for p in model.adapter_parameters()])
 gradnorm=float(g1.norm());cosine=float(torch.dot(g0,g1)/(g0.norm()*g1.norm()))
 numpy_errors=err.detach().numpy();numpy_weights=e['weights'][960:].numpy()
 numpy_loss=float(np.sum(numpy_errors*numpy_weights)/np.sum(numpy_weights))
 manual_total=0.;manual_den=0.
 for row in range(1920):
  for joint in range(42):
   w=4. if bool(mask[row]) and 28<=joint<35 else 1.
   manual_total+=float(numpy_errors[row,joint])*w;manual_den+=w
 manual_loss=manual_total/manual_den
 manual_error=max(abs(float(weighted.detach())-numpy_loss),abs(float(weighted.detach())-manual_loss))
 checks=dict(before_action_alignment=alignment['all_clock_max_error_s']<=1e-8,RH_loaded_energy_at_least_one_percent=fraction>=.01,weighted_gradient_finite_nonzero=bool(torch.isfinite(g1).all() and gradnorm>=1e-10),manual_numpy_weighted_objective_agreement=manual_error<=1e-12,frozen_parent_no_grad=all(not p.requires_grad and p.grad is None for p in model.parent.parameters()))
 source_hashes={str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','model_io.py','data.py','feasibility.py')};source_hashes[str(ROOT/'PREREG.json')]=sha256(ROOT/'PREREG.json')
 report=dict(supported=all(checks.values()),checks=checks,total_post_squared_error_energy=total,RH_loaded_error_energy=rh_loaded,RH_loaded_error_energy_fraction=fraction,per_joint_post_MSE=err.detach().mean(0).tolist(),RH_loaded_action_MSE=float(err[mask,28:35].mean().detach()),other_action_MSE=float(torch.cat((err[:,:28],err[:,35:]),dim=1).mean().detach()),uniform_loss=float(plain.detach()),weighted_loss=float(weighted.detach()),uniform_gradient_norm=float(g0.norm()),weighted_gradient_norm=gradnorm,uniform_weighted_gradient_cosine=cosine,manual_numpy_weighted_loss_max_error=manual_error,alignment=alignment,source_hashes=source_hashes,data_hashes=data_hashes,optimizer_updates=0,physics_integration_steps=0,interpretation='Teacher-distribution feasibility only; not proof loss weighting improves physical slip. No model or optimizer update performed.')
 with (HERE/'FEASIBILITY.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:report[k] for k in ('supported','checks','RH_loaded_error_energy_fraction','weighted_gradient_norm','uniform_weighted_gradient_cosine','uniform_loss','weighted_loss','manual_numpy_weighted_loss_max_error')}),flush=True)

if __name__=='__main__':main()
