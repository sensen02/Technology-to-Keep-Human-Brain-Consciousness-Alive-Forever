"""No-update full production/cache weighted objective and gradient validation."""
import json
import torch
from model_io import HERE,ROOT,load_parent,sha256,verify_hashes
from data import prepare,cached,objective,invariance


def main():
 torch.set_num_threads(1);feas=json.loads((HERE/'FEASIBILITY.json').read_text())
 if not feas['supported']:raise ValueError('fixed feasibility rule rejects training')
 verify_hashes(feas['source_hashes']);verify_hashes(feas['data_hashes'])
 model,parent=load_parent();e,hashes,alignment=prepare(model)
 with torch.no_grad():prod,states=model.sequence(e['x'][:,None],e['u'][:,None]);batch=cached(model,e)
 checks=dict(parent_actual_step2000=parent['curve'][-1]['step']==2000,full_production_output_tol=float((prod[:,0]-batch).abs().max())<=1e-12,causal_base_history_exact=torch.equal(states[0][:,0],e['base_hidden']),causal_residual_history_exact=torch.equal(states[1][:,0],e['residual_hidden']))
 # Different graph execution paths, same four-dimensional weighted sum.
 model.zero_grad(set_to_none=True);production,history=model.sequence(e['x'][:,None],e['u'][:,None]);ploss=objective(production[:,0],e);ploss.backward();g0=[p.grad.clone() for p in model.adapter_parameters()]
 model.zero_grad(set_to_none=True);bloss=objective(cached(model,e),e);bloss.backward();g1=[p.grad.clone() for p in model.adapter_parameters()]
 grad_error=max(float((a-b).abs().max()) for a,b in zip(g0,g1))
 checks['all_head_weighted_gradients_tol1e12']=grad_error<=1e-12
 checks['weighted_loss_production_tol1e12']=abs(float(ploss.detach())-float(bloss.detach()))<=1e-12
 checks['all_new_gradients_finite_nonzero']=all(torch.isfinite(g).all() and g.abs().max()>0 for g in g1)
 checks['frozen194878_no_gradient']=all(not p.requires_grad and p.grad is None for p in model.parent.parameters())
 checks.update(invariance(model))
 obs=e['x'][1200].clone().requires_grad_(True);state=(e['base_hidden'][1199:1200],e['residual_hidden'][1199:1200])
 jac=torch.autograd.functional.jacobian(lambda x:model.step(x[None],e['u'][1200:1201],state)[0][0],obs)
 checks['negative_target_position_jacobian_unchanged']=torch.equal(jac[:,126:168],-torch.diag(model.base.gain()))
 sources={str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','model_io.py','data.py','feasibility.py','FEASIBILITY.json','check_train.py')};sources[str(ROOT/'PREREG.json')]=sha256(ROOT/'PREREG.json')
 report=dict(all_passed=all(checks.values()),checks=checks,gradient_max_error=grad_error,source_hashes=sources,data_hashes=hashes,alignment=alignment,optimizer_updates=0,physics_integration_steps=0,Core_constructions=0)
 with (HERE/'PREFLIGHT.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),gradient_max_error=grad_error)),flush=True)
 if not report['all_passed']:raise ValueError('weighted preflight failed')

if __name__=='__main__':main()
