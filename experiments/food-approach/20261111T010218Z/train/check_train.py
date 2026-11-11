"""Offline readout invariance and training implementation verification; no physics."""
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from model_io import HERE,ROOT,ReadoutMotor,load_parent,save,exact,sha256
from data import prepare,cached,invariance


def main():
    torch.set_num_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
    parent,parent_payload=load_parent();model=ReadoutMotor(parent)
    e,data_hashes=prepare(model);checks={}
    checks['parent_parameter_count194878']=sum(p.numel() for p in model.parent.parameters())==194878
    checks['new_parameter_count2730']=sum(p.numel() for p in model.adapter_parameters())==2730
    checks['zero_head_weights_and_bias']=bool((model.readout.weight==0).all() and (model.readout.bias==0).all())
    checks.update({'initial_'+k:v for k,v in invariance(model,('neutral','left','right')).items()})
    before={k:v.clone() for k,v in model.parent.state_dict().items()}
    opt=torch.optim.Adam(model.adapter_parameters(),lr=.003)
    prediction=cached(model,e)
    initial_loss=(prediction[960:].tanh()-e['y'][960:]).square().mean()
    initial_loss.backward()
    checks['only_two_new_tensors_have_finite_nonzero_gradient']=all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.adapter_parameters())
    checks['frozen_parent_has_no_gradient']=all(not p.requires_grad and p.grad is None for p in model.parent.parameters())
    opt.step();checks['temporary_actual_update_changed_newhead']=bool(model.readout.weight.abs().max()>0)
    checks['all_parent_weights_after_update_exact']=exact(before,model.parent.state_dict())
    checks.update({'after_update_'+k:v for k,v in invariance(model,('neutral','right')).items()})
    # Nonzero newhead: compare the real full recurrence to vectorized cached inference.
    means,states=model.sequence(e['x'][:,None],e['u'][:,None])
    vector=cached(model,e)
    output_error=float((means[:,0]-vector).abs().max().detach())
    checks['cached_output_tol1e12']=output_error<=1e-12
    checks['parent_cached_dual_state_exact']=torch.equal(states[0][:,0],e['base_hidden']) and torch.equal(states[1][:,0],e['residual_hidden'])
    model.zero_grad(set_to_none=True)
    ((means[960:,0].tanh()-e['y'][960:]).square().mean()).backward()
    reference=[p.grad.clone() for p in model.adapter_parameters()]
    model.zero_grad(set_to_none=True)
    ((vector[960:].tanh()-e['y'][960:]).square().mean()).backward()
    gradient_error=max(float((p.grad-g).abs().max()) for p,g in zip(model.adapter_parameters(),reference))
    checks['all_new_gradient_tol1e12']=gradient_error<=1e-12
    obs=e['x'][1200].clone().requires_grad_(True)
    state=(e['base_hidden'][1199:1200],e['residual_hidden'][1199:1200])
    jac=torch.autograd.functional.jacobian(lambda x:model.step(x[None],e['u'][1200:1201],state)[0][0],obs)
    checks['direct_target_negative_jacobian_unchanged']=torch.equal(jac[:,126:168],-torch.diag(model.base.gain()))
    sources={str(HERE/n):sha256(HERE/n) for n in ('model_io.py','data.py','PREREG.json','check_train.py')}
    sources[str(ROOT/'PREREG.json')]=sha256(ROOT/'PREREG.json')
    config=dict(kind='temporary_readout_optimizer_unit_check',source_hashes=sources,data_hashes=data_hashes,physical_eval_only_step=2000)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    curve=[dict(step=1,loss=float(initial_loss.detach()))]
    with tempfile.TemporaryDirectory(prefix='readout-check-',dir=HERE) as temp:
        path=Path(temp)/'unit.pt';save(path,model,opt,config,curve,rng)
        p=torch.load(path,map_location='cpu',weights_only=False)
        new_parent,_=load_parent();new=ReadoutMotor(new_parent);new.load_state_dict(p['state_dict'])
        newopt=torch.optim.Adam(new.adapter_parameters(),lr=.003);newopt.load_state_dict(p['optimizer'])
        checks['weights_roundtrip_exact']=exact(model.state_dict(),new.state_dict())
        checks['actual_new_Adam2_state_roundtrip_exact']=len(opt.state)==2 and exact(opt.state_dict(),newopt.state_dict())
        checks['RNG_config_curve_roundtrip_exact']=exact(p['rng'],rng) and exact(p['config'],config) and exact(p['curve'],curve)
        with torch.no_grad():a=cached(model,e);b=cached(new,e)
        checks['saved_nonzero_inference_exact']=torch.equal(a,b)
    report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=sources,data_hashes=data_hashes,
                cached_output_max_error=output_error,cached_gradient_max_error=gradient_error,physical_steps=0,
                temporary_optimizer_unit_steps=1,persistent_training_updates=0,
                interpretation='offline implementation/identity checks only, no new physical result')
    (HERE/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_passed']:raise ValueError('readout preflight failed')

if __name__=='__main__':main()
