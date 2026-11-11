"""Meaningful offline invariance, cached learning and feedback-Jacobian checks."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from fast_motor import HERE,FastMotor,load_base,save,load,sha256
from fast_data import prepare,cached,neutral


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b) and a.dtype==b.dtype and a.shape==b.shape
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def main():
    torch.set_num_threads(1);torch.manual_seed(20261110);np.random.seed(20261110)
    model=FastMotor(load_base());episodes,data=prepare(model)
    checks={'four_authenticated_teacher_episodes':len(episodes)==4}
    checks.update({'neutral_initial_'+k:v for k,v in neutral(model).items()})
    indices=torch.arange(1120,1140)
    xs=[];us=[];states=[];features=[];gates=[];means=[];ys=[]
    for e in episodes:
        xs.append(e['x'][indices]);us.append(e['u'][indices]);states.append(e['base_hidden'][indices-1])
        features.append(e['features'][indices]);gates.append(e['gate'][indices]);means.append(e['mean'][indices]);ys.append(e['y'][indices])
    x=torch.cat(xs);u=torch.cat(us);state=torch.cat(states);feature=torch.cat(features);gate=torch.cat(gates);mean=torch.cat(means);y=torch.cat(ys)
    # Reference executes actual production step independently for each saved before-state.
    outputs=[]
    for row,cmd,h in zip(x,u,state):outputs.append(model.step(row[None],cmd[None],h[None])[0][0])
    sequential=torch.stack(outputs);batch=mean+model.correction(feature,gate)
    output_error=float((sequential-batch).abs().max().detach())
    checks['cache_production_output_error_le_1e12']=output_error<=1e-12
    model.zero_grad(set_to_none=True);((sequential.tanh()-y).square().mean()).backward()
    reference_grad=[p.grad.clone() for p in model.adapter_parameters()]
    model.zero_grad(set_to_none=True);loss=(batch.tanh()-y).square().mean();loss.backward()
    grad_error=max(float((p.grad-g).abs().max()) for p,g in zip(model.adapter_parameters(),reference_grad))
    checks['cache_production_gradient_error_le_1e12']=grad_error<=1e-12
    checks['all_adapter_tensors_finite_nonzero_gradient']=all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.adapter_parameters())
    before={k:v.clone() for k,v in model.base.state_dict().items()}
    opt=torch.optim.Adam(model.adapter_parameters(),lr=.001);opt.step()
    checks['base_all_weights_unchanged_after_actual_update']=equal(before,model.base.state_dict())
    checks['base_no_gradients']=all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    checks.update({'neutral_after_update_'+k:v for k,v in neutral(model).items()})
    xx=x[0].clone().requires_grad_(True)
    jac=torch.autograd.functional.jacobian(lambda v:model.step(v[None],u[:1],state[:1])[0][0],xx)
    checks['direct_target_position_jacobian_negative_exact']=torch.equal(jac[:,126:168],-torch.diag(model.base.gain()))
    rejected=0
    for cmd in ([.1,0.],[-.2,0.],[0.,1.1]):
        try:model.step(x[:1],torch.tensor([cmd],dtype=torch.float64),state[:1])
        except ValueError:rejected+=1
    checks['untrained_command_domain_rejected']=rejected==3
    sources={str(HERE/n):sha256(HERE/n) for n in ('fast_motor.py','fast_data.py','check_fast.py','PREREG.json')}
    config=dict(sources=sources,data_hashes=data,steps=1,kind='temporary_single_optimizer_verification_not_training')
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    with tempfile.TemporaryDirectory(prefix='food-fast-check-') as temp:
        path=Path(temp)/'check.pt';save(path,model,opt,config,[dict(step=1,loss=float(loss.detach()))],rng)
        loaded,p=load(path,sha256(path))
        checkopt=torch.optim.Adam(loaded.adapter_parameters(),lr=.001);checkopt.load_state_dict(p['optimizer'])
        checks['saved_weights_exact']=equal(model.state_dict(),loaded.state_dict())
        checks['saved_optimizer_exact']=equal(opt.state_dict(),checkopt.state_dict())
        checks['saved_rng_and_config_exact']=equal(p['rng'],rng) and equal(p['config'],config)
        with torch.no_grad():
            a=model.step(x,u,state)[0];b=loaded.step(x,u,state)[0]
        checks['saved_inference_exact']=torch.equal(a,b)
    report=dict(all_passed=all(checks.values()),checks=checks,output_max_error=output_error,gradient_max_error=grad_error,
                source_hashes=sources,data_hashes=data,physical_steps=0,optimizer_unit_steps=1,
                parameter_counts=dict(base=sum(p.numel() for p in model.base.parameters()),adapter=sum(p.numel() for p in model.adapter_parameters())),
                interpretation='offline preflight only, actual steering/deceleration/food not established')
    (HERE/'FAST_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_passed']:raise ValueError('fast preflight failed')

if __name__=='__main__':main()
