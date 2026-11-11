"""Offline one-step unit verification; no training run, no physics/teacher execution."""
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from residual_motor import ResidualMotor,load_baseline_actor,save_candidate,load_candidate,sha256,HERE,CHECKPOINT,BASE

REFERENCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz')

def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
    return type(a)==type(b) and a==b


def main():
    torch.set_num_threads(1);torch.manual_seed(20261110)
    inputs=[REFERENCE,CHECKPOINT,BASE/'models.py']+[HERE.parent/'teacher'/n/'SAMPLES_5MS.npz' for n in ('left_full','right_half')]
    hashes={str(p):sha256(p) for p in inputs}
    model=ResidualMotor(load_baseline_actor()).eval()
    frozen={k:v.clone() for k,v in model.base.state_dict().items()}
    checks={}
    with np.load(REFERENCE,allow_pickle=False) as z:
        x=torch.tensor(z['X238'],dtype=torch.float64)[:,None]
        expected=torch.tensor(z['mean'],dtype=torch.float64)
        hidden=torch.tensor(z['hidden_after'],dtype=torch.float64)
        actions=torch.tensor(z['policy_action'],dtype=torch.float64)
    zeros=x.new_zeros((len(x),1,2))
    with torch.no_grad():
        before,before_states=model.sequence(x,zeros)
    checks['all2880_neutral_means_bitexact_initial']=torch.equal(before[:,0],expected)
    checks['all2880_neutral_actions_bitexact_initial']=torch.equal(before[:,0].tanh(),actions)
    checks['all2880_neutral_base_hidden_bitexact_initial']=torch.equal(before_states[0][:,0],hidden)
    checks['base_frozen_at_construction']=all(not p.requires_grad for p in model.base.parameters())
    opt=torch.optim.Adam(model.adapter_parameters(),lr=.001)
    opt.zero_grad(set_to_none=True)
    losses=[]
    for condition in ('left_full','right_half'):
        with np.load(HERE.parent/'teacher'/condition/'SAMPLES_5MS.npz',allow_pickle=False) as z:
            tx=torch.tensor(z['X238'],dtype=torch.float64)[:,None]
            tu=torch.tensor(z['command2'],dtype=torch.float64)[:,None]
            ty=torch.tensor(z['Y42'],dtype=torch.float64)[:,None]
        with torch.no_grad():
            _,prime=model.sequence(tx[:1080],tu[:1080])
        state=tuple(s[-1].detach() for s in prime)
        prediction,_=model.sequence(tx[1080:1112],tu[1080:1112],state)
        loss=(prediction.tanh()-ty[1080:1112]).square().mean()/2
        losses.append(float(loss.detach()))
        loss.backward()
    gradients={name:dict(finite=bool(torch.isfinite(p.grad).all()),maxabs=float(p.grad.abs().max()),nonzero_entries=int(torch.count_nonzero(p.grad)),entries=p.numel()) for name,p in model.named_parameters() if not name.startswith('base.')}
    checks['every_adapter_tensor_finite_nonzero_gradient']=all(v['finite'] and v['maxabs']>0 for v in gradients.values())
    checks['every_adapter_scalar_connection_nonzero_gradient']=all(v['nonzero_entries']==v['entries'] for v in gradients.values())
    checks['base_has_no_gradients']=all(p.grad is None for p in model.base.parameters())
    old_adapter={n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith('base.')}
    opt.step() # Exactly ONE optimizer unit-test update. No training run.
    checks['every_adapter_tensor_updated']=all(not torch.equal(v,dict(model.named_parameters())[n]) for n,v in old_adapter.items())
    checks['base_weights_and_buffers_exact_after_update']=exact(frozen,model.base.state_dict())
    with torch.no_grad():
        after,after_states=model.sequence(x,zeros)
        commands=x.new_zeros((160,1,2));commands[:80,0,1]=.8
        changed,states=model.sequence(x[:160],commands)
        first,first_states=model.sequence(x[:73],commands[:73])
        second,second_states=model.sequence(x[73:160],commands[73:],tuple(s[-1] for s in first_states))
    checks['all2880_neutral_means_bitexact_after_update']=torch.equal(after[:,0],expected)
    checks['all2880_neutral_actions_bitexact_after_update']=torch.equal(after[:,0].tanh(),actions)
    checks['all2880_neutral_base_hidden_bitexact_after_update']=torch.equal(after_states[0][:,0],hidden)
    checks['neutral_after_nonzero_memory_bitexact']=torch.equal(changed[80:,0],expected[80:160])
    checks['nonzero_command_changes_output']=not torch.equal(changed[:80,0],expected[:80])
    checks['base_hidden_independent_of_command_bitexact']=torch.equal(states[0][:,0],hidden[:160])
    checks['residual_state_not_reset_when_command_zero']=not torch.equal(states[1][80],torch.zeros_like(states[1][80]))
    checks['split_continuation_means_bitexact']=torch.equal(torch.cat((first,second)),changed)
    checks['split_continuation_both_states_bitexact']=all(torch.equal(torch.cat((a,b)),c) for a,b,c in zip(first_states,second_states,states))
    checks['latent_correction_bounded']=bool((changed[:,0]-expected[:160]).abs().max()<=model.RESIDUAL_BOUND)
    # Direct excluded-channel intervention: base hidden and residual hidden must be identical.
    obs=x[1080].clone();perturbed=obs.clone()
    for start,stop in model.base.MASK_RANGES:perturbed[:,start:stop]+=1
    state=model.initial_state(1);u=obs.new_tensor([[.5,-.5]])
    with torch.no_grad():
        a,sa=model.step(obs,u,state);b,sb=model.step(perturbed,u,state)
    checks['excluded_channels_do_not_drive_either_hidden']=all(torch.equal(v,w) for v,w in zip(sa,sb))
    # Current target derivative remains exactly the retained inhibitory diagonal.
    differentiable=obs.clone().requires_grad_()
    mean,_=model.step(differentiable,u,state)
    rows=[torch.autograd.grad(mean[0,j],differentiable,retain_graph=True)[0][0,126:168] for j in range(42)]
    jac=torch.stack(rows)
    checks['direct_target_jacobian_exact_retained_inhibitory_diagonal']=torch.equal(jac,-torch.diag(model.base.gain()))
    # Both recurrent states are part of checkpoint runtime state, with no hidden caches.
    with tempfile.TemporaryDirectory(prefix='offline-check-',dir=HERE) as td:
        path=Path(td)/'unit-step-only.pt'
        runtime=tuple(s[-1].clone() for s in states)
        save_candidate(path,model,opt,dict(kind='single optimizer unit test only',optimizer_steps=1),runtime)
        loaded,payload=load_candidate(path)
        checks['saved_all_weights_reload_exact']=exact(model.state_dict(),loaded.state_dict())
        checks['saved_both_runtime_states_reload_exact']=exact(runtime,payload['runtime_state'])
        restored=torch.optim.Adam(loaded.adapter_parameters(),lr=.001)
        restored.load_state_dict(payload['optimizer'])
        checks['saved_optimizer_reload_exact']=exact(opt.state_dict(),restored.state_dict())
        with torch.no_grad():
            a,sa=model.step(x[160],u,runtime)
            b,sb=loaded.step(x[160],u,payload['runtime_state'])
        checks['reloaded_output_and_both_next_states_exact']=torch.equal(a,b) and exact(sa,sb)
        checks['loaded_base_stays_frozen']=all(not p.requires_grad for p in loaded.base.parameters())
    for name,badobs,badu in [('extra_observation',x[0].new_zeros((1,239)),zeros[0]),('extra_command',x[0],zeros[0].new_zeros((1,3))),('command_out_of_range',x[0],zeros[0].new_tensor([[0,1.1]]))]:
        try:model.step(badobs,badu,model.initial_state(1));rejected=False
        except ValueError:rejected=True
        checks['reject_'+name]=rejected
    checks['inputs_unchanged']=all(sha256(p)==h for p,h in hashes.items())
    report=dict(all_passed=all(checks.values()),checks=checks,gradients=gradients,
                adapter_parameter_count=sum(p.numel() for p in model.adapter_parameters()),
                frozen_base_parameter_count=sum(p.numel() for p in model.base.parameters()),
                identity=model.identity(),input_hashes=hashes,
                source_hashes={str(HERE/n):sha256(HERE/n) for n in ('residual_motor.py','check_residual.py','PREREG.json')},
                optimizer_unit_steps=1,physical_integration_steps=0,teacher_calls=0,unit_losses=losses,
                interpretation='Offline interface/learning-path candidate only. No trained candidate checkpoint retained. No turning/slip/food-approach improvement established.')
    (HERE/'CHECK_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_passed']:raise RuntimeError('offline candidate check failure')

if __name__=='__main__':main()
