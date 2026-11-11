"""Check exact verified-trajectory parity, actual learning path and checkpoint reload."""
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from conditional_motor import ConditionalMotor,OdorBehavior,load_baseline_actor,save_motor,load_motor,sha256,HERE,CHECKPOINT

def main():
    torch.set_num_threads(1); torch.manual_seed(20261110)
    source=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz')
    baseline=load_baseline_actor(); model=ConditionalMotor(baseline)
    checks={}
    with np.load(source,allow_pickle=False) as data:
        x=torch.tensor(data['X238'],dtype=torch.float64)
        expected_mean=torch.tensor(data['mean'],dtype=torch.float64)
        expected_hidden=torch.tensor(data['hidden_after'],dtype=torch.float64)
    with torch.no_grad():
        means,states=model.sequence(x[:,None],x.new_zeros((len(x),1,2)))
    checks['all_2880_saved_means_bitexact']=torch.equal(means[:,0],expected_mean)
    checks['all_2880_saved_hidden_bitexact']=torch.equal(states[:,0],expected_hidden)
    with torch.no_grad():
        # A zero-initial adapter must preserve baseline even for nonzero commands.
        nonzero,_=model.sequence(x[:80,None],x.new_ones((80,1,2)))
    checks['zero_adapter_nonzero_command_has_no_hidden_drive']=torch.equal(nonzero[:,0],expected_mean[:80])
    model.set_training_stage('adapter')
    before={k:v.clone() for k,v in model.base.state_dict().items()}
    opt=torch.optim.Adam([model.command_encoder.weight],lr=.001)
    obs=x[:120,None]; commands=x.new_zeros((120,1,2));commands[:,:,1]=1
    mean,_=model.sequence(obs,commands)
    target=mean.detach().tanh()+.001
    loss=(mean.tanh()-target).square().mean()
    loss.backward()
    grad=model.command_encoder.weight.grad
    checks['adapter_has_finite_nonzero_actual_gradient']=bool(torch.isfinite(grad).all() and grad.abs().max()>0)
    opt.step()
    checks['adapter_update_keeps_all_baseline_weights_exact']=all(torch.equal(v,model.base.state_dict()[k]) for k,v in before.items())
    with torch.no_grad():
        shifted,_=model.sequence(obs,commands)
        neutral,_=model.sequence(obs,torch.zeros_like(commands))
    checks['trained_adapter_changes_nonzero_command_outputs']=not torch.equal(shifted,mean)
    checks['trained_adapter_neutral_still_bitexact']=torch.equal(neutral[:,0],expected_mean[:120])
    with tempfile.TemporaryDirectory(prefix='food-motor-check-') as td:
        path=Path(td)/'check.pt'
        save_motor(path,model,opt,dict(kind='unit learning path only; not gait training'),[dict(loss=float(loss.detach()))])
        loaded,payload=load_motor(path)
        checks['saved_state_reload_bitexact']=all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
        restored=torch.optim.Adam([loaded.command_encoder.weight],lr=.001)
        restored.load_state_dict(payload['optimizer'])
        a=opt.state_dict(); b=restored.state_dict()
        checks['optimizer_moments_reload_exact']=a['param_groups']==b['param_groups'] and all(torch.equal(v,b['state'][k][name]) for k,state in a['state'].items() for name,v in state.items())
        with torch.no_grad(): after,_=loaded.sequence(obs,commands)
        checks['reloaded_inference_bitexact']=torch.equal(after,shifted)
    behavior=OdorBehavior()
    z=torch.tensor([[.1,.8]],dtype=torch.float64)
    bstate=behavior.initial_state(1)
    command,newstate=behavior.step(z,bstate)
    checks['behavior_neutral_initialization']=torch.equal(command,torch.zeros_like(command))
    checks['behavior_memory_driven_by_receptors']=not torch.equal(newstate,bstate)
    failed=False
    try: behavior.step(torch.zeros((1,3),dtype=torch.float64),bstate)
    except ValueError: failed=True
    checks['behavior_rejects_extra_target_channel']=failed
    report=dict(all_passed=all(checks.values()),checks=checks,baseline_checkpoint=str(CHECKPOINT),source_arrays=str(source),source_arrays_sha256=sha256(source),source_hashes={str(HERE/name):sha256(HERE/name) for name in ('conditional_motor.py','check_conditional.py','PREREG.json')},physical_steps=0,interpretation='offline neural interface and optimizer verification only; turning/food approach not established')
    (HERE/'CONDITIONAL_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_passed']: raise RuntimeError('conditional checks failed')

if __name__=='__main__':main()
