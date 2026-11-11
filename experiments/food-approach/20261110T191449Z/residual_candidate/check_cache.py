"""Full saved sequence cache/production parity, zero optimizer/physics steps."""
import json
import torch
from residual_motor import HERE,ResidualMotor,load_baseline_actor,sha256
from cached_training import prepare,cached_sequence,neutral_check,SOURCE_NAMES


def main():
    torch.set_num_threads(1);torch.manual_seed(20261110)
    model=ResidualMotor(load_baseline_actor());episodes,hashes=prepare(model);checks={};details={}
    for e in episodes:
        with torch.no_grad():
            production,state=model.sequence(e['x'],e['u']);cached,rh=cached_sequence(model,e)
        checks[e['name']+'_all2880_means_exact']=torch.equal(production,cached)
        checks[e['name']+'_all2880_base_hidden_exact']=torch.equal(state[0],e['hidden'])
        checks[e['name']+'_all2880_residual_hidden_exact']=torch.equal(state[1],rh)
        # Exact full-sequence BPTT, zero optimizer updates; production base frozen.
        model.zero_grad(set_to_none=True)
        p,_=model.sequence(e['x'],e['u']);loss=(p[960:].tanh()-e['y'][960:]).square().mean();loss.backward()
        grads=[p.grad.clone() for p in model.adapter_parameters()]
        del p,loss
        model.zero_grad(set_to_none=True)
        p,_=cached_sequence(model,e);loss=(p[960:].tanh()-e['y'][960:]).square().mean();loss.backward()
        checks[e['name']+'_all_adapter_gradients_exact']=all(torch.equal(g,p.grad) for g,p in zip(grads,model.adapter_parameters()))
        checks[e['name']+'_base_gradients_absent']=all(p.grad is None for p in model.base.parameters())
        details[e['name']]=dict(rows=len(e['x']),postprime_rows=1920,loss=float(loss.detach()))
        del p,loss,production,cached,state,rh,grads
    checks['neutral2880_exact']=all(neutral_check(model).values())
    report=dict(all_passed=all(checks.values()),checks=checks,details=details,data_hashes=hashes,source_hashes={str(HERE/n):sha256(HERE/n) for n in SOURCE_NAMES},optimizer_steps=0,physical_steps=0,baseline_cache_once_per_episode=True)
    (HERE/'CACHE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if not report['all_passed']:raise ValueError('cache check failed')

if __name__=='__main__':main()
