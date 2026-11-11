"""Independent saved fast-residual state audit, never simulate or update."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from fast_motor import load,load_base,sha256
from fast_data import neutral
from check_fast import equal


def verify(path,expected):
    torch.set_num_threads(1);model,p=load(path,expected)
    step=p['curve'][-1]['step'];checks={}
    checks['complete_curve']=len(p['curve'])==step+1 and [r['step'] for r in p['curve']]==list(range(step+1))
    checks['all_weights_reload_exact']=equal(model.state_dict(),p['model'])
    checks['frozen_base_exact']=equal(model.base.state_dict(),load_base().state_dict())
    opt=torch.optim.Adam(model.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
    checks['actual_optimizer_reload_exact']=equal(opt.state_dict(),p['optimizer'])
    checks['coverage_once']=len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==len(model.adapter_parameters()) and len(opt.state)==len(model.adapter_parameters())
    checks['all_adam_counters']=all(float(v['step'])==step for v in opt.state.values())
    checks['all_moments_finite_nonnegative']=all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() and (v['exp_avg_sq']>=0).all() for v in opt.state.values())
    checks['hyperparameters']=opt.param_groups[0]['lr']==.001 and opt.param_groups[0]['betas']==(.9,.999) and opt.param_groups[0]['eps']==1e-8 and opt.param_groups[0]['weight_decay']==0
    torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
    checks['rng_exact_roundtrip']=torch.equal(torch.get_rng_state(),p['rng']['torch']) and equal(np.random.get_state(),p['rng']['numpy'])
    checks['finite_full_curve']=all(math.isfinite(row['wall_s']) and all(math.isfinite(v) for v in row.get('postprime_MSE',{}).values()) for row in p['curve'])
    checks.update({'neutral_'+key:value for key,value in neutral(model).items()})
    checks['checkpoint_unchanged']=sha256(path)==expected
    if not all(checks.values()):raise ValueError('saved-state verification failed '+json.dumps(checks))
    return dict(all_passed=True,step=step,checks=checks,checkpoint=str(Path(path).resolve()),sha256=expected,optimizer_updates=0,physical_steps=0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out');args=a.parse_args()
    report=verify(args.checkpoint,args.sha256)
    if args.out:
        with Path(args.out).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
