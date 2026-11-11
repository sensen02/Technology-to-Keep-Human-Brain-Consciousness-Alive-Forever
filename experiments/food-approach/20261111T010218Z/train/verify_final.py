"""Independent real saved-state audit, no updates or physics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from model_io import HERE,load_checked,load_parent,exact,sha256
from data import invariance,prepare,cached


def main(a):
 torch.set_num_threads(1)
 model,p=load_checked(a.checkpoint,a.sha256,require_final=False)
 step=p['curve'][-1]['step'];checks={}
 parent,parent_payload=load_parent()
 checks['entire_frozen_parent_exact']=exact(model.parent.state_dict(),parent.state_dict())
 checks['parent_parameters_frozen']=all(not q.requires_grad for q in model.parent.parameters())
 checks['only2730_new_parameters']=sum(q.numel() for q in model.adapter_parameters())==2730
 checks['complete_fresh_curve']=len(p['curve'])==step and [r['step'] for r in p['curve']]==list(range(1,step+1))
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.003);opt.load_state_dict(p['optimizer'])
 checks['optimizer_roundtrip_exact']=exact(opt.state_dict(),p['optimizer'])
 checks['exactly_two_new_Adam_states']=len(opt.state)==2 and len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==2
 checks['fresh_counters_match_updates']=all(float(v['step'])==step for v in opt.state.values())
 checks['moments_finite_nonnegative']=all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() and (v['exp_avg_sq']>=0).all() for v in opt.state.values())
 checks['fixed_optimizer_hyperparameters']=opt.param_groups[0]['lr']==.003 and opt.param_groups[0]['betas']==(.9,.999) and opt.param_groups[0]['eps']==1e-8 and opt.param_groups[0]['weight_decay']==0
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 checks['RNG_exact_roundtrip']=torch.equal(torch.get_rng_state(),p['rng']['torch']) and exact(np.random.get_state(),p['rng']['numpy'])
 checks.update(invariance(model))
 e,_=prepare(model)
 with torch.no_grad():
  means,states=model.sequence(e['x'][:,None],e['u'][:,None]);vec=cached(model,e)
 err=float((means[:,0]-vec).abs().max())
 checks['full_left_cached_vs_production_tol1e12']=err<=1e-12
 checks['parent_both_left_states_exact']=torch.equal(states[0][:,0],e['base_hidden']) and torch.equal(states[1][:,0],e['residual_hidden'])
 checks['checkpoint_unchanged']=sha256(a.checkpoint)==a.sha256
 report=dict(all_passed=all(checks.values()),checks=checks,step=step,checkpoint=str(Path(a.checkpoint).resolve()),sha256=a.sha256,
             left_postprime_action_MSE=float((means[960:,0].tanh()-e['y'][960:]).square().mean()),
             output_max_error=err,physical_steps=0,optimizer_updates=0,
             source_hashes={str(HERE/n):sha256(HERE/n) for n in ('verify_final.py','model_io.py','data.py')},
             limitations=['curve starts1 notstep0; freshoptimizer scope explicit','constructorreload mayconsume RNG but no stochastic training operation inreadout loop; fulltraining replay not established','no actual physical qualification'])
 with Path(a.out).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),step=step,left_MSE=report['left_postprime_action_MSE'])))
 if not report['all_passed']:raise ValueError('saved-state audit failed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);main(p.parse_args())
