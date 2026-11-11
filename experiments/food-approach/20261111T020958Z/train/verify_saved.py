"""Fresh-process actual saved-state audit; no updates or physical construction."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from model_io import load_checked,load_parent,exact,sha256
from data import invariance,prepare,cached

def verify(path,digest):
 torch.set_num_threads(1);m,p=load_checked(path,digest,require_final=False);step=p['curve'][-1]['step'];checks={}
 checks['curve0_through_global']=len(p['curve'])==step+1 and [r['step'] for r in p['curve']]==list(range(step+1))
 checks['actual_model_reload_exact']=exact(m.state_dict(),p['state_dict'])
 parent,_=load_parent();checks['all197608_frozen_parent_exact']=exact(m.parent.state_dict(),parent.state_dict()) and all(not q.requires_grad for q in m.parent.parameters())
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 checks['optimizer_reload_exact']=exact(opt.state_dict(),p['optimizer'])
 checks['four_states_and_coverage']=len(opt.state)==4 and len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==4
 checks['counters']=all(float(s['step'])==step for s in opt.state.values())
 checks['moments_finite_nonnegative']=all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all() for s in opt.state.values())
 checks['moment_shapes']=all(s['exp_avg'].shape==q.shape and s['exp_avg_sq'].shape==q.shape for q,s in opt.state.items())
 checks['hyperparameters']=opt.param_groups[0]['lr']==.001 and opt.param_groups[0]['betas']==(.9,.999) and opt.param_groups[0]['eps']==1e-8
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['rng_restore_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 checks.update(invariance(m));e,_=prepare(m)
 with torch.no_grad():a,state=m.sequence(e['x'][:,None],e['u'][:,None]);b=cached(m,e)
 checks['actual_left_cache_production_tol']=float((a[:,0]-b).abs().max())<=1e-12
 checks['left_frozen_parent_states_exact']=torch.equal(state[0][:,0],e['bh']) and torch.equal(state[1][:,0],e['rh'])
 checks['checkpoint_unchanged']=sha256(path)==digest
 if not all(checks.values()):raise ValueError('saved audit failed '+json.dumps(checks))
 return dict(all_passed=True,checks=checks,step=step,checkpoint=str(path),sha256=digest,physical_steps=0,optimizer_updates=0)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out',required=True);args=a.parse_args();r=verify(args.checkpoint,args.sha256)
 with Path(args.out).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,step=r['step'],checks=len(r['checks']))))
