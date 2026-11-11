"""Fresh-process actual weighted candidate state audit, no updates or simulation."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from model_io import load_checked,load_parent,exact,sha256
from data import prepare,cached,invariance,objective


def verify(path,sha):
 torch.set_num_threads(1);model,p=load_checked(path,sha,require_final=False);parent,pp=load_parent();step=p['curve'][-1]['step'];checks={}
 checks['curve1_to_global_contiguous']=[r['step'] for r in p['curve']]==list(range(1,step+1))
 checks['parent2000_curve_prefix_exact']=exact(p['curve'][:2000],pp['curve'])
 checks['saved_model_reload_exact']=exact(model.state_dict(),p['state_dict'])
 checks['all194878_frozen_weights_exact']=exact(model.parent.state_dict(),parent.parent.state_dict())
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 checks['actual_optimizer_reload_exact']=exact(opt.state_dict(),p['optimizer'])
 checks['two_Adam_states_and_parameter_order']=len(opt.state)==2 and all(a is b for a,b in zip(opt.param_groups[0]['params'],model.adapter_parameters()))
 checks['global_counters']=all(float(s['step'])==step for s in opt.state.values())
 checks['finite_nonnegative_moments']=all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all() for s in opt.state.values())
 checks['moment_shapes']=all(s['exp_avg'].shape==q.shape and s['exp_avg_sq'].shape==q.shape for q,s in opt.state.items())
 checks['fixed_optimizer_hyperparameters']=len(opt.param_groups)==1 and opt.param_groups[0]['lr']==.001 and opt.param_groups[0]['betas']==(.9,.999) and opt.param_groups[0]['weight_decay']==0 and opt.param_groups[0]['eps']==1e-8
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 checks['rng_restore_roundtrip']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 checks.update(invariance(model))
 e,hashes,alignment=prepare(model)
 checks['same_before_action_load_alignment']=alignment==p['config']['alignment']
 checks['fixed_weight_mask_hash']=sha256(p['config']['weight_sidecar'])==p['config']['weight_sidecar_sha256']
 with np.load(p['config']['weight_sidecar'],allow_pickle=False) as w:
  checks['actual_fixed_weights_equal']=np.array_equal(w['weights'],e['weights'].numpy()) and np.array_equal(w['before_indices'],e['before_indices'])
 with torch.no_grad():means,h=model.sequence(e['x'][:,None],e['u'][:,None]);prediction=cached(model,e)
 checks['left_full_production_cache_tol1e12']=float((means[:,0]-prediction).abs().max())<=1e-12
 checks['left_frozen_dual_states_exact']=torch.equal(h[0][:,0],e['base_hidden']) and torch.equal(h[1][:,0],e['residual_hidden'])
 checks['checkpoint_unchanged']=sha256(path)==sha
 if not all(checks.values()):raise ValueError('saved state verification failed '+json.dumps(checks))
 return dict(all_passed=True,checks=checks,step=step,checkpoint=str(Path(path).resolve()),sha256=sha,weighted_postmetric=float(objective(prediction,e).detach()),uniform_postmetric=float(objective(prediction,e,False).detach()),physical_steps=0,optimizer_updates=0)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out',required=True);args=a.parse_args();r=verify(args.checkpoint,args.sha256)
 with Path(args.out).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,step=r['step'],checks=len(r['checks']))),flush=True)
