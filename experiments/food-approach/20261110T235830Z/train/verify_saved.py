"""Fresh process saved weights/Adam/RNG/curve/inference verification, no updates."""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from model_io import load_checked,load_parent,exact,sha256
from data import neutral

def verify(path,digest):
 torch.set_num_threads(1);model,p=load_checked(path,digest,require_final=False);step=p['curve'][-1]['step'];parent_model,parent=load_parent()
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.0003);opt.load_state_dict(p['optimizer'])
 checks=dict(model_reload_exact=exact(model.state_dict(),p['state_dict']),optimizer_reload_exact=exact(opt.state_dict(),p['optimizer']),eight_states=len(opt.state)==8,global_counters=all(float(s['step'])==step for s in opt.state.values()),lr_unchanged_after_declared_resume=opt.param_groups[0]['lr']==.0003,finite_moments=all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all() for s in opt.state.values()),fullcurve=[r['step'] for r in p['curve']]==list(range(step+1)),parent_prefix_exact=exact(p['curve'][:501],parent['config']['curve']),frozen_base_exact=exact(model.base.state_dict(),parent_model.base.state_dict()))
 state=torch.load(Path(path).parent/'RESUME_STATE.pt',map_location='cpu',weights_only=False)
 old=deepcopy(parent['optimizer']);old['param_groups'][0]['lr']=.0003
 checks.update(resume_model_exact=exact(state['state_dict'],parent['state_dict']),resume_optimizer_only_lr=exact(state['optimizer'],old),resume_rng_exact=exact(state['rng'],parent['config']['rng']),resume_curve_exact=exact(state['curve'],parent['config']['curve']))
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['rng_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 checks.update({'neutral_'+k:v for k,v in neutral(model).items()})
 # Saved final residual state is not deployment state; inference check uses actual source-prefix observations.
 from model_io import REFERENCE
 with np.load(REFERENCE,allow_pickle=False) as z,torch.no_grad():
  x=torch.tensor(z['X238'][:25].copy(),dtype=torch.float64);u=torch.zeros((25,2),dtype=torch.float64);u[:,1]=1
  a,ha=model.sequence(x[:,None],u[:,None]);b,hb=load_checked(path,digest,require_final=False)[0].sequence(x[:,None],u[:,None])
 checks['actual_reload_nonzero_inference_exact']=torch.equal(a,b) and exact(ha,hb)
 checks['checkpoint_unchanged']=sha256(path)==digest
 report=dict(all_passed=all(checks.values()),checks=checks,step=step,checkpoint=str(Path(path).resolve()),sha256=digest,physical_steps=0,optimizer_updates=0)
 if not report['all_passed']:raise ValueError('saved checks failed '+json.dumps(checks))
 return report
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out',required=True);p=a.parse_args();r=verify(p.checkpoint,p.sha256)
 with Path(p.out).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,step=r['step'],checks=len(r['checks']))),flush=True)
