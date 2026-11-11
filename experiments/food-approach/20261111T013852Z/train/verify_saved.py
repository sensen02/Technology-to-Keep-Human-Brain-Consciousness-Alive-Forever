"""Fresh-process actual saved states and full nonpositive-command invariance."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from model_io import load_checked,load_parent,exact,sha256
from data import invariance,prepare,cached

def verify(path,digest):
 torch.set_num_threads(1);m,p=load_checked(path,digest,require_final=False);parent,pp=load_parent();step=p['curve'][-1]['step'];checks={}
 checks['parent_curve_prefix_exact']=exact(p['curve'][:2000],pp['curve'])
 checks['weights_exact']=exact(m.state_dict(),p['state_dict'])
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer'])
 checks['optimizer_exact']=exact(opt.state_dict(),p['optimizer'])
 checks['two_Adam_states']=len(opt.state)==2
 checks['all_counters']=all(float(s['step'])==step for s in opt.state.values())
 checks['finite_moments']=all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all() for s in opt.state.values())
 checks['lr_uniform']=len(opt.param_groups)==1 and opt.param_groups[0]['lr']==.001
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 checks['rng_roundtrip']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 checks.update(invariance(m))
 e,_=prepare(m)
 with torch.no_grad():
  a,_=m.sequence(e['x'][:,None],e['u'][:,None]);b=cached(m,e)
 checks['actual_left_inference_cache_tol']=float((a[:,0]-b).abs().max())<=1e-12
 checks['frozen_parent_exact2000']=exact(m.parent.state_dict(),parent.parent.state_dict())
 if not all(checks.values()):raise ValueError('saved checks failed '+json.dumps(checks))
 return dict(all_passed=True,step=step,checks=checks,checkpoint=str(path),sha256=digest,physical_steps=0,optimizer_updates=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=verify(a.checkpoint,a.sha256)
 with Path(a.out).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,step=r['step'],checks=len(r['checks']))),flush=True)
