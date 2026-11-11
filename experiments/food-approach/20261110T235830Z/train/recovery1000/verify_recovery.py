"""Independent durability/final checkpoint audit; no optimization or physics."""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT))
from model_io import legacy,ResidualMotor,verify_hashes,sha256,exact
from data import neutral
P1000=ROOT/'TRAIN/checkpoint_001000.pt'
SHA1000='47a84f5058654f76624f897fa1d90580b2733739df64e99c234a25468260cd59'

def verify(path,digest):
 torch.set_num_threads(1)
 if sha256(path)!=digest or sha256(P1000)!=SHA1000:raise ValueError('checkpoint digest mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False);parent=torch.load(P1000,map_location='cpu',weights_only=False)
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 model=ResidualMotor(legacy.load_baseline_actor());model.load_state_dict(p['state_dict'],strict=True)
 if model.identity()!=p['identity']:raise ValueError('identity mismatch')
 opt=torch.optim.Adam(model.adapter_parameters(),lr=.0003);opt.load_state_dict(p['optimizer'])
 step=p['curve'][-1]['step'];resume=torch.load(HERE/'RESUME_STATE.pt',map_location='cpu',weights_only=False)
 checks=dict(fixed_snapshot=step in (1100,1200,1300,1400,1500),fullcurve=[r['step'] for r in p['curve']]==list(range(step+1)),parent1000prefixexact=exact(p['curve'][:1001],parent['curve']),model_reload=exact(model.state_dict(),p['state_dict']),optimizer_reload=exact(opt.state_dict(),p['optimizer']),eight_states=len(opt.state)==8,counters_exact=all(float(s['step'])==step for s in opt.state.values()),all_moments_finite=all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq']>=0).all() for s in opt.state.values()),same_lr=opt.param_groups[0]['lr']==.0003,frozen_base=exact(model.base.state_dict(),legacy.load_baseline_actor().state_dict()),resume_model=exact(resume['state_dict'],parent['state_dict']),resume_optimizer=exact(resume['optimizer'],parent['optimizer']),resume_rng=exact(resume['rng'],parent['rng']),resume_curve=exact(resume['curve'],parent['curve']))
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['rng_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 checks.update({'neutral_'+k:v for k,v in neutral(model).items()});checks['unchanged']=sha256(path)==digest
 if not all(checks.values()):raise ValueError('recovery checkpoint audit failed '+json.dumps(checks))
 return dict(all_passed=True,step=step,checks=checks,checkpoint=str(Path(path).resolve()),sha256=digest,physical_steps=0,optimizer_updates=0)

def load_durability(path,digest,*,for_recovery=False):
 """Explicit recovery-only loader for1100..1400, never deployment selection."""
 if not for_recovery:raise ValueError('durability snapshots require explicit recovery-only scope')
 verify(path,digest)
 p=torch.load(path,map_location='cpu',weights_only=False)
 model=ResidualMotor(legacy.load_baseline_actor());model.load_state_dict(p['state_dict'],strict=True)
 return model,p

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args();report=verify(a.checkpoint,a.sha256)
 with Path(a.out).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(step=report['step'],all_passed=True,checks=len(report['checks']))),flush=True)
