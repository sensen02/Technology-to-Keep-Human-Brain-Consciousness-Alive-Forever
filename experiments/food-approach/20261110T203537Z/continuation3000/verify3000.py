"""Independent exact continuation saved state verifier, no optimization or physics."""
from pathlib import Path
import argparse,json,sys
import torch,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,sha256,verify_hashes
from fast_data import neutral
from check_fast import equal

def main(args):
 torch.set_num_threads(1);m,p=load(args.checkpoint,args.sha256);cfg=p['config'];step=p['curve'][-1]['step']
 checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not v:raise ValueError('failed '+k)
 check('fixed_saved_steps',step in (1500,2000,2500,3000))
 check('configured_global3000',cfg['steps']==3000 and cfg['physical_eval_only_step']==3000 and cfg['lr']==.0001)
 _,parent=load(cfg['parent_checkpoint'],cfg['parent_sha256'])
 check('fullcurve_contiguous',len(p['curve'])==step+1 and [r['step'] for r in p['curve']]==list(range(step+1)))
 check('parentcurveprefix_exact',equal(p['curve'][:1001],parent['curve']))
 check('same_data_identity',cfg['data_hashes']==parent['config']['data_hashes'])
 check('old_sources_retained',all(cfg['sources'][q]==v for q,v in parent['config']['sources'].items()))
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.0001);opt.load_state_dict(p['optimizer'])
 check('optimizer_reload_exact',equal(opt.state_dict(),p['optimizer']))
 check('weights_reload_exact',equal(m.state_dict(),p['model']))
 check('one_uniform_lr',len(opt.param_groups)==1 and opt.param_groups[0]['lr']==.0001)
 check('moments_count6',len(opt.state)==6)
 check('counters_global',all(float(s['step'])==step for s in opt.state.values()))
 check('finite_moments',all(bool(torch.isfinite(s[k]).all()) for s in opt.state.values() for k in ('exp_avg','exp_avg_sq')))
 check('secondmoments_nonnegative',all(bool((s['exp_avg_sq']>=0).all()) for s in opt.state.values()))
 for k,v in neutral(m).items():check('neutral_'+k,v)
 resume_path=Path(args.checkpoint).parent/'RESUME_STATE.pt';resume=torch.load(resume_path,map_location='cpu',weights_only=False)
 check('resume_model_exact',equal(resume['model'],parent['model']))
 expected=parent['optimizer'];expected['param_groups'][0]['lr']=.0001
 check('resume_optimizer_onlylr_changed',equal(resume['optimizer'],expected))
 check('resume_rng_exact',equal(resume['rng'],parent['rng']))
 check('resume_curve_exact',equal(resume['curve'],parent['curve']))
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 check('rng_roundtrip',equal(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng']))
 check('checkpoint_unchanged',sha256(args.checkpoint)==args.sha256)
 report=dict(all_passed=all(checks.values()),checks=checks,checkpoint=str(Path(args.checkpoint).resolve()),sha256=args.sha256,step=step,optimizer_updates=0,physical_steps=0)
 with Path(args.out).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(step=step,all_passed=report['all_passed'],checks=len(checks))),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out',required=True);main(a.parse_args())
