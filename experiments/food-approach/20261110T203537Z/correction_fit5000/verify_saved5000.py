"""Independent saved corrective continuation audit; no updates or physics."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,sha256
from fast_data import neutral
from check_fast import equal

def main(args):
 torch.set_num_threads(1);m,p=load(args.checkpoint,args.sha256);cfg=p['config'];step=p['curve'][-1]['step'];checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not v:raise ValueError('failed '+k)
 check('fixed4000or5000',step in (4000,5000));check('configured5000onlyeval',cfg['steps']==5000 and cfg['physical_eval_only_step']==5000 and cfg['lr']==.0001)
 _,parent=load(cfg['parent_checkpoint'],cfg['parent_sha256'])
 check('parent3000',parent['curve'][-1]['step']==3000 and cfg['parent_sha256']=='17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8')
 check('completecurve',len(p['curve'])==step+1 and [r['step'] for r in p['curve']]==list(range(step+1)))
 check('parentcurveprefixexact',equal(p['curve'][:3001],parent['curve']))
 check('parentoldsourcehashesretained',all(cfg['sources'][q]==v for q,v in parent['config']['sources'].items()))
 check('parentolddatahashesretained',all(cfg['data_hashes'][q]==v for q,v in parent['config']['data_hashes'].items()))
 check('correctiveexplicitfailedobservations',len(cfg['corrective_examples'])==2 and all(v['original_dataset_safety_admitted'] is False and v['passing_demonstration'] is False and v['role']=='FAILED_OBSERVATIONS_VALID_TEACHER_LABELS_ONLY' for v in cfg['corrective_examples']))
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.0001);opt.load_state_dict(p['optimizer'])
 check('modelreloadexact',equal(m.state_dict(),p['model']));check('optimizerreloadexact',equal(opt.state_dict(),p['optimizer']))
 check('sixAdamstates',len(opt.state)==6);check('uniformunchangedlr',len(opt.param_groups)==1 and opt.param_groups[0]['lr']==.0001)
 check('countersglobal',all(float(s['step'])==step for s in opt.state.values()))
 check('momentsfinite',all(bool(torch.isfinite(s[k]).all()) for s in opt.state.values() for k in ('exp_avg','exp_avg_sq')))
 check('secondmomentsnonnegative',all(bool((s['exp_avg_sq']>=0).all()) for s in opt.state.values()))
 resume=torch.load(Path(args.checkpoint).parent/'RESUME_STATE.pt',map_location='cpu',weights_only=False)
 for key,oldkey in (('model','model'),('optimizer','optimizer'),('curve','curve'),('rng','rng')):check('exact3000resume_'+key,equal(resume[key],parent[oldkey]))
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 check('rngroundtrip',equal(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng']))
 for k,v in neutral(m).items():check('neutral_'+k,v)
 check('checkpointunchanged',sha256(args.checkpoint)==args.sha256)
 report=dict(all_passed=all(checks.values()),checks=checks,step=step,checkpoint=str(Path(args.checkpoint).resolve()),sha256=args.sha256,original_safety_flags_remain_false=True,physical_steps=0,optimizer_updates=0)
 with Path(args.out).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(step=step,checks=len(checks),all_passed=report['all_passed'])),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--sha256',required=True);a.add_argument('--out',required=True);main(a.parse_args())
