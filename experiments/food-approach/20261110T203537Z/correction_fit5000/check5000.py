"""Authenticate corrective labels and exact six-way cached objective. Zero updates/physics."""
from pathlib import Path
import sys,json
import torch,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,sha256,verify_hashes
from fast_data import cached,neutral,metrics
from check_fast import equal
from data5000 import prepare,objective,PARENT_SHA
PARENT=ROOT/'continuation3000/TRAIN/checkpoint_003000.pt'

def main():
 torch.set_num_threads(1);m,p=load(PARENT,PARENT_SHA);eps,data,admission=prepare(m)
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.0001);opt.load_state_dict(p['optimizer'])
 checks=dict(parent_model_exact=equal(m.state_dict(),p['model']),parent_optimizer_exact=equal(opt.state_dict(),p['optimizer']),parent_counters3000=all(float(s['step'])==3000 for s in opt.state.values()),unchanged_lr=opt.param_groups[0]['lr']==.0001,six_trajectories=len(eps)==6,corrective_safety_flags_false=all(v['original_dataset_safety_admitted'] is False and v['passing_demonstration'] is False for v in admission),postprime_lengths=[len(e['x'])-960 for e in eps]==[1920]*4+[960]*2)
 maxmean=maxhidden=maxgrad=0.
 for e in eps:
  full,hidden=m.sequence(e['x'][:,None],e['u'][:,None]);fast=cached(m,e)
  maxmean=max(maxmean,float((full[:,0]-fast).abs().max().detach()));maxhidden=max(maxhidden,float((hidden[:,0]-e['base_hidden']).abs().max().detach()))
  lf=(full[960:,0].tanh()-e['y'][960:]).square().mean()/6
  lc=(fast[960:].tanh()-e['y'][960:]).square().mean()/6
  gf=torch.autograd.grad(lf,m.adapter_parameters());gc=torch.autograd.grad(lc,m.adapter_parameters())
  maxgrad=max(maxgrad,max(float((a-b).abs().max()) for a,b in zip(gf,gc)))
  del full,hidden,fast,lf,lc,gf,gc
 checks.update(cached_fullsequence_mean_error_le1e12=maxmean<=1e-12,cached_causal_hidden_exact=maxhidden==0,cached_all_adapter_gradient_error_le1e12=maxgrad<=1e-12)
 with torch.no_grad():
  per=metrics(m,eps);loss=float(objective(m,eps))
 checks['six_equal_trajectory_weight']=abs(loss-sum(per.values())/6)<1e-15
 checks['not_row_pooled']=sum(len(e['x'])-960 for e in eps)==9600
 for k,v in neutral(m).items():checks['neutral_'+k]=v
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy'])
 checks['rng_exact_resume']=equal(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 src={str(HERE/n):sha256(HERE/n) for n in ('PREREG.json','data5000.py','check5000.py')}
 report=dict(all_passed=all(checks.values()),checks=checks,original_safety_admission_not_changed=True,corrective_examples=admission,corrective_total_loss_weight=1/3,per_trajectory_weight=1/6,initial_new_sixway_metrics=per,initial_new_objective=loss,cache_mean_error=maxmean,cache_hidden_error=maxhidden,cache_gradient_error=maxgrad,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,source_hashes=src,data_hashes=data,physical_steps=0,optimizer_updates=0)
 (HERE/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),mean_error=maxmean,gradient_error=maxgrad,admissions=admission,initial_MSE=per)))
 if not report['all_passed']:raise ValueError('corrective preflight failed')
if __name__=='__main__':main()
