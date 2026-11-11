"""Fresh saved LH455 model/Adam2/cache/row-preservation audit, zero physics."""
import argparse,json,hashlib
import torch,numpy as np
from model_io import *
def main():
 a=argparse.ArgumentParser();a.add_argument('--step',type=int,choices=(2000,4000),required=True);args=a.parse_args();torch.set_num_threads(1);path=HERE/'TRAIN'/f'checkpoint_{args.step:06d}.pt';m,p=load_checked(path,sha256(path),require_final=False)
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer']);checks=dict(Adam_exact=exact(opt.state_dict(),p['optimizer']),Adam2_LH455=len(opt.state)==2 and sum(q.numel() for q in m.adapter_parameters())==455)
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['RNG_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 rows=[json.loads(l) for l in (HERE/'TRAIN/CURVE.jsonl').read_text().splitlines()][:args.step];checks['curve_exact']=exact(rows,p['curve']);checks['all_steps']=list(range(1,args.step+1))==[v['step'] for v in rows]
 c=prepare(m);checks['production_means_exact']=torch.equal(c['means'],p['replay']['means']);checks['production_basehidden_exact']=torch.equal(c['base_states'],p['replay']['base_states']);checks['production_residualhidden_exact']=torch.equal(c['hidden'],p['replay']['residual_states'])
 with torch.no_grad():pred=cached_forward(m,c)
 error=float((pred-c['means']).abs().max());checks['cached_production_tolerance']=error<=1e-12;checks['hidden_cache_sha']=hashlib.sha256(c['hidden'].numpy().tobytes()).hexdigest()==p['config']['hidden_cache_sha256'];checks['loss_weight_sha']=hashlib.sha256(c['weights'].numpy().tobytes()).hexdigest()==p['config']['weights_sha256'];checks['preaction_mask']=c['time_error']<1e-8 and c['preaction_indices'][960]==5799
 parent,pp=old.load_checked(PARENT,PARENT_SHA);other=list(range(7))+list(range(14,42));checks['actual_nonLH_parent_outputs_exact']=torch.equal(c['means'][:,:,other],pp['replay']['means'][:,:,other]);checks['frozen_state_exact']=exact(frozen(m),frozen(initialize()));checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 r=dict(all_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},step=args.step,checkpoint=str(path),sha256=sha256(path),max_cached_production_error=error,weighted_LH_action_MSE=float(loss(c['means'],c)),uniform_all42_action_MSE=float((c['means'][960:].tanh()-c['y'][960:]).square().mean()),physics_steps=0,optimizer_updates=0)
 (HERE/'TRAIN'/f'VERIFY_{args.step:06d}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 if not r['all_passed']:raise ValueError('LH455 saved independent audit failed')
if __name__=='__main__':main()
