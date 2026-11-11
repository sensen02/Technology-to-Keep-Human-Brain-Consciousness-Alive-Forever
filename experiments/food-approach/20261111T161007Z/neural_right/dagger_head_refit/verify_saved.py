"""Fresh-process dense2730 two-source saved model/Adam/RNG/full production audit."""
import argparse,json,hashlib
import torch,numpy as np
from model_io import *
def main():
 a=argparse.ArgumentParser();a.add_argument('--step',type=int,choices=(2000,4000),required=True);args=a.parse_args();torch.set_num_threads(1);path=HERE/'TRAIN'/f'checkpoint_{args.step:06d}.pt';m,p=load_checked(path,sha256(path),require_final=False)
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.0003);opt.load_state_dict(p['optimizer']);checks=dict(Adam_exact=exact(opt.state_dict(),p['optimizer']),Adam2_dense2730=len(opt.state)==2 and sum(q.numel() for q in m.adapter_parameters())==2730)
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['RNG_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng']);rows=[json.loads(l) for l in (HERE/'TRAIN/CURVE.jsonl').read_text().splitlines()][:args.step];checks['curve_exact']=exact(rows,p['curve']);checks['all_steps']=list(range(1,args.step+1))==[v['step'] for v in rows]
 c=prepare(m);checks['two_source_production_means_exact']=torch.equal(c['means'],p['replay']['means']);checks['two_source_production_basehidden_exact']=torch.equal(c['base_states'],p['replay']['base_states']);checks['two_source_production_residualhidden_exact']=torch.equal(c['hidden'],p['replay']['residual_states'])
 with torch.no_grad():pred=cached_forward(m,c)
 error=float((pred-c['means']).abs().max());checks['cache_production_tolerance']=error<=1e-12;checks['cache_SHA']=hashlib.sha256(c['hidden'].numpy().tobytes()).hexdigest()==p['config']['hidden_cache_sha256'];checks['equal_two_episodes']=p['config']['episode_weights']==[.5,.5] and p['config']['post_counts']==[1920,1920];checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 result=dict(all_passed=all(checks.values()),checks=checks,step=args.step,checkpoint=str(path),sha256=sha256(path),max_cached_production_error=error,equal_objective=float(objective(c['means'],c)),episode_MSE={e['name']:float((c['means'][960:,j].tanh()-c['y'][960:,j]).square().mean()) for j,e in enumerate(c['episodes'])},physics_steps=0,optimizer_updates=0)
 (HERE/'TRAIN'/f'VERIFY_{args.step:06d}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['all_passed']:raise ValueError('two-source saved densehead audit failed')
if __name__=='__main__':main()
