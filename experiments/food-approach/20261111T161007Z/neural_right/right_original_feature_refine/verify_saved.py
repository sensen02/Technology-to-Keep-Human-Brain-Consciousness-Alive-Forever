"""Fresh-process original-only fullfeature ALL8 saved production/scan/Adam/RNG audit."""
import argparse,json,hashlib
import torch,numpy as np
from model_io import *
def main():
 a=argparse.ArgumentParser();a.add_argument('--step',type=int,choices=(1000,2000),required=True);args=a.parse_args();torch.set_num_threads(1);path=HERE/'TRAIN'/f'checkpoint_{args.step:06d}.pt';m,p=load_checked(path,sha256(path),require_final=False);opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);opt.load_state_dict(p['optimizer']);checks=dict(Adam_exact=exact(opt.state_dict(),p['optimizer']),freshALL8_statecounters=len(opt.state)==8 and all(float(s['step'])==args.step for s in opt.state.values()))
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['RNG_exact']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng']);rows=[json.loads(l) for l in (HERE/'TRAIN/CURVE.jsonl').read_text().splitlines()][:args.step];checks['curve_exact']=exact(rows,p['curve']);checks['all_steps']=list(range(1,args.step+1))==[r['step'] for r in rows]
 x,u,y=dataset()
 with torch.no_grad():means,h=m.sequence(x,u)
 checks['productionmeans_exact']=torch.equal(means,p['replay']['means']);checks['productionbasehidden_exact']=torch.equal(h[0],p['replay']['base_states']);checks['productionresidualhidden_exact']=torch.equal(h[1],p['replay']['residual_states']);pc=cached(m,x,u)
 with torch.no_grad():scan,sh=make(m)(pc['features'],pc['gate'],pc['means'],pc['valid'])
 errors=dict(means=float((scan-means).abs().max()),hidden=float((sh-h[1]).abs().max()));checks['scan_tolerance']=max(errors.values())<=1e-12;checks['base_cache_SHA']=hashlib.sha256(pc['means'].numpy().tobytes()).hexdigest()==p['config']['base_cache_sha256'];checks['qualified_only_loss']=p['config']['training_episode_count']==1 and p['config']['training_dataset']==str(DATA) and p['config']['actual_stream_not_in_loss'];checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 result=dict(all_passed=all(checks.values()),checks=checks,step=args.step,checkpoint=str(path),sha256=sha256(path),max_errors=errors,postprime_action_MSE=float((means[960:].tanh()-y[960:]).square().mean()),physics_steps=0,optimizer_updates=0)
 out=HERE/'TRAIN'/f'VERIFY_{args.step:06d}.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['all_passed']:raise ValueError('originalRIGHTfeature saved audit failed')
if __name__=='__main__':main()
