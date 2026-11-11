"""New-process saved model/Adam/RNG/curve audit and actual production replay."""
import argparse,json
import torch,numpy as np
from model import *
def main():
 a=argparse.ArgumentParser();a.add_argument('--step',type=int,required=True);args=a.parse_args();torch.set_num_threads(1)
 path=HERE/'TRAIN'/f'checkpoint_{args.step:06d}.pt';m,p=load_checked(path,sha256(path),require_final=False);checks={}
 opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);opt.load_state_dict(p['optimizer']);checks['Adam_exact_reload']=exact(opt.state_dict(),p['optimizer'])
 torch.set_rng_state(p['rng']['torch']);np.random.set_state(p['rng']['numpy']);checks['RNG_exact_reload']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),p['rng'])
 rows=[json.loads(l) for l in (HERE/'TRAIN/CURVE.jsonl').read_text().splitlines()][:args.step];checks['curve_exact']=rows==p['curve'];checks['all_steps']=list(range(1,args.step+1))==[r['step'] for r in rows]
 x,u,y=dataset()
 with torch.no_grad():means,s=m.sequence(x,u)
 checks['production_means_exact']=torch.equal(means,p['replay']['means']);checks['production_basehistory_exact']=torch.equal(s[0],p['replay']['base_states']);checks['production_residualhistory_exact']=torch.equal(s[1],p['replay']['residual_states'])
 pc=cached(m,x,u)
 with torch.no_grad():v,h=make(m)(pc['features'],pc['gate'],pc['means'],pc['valid'])
 errors=dict(scan_means=float((v-means).abs().max()),scan_states=float((h-s[1]).abs().max()))
 checks['scan_production_tolerance']=max(errors.values())<=1e-12
 from train import neutral
 checks.update({'neutral_'+k:v for k,v in neutral(m).items()});checks['base_no_grad']=all(not q.requires_grad and q.grad is None for q in m.base.parameters())
 result=dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,step=args.step,checkpoint=str(path),sha256=sha256(path),postprime_action_MSE=float((means[960:].tanh()-y[960:]).square().mean()),physics_steps=0,optimizer_updates=0)
 (HERE/'TRAIN'/f'VERIFY_{args.step:06d}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['all_passed']:raise ValueError('saved independent audit failed')
if __name__=='__main__':main()
