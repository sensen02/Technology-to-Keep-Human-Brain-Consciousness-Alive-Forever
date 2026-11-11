"""Fresh process strict refinement8Adam/RNG/curve and actual production replay audit."""
import argparse,json,hashlib
from model_io import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=(1000,2000),required=True);p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parentresource release required')
 torch.set_num_threads(1);path=HERE/'TRAIN'/f'checkpoint_{args.step:06d}.pt';out=HERE/'TRAIN'/f'VERIFY_{args.step:06d}.json'
 if out.exists():raise FileExistsError(out)
 m,payload=load_checked(path,sha256(path),require_final=False);opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);opt.load_state_dict(payload['optimizer'])
 checks=dict(Adam_exact=exact(opt.state_dict(),payload['optimizer']),Adam8states=len(opt.state)==8,base_no_grad=all(not p.requires_grad and p.grad is None for p in m.base.parameters()))
 torch.set_rng_state(payload['rng']['torch']);np.random.set_state(payload['rng']['numpy']);checks['RNG_exact_after_constructor']=exact(dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),payload['rng'])
 curve=[json.loads(v) for v in (HERE/'TRAIN/CURVE.jsonl').read_text().splitlines()][:args.step];checks['curve_exact']=exact(curve,payload['curve']);checks['new_steps1toN']=list(range(1,args.step+1))==[r['step'] for r in curve]
 x,u,y=dataset()
 with torch.no_grad():means,state=m.sequence(x,u)
 for name,v in [('means',means),('base_states',state[0]),('residual_states',state[1])]:checks['actualproduction_'+name+'_exact']=torch.equal(v,payload['replay'][name])
 pc=cached(m,x,u);scan=make(m)
 with torch.no_grad():pred,history=scan(pc['features'],pc['gate'],pc['means'],pc['valid'])
 error=float((pred-means).abs().max());herror=float((history-state[1]).abs().max());checks['scan_production_tolerance']=error<=1e-12 and herror<=1e-12
 checks['frozen_base_cache_hash']=hashlib.sha256(pc['means'].numpy().tobytes()).hexdigest()==payload['config']['base_cache_sha256'];checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 checks['43242_unfrozen_8tensors']=sum(v.numel() for v in m.parameters() if v.requires_grad)==43242 and len(m.adapter_parameters())==8
 r=dict(all_passed=all(checks.values()),checks=checks,step=args.step,sha256=sha256(path),postprime_action_MSE=float((means[960:].tanh()-y[960:]).square().mean()),production_scan_maxerror=error,production_scan_hidden_maxerror=herror,physics_steps=0,optimizer_updates=0)
 out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r),flush=True)
 if not r['all_passed']:raise ValueError('refinement saved audit failed')
if __name__=='__main__':main()
