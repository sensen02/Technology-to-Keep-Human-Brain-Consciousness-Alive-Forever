"""CODE PREPARED ONLY: explicit parent-resource-release flag required to execute."""
import argparse,json,time,os,hashlib
from copy import deepcopy
import numpy as np,torch
from model_io import *
def dump(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def preflight(m,c):
 a=deepcopy(m);b=deepcopy(m);checks={};errors={}
 def close(n,u,v):
  errors[n]=float((u-v).abs().max());checks[n]=errors[n]<=1e-12
 initial=frozen(m);pa,sa=a.sequence(c['x'],c['u']);pb=cached_forward(b,c)
 close('production_cached_means',pa,pb);close('production_cached_hidden',sa[1],c['hidden']);close('production_cached_basehidden',sa[0],c['base_states'])
 la=(pa[960:].tanh()-c['y'][960:]).square().mean();lb=(pb[960:].tanh()-c['y'][960:]).square().mean();la.backward();lb.backward()
 for n in ('weight','bias'):close('head_gradient_'+n,getattr(a.head,n).grad,getattr(b.head,n).grad)
 checks['all_frozen_no_grad']=all(not p.requires_grad and p.grad is None for n,p in a.named_parameters() if not n.startswith('head.'))
 oa=torch.optim.Adam(a.adapter_parameters(),lr=.001);ob=torch.optim.Adam(b.adapter_parameters(),lr=.001);checks['freshAdam_no_state']=len(oa.state)==0 and len(ob.state)==0;oa.step();ob.step()
 for n,v in a.state_dict().items():close('unit_model_'+n,v,b.state_dict()[n])
 for i,(u,v) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in u:close('unit_Adam_'+str(i)+'_'+k,u[k],v[k])
 checks['frozen_exact']=exact(frozen(a),initial) and exact(frozen(b),initial)
 parent,p=old.load_checked(PARENT,PARENT_SHA);checks['initial_parent_allweights_exact']=exact(m.state_dict(),parent.state_dict());checks['initial_parent_actualmeans_exact']=torch.equal(c['means'],p['replay']['means']);checks['initial_parent_actualhidden_exact']=torch.equal(c['hidden'],p['replay']['residual_states'])
 checks.update({'neutral_'+k:v for k,v in neutral(m).items()});checks['2730_head_parameters']=sum(p.numel() for p in m.adapter_parameters())==2730
 r=dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,tolerance=1e-12,objective='same uniform post960 action MSE',physics_steps=0)
 dump(HERE/'PREFLIGHT.json',r)
 if not r['all_passed']:raise ValueError('head-only preflight failed')
 return r

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parent-resource-released',action='store_true');a=parser.parse_args()
 if not a.parent_resource_released:raise ValueError('CODE_ONLY until explicit parent releases heavy resource')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 if (HERE/'TRAIN').exists():raise ValueError('do not overwrite or blindly restart')
 m=initialize();c=prepare(m);preflight(m,c)
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();c=prepare(m);opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);frozen_start=frozen(m)
 paths=[HERE/n for n in ('model_io.py','train.py','verify_saved.py','PREREG.json')]+[PARENT_DIR/'model.py',old.OLD,old.ENGINE,BASE/'models.py']
 sources={str(p):sha256(p) for p in paths};data={str(p):sha256(p) for p in (PARENT,DATA,REFERENCE,old.legacy.CHECKPOINT)}
 config=dict(kind='uniform_head_only_fixed4000',physical_eval_only_step=4000,source_hashes=sources,data_hashes=data,head_parameters=2730,Adam_states=2,lr=.001,seed=20261111,loss='uniform post960 action MSE',hidden_cache_sha256=hashlib.sha256(c['hidden'].numpy().tobytes()).hexdigest(),base_cache_sha256=hashlib.sha256(c['base'].numpy().tobytes()).hexdigest(),no_old_Adam=True)
 out=HERE/'TRAIN';out.mkdir();dump(out/'RUNTIME.json',dict(pid=os.getpid(),python=sys.executable,affinity=list(os.sched_getaffinity(0)),threads=torch.get_num_threads(),config=config));curve=[];begun=time.monotonic()
 with (out/'CURVE.jsonl').open('x') as f:
  for step in range(1,4001):
   opt.zero_grad(set_to_none=True);pred=cached_forward(m,c);loss=(pred[960:].tanh()-c['y'][960:]).square().mean();loss.backward()
   norm=torch.sqrt(sum(p.grad.square().sum() for p in m.adapter_parameters()))
   if not torch.isfinite(norm) or any(p.grad is not None for n,p in m.named_parameters() if not n.startswith('head.')):raise ValueError('invalid/frozen gradient')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(norm),wall_s=time.monotonic()-begun)
   if step%500==0 or step==1:
    with torch.no_grad():after=cached_forward(m,c);row['action_MSE_after']=float((after[960:].tanh()-c['y'][960:]).square().mean())
    print(json.dumps(row),flush=True)
   curve.append(row);f.write(json.dumps(row)+'\n');f.flush()
   if step in (2000,4000):
    verify_hashes(sources);verify_hashes(data)
    if not exact(frozen_start,frozen(m)) or not all(neutral(m).values()):raise ValueError('frozen/neutral drift')
    with torch.no_grad():means,state=m.sequence(c['x'],c['u'])
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());payload=dict(format_version='RIGHT_HEAD_REFIT_V1',identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng,replay=dict(means=means,base_states=state[0],residual_states=state[1]))
    path=out/f'checkpoint_{step:06d}.pt';tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path);digest=sha256(path);restored,p=load_checked(path,digest,require_final=False)
    checks=dict(model=exact(m.state_dict(),restored.state_dict()),Adam=exact(opt.state_dict(),p['optimizer']),curve=exact(curve,p['curve']),RNG=exact(rng,p['rng']),frozen=exact(frozen_start,frozen(restored)))
    if not all(checks.values()):raise ValueError('live saved mismatch')
    dump(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,checks=checks,step=step,sha256=digest));torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);print(json.dumps(dict(saved=str(path),sha256=digest)),flush=True)
 dump(out/'RESULT.json',dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',updates=4000,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-begun))
if __name__=='__main__':main()
