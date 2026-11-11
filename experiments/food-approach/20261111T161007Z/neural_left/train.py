"""Fixed LEFT head-only training. Execution requires explicit parent resource release."""
import argparse,os,time,json,hashlib,subprocess
from copy import deepcopy
from model_io import *

def dump(path,value):
 path=Path(path)
 if path.exists():raise FileExistsError(path)
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def preflight(m,c):
 a=deepcopy(m);b=deepcopy(m);checks={};errors={};initial=frozen(m)
 def close(name,x,y):errors[name]=float((x-y).abs().max());checks[name]=errors[name]<=1e-12
 pa,sa=a.sequence(c['x'],c['u']);pb=cached_forward(b,c)
 close('production_cached_means',pa,pb);close('production_cached_hidden',sa[1],c['hidden']);close('production_cached_basehidden',sa[0],c['base_states'])
 checks['zero_head_all_actual_commands_exact_base']=torch.equal(pa,c['base']) and all(torch.count_nonzero(p)==0 for p in m.head.parameters())
 la=(pa[960:].tanh()-c['y'][960:]).square().mean();lb=(pb[960:].tanh()-c['y'][960:]).square().mean();la.backward();lb.backward()
 for n in ('weight','bias'):close('head_gradient_'+n,getattr(a.head,n).grad,getattr(b.head,n).grad)
 checks['all_frozen_no_grad']=all(not p.requires_grad and p.grad is None for n,p in a.named_parameters() if not n.startswith('head.'))
 oa=torch.optim.Adam(a.adapter_parameters(),lr=LR);ob=torch.optim.Adam(b.adapter_parameters(),lr=LR)
 checks['fresh_Adam_no_state']=len(oa.state)==0 and len(ob.state)==0;oa.step();ob.step()
 for n,v in a.state_dict().items():close('unit_model_'+n,v,b.state_dict()[n])
 for i,(x,y) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in x:close('unit_Adam_'+str(i)+'_'+k,x[k],y[k])
 checks['frozen_exact']=exact(frozen(a),initial) and exact(frozen(b),initial)
 checks['2730_head_only']=sum(p.numel() for p in m.parameters() if p.requires_grad)==2730
 checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 # Check off-manifold commands are rejected, rather than silently clipping or inheriting RIGHT domain.
 for name,u in [('negative_turn',[0.,-1.]),('uncoupled',[0.,1.]),('forward_too_high',[.34,1.])]:
  try:m.step(c['x'][0],torch.tensor([u],dtype=torch.float64),m.initial_state(1))
  except ValueError:checks['reject_'+name]=True
  else:checks['reject_'+name]=False
 r=dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,tolerance=1e-12,physics_steps=0)
 dump(HERE/'PREFLIGHT.json',r)
 if not r['all_passed']:raise ValueError('LEFT head preflight failed')
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('CODEONLY until parent releases heavy resource')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if (HERE/'TRAIN').exists():raise FileExistsError('no overwrite or automatic restart')
 code=json.loads((HERE/'CODEONLY.json').read_text());verify_hashes(code['source_hashes']);verify_hashes(code['evidence_hashes'])
 if not code['all_static_checks_passed']:raise ValueError('static CODEONLY readiness missing')
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();c=prepare(m);preflight(m,c)
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();c=prepare(m);opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);start=frozen(m)
 sources={str(v):sha256(v) for v in source_paths()};data={str(v):sha256(v) for v in data_paths()}
 config=dict(kind=FORMAT,physical_eval_only_step=4000,source_hashes=sources,data_hashes=data,head_parameters=2730,Adam_states=2,lr=LR,seed=20261111,loss='uniform1920postprime42actionMSE',hidden_cache_sha256=hashlib.sha256(c['hidden'].numpy().tobytes()).hexdigest(),base_cache_sha256=hashlib.sha256(c['base'].numpy().tobytes()).hexdigest(),transfer_Adam_used=False)
 out=HERE/'TRAIN';out.mkdir();dump(out/'IDENTITY.json',dict(config=config,model_identity=m.identity(),pid=os.getpid(),python=sys.executable,affinity=list(os.sched_getaffinity(0))))
 curve=[];begun=time.monotonic()
 with (out/'CURVE.jsonl').open('x') as stream:
  for step in range(1,4001):
   opt.zero_grad(set_to_none=True);pred=cached_forward(m,c);loss=(pred[960:].tanh()-c['y'][960:]).square().mean();loss.backward()
   norm=torch.sqrt(sum(p.grad.square().sum() for p in m.adapter_parameters()))
   if not torch.isfinite(norm) or any(p.grad is not None for n,p in m.named_parameters() if not n.startswith('head.')):raise ValueError('nonfinite/frozen gradient')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(norm),wall_s=time.monotonic()-begun)
   if step%500==0 or step==1:
    with torch.no_grad():row['action_MSE_after']=float((cached_forward(m,c)[960:].tanh()-c['y'][960:]).square().mean())
    print(json.dumps(row),flush=True)
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step in (2000,4000):
    verify_hashes(sources);verify_hashes(data)
    if not exact(start,frozen(m)) or not all(neutral(m).values()):raise ValueError('frozen/neutral drift')
    with torch.no_grad():means,state=m.sequence(c['x'],c['u'])
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    payload=dict(format_version=FORMAT,identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng,replay=dict(means=means,base_states=state[0],residual_states=state[1]))
    path=out/f'checkpoint_{step:06d}.pt'
    if path.exists():raise FileExistsError(path)
    tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path);digest=sha256(path);fresh,saved=load_checked(path,digest,require_final=False)
    checks=dict(model=exact(m.state_dict(),fresh.state_dict()),Adam=exact(opt.state_dict(),saved['optimizer']),curve=exact(curve,saved['curve']),RNG=exact(rng,saved['rng']),frozen=exact(start,frozen(fresh)))
    if not all(checks.values()):raise ValueError('live saved model/Adam/RNG/curve mismatch')
    dump(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,checks=checks,step=step,sha256=digest))
    # Loader initializes modules; always restore the live saved RNG afterwards.
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--step',str(step),'--parent-resource-released'],check=True)
    print(json.dumps(dict(saved=str(path),sha256=digest)),flush=True)
 dump(out/'RESULT.json',dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',updates=4000,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-begun))
if __name__=='__main__':main()
