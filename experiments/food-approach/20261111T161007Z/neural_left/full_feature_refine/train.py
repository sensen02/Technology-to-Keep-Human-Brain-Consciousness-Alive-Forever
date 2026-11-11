"""Fixed additional2000 LEFT full adapter BPTT using shared-parameter Torchscript scan."""
import argparse,os,time,json,subprocess,hashlib
from copy import deepcopy
from model_io import *

def dump(path,value):
 path=Path(path)
 if path.exists():raise FileExistsError(path)
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def preflight(m,x,u,y):
 checks={};errors={};started=time.monotonic()
 def close(n,a,b):errors[n]=float((a-b).abs().max());checks[n]=errors[n]<=1e-12
 parent,payload=old.load_checked(PARENT,PARENT_SHA)
 checks['initial_parent_allweights_exact']=exact(m.state_dict(),parent.state_dict())
 with torch.no_grad():initial,history=m.sequence(x,u)
 for name,value in [('means',initial),('base_states',history[0]),('residual_states',history[1])]:checks['initial_parent_teacherflow_'+name]=torch.equal(value,payload['replay'][name])
 a=deepcopy(m);b=deepcopy(m);pc=cached(b,x,u);scan=make(b);base=frozen_base(m)
 pa,sa=a.sequence(x,u);pb,sb=scan(pc['features'],pc['gate'],pc['means'],pc['valid'])
 close('production_scan_means',pa,pb);close('production_scan_hidden',sa[1],sb);close('production_base_history',sa[0],pc['base_states'])
 la=(pa[960:].tanh()-y[960:]).square().mean();lb=(pb[960:].tanh()-y[960:]).square().mean();la.backward();lb.backward()
 for (n,p),(q,v) in zip(a.named_parameters(),b.named_parameters()):
  if not n.startswith('base.'):
   if p.grad is None or v.grad is None:raise ValueError('broken trainable gradient '+n)
   close('gradient_'+n,p.grad,v.grad);checks['gradient_nonzero_'+n]=bool(p.grad.abs().max()>0)
 checks['base_no_grad']=all(not p.requires_grad and p.grad is None for p in a.base.parameters())
 oa=torch.optim.Adam(a.adapter_parameters(),lr=LR);ob=torch.optim.Adam(b.adapter_parameters(),lr=LR)
 checks['fresh_Adam0states']=len(oa.state)==len(ob.state)==0
 torch.nn.utils.clip_grad_norm_(a.adapter_parameters(),1.);torch.nn.utils.clip_grad_norm_(b.adapter_parameters(),1.);oa.step();ob.step()
 for n,p in a.state_dict().items():close('unit_model_'+n,p,b.state_dict()[n])
 for i,(p,q) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in p:close('unit_Adam_'+str(i)+'_'+k,p[k],q[k])
 checks['unit_Adam8states']=len(oa.state)==len(ob.state)==8
 checks['base_exact_after_update']=exact(base,frozen_base(a)) and exact(base,frozen_base(b))
 checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 checks['trainable43242_8tensors']=sum(p.numel() for p in m.adapter_parameters())==43242 and len(m.adapter_parameters())==8
 result=dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,tolerance=1e-12,full2880_BPTT_no_detach=True,full_forward_backward_s=time.monotonic()-started,physics_steps=0)
 dump(HERE/'PREFLIGHT.json',result);print(json.dumps(dict(stage='PREFLIGHT',all_passed=result['all_passed'],checks=len(checks),wall_s=result['full_forward_backward_s'],max_error=max(errors.values()))),flush=True)
 if not result['all_passed']:raise ValueError('fullfeaturepreflight failed')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent exclusiveheavy release required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if (HERE/'TRAIN').exists():raise FileExistsError('no overwrite or automatic restart')
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();x,u,y=dataset();preflight(m,x,u,y)
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);pc=cached(m,x,u);engine=make(m);base=frozen_base(m)
 source={str(p):sha256(p) for p in source_paths()};data={str(p):sha256(p) for p in data_paths()}
 config=dict(kind=FORMAT,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,parent_lineage_step=4000,additional_updates=2000,physical_eval_only_step=2000,source_hashes=source,data_hashes=data,lr=LR,seed=20261111,loss='uniformpost960actionMSE_full2880BPTT',gradient_clip_norm=1.,fresh_Adam=True,trainable_tensors=8,base_cache_sha256=hashlib.sha256(pc['means'].numpy().tobytes()).hexdigest())
 out=HERE/'TRAIN';out.mkdir();dump(out/'IDENTITY.json',dict(config=config,identity=m.identity(),pid=os.getpid(),python=sys.executable,affinity=list(os.sched_getaffinity(0)),threads=torch.get_num_threads(),inter_op=torch.get_num_interop_threads()))
 curve=[];begun=time.monotonic()
 with (out/'CURVE.jsonl').open('x') as stream:
  for step in range(1,2001):
   opt.zero_grad(set_to_none=True);pred,h=engine(pc['features'],pc['gate'],pc['means'],pc['valid']);loss=(pred[960:].tanh()-y[960:]).square().mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.adapter_parameters(),1.)
   if not torch.isfinite(norm) or any(p.grad is not None for p in m.base.parameters()):raise ValueError('invalidgradient or basegradient')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(norm),wall_s=time.monotonic()-begun)
   if step==1 or step%100==0:
    with torch.no_grad():after,_=engine(pc['features'],pc['gate'],pc['means'],pc['valid']);row['action_MSE_after']=float((after[960:].tanh()-y[960:]).square().mean())
    print(json.dumps(row),flush=True)
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step in (1000,2000):
    verify_hashes(source);verify_hashes(data)
    if not exact(base,frozen_base(m)) or not all(neutral(m).values()):raise ValueError('base or neutral drift')
    with torch.no_grad():means,state=m.sequence(x,u)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    payload=dict(format_version=FORMAT,identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng,replay=dict(means=means,base_states=state[0],residual_states=state[1]))
    path=out/f'checkpoint_{step:06d}.pt'
    if path.exists():raise FileExistsError(path)
    tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path);digest=sha256(path);restored,saved=load_checked(path,digest,require_final=False)
    checks=dict(model=exact(m.state_dict(),restored.state_dict()),Adam=exact(opt.state_dict(),saved['optimizer']),RNG=exact(rng,saved['rng']),curve=exact(curve,saved['curve']),base=exact(base,frozen_base(restored)))
    if not all(checks.values()):raise ValueError('live saved mismatch')
    dump(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,checks=checks,step=step,sha256=digest))
    # Constructor consumes RNG; restore the saved live stream before any next update.
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy'])
    del restored,saved,payload,means,state
    subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--step',str(step),'--parent-resource-released'],check=True)
    print(json.dumps(dict(stage='SAVED',step=step,sha256=digest)),flush=True)
 dump(out/'RESULT.json',dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',additional_updates=2000,parent_sha256=PARENT_SHA,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-begun))
if __name__=='__main__':main()
