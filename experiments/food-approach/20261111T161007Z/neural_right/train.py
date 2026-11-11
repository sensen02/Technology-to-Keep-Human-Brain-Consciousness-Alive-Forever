"""Fixed2000 single-trajectory full BPTT, exact shared parameter scripted scan."""
import json,time,os,sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from model import *
def dump(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def neutral(m):
 with np.load(REFERENCE,allow_pickle=False) as z:x=torch.from_numpy(z['X238'].copy())[:,None];ref=torch.from_numpy(z['mean'].copy())[:,None];hb=torch.from_numpy(z['hidden_after'].copy())[:,None]
 with torch.no_grad():a,s=m.sequence(x,torch.zeros((2880,1,2),dtype=torch.float64))
 return dict(means_exact=torch.equal(a,ref),base_hidden_exact=torch.equal(s[0],hb))
def preflight(m,x,u,y):
 checks={};errors={}
 def close(name,a,b):
  e=float((a-b).abs().max());errors[name]=e;checks[name]=e<=1e-12
 # Exercise nonzero head so every gradient path is nontrivial.
 a=deepcopy(m)
 with torch.no_grad():a.head.weight.normal_(0,.02);a.head.bias.normal_(0,.02)
 b=deepcopy(a);pc=cached(b,x,u);scan=make(b)
 before=deepcopy(a.base.state_dict())
 start=time.monotonic();pa,sa=a.sequence(x,u);pb,sb=scan(pc['features'],pc['gate'],pc['means'],pc['valid'])
 close('production_scan_means',pa,pb);close('production_scan_residualhistory',sa[1],sb);close('production_cached_basehistory',sa[0],pc['base_states'])
 la=(pa[960:].tanh()-y[960:]).square().mean();lb=(pb[960:].tanh()-y[960:]).square().mean();la.backward();lb.backward()
 for (n,p),(q,v) in zip(a.named_parameters(),b.named_parameters()):
  if not n.startswith('base.'):close('gradient_'+n,p.grad,v.grad)
 checks['base_no_grad']=all(p.grad is None and not p.requires_grad for p in a.base.parameters())
 oa=torch.optim.Adam(a.adapter_parameters(),lr=.001);ob=torch.optim.Adam(b.adapter_parameters(),lr=.001)
 torch.nn.utils.clip_grad_norm_(a.adapter_parameters(),1.);torch.nn.utils.clip_grad_norm_(b.adapter_parameters(),1.);oa.step();ob.step()
 for n,p in a.state_dict().items():close('unit_model_'+n,p,b.state_dict()[n])
 for i,(p,q) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in p:close('unit_Adam_'+str(i)+'_'+k,p[k],q[k])
 checks['base_unchanged']=exact(before,a.base.state_dict())
 checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 with torch.no_grad():p,s=m.sequence(x,u)
 pc0=cached(m,x,u);checks['initial_right_zerohead_baseexact']=torch.equal(p,pc0['means'])
 checks['fresh_Adam_no_state']=len(torch.optim.Adam(m.adapter_parameters(),lr=.001).state)==0
 result=dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,full_forward_backward_s=time.monotonic()-start,tolerance=1e-12,actual_full_trajectory=True)
 dump(HERE/'PREFLIGHT.json',result)
 if not result['all_passed']:raise ValueError('preflight failed')
 return result

def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 if (HERE/'TRAIN').exists():raise ValueError('do not overwrite or blindly restart training')
 m=initialize();x,u,y=dataset();preflight(m,x,u,y)
 # Preflight uses temporary model/RNG; explicit reset makes actual initialization reproducible.
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);pc=cached(m,x,u);engine=make(m)
 sources={str(p):sha256(p) for p in [HERE/'model.py',HERE/'train.py',HERE/'verify_saved.py',HERE/'PREREG.json',OLD,ENGINE,BASE/'models.py']}
 data={str(p):sha256(p) for p in [DATA,TRANSFER,legacy.CHECKPOINT,REFERENCE,HERE.parent/'attempt02/right_yaw_m060/IDENTITY.json',HERE.parent/'attempt02/right_yaw_m060/VERIFY.json']}
 config=dict(kind='RIGHT_ONLY_new006_bound3_fresh2000',physical_eval_only_step=2000,source_hashes=sources,data_hashes=data,lr=.001,seed=20261111,loss='post960 action MSE single2880 full BPTT',gradient_clip_norm=1.,transfer_encoder_gru_only=True)
 out=HERE/'TRAIN';out.mkdir();curve=[];base=deepcopy(m.base.state_dict());begun=time.monotonic()
 dump(out/'RUNTIME.json',dict(pid=os.getpid(),python=sys.executable,affinity=list(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),inter_op=torch.get_num_interop_threads(),source_hashes=sources,data_hashes=data))
 with (out/'CURVE.jsonl').open('x') as f:
  for step in range(1,2001):
   if time.monotonic()-begun>14400:raise TimeoutError('4h wall cap')
   opt.zero_grad(set_to_none=True);pred,h=engine(pc['features'],pc['gate'],pc['means'],pc['valid']);loss=(pred[960:].tanh()-y[960:]).square().mean();loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.adapter_parameters(),1.)
   if not torch.isfinite(grad) or any(p.grad is not None for p in m.base.parameters()):raise ValueError('gradient/base invalid')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(grad),wall_s=time.monotonic()-begun)
   if step%100==0 or step==1:
    with torch.no_grad():after,_=engine(pc['features'],pc['gate'],pc['means'],pc['valid']);row['action_MSE_after']=float((after[960:].tanh()-y[960:]).square().mean())
    print(json.dumps(row),flush=True)
   curve.append(row);f.write(json.dumps(row)+'\n');f.flush()
   if step in (1000,2000):
    verify_hashes(sources);verify_hashes(data)
    if not exact(base,m.base.state_dict()) or not all(neutral(m).values()):raise ValueError('base/neutral drift')
    with torch.no_grad():means,states=m.sequence(x,u)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state())
    payload=dict(format_version='RIGHT_ONLY_V1',identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng,replay=dict(means=means,base_states=states[0],residual_states=states[1]))
    path=out/f'checkpoint_{step:06d}.pt';tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path)
    restored,p=load_checked(path,sha256(path),require_final=False)
    if not all([exact(m.state_dict(),restored.state_dict()),exact(opt.state_dict(),p['optimizer']),exact(curve,p['curve']),exact(rng,p['rng'])]):raise ValueError('live saved state mismatch')
    dump(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,step=step,sha256=sha256(path),model=True,Adam=True,RNG=True,curve=True,neutral=neutral(m)))
    print(json.dumps(dict(saved=str(path),sha256=sha256(path),step=step)),flush=True)
    # Independent process audit is deliberately run after training exits: never two training/physics processes.
    del restored,p,payload,means,states
 dump(out/'RESULT.json',dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',steps=2000,sha256=sha256(out/'checkpoint_002000.pt'),final=curve[-1],wall_s=time.monotonic()-begun))
if __name__=='__main__':main()
