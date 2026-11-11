"""New balanced qualified/CORRECTIVE densehead2730 fixed4000, LR.0003."""
import argparse,json,time,os,hashlib
import numpy as np,torch
from model_io import *
from check_train import run_checks
def dump(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('parent shared resource release required')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 if (HERE/'TRAIN').exists():raise ValueError('do not overwrite or blindly restart')
 m=initialize();c=prepare(m);pre=run_checks(m,c);dump(HERE/'PREFLIGHT.json',pre)
 if not pre['all_passed']:raise ValueError('twoepisode densehead preflight failed')
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();c=prepare(m);opt=torch.optim.Adam(m.adapter_parameters(),lr=.0003);base=frozen(m)
 sources={str(p):sha256(p) for p in [HERE/n for n in ('model_io.py','check_train.py','train.py','verify_saved.py','PREREG.json','oracle.py','verify_oracle.py')]+[PARENT_DIR/'model_io.py',HERE.parent/'model.py',BASE/'models.py']}
 oracle=json.loads((HERE/'ORACLE/RESULT.json').read_text());data={str(p):sha256(p) for p in (PARENT,DATA,QUALIFIED/'RESULT.json',QUALIFIED/'IDENTITY.json',CORRECTIVE,HERE/'ORACLE/RESULT.json',HERE/'ORACLE/VERIFY.json',HERE/'ORACLE/PREFIX_AUDIT.json',ACTUAL/'neural_decisions.npz',REFERENCE)};data.update(oracle['input_source_hashes']);data.update(oracle['artifacts'])
 config=dict(kind='RIGHT_ONLY_equal_qualified_CORRECTIVE_densehead_fixed4000',physical_eval_only_step=4000,source_hashes=sources,data_hashes=data,head_parameters=2730,Adam_states=2,lr=.0003,seed=20261111,episode_names=[e['name'] for e in c['episodes']],episode_lengths=[2880,2880],post_counts=[1920,1920],episode_weights=[.5,.5],objective='uniform42 post960 actionMSE mean of two wholetrajectories',corrective_label_status='UNEXECUTED_CORRECTIVE_EXAMPLES_NOT_QUALIFIED_TEACHER_PHYSICS',hidden_cache_sha256=hashlib.sha256(c['hidden'].numpy().tobytes()).hexdigest(),no_old_Adam=True)
 out=HERE/'TRAIN';out.mkdir();dump(out/'RUNTIME.json',dict(pid=os.getpid(),python=sys.executable,affinity=list(os.sched_getaffinity(0)),threads=torch.get_num_threads(),config=config));curve=[];begun=time.monotonic()
 with (out/'CURVE.jsonl').open('x') as f:
  for step in range(1,4001):
   opt.zero_grad(set_to_none=True);pred=cached_forward(m,c);loss=objective(pred,c);loss.backward();norm=torch.sqrt(sum(q.grad.square().sum() for q in m.adapter_parameters()))
   if not torch.isfinite(norm) or any(q.grad is not None for n,q in m.named_parameters() if not n.startswith('head.')):raise ValueError('nonfinite/frozen gradient')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(norm),wall_s=time.monotonic()-begun)
   if step%500==0 or step==1:
    with torch.no_grad():after=cached_forward(m,c);row['equal_objective_after']=float(objective(after,c));row['episode_MSE_after']={e['name']:float((after[960:,j].tanh()-c['y'][960:,j]).square().mean()) for j,e in enumerate(c['episodes'])}
    print(json.dumps(row),flush=True)
   curve.append(row);f.write(json.dumps(row)+'\n');f.flush()
   if step in (2000,4000):
    verify_hashes(sources);verify_hashes(data)
    if not exact(base,frozen(m)) or not all(neutral(m).values()):raise ValueError('frozen/neutral drift')
    ms=[];hs=[]
    with torch.no_grad():
     for e in c['episodes']:
      mean,h=m.sequence(e['x'],e['u']);ms.append(mean);hs.append(h)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());payload=dict(format_version='RIGHT_DAGGER_DENSE_HEAD_V1',identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng,replay=dict(means=torch.cat(ms,1),base_states=torch.cat([h[0] for h in hs],1),residual_states=torch.cat([h[1] for h in hs],1)))
    path=out/f'checkpoint_{step:06d}.pt';tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path);digest=sha256(path);restored,p=load_checked(path,digest,require_final=False)
    checks=dict(model=exact(m.state_dict(),restored.state_dict()),Adam=exact(opt.state_dict(),p['optimizer']),curve=exact(curve,p['curve']),RNG=exact(rng,p['rng']),frozen=exact(base,frozen(restored)))
    if not all(checks.values()):raise ValueError('live saved mismatch')
    dump(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,checks=checks,step=step,sha256=digest));torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);print(json.dumps(dict(saved=str(path),sha256=digest)),flush=True)
 dump(out/'RESULT.json',dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',updates=4000,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-begun))
if __name__=='__main__':main()
