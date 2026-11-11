"""RIGHT anatomical LH stance-only fixed4000; parent-resource-release gate."""
import argparse,json,time,os,hashlib
import numpy as np,torch
from model_io import *
from check_train import run_checks
def dump(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def main():
 a=argparse.ArgumentParser();a.add_argument('--parent-resource-released',action='store_true');args=a.parse_args()
 if not args.parent_resource_released:raise ValueError('CODE_ONLY pending parent resource release')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 if (HERE/'TRAIN').exists():raise ValueError('do not overwrite or blindly restart')
 m=initialize();c=prepare(m);pre=run_checks(m,c);dump(HERE/'PREFLIGHT.json',pre);dump(HERE/'PARENT_LH_DIAGNOSTIC.json',pre['diagnostic'])
 if not pre['all_passed']:raise ValueError('LH full production/cache/mapping/mask preflight failed')
 torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();c=prepare(m);opt=torch.optim.Adam(m.adapter_parameters(),lr=.001);frozen_start=frozen(m)
 paths=[HERE/n for n in ('model_io.py','check_train.py','train.py','verify_saved.py','PREREG.json')]+[PARENT_DIR/'model_io.py',HERE.parent/'model.py',BASE/'models.py']
 sources={str(p):sha256(p) for p in paths};data={str(p):sha256(p) for p in (PARENT,DATA,TRACE,REFERENCE,XML,MAPPING_SOURCE)}
 config=dict(kind='RIGHT_ONLY_LH_stance_fixed4000',physical_eval_only_step=4000,source_hashes=sources,data_hashes=data,LH_effective_parameters=455,Adam_states=2,lr=.001,seed=20261111,loss='LH7 post960 weighted actionMSE weight4 if preaction teacherLH>=.1W else1, normalized7*sumweights',hidden_cache_sha256=hashlib.sha256(c['hidden'].numpy().tobytes()).hexdigest(),weights_sha256=hashlib.sha256(c['weights'].numpy().tobytes()).hexdigest(),preaction_time_error=c['time_error'],mapping=c['mapping'],no_old_Adam=True)
 out=HERE/'TRAIN';out.mkdir();dump(out/'RUNTIME.json',dict(pid=os.getpid(),python=sys.executable,affinity=list(os.sched_getaffinity(0)),threads=torch.get_num_threads(),config=config));curve=[];begun=time.monotonic()
 with (out/'CURVE.jsonl').open('x') as f:
  for step in range(1,4001):
   opt.zero_grad(set_to_none=True);pred=cached_forward(m,c);objective=loss(pred,c);objective.backward();norm=torch.sqrt(sum(p.grad.square().sum() for p in m.adapter_parameters()))
   if not torch.isfinite(norm) or any(q.grad is not None for n,q in m.named_parameters() if n not in ('head.lh_weight','head.lh_bias')):raise ValueError('nonfinite/outsideLH gradient')
   opt.step();row=dict(step=step,weighted_LH_loss_before_update=float(objective.detach()),gradient_norm=float(norm),wall_s=time.monotonic()-begun)
   if step%500==0 or step==1:
    with torch.no_grad():after=cached_forward(m,c);row.update(weighted_LH_loss_after=float(loss(after,c)),uniform_all42_action_MSE=float((after[960:].tanh()-c['y'][960:]).square().mean()),uniform_LH_action_MSE=float((after[960:,:,7:14].tanh()-c['y'][960:,:,7:14]).square().mean()))
    print(json.dumps(row),flush=True)
   curve.append(row);f.write(json.dumps(row)+'\n');f.flush()
   if step in (2000,4000):
    verify_hashes(sources);verify_hashes(data)
    if not exact(frozen_start,frozen(m)) or not all(neutral(m).values()):raise ValueError('frozen/neutral drift')
    with torch.no_grad():means,state=m.sequence(c['x'],c['u'])
    other=list(range(7))+list(range(14,42))
    if not torch.equal(means[:,:,other],c['means'][:,:,other]):raise ValueError('nonLH same-input outputs changed')
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());payload=dict(format_version='RIGHT_LH_STANCE_REFIT_V1',identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng,replay=dict(means=means,base_states=state[0],residual_states=state[1]))
    path=out/f'checkpoint_{step:06d}.pt';tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path);digest=sha256(path);restored,p=load_checked(path,digest,require_final=False)
    checks=dict(model=exact(m.state_dict(),restored.state_dict()),Adam=exact(opt.state_dict(),p['optimizer']),curve=exact(curve,p['curve']),RNG=exact(rng,p['rng']),frozen=exact(frozen_start,frozen(restored)))
    if not all(checks.values()):raise ValueError('live saved mismatch')
    dump(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,checks=checks,step=step,sha256=digest));torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);print(json.dumps(dict(saved=str(path),sha256=digest)),flush=True)
 dump(out/'RESULT.json',dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',updates=4000,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-begun))
if __name__=='__main__':main()
