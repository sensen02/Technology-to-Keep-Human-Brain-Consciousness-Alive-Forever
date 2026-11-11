"""One fixed2000 original-qualified RIGHT fulladapter43242 BPTT, freshALL8 Adam."""
import argparse,json,time,os,hashlib
from copy import deepcopy
import numpy as np,torch
from model_io import *
def dump(path,value):
 if Path(path).exists():raise FileExistsError(path)
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def preflight(m,x,u,y):
 checks={};errors={};parent,p=old.load_checked(PARENT,PARENT_SHA)
 def close(n,a,b):errors[n]=float((a-b).abs().max());checks[n]=errors[n]<=1e-12
 checks['initial_base_semantic_exact']=exact(frozen_base(m),frozen_base(parent));checks['initial_encoder_exact']=exact(m.encoder.state_dict(),parent.encoder.state_dict());checks['initial_GRU_exact']=exact(m.gru.state_dict(),parent.gru.state_dict());checks['initial_densehead_fullmatrix_exact']=torch.equal(m.head.weight,parent.head.weight) and torch.equal(m.head.bias,parent.head.bias)
 with torch.no_grad():means,h=m.sequence(x,u)
 checks['initial_teacher_means_bitexact']=torch.equal(means,p['replay']['means']);checks['initial_teacher_basehidden_bitexact']=torch.equal(h[0],p['replay']['base_states']);checks['initial_teacher_residualhidden_bitexact']=torch.equal(h[1],p['replay']['residual_states'])
 with np.load(ACTUAL_STREAM,allow_pickle=False) as z:ax=torch.from_numpy(z['X238'].copy())[:,None];au=torch.from_numpy(z['command'].copy())[:,None];am=torch.from_numpy(z['mean'].copy())[:,None];ah=torch.from_numpy(z['hidden_after'].copy())[:,None]
 with torch.no_grad():aout,astate=m.sequence(ax,au)
 checks['preflightONLY_actual_means_bitexact']=torch.equal(aout,am);checks['preflightONLY_actual_basehidden_bitexact']=torch.equal(astate[0],ah[:,:,:128]);checks['preflightONLY_actual_residualhidden_bitexact']=torch.equal(astate[1],ah[:,:,128:])
 a=deepcopy(m);b=deepcopy(m);c=cached(b,x,u);engine=make(b);pa,ha=a.sequence(x,u);pb,hb=engine(c['features'],c['gate'],c['means'],c['valid']);close('production_scan_means',pa,pb);close('production_scan_hidden',ha[1],hb);close('production_cached_basehidden',ha[0],c['base_states']);la=(pa[960:].tanh()-y[960:]).square().mean();lb=(pb[960:].tanh()-y[960:]).square().mean();la.backward();lb.backward()
 for (n,q),(n2,v) in zip(a.named_parameters(),b.named_parameters()):
  if not n.startswith('base.'):
   if q.grad is None or v.grad is None:raise ValueError('broken actual trainable gradient '+n)
   close('gradient_'+n,q.grad,v.grad);checks['finite_gradient_'+n]=bool(torch.isfinite(q.grad).all())
 checks['base_no_grad']=all(not q.requires_grad and q.grad is None for q in a.base.parameters());oa=torch.optim.Adam(a.adapter_parameters(),lr=LR);ob=torch.optim.Adam(b.adapter_parameters(),lr=LR);checks['freshALL8_empty']=len(oa.state)==len(ob.state)==0;torch.nn.utils.clip_grad_norm_(a.adapter_parameters(),1.);torch.nn.utils.clip_grad_norm_(b.adapter_parameters(),1.);oa.step();ob.step()
 for n,v in a.state_dict().items():close('unit_model_'+n,v,b.state_dict()[n])
 for i,(v,w) in enumerate(zip(oa.state.values(),ob.state.values())):
  for k in v:close('unit_Adam_'+str(i)+'_'+k,v[k],w[k])
 checks['Adam8_states']=len(oa.state)==len(ob.state)==8;checks['base_unchanged']=exact(frozen_base(a),frozen_base(m)) and exact(frozen_base(b),frozen_base(m));checks['ALL8_43242']=len(m.adapter_parameters())==8 and sum(q.numel() for q in m.adapter_parameters())==43242;checks.update({'neutral_'+k:v for k,v in neutral(m).items()})
 result=dict(all_passed=all(checks.values()),checks=checks,max_errors=errors,tolerance=1e-12,full2880_BPTT_without_detach=True,training_loss_episode_count=1,actual_stream_used_only_for_initial_parity=True,physics_steps=0);dump(HERE/'PREFLIGHT.json',result)
 if not result['all_passed']:raise ValueError('originalRIGHT fullfeature preflight failed')

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent-resource-released',action='store_true');args=p.parse_args()
 if not args.parent_resource_released:raise ValueError('CODE_ONLY until parent releases exclusiveheavy resource')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
 if (HERE/'TRAIN').exists():raise FileExistsError('no overwrite/blind restart')
 m=initialize();x,u,y=dataset();preflight(m,x,u,y);torch.manual_seed(20261111);np.random.seed(20261111);m=initialize();opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);pc=cached(m,x,u);engine=make(m);base=frozen_base(m)
 source={str(p):sha256(p) for p in [HERE/n for n in ('model_io.py','train.py','verify_saved.py','PREREG.json')]+[PARENT_DIR/'model_io.py',HERE.parent/'model.py',ENGINE,BASE/'models.py']};data={str(p):sha256(p) for p in (PARENT,DATA,QUALIFIED/'RESULT.json',QUALIFIED/'IDENTITY.json',REFERENCE)};pref={str(ACTUAL_STREAM):sha256(ACTUAL_STREAM)}
 config=dict(kind=FORMAT,parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,parent_source_training_step=4000,additional_updates=2000,physical_eval_only_step=2000,source_hashes=source,data_hashes=data,preflight_only_hashes=pref,training_episode_count=1,training_dataset=str(DATA),training_dataset_kind='ORIGINAL_QUALIFIED_RIGHT_ONLY',actual_stream_not_in_loss=True,lr=LR,seed=20261111,fresh_AdamALL8=True,trainable_parameters=43242,gradient_clip_norm=1.,loss='uniformpost960actionMSE_full2880BPTT',base_cache_sha256=hashlib.sha256(pc['means'].numpy().tobytes()).hexdigest());out=HERE/'TRAIN';out.mkdir();dump(out/'RUNTIME.json',dict(config=config,identity=m.identity(),pid=os.getpid(),python=sys.executable,affinity=list(os.sched_getaffinity(0)),threads=torch.get_num_threads()));curve=[];started=time.monotonic()
 with (out/'CURVE.jsonl').open('x') as stream:
  for step in range(1,2001):
   if time.monotonic()-started>14400:raise TimeoutError('fixed4h wall cap')
   opt.zero_grad(set_to_none=True);pred,h=engine(pc['features'],pc['gate'],pc['means'],pc['valid']);loss=(pred[960:].tanh()-y[960:]).square().mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.adapter_parameters(),1.)
   if not torch.isfinite(norm) or any(q.grad is not None for q in m.base.parameters()):raise ValueError('nonfinite/base gradient')
   opt.step();row=dict(step=step,loss_before_update=float(loss.detach()),gradient_norm=float(norm),wall_s=time.monotonic()-started)
   if step==1 or step%100==0:
    with torch.no_grad():after,_=engine(pc['features'],pc['gate'],pc['means'],pc['valid']);row['action_MSE_after']=float((after[960:].tanh()-y[960:]).square().mean())
    print(json.dumps(row),flush=True)
   curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
   if step in (1000,2000):
    verify_hashes(source);verify_hashes(data);verify_hashes(pref)
    if not exact(base,frozen_base(m)) or not all(neutral(m).values()):raise ValueError('base/neutral drift')
    with torch.no_grad():means,state=m.sequence(x,u)
    rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state());payload=dict(format_version=FORMAT,identity=m.identity(),state_dict=m.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng,replay=dict(means=means,base_states=state[0],residual_states=state[1]));path=out/f'checkpoint_{step:06d}.pt';tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path);digest=sha256(path);restored,saved=load_checked(path,digest,require_final=False);checks=dict(model=exact(m.state_dict(),restored.state_dict()),Adam=exact(opt.state_dict(),saved['optimizer']),RNG=exact(rng,saved['rng']),curve=exact(curve,saved['curve']),base=exact(base,frozen_base(restored)))
    if not all(checks.values()):raise ValueError('live saved mismatch')
    dump(out/f'LIVE_SAVED_{step:06d}.json',dict(all_exact=True,checks=checks,step=step,sha256=digest));torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);print(json.dumps(dict(saved=str(path),sha256=digest)),flush=True)
 dump(out/'RESULT.json',dict(status='TRAINED_NOT_PHYSICALLY_EVALUATED',additional_updates=2000,parent_sha256=PARENT_SHA,checkpoint=str(path),sha256=digest,final=curve[-1],wall_s=time.monotonic()-started))
if __name__=='__main__':main()
