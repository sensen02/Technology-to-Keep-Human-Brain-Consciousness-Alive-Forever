"""One bounded sensory fine-tune with signed phase-correct forward and actual teacher labels."""
from common import *
import copy,math,time,hashlib
from policy_repair import read
from fit_plus_fast import load
from fit_plus_fast import objective

HERE=Path(__file__).resolve().parent

def digest(m):
 h=hashlib.sha256()
 for k,v in m.named_parameters():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()

def run():
 torch.set_num_threads(1);torch.manual_seed(29371);started=time.monotonic();out=HERE/'FIT_REPAIR';out.mkdir(exist_ok=False)
 initial=HERE/'ROUTING_FIXED.pt';identity=manifest([initial]);saved=torch.load(initial,weights_only=True,map_location='cpu');actor=read(initial)
 data,sources=load(PARENT/'TEACHER_TRAIN');original=len(data['x'])
 for folder in (PARENT/'DAGGER1',PARENT/'DAGGER2',HERE/'SHADOW'):
  extra,src=load(folder,True);sources+=src;data={k:np.concatenate((data[k],extra[k])) for k in data}
 dev,dev_sources=load(PARENT/'TEACHER_DEV')
 identity.update({s['path']:s['sha256'] for s in sources+dev_sources})
 write(out/'INPUTS.json',dict(source_sha256=identity,sources=sources,dev_sources=dev_sources,epochs=40,frames=len(data['x']),
  original_frames=original,initial_sha256=sha(initial),optimizer='resume exact parent state, first resumed LR explicitly .00025 schedule'))
 x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y']);drive=torch.from_numpy(data['lambda'].astype(np.float32));base=torch.from_numpy(actor.rhythmic(data['phase']))
 dx=torch.from_numpy(dev['x']);dy=torch.from_numpy(dev['y']);dd=torch.from_numpy(dev['lambda'].astype(np.float32));db=torch.from_numpy(actor.rhythmic(dev['phase']))
 optimizer=torch.optim.AdamW(actor.parameters());optimizer.load_state_dict(saved['optimizer'])
 rng=torch.Generator();rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng'])
 probe=copy.deepcopy(actor).double();idx=np.where(data['lambda'].max(1)>.5)[0][:128];args=(x[idx].double(),base[idx],y[idx],drive[idx].double())
 loss=objective(probe,*args);loss.backward();checks=[]
 for parameter in (probe.reflex[-1].bias,probe.reflex[-1].weight):
  index=int(parameter.grad.abs().argmax());analytic=float(parameter.grad.reshape(-1)[index]);v=parameter.data.reshape(-1);old=float(v[index]);eps=1e-6
  v[index]=old+eps;plus=float(objective(probe,*args));v[index]=old-eps;minus=float(objective(probe,*args));v[index]=old
  numeric=(plus-minus)/(2*eps);assert abs(analytic)>1e-6 and abs(analytic-numeric)<1e-7+1e-4*abs(analytic)
  checks.append(dict(analytic=analytic,numeric=numeric,error=abs(analytic-numeric)))
 write(out/'GRADIENT_CHECK.json',dict(pass_=True,checks=checks))
 def fullsave(epoch,update):
  p=out/f'epoch{epoch:03d}.pt';tmp=p.with_suffix('.tmp');torch.save(dict(state=actor.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),torch_rng=torch.get_rng_state(),epoch=epoch,update=update),tmp);tmp.replace(p);return p
 start_update=saved['update'];update=start_update;fullsave(0,update);(out/'UPDATES').mkdir()
 event=torch.from_numpy(np.where(data['lambda'].max(1)>.5)[0]);stop=torch.from_numpy(np.where(np.max(abs(data['x'][:,:2]),axis=1)<.1)[0]);history=[]
 with (out/'UPDATES.jsonl').open('w') as log:
  for epoch in range(1,41):
   assert time.monotonic()-started<600
   lr=.00025*(.2+.8*.5*(1+np.cos(np.pi*epoch/40)))
   for group in optimizer.param_groups:group['lr']=float(lr)
   for _ in range(math.ceil(len(x)/512)):
    ids=torch.cat((torch.randint(len(x),(128,),generator=rng),event[torch.randint(len(event),(128,),generator=rng)],stop[torch.randint(len(stop),(128,),generator=rng)],original+torch.randint(len(x)-original,(128,),generator=rng)))
    optimizer.zero_grad(set_to_none=True);loss=objective(actor,x[ids],base[ids],y[ids],drive[ids]);assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),10);optimizer.step();update+=1
    p=out/'UPDATES'/f'{update:06d}.pt';tmp=p.with_suffix('.tmp');torch.save(dict(state=actor.state_dict(),update=update),tmp);tmp.replace(p)
    log.write(json.dumps(dict(update=update,epoch=epoch,loss=float(loss.detach()),parameter_sha256=digest(actor),checkpoint=str(p),checkpoint_sha256=sha(p)))+'\n')
   with torch.inference_mode():pred,_,_=actor(dx,db);rmse=float(torch.sqrt(((pred-dy)**2).mean()))
   entry=dict(epoch=epoch,update=update,dev_joint_rmse=rmse)
   if epoch%10==0:entry.update(path=str(fullsave(epoch,update)));entry['sha256']=sha(entry['path']);print(entry,flush=True)
   history.append(entry);write(out/'HISTORY.json',history)
 verify_sources(identity)
 choice=min((h for h in history if 'path' in h),key=lambda h:h['dev_joint_rmse']);write(out/'SELECTED.json',choice)
 write(out/'RESULT.json',dict(actual_updates=update-start_update,start_update=start_update,end_update=update,frames=len(x),wall_seconds=time.monotonic()-started))

if __name__=='__main__':run()
