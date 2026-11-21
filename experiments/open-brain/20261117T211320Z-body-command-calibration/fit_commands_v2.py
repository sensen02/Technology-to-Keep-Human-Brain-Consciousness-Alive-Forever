"""Bounded same-body teacher imitation for command-response calibration."""
from scope import *
import math,hashlib
from fit_plus_fast import objective,load
from policy_repair import read

def digest(m):
 h=hashlib.sha256()
 for k,v in m.named_parameters():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()

def run():
 torch.set_num_threads(1);torch.manual_seed(35071);start=time.monotonic();out=HERE/'FIT';out.mkdir(exist_ok=False)
 sources=freeze([HERE/'DEV/RESULT.json']);saved=torch.load(WEIGHT,map_location='cpu',weights_only=True);m=read(WEIGHT)
 # Every complete teacher is a relative target. Its absolute task failures remain labelled.
 data={k:[] for k in ('x','phase','y','lambda')};rows=[]
 for path in sorted((HERE/'DEV').glob('*_teacher/TRACE.npz')):
  result=json.loads((path.parent/'RESULT.json').read_text());assert result['steps']==20000 and not any(result['warning'])
  assert sha(path)==result['trace_sha256'];sources[str(path)]=sha(path)
  with np.load(path,allow_pickle=False) as z:
   for k in data:data[k].append(z[k].copy())
  rows.append(dict(path=str(path),absolute_pass=result['absolute_pass'],frames=2000))
 assert len(rows)==16
 current={k:np.concatenate(v) for k,v in data.items()}
 old,old_sources=load(PARENT/'TEACHER_TRAIN')
 # PARENT in common points to the terrain-learning experiment; preserving its sensory input structure.
 for src in old_sources:sources[src['path']]=src['sha256']
 data={k:np.concatenate((current[k],old[k])) for k in current}
 x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y']);drive=torch.from_numpy(data['lambda'].astype(np.float32));base=torch.from_numpy(m.rhythmic(data['phase']))
 optimizer=torch.optim.AdamW(m.parameters());optimizer.load_state_dict(saved['optimizer'])
 rng=torch.Generator();rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng'])
 write(out/'INPUTS.json',dict(source_sha256=sources,current_teachers=rows,frames=len(x),initial_sha256=sha(WEIGHT),epochs=40,
  current_teacher_frames=len(current['x']),old_terrain_frames=len(old['x']),optimizer_resume=True,
  selection='fixed epoch40 actual policy; physical development gate first, no held used in training or selection'))
 probe=copy.deepcopy(m).double();ids=np.where(data['lambda'].max(1)>.5)[0][:128];args=(x[ids].double(),base[ids],y[ids],drive[ids].double())
 loss=objective(probe,*args);loss.backward();checks=[]
 for param in (probe.reflex[-1].bias,probe.reflex[-1].weight):
  idx=int(param.grad.abs().argmax());analytic=float(param.grad.reshape(-1)[idx]);v=param.data.reshape(-1);oldval=float(v[idx]);eps=1e-6
  v[idx]=oldval+eps;plus=float(objective(probe,*args));v[idx]=oldval-eps;minus=float(objective(probe,*args));v[idx]=oldval
  numeric=(plus-minus)/(2*eps);assert abs(analytic)>1e-6 and abs(analytic-numeric)<1e-7+1e-4*abs(analytic)
  checks.append(dict(analytic=analytic,numeric=numeric,error=abs(analytic-numeric)))
 write(out/'GRADIENT_CHECK.json',dict(pass_=True,checks=checks))
 update=saved['update'];initial_update=update
 def full(epoch):
  p=out/f'epoch{epoch:03d}.pt';tmp=p.with_suffix('.tmp');torch.save(dict(state=m.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),torch_rng=torch.get_rng_state(),epoch=epoch,update=update),tmp);tmp.replace(p);return p
 full(0);(out/'UPDATES').mkdir();event=torch.from_numpy(np.where(data['lambda'].max(1)>.5)[0]);n=len(current['x']);history=[]
 with (out/'UPDATES.jsonl').open('w') as log:
  for epoch in range(1,41):
   assert time.monotonic()-start<600
   lr=.0004*(.2+.8*.5*(1+np.cos(np.pi*epoch/40)))
   for group in optimizer.param_groups:group['lr']=float(lr)
   for _ in range(math.ceil(len(x)/512)):
    ids=torch.cat((torch.randint(len(x),(128,),generator=rng),torch.randint(n,(256,),generator=rng),event[torch.randint(len(event),(128,),generator=rng)]))
    optimizer.zero_grad(set_to_none=True);loss=objective(m,x[ids],base[ids],y[ids],drive[ids]);assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.);optimizer.step();update+=1
    p=out/'UPDATES'/f'{update:06d}.pt';tmp=p.with_suffix('.tmp');torch.save(dict(state=m.state_dict(),update=update),tmp);tmp.replace(p)
    log.write(json.dumps(dict(update=update,epoch=epoch,loss=float(loss.detach()),parameter_sha256=digest(m),checkpoint=str(p),sha256=sha(p)))+'\n')
   with torch.inference_mode():pred,_,_=m(x[:n],base[:n]);rmse=float(torch.sqrt(((pred-y[:n])**2).mean()))
   if epoch%10==0:
    p=full(epoch);entry=dict(epoch=epoch,update=update,train_joint_rmse=rmse,path=str(p),sha256=sha(p));history.append(entry);write(out/'HISTORY.json',history);print(entry,flush=True)
 verify_sources(sources);write(out/'RESULT.json',dict(actual_updates=update-initial_update,initial_update=initial_update,last_update=update,frames=len(x),wall_seconds=time.monotonic()-start))

if __name__=='__main__':run()
