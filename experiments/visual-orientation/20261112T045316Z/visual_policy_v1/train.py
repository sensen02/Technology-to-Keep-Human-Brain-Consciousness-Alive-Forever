"""One fresh visual CNN/GRU candidate, fixed500; no motor/physics loading."""
import time,resource
from data import *
from model import VisualPolicy,make,production,objective
def main():
 cpu1();f=frozen();pre=read(HERE/'PREFLIGHT.json')
 if not pre['all_passed']:raise ValueError('actual visual preflight required')
 out=HERE/'TRAIN'
 if out.exists():raise FileExistsError('no blind restart/overwrite')
 rgb,manifest=library();data=arrays('train');torch.manual_seed(20261112);m=VisualPolicy();opt=torch.optim.Adam(m.parameters(),lr=1e-4);scan=make(m);gen=np.random.Generator(np.random.PCG64(20261112));curve=[];pairs=[];batches=[];out.mkdir();start=time.monotonic()
 hashes=dict(f['source_hashes']);hashes[str(HERE/'FROZEN.json')]=sha(HERE/'FROZEN.json');hashes[str(HERE/'PREFLIGHT.json')]=sha(HERE/'PREFLIGHT.json')
 def checkpoint(step):
  verify(hashes);guard();ix=expand([0,4]);x,y,w=batch(rgb,data,ix)
  with torch.no_grad():turn,latent,h=production(m,x)
  rng=dict(torch=torch.get_rng_state(),numpy=gen.bit_generator.state)
  payload=dict(format_version=FORMAT,deployment_qualified=False,identity=m.identity(),step=step,state_dict=m.state_dict(),optimizer=opt.state_dict(),rng=rng,curve=curve,pair_schedule=pairs,batch_schedule=batches,pair_table=pair_table(),replay_indices=ix,replay=dict(turn=turn,latent=latent,hidden=h),source_hashes=hashes,data_hashes={str(HERE/n):sha(HERE/n) for n in ('train.npz','validation.npz','controls.npz','DATASET.json')},motor_sha256=MOTOR_SHA,olfaction_connected=False)
  path=out/f'checkpoint_{step:06d}.pt';tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path);copy,p=load_checked(path,sha(path),require_final=False)
  checks=dict(model=exact(m.state_dict(),copy.state_dict()),Adam=exact(opt.state_dict(),p['optimizer']),RNG=exact(rng,p['rng']),curve=exact(curve,p['curve']),pairs=exact(pairs,p['pair_schedule']),batches=exact(batches,p['batch_schedule']))
  if not all(checks.values()):raise ValueError('actual visual save mismatch')
  torch.set_rng_state(rng['torch']);gen.bit_generator.state=rng['numpy'];save(out/f'LIVE_SAVED_{step:06d}.json',dict(all_passed=True,checks=checks,sha256=sha(path),step=step));print(json.dumps(dict(saved=str(path),step=step,sha256=sha(path))),flush=True)
 curve.append(dict(step=0,status='INITIAL_RANDOM_VISUAL_POLICY_NOT_BEHAVIOR_QUALIFIED'));checkpoint(0)
 with (out/'CURVE.jsonl').open('x') as log:
  log.write(json.dumps(curve[0])+'\n');log.flush()
  for step in range(1,501):
   if time.monotonic()-start>7200:raise TimeoutError('fixed2hwallcap')
   ids=gen.choice(12,2,replace=False);ix=expand(ids);pairs.append(ids.tolist());batches.append(ix);x,y,w=batch(rgb,data,ix);opt.zero_grad(set_to_none=True);turn,latent,h=scan(x);loss=objective(turn,latent,y,w)
   if not torch.isfinite(loss):raise ValueError('nonfinite visual loss')
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('nonfinite visual gradient')
   opt.step();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
   if rss>1800:raise MemoryError('registered RSS1800MiB cap')
   row=dict(step=step,pair_ids=ids.tolist(),batch_indices=ix,objective_before_update=float(loss.detach()),gradient_norm=float(grad),peak_RSS_MiB=rss,wall_s=time.monotonic()-start);curve.append(row);log.write(json.dumps(row)+'\n');log.flush()
   del turn,latent,h,loss,x,y,w
   if step==1 or step%50==0:print(json.dumps(row),flush=True)
   if step in (250,500):checkpoint(step)
 save(out/'RESULT.json',dict(status='TRAINED_NOT_VALIDATED_OR_PHYSICAL',updates=500,checkpoint=str(out/'checkpoint_000500.pt'),sha256=sha(out/'checkpoint_000500.pt'),seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,Core_constructions=0,motor_model_loads=0))
if __name__=='__main__':main()
