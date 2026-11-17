"""Fixed400 full1920 batch7 aggregation repair, no physical or native-function claim."""
from common import *
def main():
 torch,np=backend();f=frozen();out=HERE/'TRAIN';out.mkdir(exist_ok=False)
 from data import arrays,tensors
 from model import make_scan,python_rollout,objective
 import time,resource
 d=arrays('train');x,cue,y,w=tensors(d);torch.manual_seed(20261112);m,parent=initialize();scan=make_scan(m);opt=torch.optim.Adam(m.parameters(),lr=.001);curve=[];start=time.monotonic();replay_index=5
 def scalar(v):return float(v.detach())
 def save_step(step):
  with torch.no_grad():
   replay=python_rollout(m,x[:,replay_index:replay_index+1],cue[:,replay_index:replay_index+1]);pred=scan(x,cue);loss,terms=objective(pred,y,m.normalize(x),w,m)
  if len(curve)==step:curve.append(dict(step=step,objective=scalar(loss),**{k:scalar(v) for k,v in terms.items()}))
  path=out/('checkpoint_'+str(step).zfill(6)+'.pt');payload=dict(format_version=FORMAT,parent_sha256=PARENT_SHA,step=step,state_dict=m.state_dict(),identity=m.identity(),optimizer=opt.state_dict(),curve=list(curve),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],source_binding_sha256=f['source_binding_sha256'],torch_rng_state=torch.get_rng_state(),replay_index=replay_index,replay=[v.clone() for v in replay],train_prediction=pred[0].clone(),scoped_engineering_only=True,full_neck_training_allowed=False,new_walking_verified=False,seed=20261112,lr=.001,batch_size=7,history_steps=1920,training_teacher='.8*explicitcue_OFFLINE_ONLY',loss_variant='weightedcueaction+.01reconstruction+.001DNactivity+.1decoderzero')
  with path.open('xb') as stream:torch.save(payload,stream)
  print(json.dumps(dict(saved=str(path),step=step,sha256=sha(path))),flush=True)
 save_step(0)
 for step in range(1,401):
  opt.zero_grad(set_to_none=True);pred=scan(x,cue);loss,terms=objective(pred,y,m.normalize(x),w,m);loss.backward()
  if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Nonfinite or absent allparametergradient')
  grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();curve.append(dict(step=step,objective_before_update=scalar(loss),gradient_norm=grad,**{k:scalar(v) for k,v in terms.items()}));del pred,loss,terms
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
  if time.monotonic()-start>1800 or rss>1800:raise RuntimeError('Fixed400 wall/RSS exceeded; no batchreduction or extra updates')
  if step%25==0:print(json.dumps(dict(step=step,seconds=time.monotonic()-start,peak_RSS_MiB=rss)),flush=True)
  if step in (200,400):save_step(step)
 final=out/'checkpoint_000400.pt';verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FIXED400_AGGREGATION_REPAIR_TRAINED_NOT_PHYSICAL_VALIDATED',updates=400,checkpoint=str(final),sha256=sha(final),parent_sha256=PARENT_SHA,seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,batch=7,full_neck_training_allowed=False,new_body_physics_steps=0))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-aggregation')
