"""Parent finite200 full1920 batch4 calibration only; not new walking or physiology."""
from common import *
def main():
 torch,np=backend();f=frozen();out=HERE/'TRAIN';out.mkdir(exist_ok=False)
 from data import arrays,tensors
 from model import NeuralBridge,make_scan,objective,python_rollout
 import time,resource
 d=arrays('train');x,cue,y=tensors(d);torch.manual_seed(20261112);m=NeuralBridge(torch.from_numpy(d['mean']),torch.from_numpy(d['std']));scan=make_scan(m);opt=torch.optim.Adam(m.parameters(),lr=.001);curve=[];start=time.monotonic();replay_index=2
 def scalar(v):return float(v.detach())
 def save_step(step):
  with torch.no_grad():
   replay=python_rollout(m,x[:,replay_index:replay_index+1],cue[:,replay_index:replay_index+1]);pred=scan(x,cue);loss,terms=objective(pred,y,m.normalize(x))
  curve.append(dict(step=step,objective=scalar(loss),**{k:scalar(v) for k,v in terms.items()})) if len(curve)==step else None
  p=out/('checkpoint_'+str(step).zfill(6)+'.pt')
  payload=dict(format_version=FORMAT,step=step,state_dict=m.state_dict(),identity=m.identity(),optimizer=opt.state_dict(),curve=list(curve),source_hashes=f['source_hashes'],data_hashes=f['data_hashes'],source_binding_sha256=f['source_binding_sha256'],torch_rng_state=torch.get_rng_state(),replay_index=replay_index,replay=[v.clone() for v in replay],train_prediction=pred[0].clone(),scoped_engineering_only=True,full_neck_training_allowed=False,new_walking_verified=False,seed=20261112,lr=.001,batch_size=4,history_steps=1920)
  with p.open('xb') as stream:torch.save(payload,stream)
  print(json.dumps(dict(saved=str(p),step=step,sha256=sha(p))),flush=True)
  return payload
 save_step(0)
 for step in range(1,201):
  opt.zero_grad(set_to_none=True);pred=scan(x,cue);loss,terms=objective(pred,y,m.normalize(x));loss.backward()
  if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Nonfinite or absent parameter gradient')
  grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step();curve.append(dict(step=step,objective_before_update=scalar(loss),gradient_norm=grad,**{k:scalar(v) for k,v in terms.items()}));del pred,loss,terms
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
  if time.monotonic()-start>1800 or rss>1800:raise RuntimeError('Fixed wall/RSS budget exceeded; no automatic fallback')
  if step%25==0:print(json.dumps(dict(step=step,peak_RSS_MiB=rss,seconds=time.monotonic()-start)),flush=True)
  if step in (100,200):save_step(step)
 final=out/'checkpoint_000200.pt';verify(f['source_hashes']);verify(f['data_hashes']);save(out/'RESULT.json',dict(status='FIXED200_ENGINEERING_OFFLINE_TRAIN_COMPLETE_NOT_BODY',updates=200,checkpoint=str(final),sha256=sha(final),seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,all_parameters_optimized=len(list(m.parameters())),full_neck_training_allowed=False,new_body_physics_steps=0,motor_model_loads=0))
if __name__=='__main__':raise SystemExit('Use run.py --execute-parent-calibration')
